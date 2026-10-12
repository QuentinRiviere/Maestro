"""``InferenceBackend`` that runs a task on a remote ComfyUI."""
from __future__ import annotations

import logging
import os
import re
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from typing import Any

from inference.comfy.client import ComfyClient, ComfyError
from inference.comfy.recipe import ComfyRecipe, recipe_matches, validate_graph
from inference.comfy.states import PromptState, extract_error, extract_outputs
from inference.contract import (
    CMD_GENERATION_TIME,
    CMD_PROGRESS,
    SendCmd,
    TaskRequest,
)


log = logging.getLogger("maestro.inference.comfy")

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


class ComfyBackend:
    """Uploads inputs, submits the recipe graph, polls, downloads outputs.

    Contract details (see ``inference/contract.py``):

    * outputs are written under ``save_dir()`` and appended to
      ``state["gen"]["file_list"]``;
    * ``state["gen"]["abort"]`` is checked between every remote call and
      triggers a verified remote cancel;
    * a prompt the server lost (restart, history eviction) is resubmitted at
      most once, within the same deadline.
    """

    def __init__(
        self,
        name: str,
        client: ComfyClient,
        recipes: Sequence[ComfyRecipe],
        *,
        save_dir: Callable[[], str],
        poll_interval: float = 2.0,
        deadline_seconds: float = 45 * 60,
        max_poll_failures: int = 5,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.name = name
        self._client = client
        self._recipes = list(recipes)
        self._save_dir = save_dir
        self._poll_interval = poll_interval
        self._deadline_seconds = deadline_seconds
        self._max_poll_failures = max_poll_failures
        self._clock = clock
        self._sleep = sleep
        self._available_node_types: frozenset[str] | None = None

    # ── public ─────────────────────────────────────────────────────────────

    def run_task(self, request: TaskRequest, send_cmd: SendCmd) -> None:
        recipe = self._recipe_for(request.model_type)
        tag = uuid.uuid4().hex[:8]

        send_cmd(CMD_PROGRESS, [0, f"ComfyUI ({self.name}): uploading inputs"])
        uploaded: dict[str, str] = {}
        for item in recipe.collect_inputs(request):
            if request.abort_requested():
                return
            uploaded[item.role] = self._client.upload(
                item.local_path,
                self._remote_input_name(tag, item.role, item.local_path),
            )

        graph = recipe.build_graph(request, uploaded)
        validate_graph(
            graph,
            allowed_node_types=recipe.allowed_node_types,
            output_node=recipe.output_node,
            available_node_types=self._node_types(),
        )
        if request.abort_requested():
            return

        started = self._clock()
        prompt_id = self._client.submit(graph, client_id=f"maestro-{tag}")
        log.info("comfy %s submitted %s for %s", self.name, prompt_id, request.model_type)
        resubmitted = False
        poll_failures = 0
        last_raw: str | None = None

        while True:
            if request.abort_requested():
                self._cancel_quietly(prompt_id)
                return
            if self._clock() - started > self._deadline_seconds:
                self._cancel_quietly(prompt_id)
                raise ComfyError(
                    f"ComfyUI prompt {prompt_id} exceeded {int(self._deadline_seconds)}s; cancelled"
                )

            try:
                result = self._client.poll(prompt_id)
                poll_failures = 0
            except Exception as exc:
                poll_failures += 1
                log.warning("comfy %s poll %s failed (%d): %s", self.name, prompt_id, poll_failures, exc)
                if poll_failures >= self._max_poll_failures:
                    raise ComfyError(
                        f"ComfyUI unreachable while polling {prompt_id}: {exc}"
                    ) from exc
                self._sleep(self._poll_interval)
                continue

            state = result.decision.state
            if state is PromptState.RUNNING:
                if result.decision.raw != last_raw:
                    last_raw = result.decision.raw
                    label = "queued" if last_raw == "queue_pending" else "running"
                    send_cmd(CMD_PROGRESS, [0, f"ComfyUI ({self.name}): {label}"])
                self._sleep(self._poll_interval)
                continue

            if state is PromptState.SUCCEEDED:
                files = extract_outputs(result.entry or {}, recipe.output_node)
                if not files:
                    raise ComfyError(
                        f"ComfyUI prompt {prompt_id} succeeded but node "
                        f"{recipe.output_node} produced no file"
                    )
                send_cmd(CMD_PROGRESS, [0, f"ComfyUI ({self.name}): downloading output"])
                paths = [self._download(file, tag, index) for index, file in enumerate(files)]
                self._register_outputs(request, paths)
                send_cmd(CMD_GENERATION_TIME, {
                    "seconds": self._clock() - started,
                    "outputs": paths,
                })
                return

            if state is PromptState.FAILED:
                raise ComfyError(extract_error(result.entry or {}))

            if state is PromptState.CANCELLED:
                if request.abort_requested():
                    return
                raise ComfyError(f"ComfyUI prompt {prompt_id} was interrupted on the server")

            # LOST: the server no longer knows the prompt.
            if resubmitted:
                raise ComfyError(f"ComfyUI lost prompt {prompt_id} twice; giving up")
            resubmitted = True
            log.warning("comfy %s lost %s; resubmitting once", self.name, prompt_id)
            send_cmd(CMD_PROGRESS, [0, f"ComfyUI ({self.name}): job lost, resubmitting"])
            prompt_id = self._client.submit(graph, client_id=f"maestro-{tag}")

    # ── internals ──────────────────────────────────────────────────────────

    def _recipe_for(self, model_type: str) -> ComfyRecipe:
        for recipe in self._recipes:
            if recipe_matches(recipe, model_type):
                return recipe
        raise ComfyError(
            f"No ComfyUI recipe for model '{model_type}' on backend '{self.name}'"
        )

    def _node_types(self) -> frozenset[str] | None:
        if self._available_node_types is None:
            try:
                self._available_node_types = frozenset(self._client.object_info().keys())
            except Exception as exc:
                # Not fatal: /prompt validates class types too, only later.
                log.warning("comfy %s object_info unavailable: %s", self.name, exc)
                return None
        return self._available_node_types

    @staticmethod
    def _remote_input_name(tag: str, role: str, local_path: str) -> str:
        base = os.path.basename(local_path)
        return _UNSAFE_NAME.sub("_", f"maestro_{tag}_{role}_{base}")

    def _download(self, file: Any, tag: str, index: int) -> str:
        ext = os.path.splitext(file.filename)[1] or ".bin"
        stamp = datetime.now().strftime("%Y-%m-%d-%Hh%Mm%Ss")
        suffix = f"_{index}" if index else ""
        dest = os.path.join(self._save_dir(), f"{stamp}_comfy_{tag}{suffix}{ext}")
        return self._client.download(file, dest)

    @staticmethod
    def _register_outputs(request: TaskRequest, paths: list[str]) -> None:
        gen = request.state.get("gen")
        if not isinstance(gen, dict):
            return
        audio_exts = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg"}
        for path in paths:
            is_audio = os.path.splitext(path)[1].lower() in audio_exts
            gen.setdefault("audio_file_list" if is_audio else "file_list", []).append(path)
            gen["last_was_audio"] = is_audio

    def _cancel_quietly(self, prompt_id: str) -> None:
        try:
            self._client.cancel(prompt_id)
        except Exception as exc:
            log.error("comfy %s cancel of %s failed: %s", self.name, prompt_id, exc)


def build_comfy_backend(
    name: str,
    config: Mapping[str, Any],
    recipes: Sequence[ComfyRecipe],
    *,
    save_dir: Callable[[], str],
) -> ComfyBackend:
    """Build a backend from a ``server_config["inference"]["backends"][name]`` entry.

    Example::

        {"type": "comfy",
         "base_url": "http://127.0.0.1:3000/proxy/services/main/maestro-comfy/",
         "token_env": "DSTACK_SERVICE_TOKEN",
         "poll_interval": 5, "deadline_seconds": 2700}

    The token is read from the environment, never stored in server_config.
    """
    from inference.comfy.client import ComfyTimeouts

    base_url = config.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise ValueError(f"inference backend '{name}': base_url is required")
    token_env = config.get("token_env")
    token = os.environ.get(str(token_env)) if token_env else None
    timeouts = ComfyTimeouts(**{
        key: float(value)
        for key, value in (config.get("timeouts") or {}).items()
        if key in {"upload", "submit", "poll", "download"}
    })
    client = ComfyClient(base_url, token=token, timeouts=timeouts)
    return ComfyBackend(
        name,
        client,
        recipes,
        save_dir=save_dir,
        poll_interval=float(config.get("poll_interval", 2.0)),
        deadline_seconds=float(config.get("deadline_seconds", 45 * 60)),
    )
