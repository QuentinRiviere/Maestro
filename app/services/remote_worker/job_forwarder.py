"""Run one prepared Maestro generation job on the remote worker.

The local Maestro keeps everything that is not GPU work: queueing, prompt
enhancement and window planning (already applied to ``params`` by the time
this runs), the gallery, sidecars and notifications. The worker is Maestro
itself, so the job runs there through the very same ``_run_generation``.

Steps: make sure the worker is ready (auto-start), preload the model through
the worker's download route (R2 cache aware), transfer every local file the
params reference plus the LoRAs, submit, mirror progress and cancellation,
then bring the outputs and their sidecars into the local output folder.

With R2 configured, inputs go controller -> R2 -> worker (content-addressed,
uploaded once) and outputs worker -> R2 -> controller, where they stay as a
remote copy. Without R2, files travel through the dstack service proxy.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import requests

from services.remote_worker.manager import RemoteWorkerManager
from services.lora_r2 import read_pointer
from services.remote_worker.r2_store import R2Store


log = logging.getLogger("maestro.remote_worker.forwarder")

_TERMINAL = {"completed", "failed", "cancelled"}
# Job fields mirrored from the remote job onto the local one while it runs.
_MIRRORED_FIELDS = (
    "progress", "step", "total_steps", "phase", "message",
    "eta_seconds", "eta_confidence", "eta_label",
)
# Params whose values are free text, never file paths.
_TEXT_KEYS = {"prompt", "negative_prompt", "alt_prompt", "multi_prompts", "prompt_enhancer"}


class RemoteJobError(RuntimeError):
    pass


class RemoteJobCancelled(RuntimeError):
    pass


@dataclass
class RemoteJobResult:
    output_files: list[str]
    clip_output_files: dict[str, str] = field(default_factory=dict)
    join_output_file: str | None = None
    # Local output name -> R2 key of its retained remote copy.
    r2_objects: dict[str, str] = field(default_factory=dict)


class WorkerClient:
    """Bearer-authenticated calls to a worker behind the dstack service proxy."""

    def __init__(self, base_url: str, token: str, *, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self._headers = {"Authorization": f"Bearer {token}"}
        self._session = session or requests.Session()

    def _url(self, path: str) -> str:
        return self.base_url + path.lstrip("/")

    def get_json(self, path: str, *, timeout: float = 30.0, params: Mapping[str, Any] | None = None) -> Any:
        response = self._session.get(self._url(path), headers=self._headers, timeout=timeout, params=params)
        if response.status_code == 404:
            return None
        if not response.ok:
            raise RemoteJobError(f"Worker GET {path} failed: HTTP {response.status_code} {response.text[:300]}")
        return response.json()

    def post_json(self, path: str, body: Any = None, *, timeout: float = 60.0) -> Any:
        response = self._session.post(self._url(path), headers=self._headers, json=body, timeout=timeout)
        if not response.ok:
            raise RemoteJobError(f"Worker POST {path} failed: HTTP {response.status_code} {response.text[:300]}")
        return response.json() if response.content else None

    def upload_file(self, local_path: str, *, kind: str, sha256: str, model_type: str = "") -> str:
        params = {"kind": kind, "name": os.path.basename(local_path), "sha256": sha256, "model_type": model_type}
        existing = self.get_json("api/v1/worker/files", params=params)
        if existing and existing.get("path"):
            return str(existing["path"])
        with open(local_path, "rb") as handle:
            response = self._session.post(
                self._url("api/v1/worker/files"),
                headers={**self._headers, "Content-Type": "application/octet-stream"},
                params=params,
                data=handle,
                timeout=600,
            )
        if not response.ok:
            raise RemoteJobError(
                f"Upload of {os.path.basename(local_path)} failed: HTTP {response.status_code} {response.text[:300]}"
            )
        return str(response.json()["path"])

    def fetch_from_r2(self, *, kind: str, name: str, sha256: str, model_type: str = "") -> str | None:
        """Ask the worker to pull an input from R2; None when it has no R2."""
        response = self._session.post(
            self._url("api/v1/worker/files/fetch"),
            headers=self._headers,
            json={"kind": kind, "name": name, "sha256": sha256, "model_type": model_type},
            timeout=600,
        )
        if response.status_code == 409:
            return None
        if not response.ok:
            raise RemoteJobError(f"Worker R2 fetch of {name} failed: HTTP {response.status_code} {response.text[:300]}")
        return str(response.json()["path"])

    def get_bytes(self, path: str, *, timeout: float = 30.0) -> tuple[bytes, str] | None:
        """Small in-memory GET (previews); None when the worker no longer has it."""
        response = self._session.get(self._url(path), headers=self._headers, timeout=timeout)
        if response.status_code == 404:
            return None
        if not response.ok:
            raise RemoteJobError(f"Worker GET {path} failed: HTTP {response.status_code}")
        return response.content, response.headers.get("Content-Type", "").split(";")[0].strip()

    def download(self, path: str, dest: str) -> None:
        part = dest + ".part"
        with self._session.get(self._url(path), headers=self._headers, timeout=600, stream=True) as response:
            if not response.ok:
                raise RemoteJobError(f"Download {path} failed: HTTP {response.status_code}")
            with open(part, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
        os.replace(part, dest)


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_local_name(out_dir: str, name: str) -> str:
    stem, ext = os.path.splitext(name)
    candidate, index = name, 1
    while os.path.exists(os.path.join(out_dir, candidate)):
        candidate = f"{stem}_{index}{ext}"
        index += 1
    return candidate


class RemoteJobForwarder:
    def __init__(
        self,
        manager: RemoteWorkerManager,
        *,
        client_factory: Callable[[str, str], WorkerClient] = WorkerClient,
        store_factory: Callable[[Any], R2Store | None] = R2Store.from_settings,
        poll_interval: float = 2.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._manager = manager
        self._client_factory = client_factory
        self._store_factory = store_factory
        self._poll_interval = poll_interval
        self._sleep = sleep

    def run(
        self,
        params: Mapping[str, Any],
        *,
        job_id: str = "",
        out_dir: str,
        resolve_lora: Callable[[str, str], str | None],
        update: Callable[..., Any],
        is_cancelled: Callable[[], bool],
        preview_mode: str | None = None,
        on_preview: Callable[[bytes, str, Mapping[str, Any]], Any] | None = None,
        on_preview_notice: Callable[[str | None], Any] | None = None,
    ) -> RemoteJobResult:
        """Run one prepared job remotely.

        ``preview_mode`` is the controller's Generation Preview preference;
        the worker renders previews in that mode and ``on_preview`` receives
        each new one (bytes, content type, the worker's preview descriptor).
        """
        model_type = str(params.get("model_type") or "")
        self._manager.job_started()
        try:
            update(phase="Remote worker", message="Waiting for the remote GPU worker…")
            status = self._manager.ensure_ready(
                model_type,
                should_abort=is_cancelled,
                on_update=lambda s: update(
                    phase="Remote worker",
                    message=_worker_status_message(s),
                ),
            )
            if is_cancelled():
                raise RemoteJobCancelled()
            client = self._client_factory(str(status["service_url"]), self._manager.settings.token)
            store = self._store_factory(self._manager.settings)

            self._preload(client, model_type, update=update, is_cancelled=is_cancelled)
            remote_params = self._transfer_inputs(
                client, store, params, model_type, resolve_lora=resolve_lora, update=update,
            )
            update(phase="Remote worker", message="Submitting to the remote worker…")
            body: dict[str, Any] = {"params": remote_params}
            if preview_mode is not None:
                body["preview_mode"] = preview_mode
            remote_id = str(client.post_json("api/v1/worker/jobs", body)["job_id"])
            remote = self._follow(
                client, remote_id, update=update, is_cancelled=is_cancelled,
                on_preview=on_preview, on_preview_notice=on_preview_notice,
            )
            return self._fetch_outputs(
                client, store, remote_id, remote, job_id=job_id or remote_id, out_dir=out_dir, update=update,
            )
        finally:
            self._manager.job_finished()

    # ── steps ──────────────────────────────────────────────────────────────

    def _preload(self, client: WorkerClient, model_type: str, *, update, is_cancelled) -> None:
        if not model_type:
            return
        downloads = (client.get_json("api/v1/models/downloads/status") or {}).get("downloads") or {}
        if (downloads.get(model_type) or {}).get("status") == "completed":
            return
        client.post_json(f"api/v1/models/{quote(model_type, safe='')}/download")
        while True:
            if is_cancelled():
                raise RemoteJobCancelled()
            entry = ((client.get_json("api/v1/models/downloads/status") or {}).get("downloads") or {}).get(model_type) or {}
            state = entry.get("status")
            if state == "completed":
                return
            if state in ("failed", "cancelled"):
                raise RemoteJobError(f"Remote worker could not fetch {model_type}: {entry.get('error') or state}")
            phase = {
                "pulling_cache": "Pulling weights from the R2 cache…",
                "publishing_cache": "Saving new weights to the R2 cache…",
            }.get(entry.get("phase") or "", "Downloading model weights on the worker…")
            update(phase="Remote worker", message=phase)
            self._sleep(max(self._poll_interval, 3.0))

    def _send_file(self, client, store, path: str, *, kind: str, model_type: str = "", update) -> str:
        """Put one local file on the worker, through R2 when available."""
        name = os.path.basename(path)
        sha = _sha256(path)
        if store is not None:
            key = store.asset_key(sha, name)
            if not store.exists(key):
                update(phase="Remote worker", message=f"Uploading {name} to R2…")
                store.upload(path, key, skip_if_exists=False)
            remote = client.fetch_from_r2(kind=kind, name=name, sha256=sha, model_type=model_type)
            if remote is not None:
                return remote
        update(phase="Remote worker", message=f"Uploading {name}…")
        return client.upload_file(path, kind=kind, sha256=sha, model_type=model_type)

    def _transfer_inputs(self, client, store, params, model_type, *, resolve_lora, update) -> dict[str, Any]:
        uploaded: dict[str, str] = {}

        def upload(path: str) -> str:
            real = os.path.realpath(path)
            if real not in uploaded:
                uploaded[real] = self._send_file(client, store, real, kind="input", update=update)
            return uploaded[real]

        def rewrite(value: Any, key: str = "") -> Any:
            if key in _TEXT_KEYS:
                return value
            if isinstance(value, str) and value and os.path.isfile(value):
                return upload(value)
            if isinstance(value, list):
                return [rewrite(item, key) for item in value]
            if isinstance(value, dict):
                return {k: rewrite(v, str(k)) for k, v in value.items()}
            return value

        remote_params = {k: rewrite(v, str(k)) for k, v in params.items()}
        remote_params.pop("_defer_output_publication", None)

        for name in params.get("activated_loras") or []:
            if not isinstance(name, str) or name.startswith(("http://", "https://")):
                continue  # URLs are downloaded by the worker itself.
            local = resolve_lora(model_type, name)
            if local and os.path.isfile(local):
                self._send_file(client, store, local, kind="lora", model_type=model_type, update=update)
                continue
            pointer = read_pointer(local) if local else None
            if pointer is not None:
                # Kept only in R2: the worker pulls the same content-addressed asset.
                update(phase="Remote worker", message=f"Worker fetching {name} from R2…")
                if client.fetch_from_r2(kind="lora", name=os.path.basename(local),
                                        sha256=pointer["sha256"], model_type=model_type) is None:
                    raise RemoteJobError(f"LoRA {name} is stored in R2 but the worker has no R2 access")
        return remote_params

    def _follow(self, client, remote_id: str, *, update, is_cancelled,
                on_preview=None, on_preview_notice=None) -> Mapping[str, Any]:
        cancel_sent_at: float | None = None
        missing = 0
        relayed_revision: Any = None
        relayed_notice: Any = None
        while True:
            if is_cancelled() and cancel_sent_at is None:
                try:
                    client.post_json(f"api/v1/cancel/{remote_id}")
                except RemoteJobError as exc:
                    log.warning("remote cancel of %s failed: %s", remote_id, exc)
                cancel_sent_at = time.monotonic()
            try:
                remote = client.get_json(f"api/v1/worker/jobs/{remote_id}")
            except (RemoteJobError, requests.RequestException) as exc:
                missing += 1
                if missing >= 10:
                    raise RemoteJobError(f"Lost contact with the remote worker: {exc}") from exc
                self._sleep(self._poll_interval)
                continue
            if remote is None:
                raise RemoteJobError("The remote worker no longer knows this job (worker restarted?)")
            missing = 0
            status = remote.get("status")
            if status in _TERMINAL:
                if status == "cancelled" or cancel_sent_at is not None:
                    raise RemoteJobCancelled()
                if status == "failed":
                    raise RemoteJobError(str(remote.get("error") or remote.get("message") or "Remote generation failed"))
                return remote
            if cancel_sent_at is not None and time.monotonic() - cancel_sent_at > 120:
                raise RemoteJobCancelled()
            if cancel_sent_at is None:
                update(**{k: remote[k] for k in _MIRRORED_FIELDS if k in remote})
                relayed_revision = self._relay_preview(client, remote, relayed_revision, on_preview)
                notice = remote.get("preview_notice")
                if on_preview_notice is not None and notice != relayed_notice:
                    on_preview_notice(notice)
                    relayed_notice = notice
            self._sleep(self._poll_interval)

    @staticmethod
    def _relay_preview(client, remote: Mapping[str, Any], last_revision: Any, on_preview) -> Any:
        """Forward the worker's newest preview once; preview failures never fail the job."""
        preview = remote.get("preview")
        if on_preview is None or not isinstance(preview, Mapping) or not preview.get("url"):
            return last_revision
        revision = preview.get("revision")
        if revision == last_revision:
            return last_revision
        try:
            fetched = client.get_bytes(str(preview["url"]))
            if fetched is not None:
                on_preview(fetched[0], fetched[1], preview)
        except Exception as exc:
            log.warning("remote preview relay failed: %s", exc)
        return revision

    def _fetch_outputs(self, client, store, remote_id, remote, *, job_id, out_dir, update) -> RemoteJobResult:
        os.makedirs(out_dir, exist_ok=True)
        objects: dict[str, str] = {}
        if store is not None:
            update(phase="Remote worker", message="Saving outputs to R2…")
            try:
                published = client.post_json(
                    f"api/v1/worker/jobs/{remote_id}/publish", {"job_id": job_id}, timeout=900,
                )
                objects = dict((published or {}).get("objects") or {})
            except RemoteJobError as exc:
                log.warning("R2 publish of %s failed, using the worker proxy: %s", remote_id, exc)

        def fetch(name: str, dest: str) -> None:
            key = objects.get(name)
            if store is not None and key:
                store.download(key, dest)
            else:
                client.download(f"api/v1/worker/jobs/{remote_id}/files/{quote(name)}", dest)

        renamed: dict[str, str] = {}
        for name in remote.get("files") or []:
            update(phase="Remote worker", message=f"Downloading {name}…")
            local_name = _unique_local_name(out_dir, name)
            fetch(name, os.path.join(out_dir, local_name))
            renamed[name] = local_name
            sidecar = f"{os.path.splitext(name)[0]}.json"
            if sidecar in (remote.get("sidecars") or []):
                fetch(sidecar, os.path.join(out_dir, f"{os.path.splitext(local_name)[0]}.json"))
        output_files = [renamed[n] for n in remote.get("output_files") or [] if n in renamed]
        clip_files = {
            str(k): renamed[v] for k, v in (remote.get("clip_output_files") or {}).items() if v in renamed
        }
        join = remote.get("join_output_file")
        return RemoteJobResult(
            output_files=output_files or list(renamed.values()),
            clip_output_files=clip_files,
            join_output_file=renamed.get(join) if join else None,
            r2_objects={renamed[n]: k for n, k in objects.items() if n in renamed},
        )


def _worker_status_message(status: Mapping[str, Any]) -> str:
    state = status.get("state")
    if state == "starting":
        pull = status.get("image_pull_percent")
        return f"Starting remote GPU worker… (image {pull}%)" if pull is not None else "Starting remote GPU worker…"
    if state == "booting":
        return "Remote worker booting Maestro…"
    if state == "stopping":
        return "Waiting for the previous worker to stop…"
    if state == "ready":
        return "Remote worker ready"
    return "Starting remote GPU worker…"
