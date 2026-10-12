"""HTTP client for open-source ComfyUI (``/api/*`` routes).

Polling only: the dstack proxy rejects websocket upgrades, so progress is read
from ``/queue`` and ``/history``.  The same client works against a ComfyUI on
the LAN (e.g. a DGX Spark) by pointing ``base_url`` at it.

Pitfalls handled here, all measured in videogen-n8n:

* Uploads always go to the input root (empty ``subfolder``): files written in
  a subfolder are saved and answered with 200 but never appear in
  ``LoadImage``/``LoadAudio`` choices, so the graph fails validation later.
* The server may rename an upload (``name (1).ext``); the returned ``name`` is
  the one the graph must reference.
* ``/prompt`` can answer 200 with ``node_errors``; that is a rejection.
* ``/view`` redirects are followed manually and without the auth header so a
  bearer token never leaks to a storage host.
"""
from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import requests

from inference.comfy.states import ComfyFile, PromptState, StateDecision, decide_state


class ComfyError(RuntimeError):
    pass


class ComfyRejectedError(ComfyError):
    """``/prompt`` refused the graph (validation or node errors)."""


@dataclass(frozen=True)
class ComfyTimeouts:
    upload: float = 120.0
    submit: float = 120.0
    poll: float = 60.0
    download: float = 600.0


@dataclass(frozen=True)
class PollResult:
    decision: StateDecision
    entry: Mapping[str, Any] | None


def _detail(response: requests.Response) -> str:
    try:
        return response.text[:500]
    except Exception:
        return ""


class ComfyClient:
    def __init__(
        self,
        base_url: str,
        *,
        token: str | None = None,
        timeouts: ComfyTimeouts | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self._token = token
        self._timeouts = timeouts or ComfyTimeouts()
        self._session = session or requests.Session()

    # ── helpers ────────────────────────────────────────────────────────────

    def _url(self, route: str) -> str:
        return urljoin(self.base_url, route.lstrip("/"))

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    def _get_json(self, route: str, *, params: Mapping[str, Any] | None = None) -> Any:
        response = self._session.get(
            self._url(route),
            params=params,
            headers=self._headers(),
            timeout=self._timeouts.poll,
        )
        if not response.ok:
            raise ComfyError(f"GET {route} failed: HTTP {response.status_code} {_detail(response)}")
        return response.json()

    def _post_json(self, route: str, body: Any, *, timeout: float) -> requests.Response:
        response = self._session.post(
            self._url(route),
            data=json.dumps(body),
            headers={**self._headers(), "Content-Type": "application/json"},
            timeout=timeout,
        )
        if not response.ok:
            raise ComfyError(f"POST {route} failed: HTTP {response.status_code} {_detail(response)}")
        return response

    # ── operations ─────────────────────────────────────────────────────────

    def system_stats(self) -> Mapping[str, Any]:
        return self._get_json("api/system_stats")

    def object_info(self) -> Mapping[str, Any]:
        return self._get_json("api/object_info")

    def upload(self, local_path: str, remote_name: str) -> str:
        """Upload an input file to the input root; return the server-side name."""
        with open(local_path, "rb") as handle:
            response = self._session.post(
                self._url("api/upload/image"),
                files={"image": (remote_name, handle, "application/octet-stream")},
                data={"subfolder": "", "type": "input"},
                headers=self._headers(),
                timeout=self._timeouts.upload,
            )
        if not response.ok:
            raise ComfyError(
                f"Upload of {remote_name} failed: HTTP {response.status_code} {_detail(response)}"
            )
        name = (response.json() or {}).get("name")
        if not isinstance(name, str) or not name:
            raise ComfyError(f"Upload of {remote_name} returned no name")
        return name

    def submit(self, graph: Mapping[str, Any], *, client_id: str | None = None) -> str:
        body: dict[str, Any] = {"prompt": graph}
        if client_id:
            body["client_id"] = client_id
        response = self._session.post(
            self._url("api/prompt"),
            data=json.dumps(body),
            headers={**self._headers(), "Content-Type": "application/json"},
            timeout=self._timeouts.submit,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if not response.ok:
            raise ComfyRejectedError(
                f"ComfyUI rejected the graph: HTTP {response.status_code} "
                f"{json.dumps(payload)[:1000] if payload is not None else _detail(response)}"
            )
        if not isinstance(payload, Mapping):
            raise ComfyError("ComfyUI /prompt returned a non-JSON body")
        node_errors = payload.get("node_errors")
        if node_errors:
            raise ComfyRejectedError(
                f"ComfyUI reported node errors: {json.dumps(node_errors)[:1000]}"
            )
        prompt_id = payload.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise ComfyError(f"ComfyUI /prompt returned no prompt_id: {json.dumps(payload)[:500]}")
        return prompt_id

    def poll(self, prompt_id: str) -> PollResult:
        """One status read. ``/queue`` is read before ``/history`` on purpose."""
        queue = self._get_json("api/queue")
        history = self._get_json(f"api/history/{prompt_id}")
        decision = decide_state(prompt_id, history, queue)
        entry = history.get(prompt_id) if isinstance(history, Mapping) else None
        return PollResult(decision, entry if isinstance(entry, Mapping) else None)

    def download(self, file: ComfyFile, dest_path: str) -> str:
        """Stream an output to ``dest_path`` via a ``.part`` file."""
        params = {"filename": file.filename, "subfolder": file.subfolder, "type": file.type}
        response = self._session.get(
            self._url("api/view"),
            params=params,
            headers=self._headers(),
            timeout=self._timeouts.download,
            stream=True,
            allow_redirects=False,
        )
        if response.is_redirect or response.status_code in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise ComfyError(f"Download of {file.filename} redirected without Location")
            # No auth header on the redirect target.
            response = self._session.get(
                urljoin(self.base_url, location),
                timeout=self._timeouts.download,
                stream=True,
                allow_redirects=True,
            )
        if not response.ok:
            raise ComfyError(
                f"Download of {file.filename} failed: HTTP {response.status_code} {_detail(response)}"
            )
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        part_path = dest_path + ".part"
        try:
            with open(part_path, "wb") as out:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        out.write(chunk)
            if os.path.getsize(part_path) == 0:
                raise ComfyError(f"Download of {file.filename} produced an empty file")
            os.replace(part_path, dest_path)
        finally:
            response.close()
            if os.path.exists(part_path):
                os.remove(part_path)
        return dest_path

    def cancel(
        self,
        prompt_id: str,
        *,
        confirm_attempts: int = 5,
        confirm_interval: float = 1.0,
    ) -> PromptState:
        """Cancel ``prompt_id`` and verify it actually stopped.

        A pending prompt is removed from the queue.  A running prompt is
        interrupted; ``prompt_id`` is passed so recent ComfyUI versions only
        interrupt that prompt (older ones ignore the body, which is safe with
        one job per endpoint).
        """
        before = self.poll(prompt_id).decision
        if before.state is not PromptState.RUNNING:
            return before.state
        if before.raw == "queue_pending":
            self._post_json("api/queue", {"delete": [prompt_id]}, timeout=self._timeouts.poll)
        else:
            self._post_json("api/interrupt", {"prompt_id": prompt_id}, timeout=self._timeouts.poll)

        after = before
        for attempt in range(max(1, confirm_attempts)):
            after = self.poll(prompt_id).decision
            if after.state is not PromptState.RUNNING:
                return after.state
            if attempt + 1 < confirm_attempts:
                time.sleep(confirm_interval)
        raise ComfyError(
            f"Cancel of {prompt_id} had no effect; prompt still {after.raw or 'running'} "
            "and the remote GPU keeps working"
        )
