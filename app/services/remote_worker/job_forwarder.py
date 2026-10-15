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
# Worker job key listing the clips a running multi-clip job finished so far.
_LIVE_CLIPS_KEY = "live_clip_output_files"
# Attempts at fetching one finished clip mid-job before leaving it to the end.
_LIVE_FETCH_ATTEMPTS = 3
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
    # What the pod billed around this job (see ``_billing``), for cost reports.
    billing: dict[str, Any] = field(default_factory=dict)


# The dstack proxy answers these when it briefly cannot reach the worker; the
# request never got there, so sending it again is safe.
_PROXY_RETRY_STATUSES = {502, 503, 504}
_PROXY_RETRY_DELAYS = (2.0, 5.0, 10.0)


class WorkerClient:
    """Bearer-authenticated calls to a worker behind the dstack service proxy."""

    def __init__(self, base_url: str, token: str, *, session: requests.Session | None = None,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self._headers = {"Authorization": f"Bearer {token}"}
        self._session = session or requests.Session()
        self._sleep = sleep

    def _url(self, path: str) -> str:
        return self.base_url + path.lstrip("/")

    def _send(self, method: str, path: str, *, retry: bool = True, upload: str | None = None,
              **kwargs: Any) -> requests.Response:
        """One request, sent again on a proxy hiccup (502/503/504 or a dropped connection).

        ``retry=False`` for a request that must not run twice (a job
        submission). ``upload`` is a file path sent as the body, reopened on
        each attempt.
        """
        delays = _PROXY_RETRY_DELAYS if retry else ()
        for attempt in range(len(delays) + 1):
            try:
                if upload is not None:
                    with open(upload, "rb") as handle:
                        response = self._session.request(method, self._url(path), data=handle, **kwargs)
                else:
                    response = self._session.request(method, self._url(path), **kwargs)
            except requests.ConnectionError:
                if attempt >= len(delays):
                    raise
                log.warning("worker %s %s: connection dropped, retrying", method, path)
            else:
                if response.status_code not in _PROXY_RETRY_STATUSES or attempt >= len(delays):
                    return response
                log.warning("worker %s %s: HTTP %s from the proxy, retrying", method, path,
                            response.status_code)
                response.close()
            self._sleep(delays[attempt])
        raise AssertionError("unreachable")

    def get_json(self, path: str, *, timeout: float = 30.0, params: Mapping[str, Any] | None = None) -> Any:
        response = self._send("GET", path, headers=self._headers, timeout=timeout, params=params)
        if response.status_code == 404:
            return None
        if not response.ok:
            raise RemoteJobError(f"Worker GET {path} failed: HTTP {response.status_code} {response.text[:300]}")
        return response.json()

    def post_json(self, path: str, body: Any = None, *, timeout: float = 60.0, retry: bool = True) -> Any:
        response = self._send("POST", path, retry=retry, headers=self._headers, json=body, timeout=timeout)
        if not response.ok:
            raise RemoteJobError(f"Worker POST {path} failed: HTTP {response.status_code} {response.text[:300]}")
        return response.json() if response.content else None

    def upload_file(self, local_path: str, *, kind: str, sha256: str, model_type: str = "") -> str:
        params = {"kind": kind, "name": os.path.basename(local_path), "sha256": sha256, "model_type": model_type}
        existing = self.get_json("api/v1/worker/files", params=params)
        if existing and existing.get("path"):
            return str(existing["path"])
        response = self._send(
            "POST", "api/v1/worker/files", upload=local_path,
            headers={**self._headers, "Content-Type": "application/octet-stream"},
            params=params,
            timeout=600,
        )
        if not response.ok:
            raise RemoteJobError(
                f"Upload of {os.path.basename(local_path)} failed: HTTP {response.status_code} {response.text[:300]}"
            )
        return str(response.json()["path"])

    def fetch_from_r2(self, *, kind: str, name: str, sha256: str, model_type: str = "") -> str | None:
        """Ask the worker to pull an input from R2; None when it has no R2."""
        response = self._send(
            "POST", "api/v1/worker/files/fetch",
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
        response = self._send("GET", path, headers=self._headers, timeout=timeout)
        if response.status_code == 404:
            return None
        if not response.ok:
            raise RemoteJobError(f"Worker GET {path} failed: HTTP {response.status_code}")
        return response.content, response.headers.get("Content-Type", "").split(";")[0].strip()

    def download(self, path: str, dest: str) -> None:
        part = dest + ".part"
        with self._send("GET", path, headers=self._headers, timeout=600, stream=True) as response:
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
        pool: Any = None,
    ) -> None:
        self._manager = manager
        # With a pool, each job runs on the least busy of its pods.
        self._pool = pool
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
        on_clip_output: Callable[[str, str], Any] | None = None,
    ) -> RemoteJobResult:
        """Run one prepared job remotely.

        ``preview_mode`` is the controller's Generation Preview preference;
        the worker renders previews in that mode and ``on_preview`` receives
        each new one (bytes, content type, the worker's preview descriptor).
        ``on_clip_output`` receives (clip index, local file name) for each
        clip of a multi-clip job as soon as the worker finished it; the file
        is downloaded into ``out_dir`` mid-job and refreshed at the end.
        """
        model_type = str(params.get("model_type") or "")
        update(phase="Remote worker", message="Waiting for the remote GPU worker…")
        waiting_since = time.monotonic()

        def on_update(status: Mapping[str, Any]) -> None:
            update(phase="Remote worker",
                   message=_worker_status_message(status, waited=time.monotonic() - waiting_since))

        if self._pool is not None:
            # Queue for the first pod with room; the pool starts pods as the queue needs them.
            manager = self._pool.acquire(model_type, should_abort=is_cancelled, on_update=on_update)
            if manager is None:
                raise RemoteJobCancelled()
        else:
            manager = self._manager
            manager.job_started()
        try:
            status = manager.ensure_ready(
                model_type,
                should_abort=is_cancelled,
                on_update=on_update,
            )
            if is_cancelled():
                raise RemoteJobCancelled()
            client = self._client_factory(str(status["service_url"]), manager.settings.token)
            store = self._store_factory(manager.settings)

            self._preload(client, model_type, update=update, is_cancelled=is_cancelled)
            remote_params = self._transfer_inputs(
                client, store, params, model_type, resolve_lora=resolve_lora, update=update,
            )
            update(phase="Remote worker", message="Submitting to the remote worker…")
            body: dict[str, Any] = {"params": remote_params}
            if preview_mode is not None:
                body["preview_mode"] = preview_mode
            # Never sent twice: a retried submission could start the job twice.
            remote_id = str(client.post_json("api/v1/worker/jobs", body, retry=False)["job_id"])
            work_started = time.monotonic()
            live_files: dict[str, str] = {}
            remote = self._follow(
                client, remote_id, update=update, is_cancelled=is_cancelled,
                on_preview=on_preview, on_preview_notice=on_preview_notice,
                on_clip_output=on_clip_output, out_dir=out_dir, live_files=live_files,
            )
            result = self._fetch_outputs(
                client, store, remote_id, remote, job_id=job_id or remote_id, out_dir=out_dir, update=update,
                live_files=live_files,
            )
            result.billing = _billing(manager, status, waited=work_started - waiting_since,
                                      worked=time.monotonic() - work_started)
            return result
        finally:
            manager.job_finished()

    # ── steps ──────────────────────────────────────────────────────────────

    def _preload(self, client: WorkerClient, model_type: str, *, update, is_cancelled) -> None:
        if not model_type:
            return
        downloads = (client.get_json("api/v1/models/downloads/status") or {}).get("downloads") or {}
        if (downloads.get(model_type) or {}).get("status") == "completed":
            return
        client.post_json(f"api/v1/models/{quote(model_type, safe='')}/download")
        started = time.monotonic()
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
                "pulling_cache": f"Pulling {model_type} from the R2 cache",
                "publishing_cache": f"Saving {model_type} to the R2 cache",
            }.get(entry.get("phase") or "", f"Downloading {model_type} on the worker")
            phase += format_progress(entry.get("progress")) or "…"
            # The elapsed minutes are the job's heartbeat: callers that cancel
            # a job without visible progress (Director waits) must not mistake
            # a long first download or R2 upload for a stuck job.
            minutes = int((time.monotonic() - started) // 60)
            update(phase="Remote worker", message=f"{phase} ({minutes} min)" if minutes else phase)
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
                on_preview=None, on_preview_notice=None, on_clip_output=None,
                out_dir: str = "", live_files: dict[str, str] | None = None) -> Mapping[str, Any]:
        cancel_sent_at: float | None = None
        live_attempts: dict[str, int] = {}
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
                if on_clip_output is not None and live_files is not None and out_dir:
                    self._relay_live_clips(client, remote_id, remote, out_dir=out_dir,
                                           live_files=live_files, attempts=live_attempts,
                                           on_clip_output=on_clip_output)
            self._sleep(self._poll_interval)

    @staticmethod
    def _relay_live_clips(client, remote_id: str, remote: Mapping[str, Any], *, out_dir: str,
                          live_files: dict[str, str], attempts: dict[str, int],
                          on_clip_output) -> None:
        """Download each clip the worker finished since the last poll.

        Best effort: a failed fetch is retried on the next polls, then left
        to the final download. It never fails the job.
        """
        clips = remote.get(_LIVE_CLIPS_KEY)
        if not isinstance(clips, Mapping):
            return
        for index, name in sorted(clips.items(), key=lambda item: str(item[0])):
            if not isinstance(name, str) or not name or name in live_files:
                continue
            if attempts.get(name, 0) >= _LIVE_FETCH_ATTEMPTS:
                continue
            attempts[name] = attempts.get(name, 0) + 1
            local_name = _unique_local_name(out_dir, name)
            try:
                os.makedirs(out_dir, exist_ok=True)
                client.download(f"api/v1/worker/jobs/{remote_id}/files/{quote(name)}",
                                os.path.join(out_dir, local_name))
            except Exception as exc:  # noqa: BLE001 - a preview of a clip, never fatal
                log.warning("live fetch of clip %s (%s) failed: %s", index, name, exc)
                continue
            live_files[name] = local_name
            try:
                on_clip_output(str(index), local_name)
            except Exception as exc:  # noqa: BLE001
                log.warning("live clip callback for %s failed: %s", name, exc)

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

    def _fetch_outputs(self, client, store, remote_id, remote, *, job_id, out_dir, update,
                       live_files: Mapping[str, str] | None = None) -> RemoteJobResult:
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
            # A clip fetched mid-job keeps its local name; the final copy
            # replaces it in place, so a caller that already shows it stays valid.
            local_name = (live_files or {}).get(name) or _unique_local_name(out_dir, name)
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


def _billing(manager: Any, ready: Mapping[str, Any], *, waited: float, worked: float) -> dict[str, Any]:
    """The pod's run, price and accrued cost when the job got it and when it ended.

    A project's GPU spend is then each run's cost growth over the project
    (boot and idle included), not only the seconds its jobs computed.
    """
    try:
        ended = manager.status()
    except Exception:  # a cost report must never fail a finished job
        ended = {}
    return {
        "pod": getattr(manager, "run_name", ""),
        "run": str(ended.get("job_submission_id") or ready.get("job_submission_id") or ""),
        "submitted_at": ready.get("submitted_at"),
        "price_per_hour": ended.get("price_per_hour") or ready.get("price_per_hour"),
        "cost_start": ready.get("cost"),
        "cost_end": ended.get("cost"),
        "wait_seconds": round(waited, 1),
        "work_seconds": round(worked, 1),
    }


def format_progress(progress: Mapping[str, Any] | None) -> str:
    """`` · 12.3 / 41.0 GB · 640 MB/s`` from a worker download progress, or ""."""
    if not isinstance(progress, Mapping) or not progress.get("done_bytes"):
        return ""
    done = float(progress["done_bytes"]) / 1e9
    total = float(progress.get("total_bytes") or 0) / 1e9
    text = f" · {done:.1f} / {total:.1f} GB" if total else f" · {done:.1f} GB"
    rate = float(progress.get("bytes_per_second") or 0) / 1e6
    return text + (f" · {rate:.0f} MB/s" if rate >= 1 else "")


# Waits shown in the job message, rounded to this step.
WAIT_SHOWN_AFTER_SECONDS = 10
WAIT_STEP_SECONDS = 10


def _worker_status_message(status: Mapping[str, Any], waited: float | None = None) -> str:
    """The worker's startup state, with how long the job has waited for it.

    The wait changes the message every few seconds: Director callers treat an
    unchanged job message as no progress, and a slow image pull is one long
    step that would otherwise read as a stalled job.
    """
    message = _worker_state_message(status)
    if waited is None or waited < WAIT_SHOWN_AFTER_SECONDS:
        return message
    seconds = int(waited) // WAIT_STEP_SECONDS * WAIT_STEP_SECONDS
    return f"{message} · {seconds // 60} min {seconds % 60:02d} s"


def _worker_state_message(status: Mapping[str, Any]) -> str:
    state = status.get("state")
    if state == "starting":
        pull = status.get("image_pull_percent")
        return f"Starting remote GPU worker… (image {pull}%)" if pull is not None else "Starting remote GPU worker…"
    if state == "booting":
        # The step in progress, e.g. "Weights pulled from R2: minimax_h3".
        steps = (status.get("boot") or {}).get("steps") or []
        current = next((step for step in reversed(steps) if step.get("state") == "started"), None)
        if current and current.get("phase") != "host":
            return f"Remote worker: {current.get('label')}…"
        return "Remote worker booting Maestro…"
    if state == "stopping":
        return "Waiting for the previous worker to stop…"
    if state == "ready":
        return "Remote worker ready"
    return "Starting remote GPU worker…"
