"""Lifecycle of the remote Maestro worker: start, readiness, idle stop, stop.

States exposed to the UI:

* ``not_configured`` – no dstack server/project/token yet.
* ``stopped``        – no active run (a failed last run carries ``message``).
* ``starting``       – run submitted, waiting for a GPU / pulling the image.
* ``booting``        – container running, Maestro not answering probes yet.
* ``ready``          – the service proxy routes to the worker.
* ``stopping``       – stop requested, dstack is terminating the pod.

The idle stop is Maestro's job: dstack ignores ``idle_duration`` on RunPod.
The idle clock only starts once the worker is ready and pauses while a job
is using it; ``max_duration``/``max_price`` in the run spec are the backstop
if Maestro itself is not running.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Mapping
from typing import Any

from services.remote_worker.dstack_client import DstackClient, DstackError
from services.remote_worker.settings import (
    RemoteWorkerSettings,
    RemoteWorkerSettingsStore,
    read_local_dstack_config,
)
from services.remote_worker.worker_spec import build_run_spec


log = logging.getLogger("maestro.remote_worker")

_STARTING_RUN = {"pending", "submitted", "provisioning"}
_FINISHED_RUN = {"terminated", "failed", "done"}


class RemoteWorkerError(RuntimeError):
    pass


class RemoteWorkerManager:
    def __init__(
        self,
        store: RemoteWorkerSettingsStore,
        *,
        client_factory: Callable[[RemoteWorkerSettings], DstackClient] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._store = store
        self._settings = store.load()
        self._client_factory = client_factory or (
            lambda s: DstackClient(s.server_url, s.project, s.token)
        )
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.RLock()
        self._active_jobs = 0
        self._ready_since: float | None = None
        self._last_activity: float | None = None
        self._run_id: str | None = None
        self._requested_model: str | None = None
        self._preloaded: set[tuple[str, str]] = set()
        self._preload: Callable[[str, str, str], None] = _request_worker_preload
        self._stop_reason: str | None = None
        self._watchdog: threading.Thread | None = None
        self._watchdog_stop = threading.Event()

    # ── settings ───────────────────────────────────────────────────────────

    @property
    def settings(self) -> RemoteWorkerSettings:
        with self._lock:
            return self._settings

    def update_settings(self, data: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            updated = RemoteWorkerSettings.from_mapping(vars(self._settings))
            payload = dict(data)
            # The UI sends the masked token back unchanged; ignore it.
            token = payload.get("token")
            if isinstance(token, str) and ("…" in token or token == "••••"):
                payload.pop("token")
            updated.update(payload)
            self._store.save(updated)
            self._settings = updated
            return updated.public()

    def import_local_dstack_config(self) -> dict[str, Any]:
        found = read_local_dstack_config()
        if not found:
            raise RemoteWorkerError("No usable project in ~/.dstack/config.yml")
        return self.update_settings(found)

    def client(self) -> DstackClient:
        settings = self.settings
        if not settings.configured:
            raise RemoteWorkerError("Remote worker is not configured (dstack server, project, token)")
        return self._client_factory(settings)

    def test_connection(self) -> dict[str, Any]:
        client = self.client()
        info = client.get_info()
        fleets = [f.get("name") for f in client.list_fleets() if isinstance(f, Mapping)]
        volumes = [
            {"name": v.get("name"), "status": v.get("status"),
             "region": (v.get("configuration") or {}).get("region")}
            for v in client.list_volumes() if isinstance(v, Mapping)
        ]
        return {"server_version": info.get("server_version"), "fleets": fleets, "volumes": volumes}

    def create_volume(self) -> dict[str, Any]:
        """Create the configured network volume (billed storage, explicit action)."""
        settings = self.settings
        if not settings.volume_name:
            raise RemoteWorkerError("No volume name configured")
        client = self.client()
        if self._find_volume(client, settings.volume_name):
            raise RemoteWorkerError(f"Volume {settings.volume_name} already exists")
        volume = client.create_volume(
            name=settings.volume_name,
            region=settings.volume_region,
            size=settings.volume_size,
            backend=(settings.backends or ["runpod"])[0],
        )
        return {"name": volume.get("name"), "status": volume.get("status")}

    @staticmethod
    def _find_volume(client: DstackClient, name: str) -> Mapping[str, Any] | None:
        for volume in client.list_volumes():
            if isinstance(volume, Mapping) and volume.get("name") == name and not volume.get("deleted"):
                return volume
        return None

    # ── status ─────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        settings = self.settings
        base: dict[str, Any] = {
            "state": "not_configured",
            "run_name": settings.run_name,
            "idle_ttl_minutes": settings.idle_ttl_minutes,
            "route_generations": settings.route_generations,
            "model_type": self._requested_model or settings.default_model_type or None,
            "active_jobs": self._active_jobs,
        }
        if not settings.configured:
            return base
        try:
            run = self.client().get_run(settings.run_name)
        except (DstackError, RemoteWorkerError) as exc:
            return {**base, "state": "unknown", "message": str(exc)}
        return {**base, **self._describe(run)}

    def _describe(self, run: Mapping[str, Any] | None) -> dict[str, Any]:
        if not run or run.get("deleted"):
            self._mark_not_ready()
            return {"state": "stopped"}
        run_status = str(run.get("status") or "")
        submission = run.get("latest_job_submission") or {}
        provisioning = submission.get("job_provisioning_data") or {}
        details: dict[str, Any] = {
            "run_status": run_status,
            "job_status": submission.get("status"),
            "message": run.get("status_message") or submission.get("status_message") or "",
            "submitted_at": run.get("submitted_at"),
            "cost": run.get("cost"),
            "price_per_hour": provisioning.get("price"),
            "gpu_region": provisioning.get("region"),
            "job_submission_id": submission.get("id"),
        }
        if run_status in _FINISHED_RUN:
            self._mark_not_ready()
            reason = submission.get("termination_reason_message") or submission.get("termination_reason")
            if run_status == "failed":
                details["message"] = f"Last run failed: {reason or run.get('termination_reason') or 'unknown'}"
            elif self._stop_reason:
                details["message"] = self._stop_reason
            return {**details, "state": "stopped"}
        if run_status == "terminating":
            self._mark_not_ready()
            return {**details, "state": "stopping"}
        if run_status in _STARTING_RUN or run_status != "running":
            pull = submission.get("image_pull_progress") or {}
            if pull.get("total_bytes"):
                details["image_pull_percent"] = int(
                    100 * (pull.get("downloaded_bytes") or 0) / max(1, pull["total_bytes"])
                )
            return {**details, "state": "starting"}
        if not _probes_ready(run, submission):
            return {**details, "state": "booting"}

        with self._lock:
            run_id = str(run.get("id") or "")
            if self._run_id != run_id or self._ready_since is None:
                self._run_id = run_id
                self._ready_since = self._clock()
                self._last_activity = self._ready_since
            idle_seconds = 0 if self._active_jobs else max(0.0, self._clock() - (self._last_activity or self._clock()))
        service_url = self.client().service_base_url(run)
        self._maybe_preload(service_url)
        details.update({
            "state": "ready",
            "service_url": service_url,
            "idle_seconds": int(idle_seconds),
            "idle_stop_in_seconds": (
                None if self._active_jobs
                else max(0, int(self.settings.idle_ttl_minutes * 60 - idle_seconds))
            ),
        })
        return details

    def _maybe_preload(self, service_url: str | None) -> None:
        """Ask a ready worker to fetch the selected model's weights, once per run."""
        model_type = self._requested_model or self.settings.default_model_type
        with self._lock:
            key = (self._run_id or "", model_type or "")
            if not service_url or not model_type or key in self._preloaded:
                return
            self._preloaded.add(key)
        token = self.settings.token
        threading.Thread(
            target=self._preload,
            args=(service_url, token, model_type),
            name="maestro_remote_worker_preload",
            daemon=True,
        ).start()

    def _mark_not_ready(self) -> None:
        with self._lock:
            self._ready_since = None
            self._run_id = None

    # ── start / stop ───────────────────────────────────────────────────────

    def start(self, model_type: str | None = None) -> dict[str, Any]:
        """Submit the worker run unless one is already active. Idempotent."""
        settings = self.settings
        client = self.client()
        with self._lock:
            if model_type:
                self._requested_model = model_type
            run = client.get_run(settings.run_name)
            if run and str(run.get("status") or "") not in _FINISHED_RUN and not run.get("deleted"):
                return self.status()
            if settings.volume_name:
                volume = self._find_volume(client, settings.volume_name)
                if volume is None:
                    raise RemoteWorkerError(
                        f"Network volume {settings.volume_name} does not exist; create it first"
                    )
                if str(volume.get("status") or "") != "active":
                    raise RemoteWorkerError(
                        f"Network volume {settings.volume_name} is {volume.get('status')}, not active yet"
                    )
            self._stop_reason = None
            self._last_activity = self._clock()
            client.apply(build_run_spec(settings))
            log.info("remote worker %s submitted (model %s)", settings.run_name, model_type or "-")
        return self.status()

    def stop(self, reason: str = "Stopped by user") -> dict[str, Any]:
        settings = self.settings
        client = self.client()
        with self._lock:
            run = client.get_run(settings.run_name)
            if run and str(run.get("status") or "") not in _FINISHED_RUN:
                client.stop_run(settings.run_name, abort=False)
                log.info("remote worker %s stop requested: %s", settings.run_name, reason)
            self._stop_reason = reason
            self._mark_not_ready()
        return self.status()

    def logs(self, since: str | None = None) -> list[dict[str, str]]:
        settings = self.settings
        client = self.client()
        run = client.get_run(settings.run_name)
        submission = (run or {}).get("latest_job_submission") or {}
        if not submission.get("id"):
            return []
        return client.poll_logs(settings.run_name, str(submission["id"]), start_time=since)

    # ── use by generation jobs ─────────────────────────────────────────────

    def job_started(self) -> None:
        with self._lock:
            self._active_jobs += 1
            self._last_activity = self._clock()

    def job_finished(self) -> None:
        with self._lock:
            self._active_jobs = max(0, self._active_jobs - 1)
            self._last_activity = self._clock()

    def ensure_ready(
        self,
        model_type: str | None,
        *,
        should_abort: Callable[[], bool] = lambda: False,
        on_update: Callable[[Mapping[str, Any]], None] = lambda status: None,
        timeout: float = 45 * 60,
        poll_interval: float = 10.0,
    ) -> dict[str, Any]:
        """Start the worker if needed and block until it is ready.

        Returns the ready status (with ``service_url``). Raises on timeout,
        on a failed run, or returns early with the current status on abort.
        """
        if not self.settings.auto_start:
            current = self.status()
            if current.get("state") != "ready":
                raise RemoteWorkerError("Remote worker is not running and auto-start is disabled")
            return current
        deadline = self._clock() + timeout
        submitted = False
        while True:
            current = self.status()
            state = current.get("state")
            on_update(current)
            if state == "ready":
                return current
            if should_abort():
                return current
            if state == "stopped":
                if submitted and str(current.get("message") or "").startswith("Last run failed"):
                    raise RemoteWorkerError(str(current["message"]))
                self.start(model_type)
                submitted = True
            elif state in ("not_configured", "unknown"):
                raise RemoteWorkerError(str(current.get("message") or "Remote worker unavailable"))
            if self._clock() > deadline:
                raise RemoteWorkerError(f"Remote worker not ready after {int(timeout // 60)} min")
            self._sleep(poll_interval)

    # ── idle watchdog ──────────────────────────────────────────────────────

    def idle_tick(self) -> bool:
        """Stop the worker when idle past the TTL. Returns True if stopped."""
        settings = self.settings
        if not settings.configured:
            return False
        with self._lock:
            if self._active_jobs or self._ready_since is None or self._last_activity is None:
                return False
            idle = self._clock() - self._last_activity
        if idle < settings.idle_ttl_minutes * 60:
            return False
        self.stop(reason=f"Stopped after {settings.idle_ttl_minutes} min without generation")
        return True

    def start_watchdog(self, interval: float = 30.0) -> None:
        if self._watchdog and self._watchdog.is_alive():
            return

        def _loop() -> None:
            while not self._watchdog_stop.wait(interval):
                try:
                    if self.settings.configured:
                        self.status()  # refreshes readiness bookkeeping
                        self.idle_tick()
                except Exception as exc:  # never let the watchdog die
                    log.warning("remote worker watchdog: %s", exc)

        self._watchdog_stop.clear()
        self._watchdog = threading.Thread(target=_loop, name="maestro_remote_worker_idle", daemon=True)
        self._watchdog.start()

    def shutdown_watchdog(self) -> None:
        self._watchdog_stop.set()


def _request_worker_preload(service_url: str, token: str, model_type: str) -> None:
    import requests
    from urllib.parse import quote

    try:
        response = requests.post(
            f"{service_url}api/v1/models/{quote(model_type, safe='')}/download",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60,
        )
        log.info("remote worker preload %s: HTTP %s", model_type, response.status_code)
    except Exception as exc:
        log.warning("remote worker preload %s failed: %s", model_type, exc)


def _probes_ready(run: Mapping[str, Any], submission: Mapping[str, Any]) -> bool:
    jobs = run.get("jobs") or []
    job_spec = (jobs[0] or {}).get("job_spec") if jobs else None
    specs = (job_spec or {}).get("probes") or []
    observed = submission.get("probes") or []
    if not specs:
        return True
    if len(observed) < len(specs):
        return False
    return all(
        int((observed[i] or {}).get("success_streak") or 0) >= int((specs[i] or {}).get("ready_after") or 1)
        for i in range(len(specs))
    )
