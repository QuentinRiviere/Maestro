"""Lifecycle of the remote Maestro worker: start, readiness, idle stop, stop.

States exposed to the UI:

* ``not_configured`` – no dstack server/project/token yet.
* ``stopped``        – no active run (a failed last run carries ``message``).
* ``starting``       – run submitted, waiting for a GPU / pulling the image
                       (also while a slow host is replaced, see below).
* ``booting``        – container running, Maestro not answering probes yet.
* ``ready``          – the service proxy routes to the worker.
* ``stopping``       – stop requested, dstack is terminating the pod.

The idle stop is Maestro's job: dstack ignores ``idle_duration`` on RunPod.
The idle clock only starts once the worker is ready and pauses while a job
is using it; ``max_duration``/``max_price`` in the run spec are the backstop
if Maestro itself is not running.

Slow hosts: nothing of Maestro runs on the pod while it pulls the ~20 GB
worker image, and some hosts pull it at a few tens of MB/s. A run that stays
in dstack ``provisioning`` past ``slow_pull_seconds`` is stopped and submitted
again (same models to preload), up to ``slow_pull_max_retries`` times; after
that the current host is kept. Queued jobs keep waiting meanwhile: the pod
reports ``starting``. A data center is not blamed for one slow host (its
machines sit on different networks), but two slow hosts in a row from the same
one keep it out of the run's ``regions`` for ``slow_region_exclusion_seconds``.

Static provider (``provider: static``): the worker always runs at
``static_url`` and is never started or stopped from here (see
static_worker.py for its states: ``ready``, ``busy``, ``paused``,
``unreachable``). ``start`` and ``stop`` only report the status: resuming a
paused worker is left to whoever paused it (llama-swap), since resuming it
from here could load a model next to the one that holds the GPU memory. The
idle stop does not apply.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from typing import Any

from services.remote_worker import boot_events
from services.remote_worker.dstack_client import DstackClient, DstackError
from services.remote_worker.settings import (
    SECRET_FIELDS,
    RemoteWorkerSettings,
    RemoteWorkerSettingsStore,
    env_overrides,
    is_masked,
    read_local_dstack_config,
)
from services.remote_worker.static_worker import READY_STATES as STATIC_READY_STATES, StaticWorker
from services.remote_worker.ephemeral_r2 import EphemeralKeyError, EphemeralR2Keys, duration_seconds
from services.remote_worker.vast_network import PLAN_OFFERS, VastFilterError, pin_fast_offers
from services.remote_worker.worker_spec import build_fleet_spec, build_run_spec, dstack_secrets


log = logging.getLogger("maestro.remote_worker")

_STARTING_RUN = {"pending", "submitted", "provisioning"}
_FINISHED_RUN = {"terminated", "failed", "done"}


class RemoteWorkerError(RuntimeError):
    pass


# With stop_when_queue_empty, the worker waits this long after its last job so
# a follow-up job (next queue item, next Director shot) can claim it.
QUEUE_EMPTY_GRACE_SECONDS = 30.0
# A Director project paused for the user's review keeps the worker this long,
# so continuing does not pay a cold start; past it the idle stop applies.
REVIEW_HOLD_SECONDS = 30 * 60
# Startup steps are read from the run logs at most this often.
BOOT_LOG_POLL_SECONDS = 3.0
# Slow hosts in a row from one data center before it is excluded.
SLOW_HOSTS_TO_EXCLUDE_REGION = 2


def offer_details(provisioning: Mapping[str, Any] | None) -> dict[str, Any]:
    """GPU, backend and instance of a dstack ``job_provisioning_data``.

    Only the fields dstack reported; used for the jobs' provenance.
    """
    provisioning = provisioning if isinstance(provisioning, Mapping) else {}
    instance = provisioning.get("instance_type")
    instance = instance if isinstance(instance, Mapping) else {}
    resources = instance.get("resources")
    resources = resources if isinstance(resources, Mapping) else {}
    gpus = [gpu for gpu in resources.get("gpus") or [] if isinstance(gpu, Mapping)]
    details: dict[str, Any] = {}
    if provisioning.get("backend"):
        details["gpu_backend"] = str(provisioning["backend"])
    if instance.get("name"):
        details["instance_type"] = str(instance["name"])
    if resources.get("spot") is not None:
        details["spot"] = bool(resources["spot"])
    if gpus:
        if gpus[0].get("name"):
            details["gpu_name"] = str(gpus[0]["name"])
        details["gpu_count"] = len(gpus)
        try:
            memory_mib = float(gpus[0].get("memory_mib") or 0)
        except (TypeError, ValueError):
            memory_mib = 0.0
        if memory_mib > 0:
            details["gpu_memory_gb"] = round(memory_mib / 1024, 1)
    return details


class RemoteWorkerManager:
    def __init__(
        self,
        store: RemoteWorkerSettingsStore,
        *,
        client_factory: Callable[[RemoteWorkerSettings], DstackClient] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
        proxy_check: Callable[[str, str], bool] | None = None,
        environ: Mapping[str, str] | None = None,
        pending_work: Callable[[], bool] | None = None,
        parent: "RemoteWorkerManager | None" = None,
        index: int = 0,
    ) -> None:
        # A pool's extra pods (index 1, 2...) share the first pod's settings,
        # queue hooks and R2 key store; each runs under its own run name.
        self._parent = parent
        self._index = index
        self._store = store
        self._env = env_overrides(environ)
        self._stored = store.load()
        self._settings = self._effective(self._stored)
        self._client_factory = client_factory or (
            lambda s: DstackClient(s.server_url, s.project, s.token)
        )
        self._clock = clock
        self._sleep = sleep
        # Whether the host still has queued or running generations (set by
        # launch.py, which owns the job queue).
        self.pending_work: Callable[[], bool] = pending_work or (lambda: False)
        # Seconds the oldest Director review pause has lasted, None without one.
        self.review_pause: Callable[[], float | None] = lambda: None
        # Models the queued work will need, preloaded from R2 at boot.
        self.upcoming_models: Callable[[], list[str]] = lambda: []
        if parent is not None:
            self.pending_work = lambda: parent.pending_work()
            self.review_pause = lambda: parent.review_pause()
            self.upcoming_models = lambda: parent.upcoming_models()
        self._lock = threading.RLock()
        self._active_jobs = 0
        self._ready_since: float | None = None
        self._last_activity: float | None = None
        self._run_id: str | None = None
        self._requested_model: str | None = None
        self._preloaded: set[tuple[str, str]] = set()
        self._preload: Callable[[str, str, str], None] = _request_worker_preload
        self._stop_reason: str | None = None
        self._proxy_check = proxy_check or _proxy_reachable
        self._proxy_ok_run: str | None = None
        self._watchdog: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        self.r2_keys = parent.r2_keys if parent else EphemeralR2Keys(store.path.parent / "remote_worker_r2_keys.json")
        self._network_filter: dict[str, Any] = {}
        self._preload_models: list[str] = []
        self._boot_submission: str | None = None
        self._boot_lines: list[dict[str, str]] = []
        self._boot_cursor: str | None = None
        self._boot_polled_at = 0.0
        self._route_ready_at: float | None = None
        self._fetch_downloads: Callable[[str, str], dict[str, Any]] = _worker_downloads
        self._downloads: dict[str, Any] = {}
        self._downloads_polled_at = 0.0
        # Slow image pull watchdog (per pod): when the current submission was
        # first seen provisioning, restarts so far, and a pending restart.
        self._provisioning_submission: str | None = None
        self._provisioning_since: float | None = None
        self._provisioning_settled: str | None = None
        self._slow_retries = 0
        self._slow_cap_logged = False
        self._slow_restart: dict[str, Any] | None = None
        # Shared by the pool through the first pod: the last slow host's data
        # center and the excluded ones with their expiry.
        self._last_slow_region: str | None = None
        self._excluded_regions: dict[str, float] = {}
        # Probes of a static worker (provider "static").
        self.static_worker = StaticWorker(clock=clock)

    # ── settings ───────────────────────────────────────────────────────────

    def _effective(self, stored: RemoteWorkerSettings) -> RemoteWorkerSettings:
        """Stored settings with the environment overrides applied on top."""
        effective = RemoteWorkerSettings.from_mapping(vars(stored))
        effective.update(self._env)
        return effective

    @property
    def env_locked(self) -> frozenset[str]:
        return frozenset(self._env)

    @property
    def settings(self) -> RemoteWorkerSettings:
        if self._parent is not None:
            return self._parent.settings
        with self._lock:
            return self._settings

    @property
    def run_name(self) -> str:
        return self.settings.pod_run_name(self._index)

    @property
    def active_jobs(self) -> int:
        return self._active_jobs

    @property
    def ready(self) -> bool:
        return self._ready_since is not None

    def public_settings(self) -> dict[str, Any]:
        return self.settings.public(self.env_locked)

    def update_settings(self, data: Mapping[str, Any]) -> dict[str, Any]:
        if self._parent is not None:
            return self._parent.update_settings(data)
        with self._lock:
            updated = RemoteWorkerSettings.from_mapping(vars(self._stored))
            payload = dict(data)
            # The UI sends masked secrets back unchanged; keep the stored ones.
            for name in SECRET_FIELDS:
                if is_masked(payload.get(name)):
                    payload.pop(name)
            # Environment-managed values are never stored nor overridden.
            for name in self._env:
                payload.pop(name, None)
            updated.update(payload)
            self._store.save(updated)
            self._stored = updated
            self._settings = self._effective(updated)
            return self.public_settings()

    def import_local_dstack_config(self) -> dict[str, Any]:
        found = read_local_dstack_config()
        if not found:
            raise RemoteWorkerError("No usable project in ~/.dstack/config.yml")
        return self.update_settings(found)

    def client(self) -> DstackClient:
        settings = self.settings
        if not settings.dstack_configured:
            raise RemoteWorkerError("Remote worker is not configured (dstack server, project, token)")
        return self._client_factory(settings)

    def test_connection(self) -> dict[str, Any]:
        if self.settings.is_static:
            if not self.settings.configured:
                raise RemoteWorkerError("Static worker URL is not set")
            self.static_worker.forget()
            current = self.status()
            if current.get("state") == "unknown":
                raise RemoteWorkerError(str(current.get("message")))
            return {"server_version": None, "fleets": [], "state": current.get("state"),
                    "message": current.get("message") or ""}
        client = self.client()
        info = client.get_info()
        fleets = [f.get("name") for f in client.list_fleets() if isinstance(f, Mapping)]
        return {"server_version": info.get("server_version"), "fleets": fleets}

    # ── status ─────────────────────────────────────────────────────────────

    def status(self) -> dict[str, Any]:
        settings = self.settings
        base: dict[str, Any] = {
            "state": "not_configured",
            "run_name": self.run_name,
            "idle_ttl_minutes": settings.idle_ttl_minutes,
            "route_generations": settings.route_generations,
            "model_type": self._requested_model or settings.default_model_type or None,
            "active_jobs": self._active_jobs,
            "pod_r2_keys": self.r2_keys.enabled(settings),
            "pod_r2_keys_active": len(self.r2_keys.tracked()),
            "network_filter": self._network_filter or None,
            "preload_models": list(self._preload_models),
            "review_hold": self._review_hold_active(),
            **self._slow_host_status(),
        }
        if not settings.configured:
            return base
        if settings.is_static:
            return {**base, **self._static_status(settings)}
        try:
            run = self.client().get_run(self.run_name)
        except (DstackError, RemoteWorkerError) as exc:
            return {**base, "state": "unknown", "message": str(exc)}
        described = self._describe(run)
        # Read after _describe, which may just have replaced a slow host.
        return {**base, **described, **self._slow_host_status()}

    def _static_status(self, settings: RemoteWorkerSettings) -> dict[str, Any]:
        """Probe the static worker; it counts as ready while it takes jobs."""
        described = self.static_worker.status(settings.static_url, settings.worker_token)
        if described.get("state") in STATIC_READY_STATES:
            with self._lock:
                if self._ready_since is None:
                    self._ready_since = self._last_activity = self._clock()
        else:
            self._mark_not_ready()
        return described

    def wake_static(self) -> bool:
        """Ask the static worker's wake URL to resume it (throttled); True when sent."""
        settings = self.settings
        return settings.is_static and self.static_worker.wake(settings.static_wake_url)

    def _slow_host_status(self) -> dict[str, Any]:
        return {"slow_host_retries": self._slow_retries,
                "excluded_regions": self._root.excluded_regions()}

    def _describe(self, run: Mapping[str, Any] | None) -> dict[str, Any]:
        replacing = self._replace_slow_host(run)
        if replacing is not None:
            return replacing
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
            **offer_details(provisioning),
        }
        if run_status in _FINISHED_RUN:
            self._mark_not_ready()
            reason = submission.get("termination_reason_message") or submission.get("termination_reason")
            if run_status == "failed":
                reason = reason or run.get("termination_reason") or "unknown"
                details["message"] = f"Last run failed: {reason}"
                if "no_capacity" in str(reason) or "no offers" in str(reason).lower():
                    settings = self.settings
                    details["message"] += (
                        f". No GPU offer matched gpu {settings.gpu}, disk {settings.disk}, "
                        f"max ${settings.max_price}/h"
                        + (f" and fleet {settings.fleet}" if settings.fleet else "")
                        + ". Raise the max price, relax the GPU or disk, or drop the fleet."
                    )
            elif self._stop_reason:
                details["message"] = self._stop_reason
            return {**details, "state": "stopped"}
        if run_status == "terminating":
            self._mark_not_ready()
            return {**details, "state": "stopping"}
        if run_status in _STARTING_RUN or run_status != "running":
            slow = self._check_slow_pull(run_status, submission, provisioning)
            if slow is not None:
                return {**details, **slow}
            pull = submission.get("image_pull_progress") or {}
            if pull.get("total_bytes"):
                details["image_pull_percent"] = int(
                    100 * (pull.get("downloaded_bytes") or 0) / max(1, pull["total_bytes"])
                )
            details["boot"] = self._boot(run, submission, "starting")
            return {**details, "state": "starting"}
        self._settle_provisioning(submission)
        if not _probes_ready(run, submission):
            details["boot"] = self._boot(run, submission, "booting")
            return {**details, "state": "booting"}
        # Probes passing is not enough: the dstack proxy registers the
        # replica a little later and answers 404 "Service not found" until
        # then. Only call the worker ready once a request goes through.
        run_id = str(run.get("id") or "")
        service_url = self.client().service_base_url(run)
        if self._proxy_ok_run != run_id:
            if not service_url or not self._proxy_check(service_url, self.settings.token):
                details["boot"] = self._boot(run, submission, "booting")
                return {**details, "state": "booting", "message": "Waiting for the dstack proxy route"}
            self._proxy_ok_run = run_id
            self._route_ready_at = self._clock()

        with self._lock:
            if self._run_id != run_id or self._ready_since is None:
                self._run_id = run_id
                self._ready_since = self._clock()
                self._last_activity = self._ready_since
            idle_seconds = 0 if self._active_jobs else max(0.0, self._clock() - (self._last_activity or self._clock()))
        self._maybe_preload(service_url)
        details["boot"] = self._boot(run, submission, "ready")
        details["worker_downloads"] = self._worker_downloads(service_url)
        details.update({
            "state": "ready",
            "service_url": service_url,
            "idle_seconds": int(idle_seconds),
            "idle_stop_in_seconds": (
                None if self._active_jobs
                else max(0, int(self._idle_limit_seconds() - idle_seconds))
            ),
        })
        return details

    def _boot(self, run: Mapping[str, Any], submission: Mapping[str, Any], state: str) -> dict[str, Any]:
        """Startup steps of the current pod, from its ``[Boot]`` log lines."""
        submission_id = str(submission.get("id") or "")
        with self._lock:
            if submission_id != self._boot_submission:
                self._boot_submission = submission_id
                self._boot_lines, self._boot_cursor = [], None
                if state != "ready":
                    self._route_ready_at = None
            events = boot_events.parse_events(self._boot_lines)
            # Once ready with every step finished, the logs are not read again.
            settled = state == "ready" and bool(events) and all(
                step["state"] != "started" for step in boot_events.steps(events))
            due = self._clock() - self._boot_polled_at >= BOOT_LOG_POLL_SECONDS
            poll = bool(submission_id) and state != "starting" and due and not settled
            if poll:
                self._boot_polled_at = self._clock()
            cursor = self._boot_cursor
        if poll:
            try:
                lines = self.client().poll_logs(self.run_name, submission_id,
                                                start_time=cursor, limit=1000)
            except (DstackError, RemoteWorkerError):
                lines = []
            with self._lock:
                if submission_id == self._boot_submission and lines:
                    self._boot_lines.extend(line for line in lines if boot_events.relevant(line.get("message", "")))
                    self._boot_cursor = lines[-1].get("timestamp") or self._boot_cursor
                events = boot_events.parse_events(self._boot_lines)
        with self._lock:
            route_ready = self._route_ready_at
        steps = boot_events.steps(events)
        submitted = boot_events.seconds(str(run.get("submitted_at") or ""))
        # The first line the pod prints ends the wait for a host and the image.
        container = next((e["at"] for e in events), None)
        host = {"phase": "host", "label": "GPU host found and image pulled",
                "state": "done" if container else "started", "started_at": submitted, "ended_at": container}
        api = next((step for step in steps if step["phase"] == "api"), None)
        route = None
        if api and api.get("ended_at"):
            route = {"phase": "route", "label": "dstack route open", "started_at": api["ended_at"],
                     "state": "done" if route_ready else "started", "ended_at": route_ready}
        result = [host, *steps] + ([route] if route else [])
        for step in result:
            if step.get("started_at") is not None and step.get("ended_at") is not None:
                step["seconds"] = round(max(0.0, step["ended_at"] - step["started_at"]), 1)
        return {"steps": result, "submitted_at": submitted}

    def _worker_downloads(self, service_url: str | None) -> list[dict[str, Any]]:
        """Weights the ready worker is pulling or downloading, with their progress."""
        if not service_url:
            return []
        if self._clock() - self._downloads_polled_at >= BOOT_LOG_POLL_SECONDS:
            self._downloads_polled_at = self._clock()
            self._downloads = self._fetch_downloads(service_url, self.settings.worker_token)
        return [
            {"model_type": model_type, "phase": entry.get("phase"), "progress": entry.get("progress")}
            for model_type, entry in sorted(self._downloads.items())
            if isinstance(entry, Mapping) and entry.get("status") == "downloading"
        ]

    def _maybe_preload(self, service_url: str | None) -> None:
        """Ask a ready worker to fetch the selected model's weights, once per run."""
        model_type = self._requested_model or self.settings.default_model_type
        with self._lock:
            key = (self._run_id or "", model_type or "")
            if not service_url or not model_type or key in self._preloaded:
                return
            self._preloaded.add(key)
        token = self.settings.worker_token
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
        """Submit the worker run unless one is already active. Idempotent.

        A static worker is never started (nor resumed): this only reports it.
        """
        settings = self.settings
        if settings.is_static:
            return self.status()
        client = self.client()
        with self._lock:
            if model_type:
                self._requested_model = model_type
            run_name = self.run_name
            run = client.get_run(run_name)
            if run and str(run.get("status") or "") not in _FINISHED_RUN and not run.get("deleted"):
                return self.status()
            self._slow_retries = 0
            self._slow_cap_logged = False
            self._slow_restart = None
            self._submit(settings, client, self._models_to_preload(model_type))
            log.info("remote worker %s submitted (model %s)", run_name, model_type or "-")
        return self.status()

    def _submit(self, settings: RemoteWorkerSettings, client: DstackClient,
                preload_models: list[str]) -> None:
        """Push the pod's secrets and fleet, then submit its run (caller holds the lock)."""
        run_name = self.run_name
        output_scope = f"{run_name}-{uuid.uuid4().hex[:12]}"
        keys = self.pod_r2_keys(settings, run_name, output_scope=output_scope)
        for name, value in dstack_secrets(settings, keys).items():
            client.set_secret(name, value)
        if settings.manage_fleet and settings.fleet:
            client.apply_fleet(build_fleet_spec(settings))
        self._stop_reason = None
        self._last_activity = self._clock()
        self._preload_models = list(preload_models)
        self._provisioning_submission = self._provisioning_since = None
        run_spec = build_run_spec(settings, output_scope, preload_models=self._preload_models,
                                  run_name=run_name)
        self._root.restrict_regions(run_spec, client)
        try:
            self._network_filter = pin_fast_offers(
                run_spec, lambda spec: client.get_plan(spec, max_offers=PLAN_OFFERS), settings,
            )
        except VastFilterError as exc:
            self._revoke_pod_keys(run_name)
            raise RemoteWorkerError(str(exc)) from exc
        client.apply(run_spec)

    # ── slow image pull watchdog ───────────────────────────────────────────

    @property
    def _root(self) -> "RemoteWorkerManager":
        """The first pod, which keeps the data center bookkeeping of the pool."""
        return self._parent or self

    def _check_slow_pull(self, run_status: str, submission: Mapping[str, Any],
                         provisioning: Mapping[str, Any]) -> dict[str, Any] | None:
        """Replace the host when the run pulls its image for too long.

        Returns the ``starting`` status to show while the run is replaced,
        or None to describe the run as usual.
        """
        threshold = self.settings.slow_pull_seconds
        if threshold <= 0 or run_status != "provisioning":
            return None
        submission_id = str(submission.get("id") or "")
        now = self._clock()
        with self._lock:
            if submission_id != self._provisioning_submission:
                self._provisioning_submission = submission_id
                self._provisioning_since = now
                return None
            elapsed = now - (self._provisioning_since or now)
            if elapsed <= threshold:
                return None
            region = str(provisioning.get("region") or "")
            limit = self.settings.slow_pull_max_retries
            if self._slow_retries >= limit:
                if not self._slow_cap_logged:
                    self._slow_cap_logged = True
                    log.info("remote worker %s: image pull slow for %ds on %s, %d host retries used: "
                             "keeping this host", self.run_name, elapsed, region or "?", limit)
                return None
            try:
                # Only ever this manager's own run, like stop().
                self.client().stop_run(self.run_name, abort=False)
            except (DstackError, RemoteWorkerError) as exc:
                log.warning("remote worker %s: could not stop the slow host: %s", self.run_name, exc)
                return None
            self._slow_retries += 1
            self._provisioning_settled = submission_id
            message = f"Slow image pull on {region or 'this host'}, trying another host"
            self._slow_restart = {"region": region, "message": message}
            self._mark_not_ready()
            log.info("remote worker %s: image pull still running after %ds on %s (host %s): "
                     "stopping it, slow-host retry %d/%d", self.run_name, elapsed, region or "?",
                     provisioning.get("hostname") or "?", self._slow_retries, limit)
        self._root.note_slow_host(region)
        return self._replacing_status()

    def _replace_slow_host(self, run: Mapping[str, Any] | None) -> dict[str, Any] | None:
        """While a slow host is being replaced: submit the new run once the old one ended."""
        with self._lock:
            if self._slow_restart is None:
                return None
            finished = not run or run.get("deleted") or str(run.get("status") or "") in _FINISHED_RUN
            if not finished:
                return self._replacing_status(run)
            pending, self._slow_restart = self._slow_restart, None
            try:
                self._submit(self.settings, self.client(), self._preload_models)
            except (DstackError, RemoteWorkerError) as exc:
                log.warning("remote worker %s: new host after a slow pull not submitted: %s",
                            self.run_name, exc)
                self._stop_reason = f"Could not replace the slow host: {exc}"
                return None
            log.info("remote worker %s: new run submitted after the slow host on %s",
                     self.run_name, pending.get("region") or "?")
        try:
            return self._describe(self.client().get_run(self.run_name))
        except (DstackError, RemoteWorkerError) as exc:
            return {"state": "unknown", "message": str(exc)}

    def _replacing_status(self, run: Mapping[str, Any] | None = None) -> dict[str, Any]:
        message = (self._slow_restart or {}).get("message") or "Trying another host"
        return {
            "state": "starting",
            "run_status": str((run or {}).get("status") or "terminating"),
            "message": message,
            "boot": {"steps": [{"phase": "host", "label": "GPU host found and image pulled",
                                "state": "started", "started_at": None, "ended_at": None}],
                     "submitted_at": None, "message": message},
        }

    def _settle_provisioning(self, submission: Mapping[str, Any]) -> None:
        """A run that left ``provisioning`` in time breaks the slow-host streak."""
        submission_id = str(submission.get("id") or "")
        with self._lock:
            if (not submission_id or submission_id != self._provisioning_submission
                    or submission_id == self._provisioning_settled):
                return
            self._provisioning_settled = submission_id
            elapsed = self._clock() - (self._provisioning_since or self._clock())
            slow = elapsed > self.settings.slow_pull_seconds > 0
        if not slow:
            self._root.clear_slow_streak()

    def note_slow_host(self, region: str) -> None:
        """Count a slow host; the second in a row from one data center excludes it."""
        settings = self.settings
        with self._lock:
            streak_region = self._last_slow_region
            if not region:
                self._last_slow_region = None
                return
            if region != streak_region:
                self._last_slow_region = region
                log.info("remote worker: slow host in %s, data center kept (first in a row)", region)
                return
            self._last_slow_region = None
            if settings.slow_region_exclusion_seconds <= 0:
                return
            self._excluded_regions[region] = self._clock() + settings.slow_region_exclusion_seconds
            log.info("remote worker: %d slow hosts in a row in %s, excluding it for %ds",
                     SLOW_HOSTS_TO_EXCLUDE_REGION, region, settings.slow_region_exclusion_seconds)

    def clear_slow_streak(self) -> None:
        with self._lock:
            self._last_slow_region = None

    def excluded_regions(self) -> list[dict[str, Any]]:
        """Data centers kept out for now, with their expiry (expired ones are dropped)."""
        now = self._clock()
        with self._lock:
            for region, until in list(self._excluded_regions.items()):
                if until <= now:
                    del self._excluded_regions[region]
                    log.info("remote worker: data center %s no longer excluded", region)
            return [{"region": region, "until": until, "expires_in_seconds": int(until - now)}
                    for region, until in sorted(self._excluded_regions.items())]

    def restrict_regions(self, run_spec: dict[str, Any], client: DstackClient) -> None:
        """Limit ``run_spec`` to the data centers its offers come from, minus the excluded ones.

        dstack only takes an allow list (``regions``), so it is read from the
        plan's offers, one plan per backend like ``pin_fast_offers``.
        """
        excluded = {entry["region"] for entry in self.excluded_regions()}
        if not excluded:
            return
        configuration = run_spec["configuration"]
        regions: list[str] = []
        try:
            for backend in configuration.get("backends") or []:
                planned = {k: v for k, v in configuration.items() if k != "fleets"}
                planned["backends"] = [backend]
                job_plans = client.get_plan({**run_spec, "configuration": planned},
                                            max_offers=PLAN_OFFERS).get("job_plans") or [{}]
                for offer in job_plans[0].get("offers") or []:
                    region = str(offer.get("region") or "")
                    if region and region not in regions:
                        regions.append(region)
        except DstackError as exc:
            log.warning("remote worker: data center exclusion skipped, no plan: %s", exc)
            return
        allowed = [region for region in regions if region not in excluded]
        if not allowed:
            # Never exclude every data center: drop the exclusion instead.
            with self._lock:
                self._excluded_regions.clear()
            log.info("remote worker: excluding %s would leave no offer, exclusion dropped",
                     ", ".join(sorted(excluded)))
            return
        configuration["regions"] = allowed
        log.info("remote worker: data centers %s excluded, run limited to %s",
                 ", ".join(sorted(excluded)), ", ".join(allowed))

    def _models_to_preload(self, model_type: str | None) -> list[str]:
        models = [model_type] if model_type else []
        try:
            models += list(self.upcoming_models())
        except Exception as exc:  # preloading is only a head start
            log.warning("remote worker upcoming-models check failed: %s", exc)
        if not models and self.settings.default_model_type:
            models.append(self.settings.default_model_type)
        return list(dict.fromkeys(m for m in models if isinstance(m, str) and m and not m.startswith("cloud_")))

    def stop(self, reason: str = "Stopped by user") -> dict[str, Any]:
        settings = self.settings
        if settings.is_static:
            # Never stops the static worker's process: it may be rendering for
            # another controller, and llama-swap owns its GPU lease.
            return self.status()
        client = self.client()
        with self._lock:
            run = client.get_run(self.run_name)
            if run and str(run.get("status") or "") not in _FINISHED_RUN:
                client.stop_run(self.run_name, abort=False)
                log.info("remote worker %s stop requested: %s", self.run_name, reason)
            self._stop_reason = reason
            self._slow_restart = None
            self._mark_not_ready()
        self._revoke_pod_keys(self.run_name)
        return self.status()

    # ── per-pod R2 keys ────────────────────────────────────────────────────

    def pod_r2_keys(self, settings: RemoteWorkerSettings, run_name: str,
                    max_duration: str | None = None, output_scope: str = "",
                    trusted: bool = False) -> dict[str, str] | None:
        """A key for one pod, or None to fall back to the long-lived keys.

        ``trusted`` is for pods the caller has kept off untrusted hosts (the
        cache warmer): they get a bucket-wide key even when GPU workers do not.
        """
        untrusted = settings.untrusted_hosts and not trusted
        if not self.r2_keys.enabled(settings):
            if untrusted and settings.r2_configured:
                raise RemoteWorkerError(
                    "Untrusted hosts need a Cloudflare API token so pods get scoped keys, not the account's"
                )
            if settings.r2_configured:
                log.warning("no Cloudflare API token: %s gets the long-lived R2 keys", run_name)
            return None
        ttl = duration_seconds(max_duration or settings.max_duration)
        try:
            if untrusted:
                return self.r2_keys.issue_untrusted(settings, run_name, ttl, output_scope or run_name)
            return self.r2_keys.issue(settings, run_name, ttl)
        except EphemeralKeyError as exc:
            # Never hand out the long-lived keys silently once ephemeral keys
            # are configured: the pod would run on untrusted hardware.
            raise RemoteWorkerError(f"Could not issue a per-pod R2 key: {exc}") from exc

    def _revoke_pod_keys(self, run_name: str) -> None:
        try:
            self.r2_keys.revoke(self.settings, run_name)
        except Exception as exc:  # the key also expires on its own
            log.warning("R2 key revoke for %s failed: %s", run_name, exc)

    def run_active(self, run_name: str) -> bool:
        run = self.client().get_run(run_name)
        return bool(run) and not run.get("deleted") and str(run.get("status") or "") not in _FINISHED_RUN

    def reconcile_pod_keys(self) -> int:
        return self.r2_keys.reconcile(self.settings, self.run_active)

    def logs(self, since: str | None = None) -> list[dict[str, str]]:
        if self.settings.is_static:
            return []
        client = self.client()
        run = client.get_run(self.run_name)
        submission = (run or {}).get("latest_job_submission") or {}
        if not submission.get("id"):
            return []
        return client.poll_logs(self.run_name, str(submission["id"]), start_time=since)

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
        poll_interval: float = 3.0,
    ) -> dict[str, Any]:
        """Start the worker if needed and block until it is ready.

        Returns the ready status (with ``service_url``). Raises on timeout,
        on a failed run, or returns early with the current status on abort.
        """
        if self.settings.is_static:
            return self._wait_static(should_abort=should_abort, on_update=on_update,
                                     timeout=timeout, poll_interval=poll_interval)
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

    def _wait_static(self, *, should_abort, on_update, timeout: float, poll_interval: float) -> dict[str, Any]:
        """Block until the static worker takes jobs.

        Unreachable: wait up to ``timeout`` (as for a starting pod). Paused:
        wait without limit, and ask the wake URL (if any) to resume it.
        """
        deadline = self._clock() + timeout
        while True:
            current = self.status()
            state = current.get("state")
            on_update(current)
            if state in STATIC_READY_STATES:
                return current
            if should_abort():
                return current
            if state in ("not_configured", "unknown"):
                raise RemoteWorkerError(str(current.get("message") or "Remote worker unavailable"))
            if state == "paused":
                self.wake_static()
                deadline = self._clock() + timeout  # a pause never fails a job
            elif self._clock() > deadline:
                raise RemoteWorkerError(
                    f"Static worker unreachable for {int(timeout // 60)} min: {current.get('message') or ''}")
            self._sleep(poll_interval)

    # ── idle watchdog ──────────────────────────────────────────────────────

    def _idle_limit_seconds(self) -> float:
        settings = self.settings
        ttl = settings.idle_ttl_minutes * 60
        if settings.stop_when_queue_empty and not self._has_pending_work():
            return min(ttl, QUEUE_EMPTY_GRACE_SECONDS)
        return ttl

    def _has_pending_work(self) -> bool:
        try:
            return bool(self.pending_work())
        except Exception as exc:  # an unknown queue state must not stop the worker
            log.warning("remote worker pending-work check failed: %s", exc)
            return True

    def _review_hold_active(self) -> bool:
        try:
            waited = self.review_pause()
        except Exception as exc:  # an unknown review state must not keep a pod billing
            log.warning("remote worker review-pause check failed: %s", exc)
            return False
        return waited is not None and waited < REVIEW_HOLD_SECONDS

    def idle_tick(self) -> bool:
        """Stop the worker when idle past its limit. Returns True if stopped."""
        settings = self.settings
        if not settings.configured or settings.is_static:
            return False
        if self._review_hold_active():
            return False
        with self._lock:
            if self._active_jobs or self._ready_since is None or self._last_activity is None:
                return False
            idle = self._clock() - self._last_activity
        limit = self._idle_limit_seconds()
        if idle < limit:
            return False
        if limit < settings.idle_ttl_minutes * 60:
            self.stop(reason="Stopped: generation queue is empty")
        else:
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
                        if not self.settings.is_static:
                            self.idle_tick()
                            self.reconcile_pod_keys()  # revoke keys of finished pods
                except Exception as exc:  # never let the watchdog die
                    log.warning("remote worker watchdog: %s", exc)

        self._watchdog_stop.clear()
        self._watchdog = threading.Thread(target=_loop, name="maestro_remote_worker_idle", daemon=True)
        self._watchdog.start()

    def shutdown_watchdog(self) -> None:
        self._watchdog_stop.set()


def _proxy_reachable(service_url: str, token: str) -> bool:
    import requests

    from services.remote_worker.worker_spec import HEALTH_PATH

    try:
        response = requests.get(
            service_url + HEALTH_PATH.lstrip("/"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )
        return response.status_code == 200
    except Exception:
        return False


def _worker_downloads(service_url: str, token: str) -> dict[str, Any]:
    import requests

    try:
        response = requests.get(f"{service_url}api/v1/models/downloads/status",
                                headers={"Authorization": f"Bearer {token}"}, timeout=5)
        return (response.json() or {}).get("downloads") or {} if response.ok else {}
    except Exception:
        return {}


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
