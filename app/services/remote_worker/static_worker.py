"""A static remote worker: Maestro always running at a fixed URL.

No dstack: nothing is started or stopped from here. The state comes from the
worker's own control API (``GET /api/v1/worker/state``, see worker_control.py):

* ``ready``       – reachable, not paused, idle.
* ``busy``        – reachable, not paused, running a job (it still takes
                    jobs: they queue on the worker).
* ``paused``      – the worker lent its GPU memory to another service (e.g.
                    llama-swap loaded a large LLM). Jobs wait in the
                    controller queue, without a timeout.
* ``unreachable`` – no answer. Jobs wait like for a starting pod, with the
                    same timeout.
* ``unknown``     – the worker refused the token: a configuration error that
                    fails jobs at once instead of waiting.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

log = logging.getLogger("maestro.remote_worker.static")

STATE_PATH = "api/v1/worker/state"
# States in which the worker takes jobs.
READY_STATES = ("ready", "busy")
# Probe results are reused this long: the UI, the pool and each waiting job poll.
PROBE_CACHE_SECONDS = 2.0
# A paused worker is woken at most this often (each wake may swap models).
WAKE_INTERVAL_SECONDS = 60.0
WAKE_TIMEOUT_SECONDS = 15 * 60

Probe = Callable[[str, str], tuple[int, Any, str]]


def normalize_url(url: str) -> str:
    url = (url or "").strip()
    return url.rstrip("/") + "/" if url else ""


def _http_probe(base_url: str, token: str) -> tuple[int, Any, str]:
    """(HTTP status, JSON body, error); status 0 when the worker did not answer."""
    import requests

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        response = requests.get(base_url + STATE_PATH, headers=headers, timeout=5)
    except Exception as exc:  # connection refused, DNS, timeout
        return 0, None, str(exc)
    try:
        body = response.json()
    except ValueError:
        body = None
    return response.status_code, body, ""


def redact(url: str) -> str:
    """``url`` without the credentials a wake URL may carry (``user:key@host``)."""
    parts = urlsplit(url)
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc="***@" + parts.netloc.rsplit("@", 1)[1]))


def _http_wake(url: str) -> None:
    import requests

    try:
        response = requests.get(url, timeout=WAKE_TIMEOUT_SECONDS)
        log.info("static worker wake %s: HTTP %s", redact(url), response.status_code)
    except Exception as exc:
        log.warning("static worker wake %s failed: %s", redact(url), type(exc).__name__)


def describe(code: int, body: Any, error: str, url: str) -> dict[str, Any]:
    """Controller status fields from one probe of the worker."""
    if code == 0:
        return {"state": "unreachable", "message": f"Worker at {url} unreachable: {error}"}
    if code in (401, 403):
        return {"state": "unknown",
                "message": f"Worker at {url} refused the token (HTTP {code}): check the static worker token"}
    if code == 404:
        # A worker without the control API (older Maestro): it cannot pause.
        return {"state": "ready", "message": "Worker has no control API (older Maestro)"}
    if code != 200 or not isinstance(body, Mapping):
        return {"state": "unreachable", "message": f"Worker at {url} answered HTTP {code}"}
    worker = {key: body.get(key) for key in ("paused", "running_job", "queued", "loaded_models", "memory")}
    if body.get("paused"):
        state, message = "paused", "Worker paused: its GPU is lent to another service. Jobs wait here."
    elif body.get("running_job"):
        state, message = "busy", ""
    else:
        state, message = "ready", ""
    return {"state": state, "message": message, "worker": worker}


class StaticWorker:
    """Probe cache and wake throttle of one static worker."""

    def __init__(self, *, clock: Callable[[], float], probe: Probe | None = None,
                 wake: Callable[[str], None] | None = None) -> None:
        self._clock = clock
        self._probe = probe or _http_probe
        self._wake = wake or _http_wake
        self._lock = threading.Lock()
        self._cached: tuple[tuple[str, str], float, dict[str, Any]] | None = None
        self._woken_at: float | None = None

    def status(self, url: str, token: str) -> dict[str, Any]:
        base = normalize_url(url)
        key = (base, token)
        now = self._clock()
        with self._lock:
            if self._cached and self._cached[0] == key and now - self._cached[1] < PROBE_CACHE_SECONDS:
                return dict(self._cached[2])
        code, body, error = self._probe(base, token)
        described = {
            **describe(code, body, error, base),
            "service_url": base,
            # An owned machine: no hourly price, no accrued cost.
            "price_per_hour": 0.0,
            "provider": "static",
        }
        with self._lock:
            self._cached = (key, now, described)
        return dict(described)

    def forget(self) -> None:
        with self._lock:
            self._cached = None

    def wake(self, url: str) -> bool:
        """Request ``url`` in the background, at most once per interval. True when sent."""
        if not url:
            return False
        now = self._clock()
        with self._lock:
            if self._woken_at is not None and now - self._woken_at < WAKE_INTERVAL_SECONDS:
                return False
            self._woken_at = now
        log.info("static worker paused with jobs waiting: waking it through %s", redact(url))
        threading.Thread(target=self._wake, args=(url,), name="maestro_static_worker_wake",
                         daemon=True).start()
        return True
