"""Several remote worker pods sharing one queue.

The first pod is the ordinary ``RemoteWorkerManager`` (settings, UI status,
start/stop endpoints). With ``worker_count`` above 1 the pool adds pods named
``<run_name>-2``, ``<run_name>-3``... that share its settings and start on
demand. Jobs wait in one queue: each takes the first ready pod with room, so
work starts on the first pod up instead of waiting for the one it was dealt.
Pods start as the queue needs them (one per ``jobs_per_worker`` waiting or
running jobs), so a single job at a time keeps using the first pod while a
burst (a bible, a batch) spreads over all of them. Each pod stops on its own
idle clock.

``Slots`` bounds how many remote jobs the controller runs at once:
``worker_count × jobs_per_worker``. Holding more than one job per pod lets the
next job's transfers and preparation overlap the previous job's GPU work.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Mapping

from services.remote_worker.manager import RemoteWorkerError, RemoteWorkerManager

# How far each state is from taking a job, to report the closest pod while waiting.
_PROGRESS = {"ready": 4, "booting": 3, "starting": 2, "stopping": 1}


class RemoteWorkerPool:
    def __init__(self, primary: RemoteWorkerManager) -> None:
        self.primary = primary
        self._pods: list[RemoteWorkerManager] = [primary]
        self._lock = threading.Lock()
        # Jobs waiting for a pod, the pods this burst started, and the ones whose run failed.
        self._waiting = 0
        self._started: set[int] = set()
        self._failed: dict[int, str] = {}
        # One thread polls dstack for every waiting job.
        self._poll_lock = threading.Lock()
        self._statuses: dict[int, Mapping[str, Any]] = {}
        self._polled_at = float("-inf")

    def _extend(self, count: int) -> None:
        while len(self._pods) < count:
            pod = RemoteWorkerManager(self.primary._store, parent=self.primary, index=len(self._pods),
                                      client_factory=self.primary._client_factory)
            if self.primary._watchdog is not None:
                pod.start_watchdog()
            self._pods.append(pod)

    def pods(self) -> list[RemoteWorkerManager]:
        """The pods jobs may use now (``worker_count`` of them)."""
        count = self.primary.settings.pool_size
        with self._lock:
            self._extend(count)
            return list(self._pods[:count])

    def all_pods(self) -> list[RemoteWorkerManager]:
        """Every pod created so far, including ones beyond a lowered count."""
        with self._lock:
            return list(self._pods)

    def acquire(
        self,
        model_type: str | None = None,
        *,
        should_abort: Callable[[], bool] = lambda: False,
        on_update: Callable[[Mapping[str, Any]], None] = lambda status: None,
        timeout: float = 45 * 60,
        poll_interval: float = 3.0,
    ) -> RemoteWorkerManager | None:
        """Wait in the queue for a ready pod with room; returns it with its job count raised.

        ``job_finished`` releases it. Returns None when ``should_abort`` turns
        true. Without auto-start the least busy pod is returned at once (the
        caller's ``ensure_ready`` reports a stopped one).
        """
        primary = self.primary
        if not primary.settings.auto_start:
            with self._lock:
                self._extend(primary.settings.pool_size)
                pod = min(self._pods[:primary.settings.pool_size], key=lambda p: (p.active_jobs, not p.ready, p._index))
                pod.job_started()
                return pod
        deadline = primary._clock() + timeout
        with self._lock:
            self._waiting += 1
        try:
            while True:
                pod = self._take_free_pod()
                if pod is not None:
                    return pod
                if should_abort():
                    return None
                on_update(self._poll(model_type, poll_interval))
                pod = self._take_free_pod()
                if pod is not None:
                    return pod
                if primary._clock() > deadline:
                    raise RemoteWorkerError(f"No remote worker ready after {int(timeout // 60)} min")
                primary._sleep(poll_interval)
        finally:
            with self._lock:
                self._waiting -= 1
                if not self._waiting:
                    self._started.clear()
                    self._failed.clear()

    def _take_free_pod(self) -> RemoteWorkerManager | None:
        depth = self.primary.settings.queue_depth
        with self._lock:
            self._extend(self.primary.settings.pool_size)
            free = [p for p in self._pods[:self.primary.settings.pool_size] if p.ready and p.active_jobs < depth]
            if not free:
                return None
            pod = min(free, key=lambda p: (p.active_jobs, p._index))
            pod.job_started()
            return pod

    def _poll(self, model_type: str | None, poll_interval: float) -> Mapping[str, Any]:
        """Refresh the pods' states (once per interval for all waiting jobs) and start the ones the queue needs.

        Returns the status of the pod closest to taking a job, for progress messages.
        """
        with self._poll_lock:
            pods = self.pods()
            if self.primary._clock() - self._polled_at >= poll_interval:
                self._statuses = {pod._index: pod.status() for pod in pods}
                self._polled_at = self.primary._clock()
                depth = self.primary.settings.queue_depth
                with self._lock:
                    demand = sum(p.active_jobs for p in pods) + self._waiting
                for pod in pods:
                    status = self._statuses.get(pod._index) or {}
                    if (pod._index in self._started and status.get("state") == "stopped"
                            and str(status.get("message") or "").startswith("Last run failed")):
                        self._started.discard(pod._index)
                        self._failed[pod._index] = str(status["message"])
                live = [p for p in pods if (self._statuses.get(p._index) or {}).get("state") not in ("stopped", None)]
                wanted = min(len(pods), -(-demand // depth))
                for pod in pods:
                    if len(live) >= wanted:
                        break
                    state = (self._statuses.get(pod._index) or {}).get("state")
                    if state == "stopped" and pod._index not in self._failed:
                        self._statuses[pod._index] = pod.start(model_type)
                        self._started.add(pod._index)
                        live.append(pod)
                    elif state in ("not_configured", "unknown") and pod is self.primary:
                        raise RemoteWorkerError(str((self._statuses[pod._index]).get("message") or "Remote worker unavailable"))
                if not live and self._failed and len(self._failed) >= len(pods):
                    raise RemoteWorkerError(next(iter(self._failed.values())))
            statuses = [self._statuses.get(pod._index) or {} for pod in pods]
        return max(statuses, key=lambda s: _PROGRESS.get(str(s.get("state")), 0), default={})

    def capacity(self) -> int:
        settings = self.primary.settings
        return settings.pool_size * settings.queue_depth

    def status(self) -> list[dict[str, Any]]:
        """One line per pod for the UI (the first pod's full status stays the main one)."""
        rows = []
        for pod in self.all_pods():
            current = pod.status()
            rows.append({key: current.get(key) for key in (
                "run_name", "state", "active_jobs", "price_per_hour", "gpu_region", "cost", "message")})
        return rows

    def stop(self, reason: str = "Stopped by user") -> None:
        for pod in self.all_pods()[1:]:
            pod.stop(reason=reason)
        self.primary.stop(reason=reason)


class Slots:
    """A counting slot whose size follows a callable (the pool's capacity).

    Same ``acquire(timeout=)`` / ``release()`` shape as ``threading.Lock`` so
    the generation queue can hold it like its other locks.
    """

    def __init__(self, capacity) -> None:
        self._capacity = capacity
        self._taken = 0
        self._condition = threading.Condition()

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        deadline = None if timeout is None or timeout < 0 else time.monotonic() + timeout
        with self._condition:
            while self._taken >= max(1, int(self._capacity())):
                if not blocking:
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return False
                self._condition.wait(remaining if remaining is not None else 1.0)
            self._taken += 1
            return True

    def release(self) -> None:
        with self._condition:
            self._taken = max(0, self._taken - 1)
            self._condition.notify()

    @property
    def taken(self) -> int:
        return self._taken
