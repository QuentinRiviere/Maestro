"""Several remote worker pods sharing one queue.

The first pod is the ordinary ``RemoteWorkerManager`` (settings, UI status,
start/stop endpoints). With ``worker_count`` above 1 the pool adds pods named
``<run_name>-2``, ``<run_name>-3``... that share its settings and start on
demand: each job goes to the least busy pod, so a single job at a time keeps
using the first pod while a burst (a bible, a batch) spreads over all of them.
Each pod stops on its own idle clock.

``Slots`` bounds how many remote jobs the controller runs at once:
``worker_count × jobs_per_worker``. Holding more than one job per pod lets the
next job's transfers and preparation overlap the previous job's GPU work.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from services.remote_worker.manager import RemoteWorkerManager


class RemoteWorkerPool:
    def __init__(self, primary: RemoteWorkerManager) -> None:
        self.primary = primary
        self._pods: list[RemoteWorkerManager] = [primary]
        self._lock = threading.Lock()

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

    def acquire(self) -> RemoteWorkerManager:
        """The least busy pod, its job count already raised (``job_finished`` releases it)."""
        with self._lock:
            self._extend(self.primary.settings.pool_size)
            candidates = self._pods[:self.primary.settings.pool_size]
            pod = min(candidates, key=lambda p: (p.active_jobs, not p.ready, p._index))
            pod.job_started()
            return pod

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
