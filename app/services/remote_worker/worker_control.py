"""Worker-side control of a Maestro worker that shares its machine.

A static worker (e.g. on a DGX Spark) runs next to other inference servers
that llama-swap starts and stops. To lend the GPU memory, the worker can:

* pause: queued jobs stay queued (the job that runs finishes), new
  submissions are refused with 423 so the controller keeps them;
* hand back its queue: every job not started yet ends ``cancelled`` with
  ``handed_back``, which the controller's forwarder resubmits later;
* unload its models: by default once the running job is done (never
  interrupting it), or 409 at once with ``wait=False``.

``release`` does all three and ``acquire`` resumes; llama-swap calls them
through docker/spark-stack/maestro_worker_lease.py. ``idle_unload_seconds``
optionally unloads the models once the worker has been idle that long.

Jobs reach the GPU through ``gate``, which wraps the generation lock and
never hands it out while paused.
"""
from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Iterable, MutableMapping
from typing import Any

from services.job_lifecycle import try_hand_back

log = logging.getLogger("maestro.remote_worker.control")

PAUSED_ENV = "MAESTRO_WORKER_START_PAUSED"
IDLE_UNLOAD_ENV = "MAESTRO_WORKER_IDLE_UNLOAD_SECONDS"
HANDED_BACK_MESSAGE = "Handed back to the controller"


class WorkerBusy(RuntimeError):
    """A generation holds the GPU and the caller asked not to wait."""


def gpu_memory_stats(
    cuda_free: int,
    cuda_total: int,
    *,
    unified: bool,
    mem_available: int | None,
) -> dict[str, Any]:
    """GPU memory figures for the worker's state.

    On a unified-memory GPU (DGX Spark GB10, Jetson) the CPU and the GPU share
    the system RAM, and cudaMemGetInfo counts the kernel page cache as used:
    after reading tens of GB of weights it reports far less free memory than
    the system can actually hand out. There, ``gpu_free_gb`` is the kernel's
    MemAvailable (page cache included), capped at the CUDA total, and the raw
    CUDA figure stays in ``cuda_free_gb``.
    """
    free = cuda_free
    if unified and mem_available is not None:
        free = min(mem_available, cuda_total)
    stats: dict[str, Any] = {
        "gpu_free_gb": round(free / 1e9, 2),
        "gpu_total_gb": round(cuda_total / 1e9, 2),
        "unified_memory": unified,
    }
    if unified:
        stats["cuda_free_gb"] = round(cuda_free / 1e9, 2)
    return stats


class WorkerControl:
    def __init__(
        self,
        gen_lock: threading.Lock,
        *,
        jobs: Callable[[], Iterable[MutableMapping[str, Any]]],
        release_models: Callable[[], list[str]],
        loaded_models: Callable[[], list[str]],
        memory: Callable[[], dict[str, Any]] = dict,
        hand_back: Callable[..., bool] = try_hand_back,
        clock: Callable[[], float] = time.monotonic,
        idle_unload_seconds: float = 0.0,
        start_paused: bool = False,
    ) -> None:
        self._gen_lock = gen_lock
        self._jobs = jobs
        self._release_models = release_models
        self._loaded_models = loaded_models
        self._memory = memory
        self._hand_back = hand_back
        self._clock = clock
        self.idle_unload_seconds = max(0.0, float(idle_unload_seconds or 0))
        self._lock = threading.Lock()
        self._resumed = threading.Event()
        self._paused = bool(start_paused)
        if not self._paused:
            self._resumed.set()
        self._last_activity = clock()
        self.gate = _PausableGate(self)
        self._idle_thread: threading.Thread | None = None

    # ── pause / resume ─────────────────────────────────────────────────────

    @property
    def paused(self) -> bool:
        return self._paused

    def pause(self) -> bool:
        """Stop starting jobs. Idempotent; returns whether it was running."""
        with self._lock:
            changed = not self._paused
            self._paused = True
            self._resumed.clear()
        if changed:
            log.info("worker paused: queued jobs wait, the running one finishes")
        return changed

    def resume(self) -> bool:
        """Start jobs again. Idempotent; returns whether it was paused."""
        with self._lock:
            changed = self._paused
            self._paused = False
            self._last_activity = self._clock()
            self._resumed.set()
        if changed:
            log.info("worker resumed")
        return changed

    # ── jobs ───────────────────────────────────────────────────────────────

    def _with_status(self, status: str) -> list[MutableMapping[str, Any]]:
        return [job for job in list(self._jobs()) if job.get("status") == status]

    def running_job(self) -> str | None:
        running = self._with_status("running")
        return str(running[0].get("id")) if running else None

    def queued(self) -> list[str]:
        return [str(job.get("id")) for job in self._with_status("queued")]

    def busy(self) -> bool:
        return self._gen_lock.locked() or self.running_job() is not None

    def unload_queue(self) -> list[str]:
        """Hand back every job that has not started; the running one is untouched."""
        handed = [
            str(job.get("id")) for job in self._with_status("queued")
            if self._hand_back(job, message=HANDED_BACK_MESSAGE, phase="")
        ]
        if handed:
            log.info("worker handed back %d queued job(s): %s", len(handed), ", ".join(handed))
        return handed

    # ── models ─────────────────────────────────────────────────────────────

    def unload_models(self, *, wait: bool = True, timeout: float | None = None) -> list[str]:
        """Release every loaded model once no generation holds the GPU.

        ``wait=False`` raises ``WorkerBusy`` instead of waiting for the
        running job. A running generation is never interrupted.
        """
        if not wait:
            if self.running_job() is not None or not self._gen_lock.acquire(blocking=False):
                raise WorkerBusy("A generation is running")
        elif not self._gen_lock.acquire(timeout=-1 if timeout is None else timeout):
            raise WorkerBusy("Timed out waiting for the running generation")
        try:
            released = list(self._release_models())
        finally:
            self._gen_lock.release()
        with self._lock:
            self._last_activity = self._clock()
        log.info("worker unloaded models: %s", ", ".join(released) or "none loaded")
        return released

    def loaded_models(self) -> list[str]:
        try:
            return list(self._loaded_models())
        except Exception as exc:  # a state report never fails
            log.warning("loaded models unknown: %s", exc)
            return []

    def memory(self) -> dict[str, Any]:
        try:
            return dict(self._memory())
        except Exception as exc:
            log.warning("memory stats unknown: %s", exc)
            return {}

    # ── lease (llama-swap) ─────────────────────────────────────────────────

    def release(self, *, wait: bool = True) -> dict[str, Any]:
        """Pause, hand back the queue, then unload the models (after the running job)."""
        self.pause()
        handed = self.unload_queue()
        released = self.unload_models(wait=wait)
        return {**self.state(), "handed_back": handed, "released": released}

    def acquire(self) -> dict[str, Any]:
        self.resume()
        return self.state()

    def state(self) -> dict[str, Any]:
        return {
            "paused": self._paused,
            "running_job": self.running_job(),
            "queued": self.queued(),
            "loaded_models": self.loaded_models(),
            "memory": self.memory(),
            "idle_unload_seconds": self.idle_unload_seconds,
        }

    # ── idle unload ────────────────────────────────────────────────────────

    def note_activity(self) -> None:
        with self._lock:
            self._last_activity = self._clock()

    def idle_tick(self) -> bool:
        """Unload the models after ``idle_unload_seconds`` without work. True when unloaded."""
        if self.idle_unload_seconds <= 0:
            return False
        if self.busy() or self.queued():
            self.note_activity()
            return False
        with self._lock:
            idle = self._clock() - self._last_activity
        if idle < self.idle_unload_seconds or not self.loaded_models():
            return False
        try:
            self.unload_models(wait=False)
        except WorkerBusy:
            return False
        log.info("worker idle for %ds: models unloaded", int(idle))
        return True

    def start_idle_watch(self, interval: float = 15.0) -> None:
        if self.idle_unload_seconds <= 0 or (self._idle_thread and self._idle_thread.is_alive()):
            return

        def _loop() -> None:
            while True:
                time.sleep(interval)
                try:
                    self.idle_tick()
                except Exception as exc:  # never let the watch die
                    log.warning("idle unload: %s", exc)

        self._idle_thread = threading.Thread(target=_loop, name="maestro_worker_idle_unload", daemon=True)
        self._idle_thread.start()


class _PausableGate:
    """The generation lock as jobs see it: never acquired while paused.

    Same ``acquire(blocking, timeout)`` / ``release()`` shape as
    ``threading.Lock``, so ``job_lifecycle.generation_slot`` takes it as is
    (it polls with a short timeout and re-checks cancellation in between).
    """

    def __init__(self, control: WorkerControl) -> None:
        self._control = control

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        control = self._control
        deadline = None if timeout is None or timeout < 0 else time.monotonic() + timeout
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if control.paused:
                if not blocking or remaining == 0:
                    return False
                control._resumed.wait(remaining)
                continue
            if not blocking:
                got = control._gen_lock.acquire(blocking=False)
            else:
                got = control._gen_lock.acquire(timeout=-1 if remaining is None else remaining)
            if not got:
                return False
            if control.paused:
                # Paused while this job waited for the lock: leave it queued.
                control._gen_lock.release()
                continue
            control.note_activity()
            return True

    def release(self) -> None:
        self._control.note_activity()
        self._control._gen_lock.release()

    def locked(self) -> bool:
        return self._control._gen_lock.locked()


def create_router(control: WorkerControl, *, enabled: Callable[[], bool] = lambda: True):
    """HTTP routes of ``control``: the probes and ``/api/v1/worker/...``.

    ``enabled`` is false outside a worker process: the control routes then
    answer 404 (the probes always answer). Handlers are sync, so a release
    waiting for a long render only holds one threadpool thread.
    """
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import JSONResponse

    router = APIRouter()

    def require_worker() -> None:
        if not enabled():
            raise HTTPException(status_code=404, detail="Not a remote worker")

    @router.get("/healthz", include_in_schema=False)
    def liveness():
        """Unauthenticated liveness probe: the process answers HTTP."""
        return {"status": "ok"}

    @router.get("/readyz", include_in_schema=False)
    def readiness():
        """Unauthenticated readiness probe: 503 while paused (llama-swap's checkEndpoint)."""
        if control.paused:
            return JSONResponse({"ready": False}, status_code=503)
        return {"ready": True}

    @router.get("/api/v1/worker/state")
    def state():
        require_worker()
        return control.state()

    @router.post("/api/v1/worker/pause")
    def pause():
        """Start no new job; the running one finishes. Idempotent."""
        require_worker()
        control.pause()
        return control.state()

    @router.post("/api/v1/worker/resume")
    def resume():
        require_worker()
        control.resume()
        return control.state()

    @router.post("/api/v1/worker/unload-queue")
    def unload_queue():
        """Hand every unstarted job back to its controller, which resubmits it later."""
        require_worker()
        return {"handed_back": control.unload_queue(), **control.state()}

    @router.post("/api/v1/worker/unload-models")
    def unload_models(wait: bool = True):
        """Release every model once the running job is done (409 at once with wait=false)."""
        require_worker()
        try:
            released = control.unload_models(wait=wait)
        except WorkerBusy as error:
            raise HTTPException(status_code=409, detail=str(error)) from None
        return {"released": released, **control.state()}

    @router.post("/api/v1/worker/release")
    def release(wait: bool = True):
        """Pause, hand back the queue, unload the models: what llama-swap calls to take the GPU."""
        require_worker()
        try:
            return control.release(wait=wait)
        except WorkerBusy as error:
            raise HTTPException(status_code=409, detail=str(error)) from None

    @router.post("/api/v1/worker/acquire")
    def acquire():
        """Resume after a release: what llama-swap calls when it gives the GPU back."""
        require_worker()
        return control.acquire()

    return router
