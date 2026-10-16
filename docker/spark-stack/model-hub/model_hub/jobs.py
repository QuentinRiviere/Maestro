"""Background download queue: one job at a time, cancel and resume.

A job downloads a list of files of one repo at one commit through the shared
verified fetcher, then calls `on_done(job)` (the app records the entry in
models.yml and re-renders llama-swap.yaml). Cancelling keeps the partial
files in <models>/.staging, so resuming continues where it stopped.
"""
from __future__ import annotations

import itertools
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable

from . import fetcher

ACTIVE = ("queued", "running")


@dataclass
class Job:
    id: str
    repo: str
    commit: str
    files: list[fetcher.RemoteFile]
    serve: dict | None = None
    state: str = "queued"          # queued | running | done | failed | cancelled
    error: str = ""
    message: str = ""
    current: str = ""
    phase: str = ""
    done_bytes: int = 0
    rate: float = 0.0
    created: float = field(default_factory=time.time)
    finished: float | None = None
    cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _file_done: int = 0

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    def as_dict(self) -> dict:
        remaining = max(self.total_bytes - self.done_bytes, 0)
        eta = remaining / self.rate if self.rate > 0 and self.phase == "download" else None
        return {"id": self.id, "repo": self.repo, "commit": self.commit, "files": [f.name for f in self.files],
                "state": self.state, "error": self.error, "message": self.message, "current": self.current,
                "phase": self.phase, "done_bytes": self.done_bytes, "total_bytes": self.total_bytes,
                "rate": self.rate, "eta_seconds": eta, "created": self.created, "finished": self.finished,
                "serve": self.serve}


class Queue:
    def __init__(self, store: fetcher.Store, on_done: Callable[[Job], str] | None = None):
        self.store = store
        self.on_done = on_done
        self._jobs: dict[str, Job] = {}
        self._pending: deque[str] = deque()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._ids = itertools.count(1)
        self._thread = threading.Thread(target=self._run, name="downloads", daemon=True)
        self._thread.start()

    def submit(self, repo: str, commit: str, files: list[fetcher.RemoteFile], serve: dict | None = None) -> Job:
        with self._lock:
            for job in self._jobs.values():
                if job.repo == repo and job.state in ACTIVE:
                    raise ValueError(f"{repo} is already queued or downloading")
            job = Job(f"j{next(self._ids)}", repo, commit, files, serve)
            self._jobs[job.id] = job
            self._pending.append(job.id)
            self._wake.notify()
            return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self) -> list[dict]:
        return [j.as_dict() for j in sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)][:50]

    def cancel(self, job_id: str) -> Job:
        job = self._jobs[job_id]
        with self._lock:
            if job.state == "queued":
                self._pending.remove(job.id)
                job.state, job.finished = "cancelled", time.time()
            elif job.state == "running":
                job.cancel.set()
        return job

    def resume(self, job_id: str) -> Job:
        """Queue a cancelled or failed job again; partial files are reused."""
        old = self._jobs[job_id]
        if old.state not in ("cancelled", "failed"):
            raise ValueError(f"job {job_id} is {old.state}")
        return self.submit(old.repo, old.commit, old.files, old.serve)

    def active_repos(self) -> set[str]:
        return {j.repo for j in self._jobs.values() if j.state in ACTIVE}

    def _run(self) -> None:
        while True:
            with self._lock:
                while not self._pending:
                    self._wake.wait()
                job = self._jobs[self._pending.popleft()]
                job.state = "running"
            self._execute(job)

    def _execute(self, job: Job) -> None:
        try:
            self.store.check_disk(job.repo, job.files)
            for f in job.files:
                job.current = f.name

                def progress(p: dict, job=job) -> None:
                    job.phase = p["phase"]
                    job.done_bytes = job._file_done + p["done"]
                    job.rate = p["rate"]

                result = self.store.fetch_file(job.repo, job.commit, f, progress=progress, cancel=job.cancel)
                job._file_done += f.size
                job.done_bytes = job._file_done
                job.message = f"{f.name}: {result}"
            job.phase, job.current = "", ""
            if self.on_done is not None:
                job.message = self.on_done(job) or "done"
            job.state = "done"
        except fetcher.Cancelled:
            job.state, job.message = "cancelled", "cancelled; partial files kept for resume"
        except Exception as exc:  # noqa: BLE001 - reported to the UI
            job.state, job.error = "failed", str(exc)
        finally:
            job.finished = time.time()
            job.rate = 0.0
