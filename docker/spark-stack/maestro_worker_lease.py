#!/usr/bin/env python3
"""Hold the Maestro worker's GPU lease for llama-swap (stdlib only).

llama-swap starts this as the ``cmd`` of its ``maestro-worker`` model. The
Maestro worker itself runs in its own container and is never started or
killed by llama-swap: this process only lends it the GPU.

* On start: ``POST /api/v1/worker/acquire`` (resume), retried until the
  worker answers. llama-swap's ``checkEndpoint: /readyz`` turns healthy once
  the worker is resumed.
* While running: every ``--check-interval`` seconds, re-acquire if the worker
  reports paused (e.g. its container restarted with
  MAESTRO_WORKER_START_PAUSED=1).
* On SIGTERM/SIGINT (llama-swap unloads or swaps the model):
  ``POST /api/v1/worker/release``, which pauses the worker, hands its
  unstarted jobs back to the controller, waits for the running job to finish
  and unloads every model. This process exits once the release returned, so
  llama-swap only starts the next model when the memory is free.

Never kills a job. If llama-swap kills this process hard before the release
returned (its stop timeout is shorter than the render), the worker still
completes the release on its own (pause, then unload after the job) and stays
paused but alive; the next start of this lease resumes it.

Environment: MAESTRO_WORKER_URL (default http://127.0.0.1:7860) and
MAESTRO_WORKER_TOKEN (the worker's bearer token, if it requires one).
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

DEFAULT_URL = "http://127.0.0.1:7860"


def log(message: str) -> None:
    print(f"[maestro-lease] {time.strftime('%H:%M:%S')} {message}", file=sys.stderr, flush=True)


class AuthError(RuntimeError):
    pass


class Worker:
    def __init__(self, url: str, token: str) -> None:
        self.url = url.rstrip("/")
        self.token = token

    def call(self, method: str, path: str, timeout: float | None) -> dict:
        request = urllib.request.Request(self.url + path, method=method, data=b"" if method == "POST" else None)
        if self.token:
            request.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            if error.code in (401, 403):
                raise AuthError(f"worker refused the token (HTTP {error.code})") from None
            raise
        return json.loads(body or b"{}")

    def acquire(self) -> dict:
        return self.call("POST", "/api/v1/worker/acquire", timeout=30)

    def state(self) -> dict:
        return self.call("GET", "/api/v1/worker/state", timeout=10)

    def release(self, timeout: float | None) -> dict:
        return self.call("POST", "/api/v1/worker/release", timeout=timeout)


def acquire_until_done(worker: Worker, stop: threading.Event, retry: float) -> bool:
    """Resume the worker, retrying while it is unreachable. False if asked to stop first."""
    while not stop.is_set():
        try:
            state = worker.acquire()
            log(f"acquired: loaded {state.get('loaded_models') or []}")
            return True
        except AuthError:
            raise
        except (OSError, ValueError, http.client.HTTPException) as error:  # URLError, HTTPError, timeouts, bad JSON
            log(f"worker not reachable yet ({error}), retrying")
        stop.wait(retry)
    return False


def release_until_done(worker: Worker, timeout: float | None, retry: float) -> bool:
    """Release the worker; True once released, or when it is not running at all."""
    deadline = None if not timeout else time.monotonic() + timeout
    while True:
        remaining = None if deadline is None else max(1.0, deadline - time.monotonic())
        try:
            state = worker.release(remaining)
            log(f"released: unloaded {state.get('released') or []}, handed back {state.get('handed_back') or []}")
            return True
        except urllib.error.HTTPError as error:
            log(f"release answered HTTP {error.code}, retrying")
        except urllib.error.URLError as error:
            if isinstance(error.reason, ConnectionRefusedError):
                # Nothing listens: the worker process is down, so it holds no memory.
                log("worker not running: nothing to release")
                return True
            log(f"release failed ({error.reason}), retrying")
        except (OSError, ValueError, http.client.HTTPException) as error:  # read timeout of a long render
            log(f"release still waiting ({error}), retrying")
        if deadline is not None and time.monotonic() >= deadline:
            log("release timed out; the worker finishes it on its own")
            return False
        time.sleep(retry)


def run(argv: list[str] | None = None, stop: threading.Event | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default=os.environ.get("MAESTRO_WORKER_URL", DEFAULT_URL))
    parser.add_argument("--check-interval", type=float, default=15.0,
                        help="seconds between checks that the worker is still resumed")
    parser.add_argument("--release-timeout", type=float, default=0.0,
                        help="give up waiting for the release after this many seconds (0: never)")
    parser.add_argument("--retry", type=float, default=2.0, help="seconds between retries")
    args = parser.parse_args(argv)
    worker = Worker(args.url, os.environ.get("MAESTRO_WORKER_TOKEN", "").strip())
    stop = stop or threading.Event()
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, lambda *_: stop.set())

    try:
        if acquire_until_done(worker, stop, args.retry):
            while not stop.wait(args.check_interval):
                try:
                    if worker.state().get("paused"):
                        log("worker paused while leased: resuming it")
                        worker.acquire()
                except AuthError:
                    raise
                except (OSError, ValueError, http.client.HTTPException) as error:
                    log(f"state check failed ({error})")
        log("stop requested: releasing the worker")
        released = release_until_done(worker, args.release_timeout or None, args.retry)
    except AuthError as error:
        log(str(error))
        return 2
    return 0 if released else 1


if __name__ == "__main__":
    sys.exit(run())
