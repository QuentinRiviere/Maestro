"""Pull the weights a worker will need from R2 while Maestro is still starting.

The boot script runs ``python -m services.remote_worker.prefetch a,b`` in the
background before launching Maestro, with the models of the work that made
the controller start the pod (``MAESTRO_PRELOAD_MODELS``). Importing the
engine and waiting for the dstack route take a minute or two; the R2 pull of
a ~20 GB model fits in that window instead of following it.

Only the cache is touched (no engine import), so this starts in about a
second. Maestro's own download of a model waits for its prefetch to finish
(``wait_for``), then finds the files on disk and only checks them.

Progress goes to stdout as ``[Boot] {json}`` lines, which the controller reads
from the run logs to show the worker's startup steps.
"""
from __future__ import annotations

import json
import os
import sys
import time

from services.remote_worker.checkpoint_cache import CheckpointCacheError, R2CheckpointCache

BOOT_MARKER = "[Boot]"
WAIT_TIMEOUT_SECONDS = 30 * 60


def boot_event(phase: str, state: str, **extra) -> None:
    print(f"{BOOT_MARKER} " + json.dumps({"phase": phase, "state": state, **extra}), flush=True)


def _marker_dir() -> str:
    return os.path.join(os.environ.get("MAESTRO_DATA", "/workspace/data"), ".prefetch")


def _marker(model_type: str, state: str) -> str:
    return os.path.join(_marker_dir(), f"{model_type}.{state}")


def worker_roots() -> dict[str, str]:
    """The cache roots of a worker, as the boot script configures them."""
    data = os.environ.get("MAESTRO_DATA", "/workspace/data")
    roots = {"ckpts": os.path.join(data, "ckpts"), "loras": os.path.join(data, "loras")}
    if os.environ.get("HF_HOME"):
        roots["hf"] = os.environ["HF_HOME"]
    return roots


def wait_for(model_type: str, timeout: float = WAIT_TIMEOUT_SECONDS) -> None:
    """Block while the boot prefetch of ``model_type`` is still pulling files."""
    deadline = time.time() + timeout
    while os.path.exists(_marker(model_type, "running")) and time.time() < deadline:
        time.sleep(1)


def prefetch(model_types: list[str], cache: R2CheckpointCache | None) -> int:
    os.makedirs(_marker_dir(), exist_ok=True)
    if cache is None:
        boot_event("prefetch", "skipped", reason="no R2 cache on this pod")
        return 0
    # Claim every model first, so Maestro waits even for those still queued.
    for model_type in model_types:
        open(_marker(model_type, "running"), "w").close()
    failures = 0
    for model_type in model_types:
        started = time.time()
        boot_event("prefetch", "started", model_type=model_type)
        try:
            files, _links = cache.read_manifest(model_type)
            pulled = cache.hydrate(model_type)
        except CheckpointCacheError as exc:
            failures += 1
            boot_event("prefetch", "failed", model_type=model_type, error=str(exc)[:300])
        else:
            boot_event("prefetch", "done", model_type=model_type, files=pulled,
                       gigabytes=round(sum(f.size for f in files) / 1e9, 1),
                       seconds=round(time.time() - started, 1))
        finally:
            os.replace(_marker(model_type, "running"), _marker(model_type, "done"))
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    model_types = [m for m in (argv[1] if len(argv) > 1 else "").split(",") if m]
    if not model_types:
        return 0
    return prefetch(model_types, R2CheckpointCache.from_environment(worker_roots()))


if __name__ == "__main__":
    sys.exit(main(sys.argv))
