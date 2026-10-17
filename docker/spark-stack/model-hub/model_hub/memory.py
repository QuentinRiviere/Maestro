"""One view of the Spark's memory: the host, llama-swap's models and what the Maestro worker holds.

llama-swap only knows its own entries; the Maestro worker is one of them
(a proxy) and loads image and video models of its own behind it. The hub
puts the three together: the kernel's figures (``/proc/meminfo``), each
running llama-swap model with the hub's own memory estimate, and the
worker's state (``GET /api/v1/worker/state``: loaded models, free memory,
process size, the running job).
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping

GB = 1e9
WORKER_MODEL = "maestro-worker"


def parse_meminfo(text: str) -> dict[str, float]:
    """``/proc/meminfo`` in GB: total, available, used, swap_total, swap_used ({} when unreadable)."""
    values: dict[str, int] = {}
    for line in (text or "").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if not parts:
            continue
        try:
            number = int(parts[0])
        except ValueError:
            continue
        values[key.strip()] = number * 1024 if (len(parts) > 1 and parts[1].lower() == "kb") else number
    if "MemTotal" not in values:
        return {}
    total = values["MemTotal"]
    available = values.get("MemAvailable", values.get("MemFree", 0))
    swap_total = values.get("SwapTotal", 0)
    swap_free = values.get("SwapFree", 0)
    cached = values.get("Cached", 0) + values.get("Buffers", 0) + values.get("SReclaimable", 0)
    return {
        "total_gb": round(total / GB, 1),
        "available_gb": round(available / GB, 1),
        # used = what cannot be reclaimed: processes, shared/unified allocations, kernel.
        "used_gb": round((total - available) / GB, 1),
        "cache_gb": round(cached / GB, 1),
        "shmem_gb": round(values.get("Shmem", 0) / GB, 1),
        "swap_total_gb": round(swap_total / GB, 1),
        "swap_used_gb": round(max(0, swap_total - swap_free) / GB, 1),
    }


def read_meminfo(path: str = "/proc/meminfo") -> dict[str, float]:
    try:
        with open(path, encoding="utf-8") as handle:
            return parse_meminfo(handle.read())
    except OSError:
        return {}


def fetch_json(url: str, token: str = "", timeout: float = 3.0) -> tuple[dict | None, str]:
    """(body, "") or (None, error) for a GET with an optional bearer token."""
    request = urllib.request.Request(url)
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read(1 << 20))
    except urllib.error.HTTPError as error:
        return None, f"HTTP {error.code}"
    except (urllib.error.URLError, OSError, ValueError) as error:
        return None, str(getattr(error, "reason", error))
    return (body if isinstance(body, dict) else None), ("" if isinstance(body, dict) else "not a JSON object")


def worker_summary(state: Mapping[str, Any] | None, error: str = "") -> dict[str, Any]:
    """The worker's state reduced to what the memory view shows."""
    if state is None:
        return {"reachable": False, "error": error or "no answer"}
    memory = state.get("memory") if isinstance(state.get("memory"), Mapping) else {}
    job = state.get("running_job") if isinstance(state.get("running_job"), Mapping) else None
    summary: dict[str, Any] = {
        "reachable": True,
        "paused": bool(state.get("paused")),
        "loaded_models": [str(name) for name in state.get("loaded_models") or []],
        "queued": len(state.get("queued") or []) if isinstance(state.get("queued"), list) else state.get("queued"),
        "running_job": ({"id": job.get("job_id") or job.get("id"), "phase": job.get("phase") or job.get("message")}
                        if job else None),
        "idle_unload_seconds": state.get("idle_unload_seconds"),
    }
    for key in ("gpu_free_gb", "gpu_total_gb", "cuda_free_gb", "unified_memory"):
        if key in memory:
            summary[key] = memory[key]
    if state.get("process_rss_gb") is not None:
        summary["process_rss_gb"] = state["process_rss_gb"]
    return summary


def memory_view(
    host: Mapping[str, float],
    running: list[Mapping[str, Any]] | None,
    estimate_gb: Callable[[str], float | None],
    resident: Callable[[str], bool],
    worker: Mapping[str, Any],
    llama_swap_error: str = "",
) -> dict[str, Any]:
    """The unified view: host figures, llama-swap models with estimates, the worker's own models.

    ``running`` is llama-swap's ``/running`` list (None when unreachable);
    ``estimate_gb(model_id)`` and ``resident(model_id)`` come from the hub's
    entries. The worker entry of llama-swap carries the worker summary, so
    its models appear under it. ``accounted_gb`` sums the known estimates and
    the worker's process size; the rest of ``used_gb`` (``other_gb``) is
    other processes and containers, kernel and shared memory the hub cannot
    name. Page cache is not in ``used_gb`` (it is reclaimable) and shows
    apart as ``cache_gb``.
    """
    models: list[dict[str, Any]] = []
    accounted = 0.0
    for item in running or []:
        if not isinstance(item, Mapping):
            continue
        model_id = str(item.get("model") or "")
        entry: dict[str, Any] = {"model": model_id, "state": item.get("state"), "ttl": item.get("ttl"),
                                 "resident": bool(resident(model_id))}
        if model_id == WORKER_MODEL:
            entry["worker"] = dict(worker)
            rss = worker.get("process_rss_gb") if worker.get("reachable") else None
            if isinstance(rss, (int, float)):
                entry["estimated_gb"] = round(float(rss), 1)
                accounted += float(rss)
        else:
            estimate = estimate_gb(model_id)
            if estimate is not None:
                entry["estimated_gb"] = round(float(estimate), 1)
                accounted += float(estimate)
        models.append(entry)
    used = float(host.get("used_gb") or 0.0)
    return {
        "host": dict(host),
        "llama_swap": {"reachable": running is not None, "error": llama_swap_error, "models": models},
        "worker": dict(worker),
        "accounted_gb": round(accounted, 1),
        "other_gb": round(max(0.0, used - accounted), 1) if host else None,
    }
