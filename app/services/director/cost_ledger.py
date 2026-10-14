"""What a Director project spent: GPU pod time, cloud API calls, $ per final minute.

Every billed job of a project appends one JSON line to
``settings/director_costs/<pid>.jsonl``:

- ``gpu``: a job on a remote worker pod, with the pod's run, price and
  accrued cost when the job got the pod and when it ended;
- ``cloud``: a hosted API call (fal, Replicate) with its estimated price, or
  ``usd: null`` when the model has no known price.

``summary`` turns the lines into the project's spend. A pod run is billed for
its whole cost growth over the project, boot and idle included; a run the
project itself started counts from zero. The share no job computed is shown
as ``overhead`` (boot, model loading, transfers, waiting).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

COSTS_DIR = Path(__file__).resolve().parents[2] / "settings" / "director_costs"
_LOCK = threading.Lock()
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


def _path(pid: str) -> Path:
    return COSTS_DIR / f"{_SAFE_ID.sub('_', str(pid))[:120] or 'project'}.jsonl"


def record(pid: str, kind: str, **entry: Any) -> None:
    """Append one billed job of project ``pid`` (never raises: a report must not fail a render)."""
    if not pid:
        return
    line = json.dumps({"ts": time.time(), "kind": kind, **entry}, default=str)
    try:
        with _LOCK:
            COSTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(_path(pid), "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except OSError as error:
        print(f"[Costs] Could not record a {kind} cost for {pid}: {error}")


def entries(pid: str) -> list[dict]:
    try:
        with open(_path(pid), encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            item = json.loads(line)
        except ValueError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def _number(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _epoch(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _gpu_runs(items: Iterable[dict], project_start: float) -> dict[str, float]:
    """Each pod run's cost over the project."""
    runs: dict[str, dict] = {}
    for item in items:
        run = str(item.get("run") or item.get("pod") or "")
        start, end = _number(item.get("cost_start")), _number(item.get("cost_end"))
        price = _number(item.get("price_per_hour")) or 0.0
        if end is None:
            # No accrued cost from dstack: bill the job's own time.
            seconds = (_number(item.get("wait_seconds")) or 0) + (_number(item.get("work_seconds")) or 0)
            runs.setdefault(f"{run}#{item.get('ts')}", {"fixed": price * seconds / 3600})
            continue
        current = runs.setdefault(run, {"low": start if start is not None else end, "high": end,
                                        "submitted": _epoch(item.get("submitted_at"))})
        current["low"] = min(current["low"], start if start is not None else end)
        current["high"] = max(current["high"], end)
    costs = {}
    for run, data in runs.items():
        if "fixed" in data:
            costs[run] = data["fixed"]
            continue
        started_here = data["submitted"] is not None and data["submitted"] >= project_start
        costs[run] = max(0.0, data["high"] - (0.0 if started_here else data["low"]))
    return costs


def summary(pid: str, *, final_seconds: float | None = None, rendered_seconds: float | None = None) -> dict:
    """The project's spend by stage, with $ per final minute when its length is known."""
    items = entries(pid)
    gpu = [i for i in items if i.get("kind") == "gpu"]
    cloud = [i for i in items if i.get("kind") == "cloud"]
    project_start = min((_number(i.get("queued_at")) or _number(i.get("ts")) or time.time() for i in items),
                        default=time.time())
    run_costs = _gpu_runs(gpu, project_start)
    stages: dict[str, dict] = {}

    def stage(name: str) -> dict:
        return stages.setdefault(name or "other", {"usd": 0.0, "jobs": 0, "gpu_seconds": 0.0, "unpriced": 0})

    computed = 0.0
    for item in gpu:
        seconds = _number(item.get("work_seconds")) or 0.0
        usd = (_number(item.get("price_per_hour")) or 0.0) * seconds / 3600
        computed += usd
        bucket = stage(str(item.get("stage") or ""))
        bucket["usd"] += usd
        bucket["jobs"] += 1
        bucket["gpu_seconds"] += seconds
    gpu_total = sum(run_costs.values())
    if gpu:
        stage("overhead")["usd"] += max(0.0, gpu_total - computed)
    for item in cloud:
        bucket = stage(str(item.get("stage") or ""))
        bucket["jobs"] += 1
        usd = _number(item.get("usd"))
        if usd is None:
            bucket["unpriced"] += 1
        else:
            bucket["usd"] += usd
    total = max(gpu_total, computed) + sum(_number(i.get("usd")) or 0.0 for i in cloud)
    for bucket in stages.values():
        bucket["usd"] = round(bucket["usd"], 4)
        bucket["gpu_seconds"] = round(bucket["gpu_seconds"], 1)
    final_minutes = (final_seconds or 0) / 60
    return {
        "usd": round(total, 4),
        "gpu_usd": round(max(gpu_total, computed), 4),
        "cloud_usd": round(sum(_number(i.get("usd")) or 0.0 for i in cloud), 4),
        "unpriced_calls": sum(1 for i in cloud if _number(i.get("usd")) is None),
        "pods": len(run_costs),
        "stages": stages,
        "final_seconds": round(final_seconds, 2) if final_seconds else None,
        "rendered_seconds": round(rendered_seconds, 2) if rendered_seconds else None,
        "usd_per_final_minute": round(total / final_minutes, 3) if final_minutes > 0 else None,
    }


def project_lengths(clips: Iterable[dict]) -> tuple[float, float]:
    """(final seconds the edit keeps, seconds rendered) of a project's clips."""
    from .edit_cut import normalize_keep

    final = rendered = 0.0
    for clip in clips or []:
        if not isinstance(clip, dict):
            continue
        duration = _number(clip.get("_director_duration_sec") or clip.get("duration_sec"))
        if duration is None:
            duration = max(0.0, (_number(clip.get("end")) or 0.0) - (_number(clip.get("start")) or 0.0))
        rendered += duration
        keep = normalize_keep(clip.get("_director_keep_sec"), duration)
        final += (keep[1] - keep[0]) if keep else duration
    return final, rendered


def stage_of(params: dict) -> str:
    """The Director stage a job belongs to: its own tag, else image or video."""
    tag = str(params.get("_director_cost_stage") or "")
    if tag:
        return tag
    return "image" if int(params.get("image_mode") or 0) > 0 else "video"


def clear(pid: str) -> None:
    try:
        os.remove(_path(pid))
    except OSError:
        pass
