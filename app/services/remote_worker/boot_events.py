"""The worker's startup steps, read from its run logs for the UI.

The boot script and the R2 prefetch print ``[Boot] {json}`` lines; a few of
Maestro's own startup lines mark the engine and the API coming up. Each log
line carries dstack's timestamp, so every step gets a start, an end and a
duration without the worker having to report clock times.
"""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable, Mapping
from typing import Any

MARKER = "[Boot]"
# Plain startup lines of Maestro itself: (substring, phase, state).
_MAESTRO_LINES = (
    ("Importing WanGP engine", "engine", "started"),
    ("WanGP loaded", "engine", "done"),
    ("Application startup complete", "api", "done"),
)
LABELS = {
    "clone": "Code cloned",
    "container": "Container started",
    "compile": "Python bytecode compiled",
    "prefetch": "Weights pulled from R2",
    "kernels": "Compiled kernels pulled from R2",
    "maestro": "Maestro starting",
    "engine": "Engine loaded",
    "api": "API answering",
}


def relevant(message: str) -> bool:
    """Whether a log message holds a startup step (the rest is not kept)."""
    return MARKER in message or any(needle in message for needle, *_ in _MAESTRO_LINES)


def seconds(timestamp: str) -> float | None:
    try:
        return dt.datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return None


def parse_events(lines: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    """``{phase, state, at, ...}`` events from dstack log lines, in order."""
    events: list[dict[str, Any]] = []
    for line in lines:
        at = seconds(str(line.get("timestamp") or ""))
        for text in str(line.get("message") or "").splitlines():
            index = text.find(MARKER)
            if index >= 0:
                try:
                    event = json.loads(text[index + len(MARKER):].strip())
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("phase") and event.get("state"):
                    events.append({**event, "at": at})
                continue
            for needle, phase, state in _MAESTRO_LINES:
                if needle in text:
                    events.append({"phase": phase, "state": state, "at": at})
    return events


def steps(events: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One step per phase (per model for the prefetch), with its state and duration."""
    ordered: dict[tuple[str, str], dict[str, Any]] = {}
    for event in events:
        phase = str(event["phase"])
        model = str(event.get("model_type") or "")
        key = (phase, model)
        step = ordered.setdefault(key, {
            "phase": phase,
            "label": LABELS.get(phase, phase) + (f": {model}" if model else ""),
            "state": "started",
            "started_at": event.get("at"),
        })
        state = str(event["state"])
        if state == "progress":
            # Progress of a running step: only its details change.
            step.setdefault("details", {}).update(
                {k: v for k, v in event.items() if k not in {"phase", "state", "at", "model_type"}})
            continue
        if state == "started":
            step["started_at"] = step.get("started_at") or event.get("at")
            if phase == "container":
                step["state"] = "done"
                step["ended_at"] = event.get("at")
        else:
            step["state"] = state
            step["ended_at"] = event.get("at")
        details = {k: v for k, v in event.items() if k not in {"phase", "state", "at", "model_type"}}
        if details:
            step.setdefault("details", {}).update(details)
    # Maestro "started" ends when the engine starts importing.
    maestro, engine = ordered.get(("maestro", "")), ordered.get(("engine", ""))
    if maestro and engine and maestro["state"] == "started":
        maestro.update(state="done", ended_at=engine.get("started_at"))
    for step in ordered.values():
        if step.get("started_at") is not None and step.get("ended_at") is not None:
            step["seconds"] = round(max(0.0, step["ended_at"] - step["started_at"]), 1)
    return list(ordered.values())
