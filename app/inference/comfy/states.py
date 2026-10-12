"""Pure decoding of ComfyUI ``/queue`` and ``/history`` snapshots.

Ported from videogen-n8n (``server/rendu/executeur/contrat.ts``), where the
rules below were measured against open-source ComfyUI:

* A history entry is only written at ``task_done``, so its presence proves the
  prompt finished.  ``{}`` means queued, running, never submitted *or* evicted.
* ``LOST`` (the only verdict that leads to a paid resubmission) therefore needs
  "absent from history" AND "absent from both queues".  The caller must read
  ``/queue`` *before* ``/history``: a prompt finishing between the two reads
  then shows up in history instead of looking lost.
* A user interrupt and a crash both end as ``status_str == "error"``; only the
  ``execution_interrupted`` message tells them apart.
* An unknown ``status_str`` is never treated as a failure: the job may still be
  running (and billing).  It is reported as running with a trace.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


log = logging.getLogger("maestro.inference.comfy")

INTERRUPT_MESSAGE = "execution_interrupted"


class PromptState(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"


_STATUS_STR = {
    "success": PromptState.SUCCEEDED,
    "error": PromptState.FAILED,
}


@dataclass(frozen=True)
class StateDecision:
    state: PromptState
    # Raw external vocabulary kept for diagnostics: status_str, or which queue.
    raw: str
    origin: str  # "history" | "queue"
    trace: str | None = None


@dataclass(frozen=True)
class ComfyFile:
    filename: str
    subfolder: str = ""
    type: str = "output"


def _contains_interrupt(messages: Any, depth: int = 0) -> bool:
    # Only descend into lists: the payload dicts come from the graph and any
    # value there could carry the marker string.
    if depth > 4 or not isinstance(messages, list):
        return False
    for item in messages:
        if item == INTERRUPT_MESSAGE:
            return True
        if isinstance(item, list) and _contains_interrupt(item, depth + 1):
            return True
    return False


def queue_position(queue: Mapping[str, Any] | None, prompt_id: str) -> str | None:
    """Return ``queue_running``/``queue_pending`` holding ``prompt_id``.

    Queue items are tuples whose index 1 is the prompt id; other indexes carry
    extra_data and output node lists and must not be searched.
    """
    if not isinstance(queue, Mapping):
        return None
    for name in ("queue_running", "queue_pending"):
        items = queue.get(name)
        if not isinstance(items, list):
            continue
        for item in items:
            if isinstance(item, list) and len(item) > 1 and item[1] == prompt_id:
                return name
    return None


def decide_state(
    prompt_id: str,
    history: Mapping[str, Any] | None,
    queue: Mapping[str, Any] | None,
) -> StateDecision:
    """Translate two already-read snapshots into a state. No I/O."""
    entry = history.get(prompt_id) if isinstance(history, Mapping) else None
    if isinstance(entry, Mapping):
        status = entry.get("status")
        if not isinstance(status, Mapping):
            # ``status`` is Optional upstream; an entry without it concludes nothing.
            return StateDecision(PromptState.RUNNING, "", "history")
        raw = status.get("status_str")
        raw = raw if isinstance(raw, str) else ""
        state = _STATUS_STR.get(raw)
        if state is None:
            trace = f"comfy unknown status_str for {prompt_id}: {raw[:200]!r}"
            log.warning(trace)
            return StateDecision(PromptState.RUNNING, raw, "history", trace)
        if state is PromptState.FAILED and _contains_interrupt(status.get("messages")):
            return StateDecision(PromptState.CANCELLED, raw, "history")
        return StateDecision(state, raw, "history")

    place = queue_position(queue, prompt_id)
    if place is not None:
        return StateDecision(PromptState.RUNNING, place, "queue")
    return StateDecision(PromptState.LOST, "", "queue")


def extract_outputs(entry: Mapping[str, Any], output_node: str) -> list[ComfyFile]:
    """Files produced by the declared output node only.

    Scanning every node would also pick up previews and intermediate saves.
    """
    outputs = entry.get("outputs") if isinstance(entry, Mapping) else None
    node_out = outputs.get(output_node) if isinstance(outputs, Mapping) else None
    files: list[ComfyFile] = []
    if not isinstance(node_out, Mapping):
        return files
    for value in node_out.values():
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, Mapping):
                continue
            filename = item.get("filename")
            if not isinstance(filename, str) or not filename:
                continue
            subfolder = item.get("subfolder")
            file_type = item.get("type")
            files.append(ComfyFile(
                filename=filename,
                subfolder=subfolder if isinstance(subfolder, str) else "",
                type=file_type if isinstance(file_type, str) else "output",
            ))
    return files


def extract_error(entry: Mapping[str, Any]) -> str:
    """Best-effort human message from a failed history entry."""
    status = entry.get("status") if isinstance(entry, Mapping) else None
    messages = status.get("messages") if isinstance(status, Mapping) else None
    if isinstance(messages, list):
        for item in messages:
            if (
                isinstance(item, list)
                and len(item) == 2
                and item[0] == "execution_error"
                and isinstance(item[1], Mapping)
            ):
                payload = item[1]
                node_type = payload.get("node_type") or "?"
                node_id = payload.get("node_id") or "?"
                message = str(payload.get("exception_message") or "").strip()
                return f"ComfyUI node {node_id} ({node_type}) failed: {message[:500]}"
    return "ComfyUI reported an execution error without details"
