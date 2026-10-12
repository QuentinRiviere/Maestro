"""Backend-neutral contract for running one generation task.

``_run_generation`` in ``launch.py`` owns the job: queueing, the GPU slot,
progress bookkeeping, sidecars and output collection.  A backend owns only
the execution of a single validated task.  It reports through the existing
``send_cmd(cmd, data)`` stream protocol that WanGP already speaks, so the
consumer loop in ``launch.py`` does not need to know where inference runs
(in-process WanGP today, a remote ComfyUI tomorrow).

Rules every backend must follow:

* ``run_task`` blocks until the task is finished, failed or cancelled.
* Progress and previews go through ``send_cmd`` using the command names below.
* Every produced media file is written under the task's save path and its
  path is appended to ``state["gen"]["file_list"]`` (or ``audio_file_list``),
  exactly like ``wgp.generate_video`` does, so job output collection stays
  ownership-based.
* Cancellation is cooperative: the caller sets ``state["gen"]["abort"]`` and
  the backend must stop as soon as it observes it.
* Failures are raised as exceptions.  The caller turns them into an
  ``error`` command and always emits ``exit``; a backend never sends ``exit``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


SendCmd = Callable[[str, Any], None]

# Stream commands understood by the consumer loop in ``_run_generation``.
CMD_PROGRESS = "progress"
CMD_STATUS = "status"
CMD_INFO = "info"
CMD_PREVIEW = "preview"
CMD_OUTPUT = "output"
CMD_GENERATION_TIME = "generation_time"


@dataclass(frozen=True)
class TaskRequest:
    """One validated generation task, as produced by ``wgp.validate_task``."""

    task: Mapping[str, Any]
    params: Mapping[str, Any]

    @property
    def model_type(self) -> str:
        return str(self.params.get("model_type") or "")

    @property
    def state(self) -> Mapping[str, Any]:
        return self.params.get("state") or {}

    def abort_requested(self) -> bool:
        gen = self.state.get("gen") or {}
        return bool(gen.get("abort", False))


@runtime_checkable
class InferenceBackend(Protocol):
    """Executes one task and reports through ``send_cmd``."""

    name: str

    def run_task(self, request: TaskRequest, send_cmd: SendCmd) -> None:
        ...
