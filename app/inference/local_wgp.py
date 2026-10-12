"""In-process WanGP backend: the historical ``wgp.generate_video`` path."""
from __future__ import annotations

import inspect
from typing import Any

from inference.contract import SendCmd, TaskRequest


class LocalWgpBackend:
    """Runs a task with the WanGP engine loaded in this process.

    ``generate_video`` already speaks the stream protocol, registers its
    outputs in ``state["gen"]`` and honours the abort flag, so this backend is
    a thin adapter.  The ``wgp`` module is injected rather than imported so
    this file stays importable without the engine (tests, remote-only hosts).
    """

    name = "local"

    def __init__(self, wgp_module: Any) -> None:
        self._wgp = wgp_module

    def run_task(self, request: TaskRequest, send_cmd: SendCmd) -> None:
        generate_video = self._wgp.generate_video
        expected_args = set(inspect.signature(generate_video).parameters.keys())
        filtered_params = {
            k: v for k, v in request.params.items() if k in expected_args
        }
        plugin_data = request.task.get("plugin_data", {})
        generate_video(
            request.task,
            send_cmd,
            plugin_data=plugin_data,
            **filtered_params,
        )
