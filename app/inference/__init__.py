"""Inference backends: where a validated generation task actually runs."""
from inference.contract import InferenceBackend, SendCmd, TaskRequest
from inference.local_wgp import LocalWgpBackend
from inference.registry import (
    BackendRegistry,
    UnknownBackendError,
    resolve_backend_name,
)

__all__ = [
    "BackendRegistry",
    "InferenceBackend",
    "LocalWgpBackend",
    "SendCmd",
    "TaskRequest",
    "UnknownBackendError",
    "resolve_backend_name",
]
