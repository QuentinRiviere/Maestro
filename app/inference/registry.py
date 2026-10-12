"""Backend selection per model type.

Routing lives in ``server_config["inference"]``::

    "inference": {
        "default": "local",
        "routes": {"minimax_h3*": "comfy", "ltx25*": "comfy"}
    }

Routes are ``fnmatch`` patterns on ``model_type``, checked in declaration
order; the first match wins.  An unknown backend name is an error rather than
a silent fallback to ``local``: running a model on the wrong host has cost
and VRAM consequences the user must not discover after the fact.
"""
from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from fnmatch import fnmatchcase
from typing import Any

from inference.contract import InferenceBackend


DEFAULT_BACKEND = "local"


def resolve_backend_name(model_type: str, config: Mapping[str, Any] | None) -> str:
    inference_cfg = (config or {}).get("inference") or {}
    routes = inference_cfg.get("routes") or {}
    if isinstance(routes, Mapping):
        for pattern, backend_name in routes.items():
            if fnmatchcase(model_type or "", str(pattern)):
                return str(backend_name)
    return str(inference_cfg.get("default") or DEFAULT_BACKEND)


class UnknownBackendError(ValueError):
    pass


class BackendRegistry:
    """Lazily builds and caches one backend instance per name."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], InferenceBackend]] = {}
        self._instances: dict[str, InferenceBackend] = {}
        self._lock = threading.Lock()

    def register(self, name: str, factory: Callable[[], InferenceBackend]) -> None:
        with self._lock:
            self._factories[name] = factory
            self._instances.pop(name, None)

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._factories)

    def get(self, name: str) -> InferenceBackend:
        with self._lock:
            instance = self._instances.get(name)
            if instance is not None:
                return instance
            factory = self._factories.get(name)
            if factory is None:
                known = ", ".join(sorted(self._factories)) or "none"
                raise UnknownBackendError(
                    f"Unknown inference backend '{name}' (registered: {known}). "
                    "Check server_config['inference']."
                )
            instance = factory()
            self._instances[name] = instance
            return instance

    def for_model(
        self,
        model_type: str,
        config: Mapping[str, Any] | None,
    ) -> InferenceBackend:
        return self.get(resolve_backend_name(model_type, config))
