"""Register the backends declared in ``server_config["inference"]["backends"]``.

Example::

    "inference": {
        "default": "local",
        "routes": {"minimax_h3*": "runpod"},
        "backends": {
            "runpod": {"type": "comfy", "base_url": "...", "token_env": "DSTACK_SERVICE_TOKEN"},
            "spark": {"type": "comfy", "base_url": "http://spark.local:8188/"}
        }
    }

Factories are lazy: a misconfigured backend fails the first job routed to it
with a clear message instead of preventing Maestro from starting.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from inference.registry import BackendRegistry


def register_configured_backends(
    registry: BackendRegistry,
    config: Mapping[str, Any] | None,
    *,
    comfy_recipes: Sequence[Any],
    save_dir: Callable[[], str],
) -> list[str]:
    backends = ((config or {}).get("inference") or {}).get("backends") or {}
    registered: list[str] = []
    if not isinstance(backends, Mapping):
        return registered
    for name, backend_config in backends.items():
        name = str(name)
        if name == "local":
            continue

        def factory(name: str = name, backend_config: Any = backend_config):
            if not isinstance(backend_config, Mapping):
                raise ValueError(f"inference backend '{name}': config must be an object")
            backend_type = backend_config.get("type")
            if backend_type == "comfy":
                from inference.comfy.backend import build_comfy_backend

                return build_comfy_backend(
                    name, backend_config, comfy_recipes, save_dir=save_dir,
                )
            raise ValueError(f"inference backend '{name}': unknown type {backend_type!r}")

        registry.register(name, factory)
        registered.append(name)
    return registered
