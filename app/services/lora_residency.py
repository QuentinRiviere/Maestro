"""Make freshly loaded LoRAs visible to a transformer that is already in VRAM.

MMGP keeps two copies of every adapter: the loaded tensors and a per-module
``"<adapter>_GPU"`` entry that its LoRA forward hooks actually read. The GPU
entries are only built when MMGP moves a model's blocks to the GPU
(``gpu_load_blocks``). A model that is still resident from the previous
generation (``MAESTRO_KEEP_MODELS_IN_VRAM=1`` on remote workers) is never
moved again, so LoRAs loaded for the next generation have no GPU entry and
every hook silently falls back to the base weights: the LoRA has no effect.

``sync_resident_loras`` builds those entries for a resident model right after
``load_loras_into_model``. For a model that is not resident it does nothing:
MMGP builds the entries itself when the model is loaded.
"""

from __future__ import annotations

from typing import Any


def _model_id(manager: Any, model: Any) -> str | None:
    for model_id, candidate in (getattr(manager, "models", None) or {}).items():
        if candidate is model:
            return model_id
    return None


def is_model_resident(manager: Any, model: Any) -> bool:
    """True when MMGP considers ``model`` loaded on the GPU."""
    if manager is None or model is None:
        return False
    model_id = _model_id(manager, model)
    return model_id is not None and model_id in (getattr(manager, "active_models_ids", None) or ())


def sync_resident_loras(manager: Any, model: Any) -> int:
    """Copy the active adapters of a resident ``model`` to the GPU.

    Returns the number of modules whose adapter entries were (re)built.
    """
    if not is_model_resident(manager, model):
        return 0
    move = getattr(manager, "_move_loras", None)
    adapters = list(getattr(model, "_loras_active_adapters", None) or ())
    lora_data = getattr(model, "_loras_model_data", None) or {}
    if not callable(move) or not adapters or not lora_data:
        return 0
    modules = {
        module: data
        for module, data in lora_data.items()
        if any(adapter in data for adapter in adapters)
    }
    if not modules:
        return 0
    move(adapters, modules, True, model)
    return len(modules)
