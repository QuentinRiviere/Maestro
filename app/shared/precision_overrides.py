"""Per-job model precision overrides.

Maestro's transformer quantization, video VAE and text encoder are normally
global settings (``transformer_quantization`` in the server config, the H3
video VAE derived from it, the H3 text encoder recommended per machine). A
generation job may override them so the same machine can A/B checkpoint
formats without changing its configuration:

* ``override_transformer_quantization``: ``int8`` | ``bf16`` | ``fp8``
* ``override_video_vae_precision`` (MiniMax H3 only): ``int8`` | ``fp16``
* ``override_text_encoder`` (MiniMax H3 only): one of the model's
  ``minimax_h3_text_encoder_variants`` ids.

An empty value means "no override": the job keeps today's behavior exactly.

The overrides active for the model being loaded live in a context variable so
the engine's existing resolvers (``get_runtime_model_def``,
``download_models``, ``get_compatible_local_model_filename``) can honor them
without threading a new argument through every call.
"""

from __future__ import annotations

import contextlib
import contextvars
import os
from collections.abc import Iterator, Mapping
from typing import Any


# Job parameter name -> internal override key.
JOB_PARAMS = {
    "override_transformer_quantization": "transformer_quantization",
    "override_video_vae_precision": "video_vae_precision",
    "override_text_encoder": "text_encoder",
}

TRANSFORMER_QUANTIZATIONS = ("int8", "bf16", "fp8")
# Accepted spelling -> canonical value. ``fp16`` is the published Comfy
# decoder; ``int8`` is Kijai's compact INT8 ConvRot decoder.
VIDEO_VAE_PRECISIONS = {
    "int8": "int8",
    "int8_convrot": "int8",
    "fp16": "fp16",
}

# Filename tokens used to check that a requested transformer format exists.
_QUANT_TOKENS = {"int8": ("int8",), "fp8": ("fp8",)}

_current: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "maestro_precision_overrides", default=None,
)


def current() -> dict:
    """The overrides of the model load in progress (empty when none)."""
    return dict(_current.get() or {})


@contextlib.contextmanager
def scope(overrides: Mapping[str, Any] | None) -> Iterator[dict]:
    """Make ``overrides`` the active ones for the enclosed resolution/load."""
    active = dict(overrides or {})
    token = _current.set(active)
    try:
        yield active
    finally:
        _current.reset(token)


def effective_transformer_quantization(default: str) -> str:
    return current().get("transformer_quantization") or default


def from_job_params(params: Mapping[str, Any] | None) -> dict:
    """Raw (unvalidated) override values present in job params."""
    params = params or {}
    raw = {}
    for param, key in JOB_PARAMS.items():
        value = params.get(param)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            raw[key] = value
    return raw


def is_minimax_h3(model_def: Mapping[str, Any] | None) -> bool:
    return str((model_def or {}).get("architecture") or "").startswith("minimax_h3")


def normalize(
    model_def: Mapping[str, Any] | None,
    *,
    transformer_quantization: Any = "",
    video_vae_precision: Any = "",
    text_encoder: Any = "",
) -> dict:
    """Validate override values for one model; return the canonical dict.

    Raises ``ValueError`` with an actionable message for an unknown value or
    an override the model does not support.
    """
    model_def = model_def or {}
    overrides: dict[str, str] = {}

    quantization = str(transformer_quantization or "").strip().lower()
    if quantization:
        if quantization not in TRANSFORMER_QUANTIZATIONS:
            raise ValueError(
                f"override_transformer_quantization must be one of "
                f"{', '.join(TRANSFORMER_QUANTIZATIONS)}; received {transformer_quantization!r}."
            )
        overrides["transformer_quantization"] = quantization

    vae = str(video_vae_precision or "").strip().lower()
    if vae:
        if not is_minimax_h3(model_def) or "minimax_h3_video_vae_filename" not in model_def:
            raise ValueError("override_video_vae_precision is supported only by MiniMax H3 video models.")
        if vae not in VIDEO_VAE_PRECISIONS:
            raise ValueError(
                "override_video_vae_precision must be one of "
                f"{', '.join(sorted(set(VIDEO_VAE_PRECISIONS.values())))}; received {video_vae_precision!r}."
            )
        overrides["video_vae_precision"] = VIDEO_VAE_PRECISIONS[vae]

    encoder = str(text_encoder or "").strip()
    if encoder:
        variants = model_def.get("minimax_h3_text_encoder_variants") or {}
        if len(variants) < 2:
            raise ValueError("override_text_encoder is supported only by models with several text encoder variants (MiniMax H3).")
        if encoder not in variants:
            raise ValueError(
                f"Unknown text encoder '{encoder}'. Choose one of: {', '.join(variants)}."
            )
        overrides["text_encoder"] = encoder
    return overrides


def transformer_file_matches(filename: str, quantization: str) -> bool:
    """Whether a resolved transformer file really has the requested format.

    ``get_model_filename`` silently falls back to another format when the
    requested one is not published (H3 Full has no FP8 file), which would
    turn an A/B comparison into a comparison of the same file.
    """
    name = os.path.basename(str(filename or "")).lower()
    if not name:
        return False
    has = {quant: any(token in name for token in tokens) for quant, tokens in _QUANT_TOKENS.items()}
    if quantization == "bf16":
        return not any(has.values())
    return has.get(quantization, False)


def text_encoder_variant_filename(model_def: Mapping[str, Any] | None, variant: str) -> str:
    variants = (model_def or {}).get("minimax_h3_text_encoder_variants") or {}
    urls = (variants.get(variant) or {}).get("URLs") or []
    if isinstance(urls, str):
        urls = [urls]
    return os.path.basename(str(urls[0])) if urls else ""


_KEY_TAGS = {"transformer_quantization": "tq", "video_vae_precision": "vae", "text_encoder": "te"}


def cache_key(model_type: str, overrides: Mapping[str, Any] | None) -> str:
    """Download/cache key of a model variant; the model type when none."""
    overrides = {k: v for k, v in (overrides or {}).items() if v}
    if not overrides:
        return model_type
    parts = [f"{_KEY_TAGS.get(key, key)}-{overrides[key]}" for key in sorted(overrides)]
    return f"{model_type}@{'+'.join(parts)}"


def describe_files(model_type: str, files: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> str:
    """One log line naming the files a job resolves to."""
    def name(key: str) -> str:
        value = files.get(key)
        return os.path.basename(str(value)) if value else "-"

    line = (
        f"[Model Files] {model_type}: transformer={name('transformer')} "
        f"video_vae={name('video_vae')} text_encoder={name('text_encoder')}"
    )
    if overrides:
        line += " (job overrides: " + ", ".join(f"{k}={v}" for k, v in sorted(overrides.items())) + ")"
    return line
