"""Generation provenance: what produced one output, on which GPU, in how long.

One ``provenance`` record per job answers, for every generated asset:

* ``model``          model type, display name, architecture, checkpoint files
* ``runtime``        transformer quantization read from the loaded file name,
                     text encoder / video VAE, attention backend, compile,
                     inference backend and per-job precision overrides
* ``loras``          name + multiplier of every active LoRA
* ``acceleration``   step count, Turbo preset, fused Turbo checkpoint,
                     step caching, sparse attention
* ``postprocessing`` spatial / temporal upsampling, film grain
* ``hardware``       GPU name and VRAM; for remote workers also the dstack
                     backend, region, instance type and $/h
* ``timing``         total wall time, active generation time and a phase
                     breakdown (download, model load, text encode, denoise,
                     VAE decode, postprocess) when the stream revealed it
* ``cost``           estimated GPU cost of a remote job (work time x $/h)

The record is built where the facts are known: on the machine that runs the
model (local Maestro or the remote worker). A controller forwarding the job
merges the pod's GPU offer and price into the worker's record.

Everything is best effort: a field whose value is unknown is omitted, never
guessed. ``prune`` drops empty values so the UI can show what is there.
This module has no dependency on ``launch.py``, ``wgp`` or torch.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

PROVENANCE_VERSION = 1

# Ordered phase buckets of the timing breakdown.
PHASES = ("download", "model_load", "text_encode", "denoise", "vae_decode", "postprocess")

# First matching rule wins; the message is lower-cased first.
_PHASE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("download", ("downloading model", "downloading lora", "pulling ", "downloading ")),
    ("model_load", ("loading model",)),
    ("text_encode", ("encoding prompt", "text encod", "encoding text")),
    ("vae_decode", ("vae decod", "decoding")),
    # Before "denoise": "upsampling" contains "sampling".
    ("postprocess", ("upscal", "upsampl", "film grain", "flashvsr", "interpolat",
                     "voice clon", "refining faces", "face refin")),
    ("denoise", ("denois", "sampling", "diffusion")),
)
# Messages that end the current phase without starting a known one.
_PHASE_END = ("model loaded",)

# Filename tokens -> quantization label, most specific first.
_QUANT_TOKENS: tuple[tuple[str, str], ...] = (
    ("nvfp4", "NVFP4"),
    ("mxfp4", "MXFP4"),
    ("fp4", "FP4"),
    ("int4", "INT4"),
    ("int8", "INT8"),
    ("fp8", "FP8"),
    ("bf16", "BF16"),
    ("fp16", "FP16"),
)
_GGUF_RE = re.compile(r"(?:^|[^a-z0-9])(q\d(?:_[a-z0-9])*)(?=[^a-z0-9]|$)", re.I)


# ── phase clock ─────────────────────────────────────────────────────────────

def classify_phase(message: Any) -> str | None:
    """The timing bucket a progress/status message belongs to, if any."""
    text = str(message or "").strip().lower()
    if not text:
        return None
    for phase, needles in _PHASE_RULES:
        if any(needle in text for needle in needles):
            return phase
    return None


class PhaseClock:
    """Attribute wall time to phases from the job's progress/status stream.

    The current phase runs until a message names another one. A message that
    matches no phase leaves the current phase running, except an explicit
    phase end (``Model loaded``) which stops it.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._current: str | None = None
        self._since: float | None = None
        self._seconds: dict[str, float] = {}

    @property
    def current(self) -> str | None:
        return self._current

    def observe(self, message: Any, now: float | None = None) -> None:
        text = str(message or "").strip().lower()
        if not text:
            return
        phase = classify_phase(text)
        if phase is None:
            if any(text.startswith(end) for end in _PHASE_END):
                self.stop(now)
            return
        self.enter(phase, now)

    def enter(self, phase: str, now: float | None = None) -> None:
        if phase == self._current:
            return
        now = self._clock() if now is None else now
        self._close(now)
        self._current, self._since = phase, now

    def stop(self, now: float | None = None) -> None:
        self._close(self._clock() if now is None else now)
        self._current, self._since = None, None

    def _close(self, now: float) -> None:
        if self._current is not None and self._since is not None:
            self._seconds[self._current] = self._seconds.get(self._current, 0.0) + max(0.0, now - self._since)

    def seconds(self, now: float | None = None) -> dict[str, float]:
        """Seconds per phase so far (the running phase included), in order."""
        totals = dict(self._seconds)
        if self._current is not None and self._since is not None:
            now = self._clock() if now is None else now
            totals[self._current] = totals.get(self._current, 0.0) + max(0.0, now - self._since)
        return {phase: round(totals[phase], 1) for phase in PHASES if totals.get(phase, 0) >= 0.05}


# ── helpers ─────────────────────────────────────────────────────────────────

def prune(value: Any) -> Any:
    """Drop ``None``, empty strings and empty containers, recursively."""
    if isinstance(value, Mapping):
        cleaned = {str(k): prune(v) for k, v in value.items()}
        return {k: v for k, v in cleaned.items() if not _empty(v)}
    if isinstance(value, (list, tuple)):
        cleaned_list = [prune(v) for v in value]
        return [v for v in cleaned_list if not _empty(v)]
    return value


def _empty(value: Any) -> bool:
    return value is None or value == "" or (isinstance(value, (dict, list, tuple)) and not value)


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _positive(value: Any) -> float | None:
    number = _number(value)
    return number if number is not None and number > 0 else None


def _basename(value: Any) -> str:
    text = str(value or "").replace("\\", "/").strip()
    return os.path.basename(text) if text else ""


def quantization_from_filename(filename: Any) -> str:
    """Quantization named by a checkpoint file name; empty when it names none."""
    name = _basename(filename).lower()
    if not name:
        return ""
    if name.endswith(".gguf"):
        match = _GGUF_RE.search(os.path.splitext(name)[0])
        return f"GGUF {match.group(1).upper()}" if match else "GGUF"
    tokens = set(re.split(r"[^a-z0-9]+", name))
    for token, label in _QUANT_TOKENS:
        if token in tokens or any(part.startswith(token) for part in tokens):
            return label
    return ""


def parse_loras(params: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Active LoRAs with their multiplier (raw text when it is phased)."""
    params = params or {}
    names = params.get("activated_loras") or []
    if isinstance(names, str):
        names = [names]
    if not isinstance(names, (list, tuple)):
        return []
    raw = params.get("loras_multipliers")
    if isinstance(raw, (list, tuple)):
        multipliers = [str(item) for item in raw]
    else:
        # WanGP: one value per LoRA separated by spaces or new lines;
        # "1;0.5" is a per-phase schedule and "1,0.8" a per-step ramp.
        text = re.sub(r"#.*", "", str(raw or ""))
        multipliers = [item for item in re.split(r"[\s]+", text) if item]
    loras = []
    for index, name in enumerate(names):
        label = _basename(name)
        if not label:
            continue
        entry: dict[str, Any] = {"name": label}
        if index < len(multipliers):
            value = _number(multipliers[index])
            entry["multiplier"] = value if value is not None else multipliers[index]
        loras.append(entry)
    return loras


def acceleration_from_params(
    params: Mapping[str, Any] | None,
    *,
    model_def: Mapping[str, Any] | None = None,
    turbo_label: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Step count, Turbo/distillation, step caching and sparse attention."""
    params = params or {}
    model_def = model_def or {}
    model_type = str(params.get("model_type") or "")
    acceleration: dict[str, Any] = {}
    steps = _positive(params.get("num_inference_steps"))
    if steps is not None:
        acceleration["steps"] = int(steps)
    # The H3 Turbo preset is stored even while Turbo is off: only report it
    # when Turbo is on, as the gallery search does.
    preset = str(params.get("minimax_h3_turbo_preset") or "").strip()
    if params.get("minimax_h3_turbo_mode") is True:
        acceleration["turbo"] = True
        if preset:
            acceleration["turbo_preset"] = preset
            label = None
            if turbo_label is not None:
                try:
                    label = turbo_label(preset)
                except Exception:
                    label = None
            if label:
                acceleration["turbo_preset_label"] = str(label)
    if model_def.get("minimax_h3_fused_turbo") or "fused_turbo" in model_type.lower():
        acceleration["fused_turbo"] = True
    if model_def.get("minimax_h3_baked_turbo"):
        acceleration["baked_turbo"] = True
    cache = str(params.get("skip_steps_cache_type") or "").strip()
    if cache:
        acceleration["step_cache"] = cache
        multiplier = _number(params.get("skip_steps_multiplier"))
        if multiplier is not None:
            acceleration["step_cache_multiplier"] = multiplier
        start = _number(params.get("skip_steps_start_step_perc"))
        if start is not None:
            acceleration["step_cache_start_percent"] = start
    attention = str(params.get("override_attention") or "").strip().lower()
    if attention in {"sol", "sla"}:
        acceleration["sparse_attention"] = attention
    return acceleration


def postprocessing_from_params(params: Mapping[str, Any] | None) -> dict[str, Any]:
    params = params or {}
    post: dict[str, Any] = {}
    spatial = str(params.get("spatial_upsampling") or "").strip()
    if spatial:
        post["spatial_upsampling"] = spatial
    temporal = str(params.get("temporal_upsampling") or "").strip()
    if temporal:
        post["temporal_upsampling"] = temporal
    grain = _positive(params.get("film_grain_intensity"))
    if grain is not None:
        post["film_grain_intensity"] = grain
        saturation = _number(params.get("film_grain_saturation"))
        if saturation is not None:
            post["film_grain_saturation"] = saturation
    return post


def hardware_from_probe(probe: Mapping[str, Any] | None, *, location: str = "local") -> dict[str, Any]:
    """GPU facts of this machine from ``hardware_detect.detect_hardware``."""
    probe = probe or {}
    hardware: dict[str, Any] = {"location": location}
    name = str(probe.get("gpu_name") or "").strip()
    if name:
        hardware["gpu"] = name
        vram = _positive(probe.get("gpu_vram_gb"))
        if vram is not None:
            hardware["vram_gb"] = round(vram, 1)
        capability = str(probe.get("gpu_capability") or "").strip()
        if capability:
            hardware["compute_capability"] = capability
    return hardware


def runtime_from_metadata(meta: Mapping[str, Any] | None) -> tuple[dict, dict]:
    """Split the engine's ``runtime_metadata`` event into model and runtime parts."""
    meta = meta or {}
    files = {k: _basename(v) for k, v in (meta.get("files") or {}).items() if _basename(v)}
    model = {
        "type": meta.get("model_type"),
        "name": meta.get("model_name"),
        "architecture": meta.get("architecture"),
        "files": files,
    }
    runtime: dict[str, Any] = {
        "inference_backend": meta.get("inference_backend"),
        "transformer_quantization": quantization_from_filename(files.get("transformer")),
        "configured_quantization": meta.get("configured_quantization"),
        "transformer_dtype": meta.get("transformer_dtype"),
        "text_encoder": meta.get("text_encoder_variant") or files.get("text_encoder"),
        "text_encoder_quantization": quantization_from_filename(files.get("text_encoder")),
        "video_vae": files.get("video_vae"),
        "video_vae_precision": meta.get("video_vae_precision"),
        "vae_precision": meta.get("vae_precision"),
        "attention": meta.get("attention"),
        "compile": meta.get("compile"),
        "memory_profile": meta.get("profile"),
        "precision_overrides": dict(meta.get("precision_overrides") or {}),
    }
    if meta.get("model_reloaded") is not None:
        runtime["model_reloaded"] = bool(meta["model_reloaded"])
    return model, runtime


# ── build / merge ───────────────────────────────────────────────────────────

def build_provenance(
    params: Mapping[str, Any] | None,
    *,
    runtime_metadata: Mapping[str, Any] | None = None,
    model_name: str | None = None,
    model_def: Mapping[str, Any] | None = None,
    hardware: Mapping[str, Any] | None = None,
    phases: Mapping[str, float] | None = None,
    total_seconds: float | None = None,
    generation_seconds: float | None = None,
    postprocessing: Mapping[str, Any] | None = None,
    turbo_label: Callable[[str], str | None] | None = None,
    captured_at: float | None = None,
) -> dict[str, Any]:
    """One job's provenance record from what this machine observed."""
    params = params or {}
    model, runtime = runtime_from_metadata(runtime_metadata)
    model["type"] = model.get("type") or params.get("model_type")
    model["name"] = model.get("name") or model_name
    model["architecture"] = model.get("architecture") or (model_def or {}).get("architecture")
    # The engine reports the attention it really used; the request's
    # override only stands in when the engine did not report one.
    if not runtime.get("attention"):
        runtime["attention"] = str(params.get("override_attention") or "").strip() or None
    post = postprocessing_from_params(params)
    post.update(postprocessing or {})
    timing: dict[str, Any] = {
        "total_seconds": _round(total_seconds),
        "generation_seconds": _round(generation_seconds),
        "phases": dict(phases or {}),
    }
    if runtime_metadata and _positive(runtime_metadata.get("model_load_seconds")) is not None:
        timing["phases"].setdefault("model_load", _round(runtime_metadata["model_load_seconds"]))
    record = {
        "version": PROVENANCE_VERSION,
        "captured_at": captured_at if captured_at is not None else time.time(),
        "model": model,
        "runtime": runtime,
        "loras": parse_loras(params),
        "acceleration": acceleration_from_params(
            _with_engine_steps(params, runtime_metadata), model_def=model_def, turbo_label=turbo_label,
        ),
        "postprocessing": post,
        "hardware": dict(hardware or {}),
        "timing": timing,
    }
    return prune(record)


def _with_engine_steps(params: Mapping[str, Any], meta: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """Prefer the step count the engine ran over the requested one."""
    steps = _positive((meta or {}).get("num_inference_steps"))
    return {**params, "num_inference_steps": int(steps)} if steps is not None else params


def _round(value: Any) -> float | None:
    number = _number(value)
    return round(max(0.0, number), 1) if number is not None else None


def remote_hardware(billing: Mapping[str, Any] | None) -> dict[str, Any]:
    """Pod facts the controller knows from dstack (see ``job_forwarder._billing``)."""
    billing = billing or {}
    hardware = {
        "location": "remote",
        "gpu": billing.get("gpu_name"),
        "gpu_count": billing.get("gpu_count"),
        "vram_gb": billing.get("gpu_memory_gb"),
        "backend": billing.get("gpu_backend"),
        "region": billing.get("gpu_region"),
        "instance_type": billing.get("instance_type"),
        "spot": billing.get("spot"),
        "price_per_hour": _positive(billing.get("price_per_hour")),
        "pod": billing.get("pod"),
    }
    return prune(hardware)


def merge_remote(
    worker_provenance: Mapping[str, Any] | None,
    billing: Mapping[str, Any] | None,
    *,
    total_seconds: float | None = None,
) -> dict[str, Any]:
    """The worker's record plus the pod's offer, price and estimated cost.

    The worker measured its own GPU (torch device name and VRAM); dstack's
    offer only fills what the worker did not report. Cost is an estimate:
    the job's work time on the pod times the pod's hourly price.
    """
    record = json.loads(json.dumps(dict(worker_provenance or {}), default=str))
    billing = billing or {}
    pod = remote_hardware(billing)
    hardware = dict(record.get("hardware") or {})
    for key, value in pod.items():
        if key == "location" or key not in hardware:
            hardware[key] = value
    record["hardware"] = hardware
    timing = dict(record.get("timing") or {})
    if "total_seconds" in timing:
        timing["worker_seconds"] = timing.pop("total_seconds")
    timing["total_seconds"] = _round(total_seconds)
    timing["remote_wait_seconds"] = _round(billing.get("wait_seconds"))
    timing["remote_work_seconds"] = _round(billing.get("work_seconds"))
    record["timing"] = timing
    price = _positive(billing.get("price_per_hour"))
    work = _positive(billing.get("work_seconds"))
    if price is not None and work is not None:
        record["cost"] = {
            "estimated_usd": round(price * work / 3600.0, 4),
            "price_per_hour": price,
            "basis": "remote work time x hourly price",
        }
    record.setdefault("version", PROVENANCE_VERSION)
    return prune(record)


def with_output_timing(provenance: Mapping[str, Any] | None, sidecar: Mapping[str, Any] | None) -> dict[str, Any]:
    """A job record narrowed to one output: its own active generation time."""
    record = dict(provenance or {})
    sidecar = sidecar or {}
    seconds = _positive(sidecar.get("generation_time"))
    if seconds is not None and sidecar.get("generation_time_basis") == "active":
        timing = dict(record.get("timing") or {})
        timing["generation_seconds"] = round(seconds, 1)
        record["timing"] = timing
    return record


# ── sidecars ────────────────────────────────────────────────────────────────

def stamp_sidecar(meta_path: str, provenance: Mapping[str, Any] | None, **fields: Any) -> bool:
    """Write ``provenance`` (and ``fields``) into an output's ``.meta.json``.

    Creates a minimal sidecar when there is none: the gallery then still
    reads the generation settings embedded in the media file.
    """
    if not provenance:
        return False
    sidecar: dict[str, Any] = {}
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                sidecar = loaded
        except (OSError, ValueError):
            return False
    else:
        sidecar["created_at"] = time.time()
    sidecar.update({k: v for k, v in fields.items() if v is not None})
    sidecar["provenance"] = with_output_timing(provenance, sidecar)
    temporary = f"{meta_path}.{os.getpid()}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, indent=2, default=str)
        os.replace(temporary, meta_path)
    except OSError:
        try:
            os.remove(temporary)
        except OSError:
            pass
        return False
    return True


_SIDECAR_CACHE: dict[str, tuple[float, dict | None]] = {}


def read_output_provenance(out_dir: str, filename: Any) -> dict | None:
    """The provenance stored beside one output (cached by sidecar mtime)."""
    name = _basename(filename)
    if not name or not out_dir:
        return None
    meta_path = os.path.join(out_dir, os.path.splitext(name)[0] + ".meta.json")
    try:
        mtime = os.path.getmtime(meta_path)
    except OSError:
        return None
    cached = _SIDECAR_CACHE.get(meta_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    record = None
    try:
        with open(meta_path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict) and isinstance(loaded.get("provenance"), dict):
            record = loaded["provenance"]
    except (OSError, ValueError):
        record = None
    if len(_SIDECAR_CACHE) > 4096:
        _SIDECAR_CACHE.clear()
    _SIDECAR_CACHE[meta_path] = (mtime, record)
    return record


def first_provenance(out_dir: str, filenames: Iterable[Any]) -> dict | None:
    for name in filenames:
        record = read_output_provenance(out_dir, name)
        if record:
            return record
    return None
