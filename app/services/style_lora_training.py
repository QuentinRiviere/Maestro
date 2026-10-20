"""Style LoRA training for Qwen Image 2.1 on the GPU worker.

A ``lora_training`` job trains a low-rank adapter on the attention
projections of the bf16 transformer from a zipped dataset of (image,
caption) pairs (``lora_dataset`` in the job params, see
``services.director.style_lora`` for how the controller builds it). The
adapter is saved in the key format MMGP reads for Qwen 2.1
(``transformer.<module>.lora_A.weight`` / ``lora_B.weight`` / ``alpha``), so a
run loads it like any other LoRA of the ``qwen21`` folder.

Two phases, each with the memory of the other released:

1. With the ordinary generation stack loaded (MMGP, quantised text encoder),
   every caption is encoded and every image goes through the VAE once.
2. The bf16 transformer alone (14 GB) with the adapters: flow-matching loss
   on the cached latents, the base weights frozen, gradient checkpointing on.

The pure pieces (``shift_sigma``, ``LoRALinear``, ``inject_lora``,
``lora_state_dict``, ``read_manifest``, ``fit_size``) have no CUDA
dependency and are unit tested on the controller.
"""
from __future__ import annotations

import gc
import json
import math
import os
import random
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_TYPE = "qwen_image_21_7B"
VAE_STRIDE = 16
# Target latent tokens group by 2x2 in the transformer, so sizes are multiples of 32 px.
SIZE_UNIT = 32
DEFAULT_STEPS = 1500
DEFAULT_RANK = 16
DEFAULT_LR = 1e-4
DEFAULT_RESOLUTION = 768
DEFAULT_BATCH = 1
MAX_STEPS = 20000
MAX_RANK = 128
MIN_FREE_GB = 36
ATTENTION_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0")
MLP_TARGETS = ("img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out")
TARGET_SETS = {"attention": ATTENTION_TARGETS, "all": ATTENTION_TARGETS + MLP_TARGETS}
# Scheduler constants of Qwen Image 2.1 (configs/scheduler_scheduler_config.json).
BASE_SEQ_LEN, MAX_SEQ_LEN, BASE_SHIFT, MAX_SHIFT = 256, 8192, 0.5, 0.9


class TrainingError(RuntimeError):
    pass


# ── pure helpers ───────────────────────────────────────────────────────────


def training_config(raw: Mapping[str, Any] | None) -> dict:
    """The bounded training settings of a job (``lora_training`` in its params)."""
    raw = dict(raw or {})

    def number(key: str, default: float, low: float, high: float, cast=float):
        value = raw.get(key, default)
        try:
            value = cast(value)
        except (TypeError, ValueError):
            value = cast(default)
        return max(low, min(high, value))

    name = str(raw.get("name") or "style_lora").strip()
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "style_lora"
    targets = str(raw.get("targets") or "attention")
    return {
        "name": safe[:80],
        "trigger": str(raw.get("trigger") or "").strip()[:80],
        "steps": number("steps", DEFAULT_STEPS, 1, MAX_STEPS, int),
        "rank": number("rank", DEFAULT_RANK, 1, MAX_RANK, int),
        "learning_rate": number("learning_rate", DEFAULT_LR, 1e-6, 1e-2),
        "resolution": number("resolution", DEFAULT_RESOLUTION, 256, 1536, int),
        "batch": number("batch", DEFAULT_BATCH, 1, 8, int),
        "targets": targets if targets in TARGET_SETS else "attention",
        "seed": number("seed", 0, 0, 2**31 - 1, int),
    }


def read_manifest(folder: str | Path) -> list[dict]:
    """The (image, caption) items of an unpacked dataset; image paths resolved, missing ones dropped."""
    folder = Path(folder)
    try:
        data = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise TrainingError(f"The dataset has no readable manifest: {error}") from error
    items = data.get("items") if isinstance(data, dict) else data
    out = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        rel = str(item.get("image") or "")
        path = (folder / rel).resolve()
        if not rel or ".." in Path(rel).parts or not path.is_file():
            continue
        out.append({"image": str(path), "caption": str(item.get("caption") or "").strip()})
    if not out:
        raise TrainingError("The dataset is empty")
    return out


def fit_size(width: int, height: int, resolution: int) -> tuple[int, int]:
    """The training size of an image: long side at most ``resolution``, both sides multiples of 32, never upscaled."""
    scale = min(1.0, resolution / max(width, height))
    w = max(SIZE_UNIT, int(width * scale) // SIZE_UNIT * SIZE_UNIT)
    h = max(SIZE_UNIT, int(height * scale) // SIZE_UNIT * SIZE_UNIT)
    return w, h


def calculate_mu(seq_len: int) -> float:
    """The dynamic shift of the scheduler for ``seq_len`` latent tokens (diffusers' calculate_shift)."""
    m = (MAX_SHIFT - BASE_SHIFT) / (MAX_SEQ_LEN - BASE_SEQ_LEN)
    b = BASE_SHIFT - m * BASE_SEQ_LEN
    return seq_len * m + b


def shift_sigma(sigma: float, mu: float) -> float:
    """Exponential time shift of a sigma in (0, 1), as the scheduler applies at inference."""
    sigma = min(max(sigma, 1e-5), 1 - 1e-5)
    return math.exp(mu) / (math.exp(mu) + (1 / sigma - 1))


def sample_sigma(rng: random.Random, mu: float) -> float:
    """A training sigma: logit-normal around the middle of the schedule, then shifted like inference."""
    return shift_sigma(1 / (1 + math.exp(-rng.gauss(0.0, 1.0))), mu)


class LoRALinear(nn.Module):
    """A frozen linear with a trainable low-rank update (fp32 factors, output cast to the input dtype)."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float | None = None):
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = float(alpha if alpha is not None else rank)
        device = base.weight.device
        self.lora_A = nn.Parameter(torch.randn(rank, base.in_features, device=device, dtype=torch.float32)
                                   / math.sqrt(base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=device, dtype=torch.float32))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        update = F.linear(F.linear(values.float(), self.lora_A), self.lora_B) * (self.alpha / self.rank)
        return self.base(values) + update.to(values.dtype)


def inject_lora(model: nn.Module, rank: int, targets: Sequence[str] = ATTENTION_TARGETS) -> dict[str, LoRALinear]:
    """Wrap the linears of every transformer block whose path ends with one of ``targets``; returns them by path."""
    wrapped: dict[str, LoRALinear] = {}
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear) or not name.startswith("transformer_blocks."):
            continue
        if not any(name.endswith("." + target) for target in targets):
            continue
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        layer = LoRALinear(module, rank)
        if attr.isdigit():
            parent[int(attr)] = layer
        else:
            setattr(parent, attr, layer)
        wrapped[name] = layer
    if not wrapped:
        raise TrainingError("No linear layer matched the LoRA targets")
    return wrapped


def lora_state_dict(wrapped: Mapping[str, LoRALinear], dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
    """The adapters in the key format MMGP reads for Qwen Image 2.1 (``transformer.<module>.lora_A.weight``)."""
    state: dict[str, torch.Tensor] = {}
    for name, layer in wrapped.items():
        key = f"transformer.{name}"
        state[f"{key}.lora_A.weight"] = layer.lora_A.detach().to("cpu", dtype).contiguous()
        state[f"{key}.lora_B.weight"] = layer.lora_B.detach().to("cpu", dtype).contiguous()
        state[f"{key}.alpha"] = torch.tensor(layer.alpha, dtype=torch.float32)
    return state


# ── the job ────────────────────────────────────────────────────────────────


def _free_memory_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1e9
    except Exception:  # noqa: BLE001
        return float("inf")


def _release_everything() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _unpack_dataset(path: str, work_dir: str) -> str:
    folder = os.path.join(work_dir, "dataset")
    os.makedirs(folder, exist_ok=True)
    try:
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                target = os.path.realpath(os.path.join(folder, member))
                if not target.startswith(os.path.realpath(folder) + os.sep):
                    raise TrainingError("The dataset archive has an unsafe path")
            archive.extractall(folder)
    except zipfile.BadZipFile as error:
        raise TrainingError(f"The dataset is not a zip file: {error}") from error
    return folder


def _load_image(path: str, resolution: int):
    from PIL import Image

    with Image.open(path) as source:
        image = source.convert("RGB")
    width, height = fit_size(image.width, image.height, resolution)
    # Resize to cover the target, then centre-crop: no padding, no distortion.
    scale = max(width / image.width, height / image.height)
    resized = image.resize((max(width, round(image.width * scale)), max(height, round(image.height * scale))),
                           Image.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def _image_tensor(image, device, dtype) -> torch.Tensor:
    import numpy as np

    array = torch.from_numpy(np.asarray(image, dtype=np.float32) / 127.5 - 1.0)
    return array.permute(2, 0, 1)[None, :, None].to(device=device, dtype=dtype)  # (1, 3, 1, H, W)


def _encode_dataset(items: Sequence[Mapping[str, Any]], config: Mapping[str, Any], *,
                    report: Callable[[str, float], bool]) -> list[dict]:
    """Phase 1: prompt embeddings and VAE latents of every item, cached on the CPU."""
    import wgp

    report("Loading the image model to encode the dataset", 0.0)
    # load_models returns the pair; the module globals are what release_model frees.
    wgp.wan_model, wgp.offloadobj = wgp.load_models(MODEL_TYPE, output_type="image")
    pipeline = wgp.wan_model.pipeline
    device = torch.device("cuda")
    encoded = []
    try:
        with torch.no_grad():
            for index, item in enumerate(items):
                if not report(f"Encoding image {index + 1} of {len(items)}", 100.0 * index / len(items)):
                    raise TrainingError("Cancelled")
                image = _load_image(item["image"], int(config["resolution"]))
                prompt_embeds, prompt_mask, image_pad_mask = pipeline._get_qwen_prompt_embeds(
                    item["caption"] or " ", None, device)
                pixels = _image_tensor(image, device, pipeline.vae.dtype)
                latents = pipeline._encode_vae_image(pixels, None)  # (1, 64, 1, h, w), normalised
                _, channels, _, h, w = latents.shape
                packed = latents.view(1, channels, h * w).transpose(1, 2)
                encoded.append({
                    "embeds": prompt_embeds.to("cpu", torch.bfloat16),
                    "mask": prompt_mask.to("cpu"),
                    "pad_mask": image_pad_mask.to("cpu"),
                    "latents": packed.to("cpu", torch.bfloat16),
                    "shape": (1, h, w),
                })
    finally:
        wgp.release_model()
        _release_everything()
    return encoded


def _transformer_file() -> str:
    """The bf16 transformer checkpoint, downloaded when missing."""
    import wgp

    url = wgp.get_model_filename(model_type=MODEL_TYPE, quantization="bf16", dtype_policy=wgp.transformer_dtype_policy)
    if not url:
        raise TrainingError("No bf16 checkpoint is declared for Qwen Image 2.1")
    local = wgp.get_compatible_local_model_filename(url, MODEL_TYPE, file_type=0)
    if local is None:
        wgp.download_models(url, MODEL_TYPE)
        local = wgp.get_compatible_local_model_filename(url, MODEL_TYPE, file_type=0)
    if local is None or not os.path.isfile(local):
        raise TrainingError("The bf16 transformer could not be located after download")
    return local


def _build_transformer(path: str):
    from accelerate import init_empty_weights
    from mmgp import offload

    from models.qwen21.runtime import _config
    from models.qwen21.transformer_qwenimage21 import QwenImage21Transformer2DModel

    with torch.device("cpu"), init_empty_weights(include_buffers=False):
        transformer = QwenImage21Transformer2DModel.from_config(_config("transformer_config.json"))
    offload.load_model_data(transformer, path, writable_tensors=False, default_dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    return transformer


def _train(encoded: Sequence[Mapping[str, Any]], config: Mapping[str, Any], *, report: Callable[[str, float], bool],
           is_cancelled: Callable[[], bool]) -> tuple[dict[str, torch.Tensor], dict]:
    """Phase 2: the flow-matching loop; returns the adapter state and a training summary."""
    device = torch.device("cuda")
    report("Loading the bf16 transformer", 0.0)
    transformer = _build_transformer(_transformer_file()).to(device)
    transformer.train()
    transformer.enable_gradient_checkpointing()
    wrapped = inject_lora(transformer, int(config["rank"]), TARGET_SETS[config["targets"]])
    params = [p for layer in wrapped.values() for p in (layer.lora_A, layer.lora_B)]
    optimizer = torch.optim.AdamW(params, lr=float(config["learning_rate"]), betas=(0.9, 0.99), weight_decay=0.01)
    steps, batch = int(config["steps"]), int(config["batch"])
    rng = random.Random(int(config["seed"]) or int(time.time()))
    order: list[int] = []
    losses: list[float] = []
    ema = None
    started = time.time()
    try:
        for step in range(1, steps + 1):
            if is_cancelled():
                raise TrainingError("Cancelled")
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for _ in range(batch):
                if not order:
                    order = list(range(len(encoded)))
                    rng.shuffle(order)
                item = encoded[order.pop()]
                latents = item["latents"].to(device)
                embeds = item["embeds"].to(device)
                mask = item["mask"].to(device)
                pad_mask = item["pad_mask"].to(device)
                tokens = latents.shape[1]
                sigma = sample_sigma(rng, calculate_mu(tokens))
                noise = torch.randn_like(latents)
                noisy = (1 - sigma) * latents + sigma * noise
                target = noise - latents
                img_mask = torch.cat([pad_mask, pad_mask.new_ones(1, tokens // 4)], dim=1)
                timestep = torch.tensor([sigma], device=device, dtype=torch.bfloat16)
                prediction = transformer(
                    hidden_states=noisy, timestep=timestep, encoder_hidden_states=embeds,
                    encoder_hidden_states_mask=mask, img_shapes=[[tuple(item["shape"])]], img_mask=img_mask,
                    return_dict=False,
                )[0][:, -tokens:]
                loss = F.mse_loss(prediction.float(), target.float()) / batch
                loss.backward()
                total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            ema = total if ema is None else 0.98 * ema + 0.02 * total
            if step % 10 == 0 or step == steps:
                losses.append(round(total, 5))
            elapsed = time.time() - started
            eta = elapsed / step * (steps - step)
            if not report(f"Step {step} of {steps}, loss {ema:.4f}, {eta / 60:.0f} min left", 100.0 * step / steps,
                          step=step, total_steps=steps):
                raise TrainingError("Cancelled")
        state = lora_state_dict(wrapped)
    finally:
        del optimizer, params, wrapped, transformer
        _release_everything()
    return state, {"losses": losses, "final_loss": ema, "seconds": round(time.time() - started), "items": len(encoded)}


def run_training_job(job: dict, raw_params: Mapping[str, Any]) -> bool:
    """Train the style LoRA described by ``raw_params`` and publish it as the job's output."""
    from safetensors.torch import save_file

    import wgp
    from services.job_lifecycle import finish_job, is_cancel_requested, update_job

    config = training_config(raw_params.get("lora_training"))
    dataset = str(raw_params.get("lora_dataset") or "")
    out_dir = job.get("out_dir") or wgp.save_path
    os.makedirs(out_dir, exist_ok=True)

    def report(message: str, percent: float, **fields: Any) -> bool:
        return update_job(job, message=message, phase="Training", progress=round(percent, 1), **fields)

    work_dir = tempfile.mkdtemp(prefix="style_lora_", dir=out_dir)
    try:
        if not torch.cuda.is_available():
            raise TrainingError("LoRA training needs the GPU worker")
        if not dataset or not os.path.isfile(dataset):
            raise TrainingError("The dataset file is missing")
        free = _free_memory_gb()
        if free < MIN_FREE_GB:
            raise TrainingError(f"Only {free:.0f} GB of memory is free; training needs {MIN_FREE_GB} GB")
        report("Unpacking the dataset", 0.0)
        items = read_manifest(_unpack_dataset(dataset, work_dir))
        if wgp.wan_model is not None or wgp.offloadobj is not None:
            wgp.release_model()
        encoded = _encode_dataset(items, config, report=report)
        if is_cancel_requested(job):
            return False
        state, summary = _train(encoded, config, report=report, is_cancelled=lambda: is_cancel_requested(job))
        name = f"{config['name']}.safetensors"
        metadata = {"maestro_style_lora": json.dumps({**config, **summary, "losses": summary["losses"][-50:]}),
                    "trigger": config["trigger"]}
        lora_dir = wgp.get_lora_dir(MODEL_TYPE)
        os.makedirs(lora_dir, exist_ok=True)
        output = os.path.join(out_dir, name)
        save_file(state, output, metadata=metadata)
        shutil.copyfile(output, os.path.join(lora_dir, name))
        with open(os.path.join(out_dir, f"{config['name']}.json"), "w", encoding="utf-8") as handle:
            json.dump({"lora_training": config, **summary}, handle)
        finish_job(job, "completed", progress=100, step=0, total_steps=0, phase="", message="Done",
                   output_files=[name])
        print(f"[Style LoRA] {name}: {summary['items']} images, {config['steps']} steps, "
              f"loss {summary['final_loss']:.4f}, {summary['seconds']} s")
        return job.get("status") == "completed"
    except TrainingError as error:
        if str(error) == "Cancelled" or is_cancel_requested(job):
            return False
        finish_job(job, "failed", error=str(error), message=f"Error: {error}")
        return False
    except Exception as error:  # noqa: BLE001 - recorded on the job
        import traceback
        traceback.print_exc()
        if not is_cancel_requested(job):
            finish_job(job, "failed", error=str(error)[:400], message=f"Error: {str(error)[:200]}")
        return False
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        _release_everything()
