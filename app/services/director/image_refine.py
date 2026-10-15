"""Director texture refine: a low-strength base-model re-render of each start image.

Qwen Image 2.1 Turbo frames a shot well in 8 steps but leaves skin, hair and
fabric smooth. After the storyboard review is approved, and before any video
uses the start images, each image is re-rendered once by the base model as a
full-frame masked edit: an all-white mask over the image itself, so denoising
starts from the image's own latents partway down the schedule
(``denoising_strength``) and the whole frame takes new texture while the
composition stays. This is the standard Qwen Image 2.1 inpaint contract
(``image_mode`` 2, ``V``/``A``/``G`` letters), so it runs unchanged on a remote
worker. The character sheets of the people in the shot are sent along so faces
keep their identity.

A shot whose refine fails keeps its unrefined image. A refined shot records
the original as ``start_image_unrefined_filename`` and is never refined again.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

log = logging.getLogger(__name__)

REFINE_STAGE = "image_refine"
DEFAULT_MODEL = "qwen_image_21_7B"
DEFAULT_STRENGTH = 0.35
DEFAULT_STEPS = 30
# The base checkpoint's own CFG (app/defaults/qwen_image_21_7B.json).
DEFAULT_GUIDANCE = 4.0
MIN_STRENGTH, MAX_STRENGTH = 0.05, 0.95
MIN_STEPS, MAX_STEPS = 4, 100
# Reference 1 is the image itself; a few character sheets keep faces without
# spending the whole 10-image budget of the editor.
MAX_IDENTITY_REFS = 4

TEXTURE_CLAUSE = (
    "Re-render reference image 1 as a true photograph of this exact frame: true-to-life skin texture "
    "with visible pores, fine individual hair strands, visible fabric weave, natural film grain. "
    "Do not change the composition, framing, poses, faces, expressions, clothing, props or lighting."
)
IDENTITY_CLAUSE = "The other reference images only show who the people are: keep every face identical."

_ON = {"on", "true", "1", "yes", "enabled"}
_OFF = {"off", "false", "0", "no", "disabled", ""}


@dataclass(frozen=True)
class RefineSettings:
    enabled: bool = False
    model: str = DEFAULT_MODEL
    strength: float = DEFAULT_STRENGTH
    steps: int = DEFAULT_STEPS

    @property
    def effective_steps(self) -> int:
        """Steps the model actually runs, as the Qwen pipeline computes its first step."""
        first_step = min(int(self.steps * (1 - self.strength)), self.steps - 1)
        return self.steps - first_step


def _enabled(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value if value is not None else "").strip().lower()
    if text in _ON:
        return True
    if text not in _OFF:
        log.warning("Unknown director_image_refine value %r; the refine stays off", value)
    return False


def _number(value, default, cast, low, high):
    try:
        number = cast(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN
        return default
    return min(high, max(low, number))


def refine_settings(params: dict) -> RefineSettings:
    """The refine settings of a Director run, with defaults for missing or invalid values."""
    params = params or {}
    model = str(params.get("director_image_refine_model") or "").strip() or DEFAULT_MODEL
    return RefineSettings(
        enabled=_enabled(params.get("director_image_refine", "off")),
        model=model,
        strength=_number(params.get("director_image_refine_strength", DEFAULT_STRENGTH),
                         DEFAULT_STRENGTH, float, MIN_STRENGTH, MAX_STRENGTH),
        steps=_number(params.get("director_image_refine_steps", DEFAULT_STEPS),
                      DEFAULT_STEPS, lambda v: int(float(v)), MIN_STEPS, MAX_STEPS),
    )


def refine_prompt(image_prompt: str, *, with_identity: bool) -> str:
    """The shot's own image prompt followed by the texture instruction."""
    parts = [str(image_prompt or "").strip(), TEXTURE_CLAUSE]
    if with_identity:
        parts.append(IDENTITY_CLAUSE)
    return " ".join(part.rstrip() for part in parts if part)


def full_frame_mask(out_dir: str, width: int, height: int, pid: str = "director") -> str:
    """An all-white mask of the image size (white = re-render), created once per size."""
    from PIL import Image

    prefix = re.sub(r"[^A-Za-z0-9_-]+", "_", str(pid))
    path = os.path.join(out_dir, f"{prefix}_refine_mask_{int(width)}x{int(height)}.png")
    if not os.path.isfile(path):
        os.makedirs(out_dir, exist_ok=True)
        partial = f"{path}.{os.getpid()}.part.png"
        Image.new("L", (int(width), int(height)), 255).save(partial)
        os.replace(partial, path)
    return path


def build_refine_job(
    *,
    source_path: str,
    mask_path: str,
    prompt: str,
    identity_refs: Sequence[str],
    settings: RefineSettings,
    width: int,
    height: int,
    pid: str,
) -> dict:
    """Job parameters of one refine: a full-frame masked edit of ``source_path``."""
    refs = [ref for ref in dict.fromkeys(identity_refs or ()) if ref][:MAX_IDENTITY_REFS]
    # V: the source image is the control image (reference 1 and the latents
    # denoising starts from); A: its mask; G: honour denoising_strength;
    # I: the character sheets.
    prompt_type = "VAG" + ("I" if refs else "")
    return {
        "model_type": settings.model,
        "prompt": prompt,
        "negative_prompt": "",
        "image_mode": 2,
        "image_prompt_type": "",
        "video_prompt_type": prompt_type,
        "image_guide": source_path,
        "image_mask": mask_path,
        "image_refs": refs,
        "denoising_strength": settings.strength,
        "masking_strength": 1.0,
        "model_mode": 0,
        "mask_expand": 0,
        "num_inference_steps": settings.steps,
        "guidance_scale": DEFAULT_GUIDANCE,
        "sample_solver": "default",
        "resolution": f"{int(width)}x{int(height)}",
        "seed": -1,
        "settings_version": 2.52,
        "generation_mode": "image",
        "repeat_generation": 1,
        "video_length": 1,
        "activated_loras": [],
        "loras_multipliers": "",
        "_director_pipeline_id": pid,
        "_director_cost_stage": REFINE_STAGE,
    }


def _match_size(path: str, size: tuple[int, int]) -> None:
    """Bring a refined image back to the start image's size, keeping its metadata.

    The pipeline renders on a 32-pixel canvas, so a 480x848 start image comes
    back 480x832; the video stage expects the size it planned for.
    """
    from PIL import Image, PngImagePlugin

    with Image.open(path) as image:
        if image.size == tuple(size):
            return
        info = dict(image.info)
        resized = image.convert("RGB" if image.mode not in ("RGB", "RGBA") else image.mode).resize(
            size, Image.Resampling.LANCZOS)
    png_info = PngImagePlugin.PngInfo()
    for key, value in info.items():
        if isinstance(value, str):
            png_info.add_text(key, value)
    resized.save(path, pnginfo=png_info)


def run_refine_stage(
    *,
    pid: str,
    clip_images: list,
    clip_plans: Sequence[dict],
    out_dir: str,
    settings: RefineSettings,
    unrefined: Optional[list] = None,
    identity_refs_for: Callable[[int], Sequence[str]],
    submit: Callable[[dict], list],
    is_cancelled: Callable[[], bool] = lambda: False,
    on_progress: Callable[[int, int, str], None] = lambda current, total, message: None,
    on_shot_done: Callable[[list, list], None] = lambda images, originals: None,
) -> tuple[list, list]:
    """Refine every start image once; return ``(clip_images, unrefined)``.

    ``clip_images`` holds filenames relative to ``out_dir`` and is updated in
    place. ``unrefined[i]`` is the original filename of a refined shot, empty
    otherwise; a shot that already has one is skipped. ``submit`` runs one
    generation job and returns its output filenames.
    """
    from PIL import Image

    unrefined = list(unrefined or [])
    unrefined.extend([""] * (len(clip_images) - len(unrefined)))
    pending = [i for i, name in enumerate(clip_images) if name and not unrefined[i]]
    for done, index in enumerate(pending):
        if is_cancelled():
            break
        original = clip_images[index]
        source_path = original if os.path.isabs(original) else os.path.join(out_dir, original)
        on_progress(done, len(pending), f"Shot {index + 1}: texture refine")
        try:
            with Image.open(source_path) as image:
                size = image.size
            plan = clip_plans[index] if index < len(clip_plans) else {}
            refs = [ref for ref in identity_refs_for(index) if ref and os.path.isfile(ref)]
            job = build_refine_job(
                source_path=source_path,
                mask_path=full_frame_mask(out_dir, *size, pid=pid),
                prompt=refine_prompt((plan or {}).get("image_prompt", ""), with_identity=bool(refs)),
                identity_refs=refs,
                settings=settings,
                width=size[0],
                height=size[1],
                pid=pid,
            )
            outputs = submit(job)
            refined = outputs[0] if outputs else ""
            refined_path = os.path.join(out_dir, refined) if refined else ""
            if not refined or not os.path.isfile(refined_path):
                raise RuntimeError("the refine job recorded no output image")
            _match_size(refined_path, size)
        except Exception as exc:
            if is_cancelled():
                break
            log.warning("[Pipeline %s] Shot %d texture refine failed, keeping the unrefined image: %s",
                        pid, index + 1, exc)
            print(f"[Pipeline {pid}] WARNING: shot {index + 1} texture refine failed, "
                  f"keeping the unrefined image: {exc}")
            continue
        clip_images[index] = refined
        unrefined[index] = original
        print(f"[Pipeline {pid}] Shot {index + 1}: texture refined {original} -> {refined}")
        on_shot_done(clip_images, unrefined)
    on_progress(len(pending), len(pending), "Texture refine done")
    return clip_images, unrefined
