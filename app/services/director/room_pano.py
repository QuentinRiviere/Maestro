"""A place's 360° panorama and the angles cut from it.

Edited from its plate, a place's other angles come back as the plate; painted
from text, their volumes drift. A panorama fixes the geometry: one
equirectangular image of the room, then every angle is a perspective crop of
it (``crop_view``: yaw per wall, any yaw the scene lock asks for later), so
the walls, windows and furniture sit where one picture put them.

Two ways to paint the panorama:

* ``outpaint`` (preferred): the approved plate is projected onto a green
  equirectangular canvas (``outpaint_canvas``) and FLUX.2 Klein 9B base with
  nomadoor's 360 ERP outpaint LoRA fills the green. The plate's pixels are
  the panorama's centre, so the place keeps its furniture and light; the
  rest of the room is invented around them (Matrix-3D, HY-World 2.0 and the
  LTX 2.3 Movie Builder all start a set this way).
* ``text``: Qwen-Image 2.1 with the panorama LoRA trained on Poly Haven's
  CC0 HDRIs (``lora_training``), prompted from the room sheet; the style
  LoRA can stack on it. Nothing of the plate survives but its description.

The panorama is kept in the bible folder (``pano_<element>.png``) and the
crops become candidates of the place's ``reverse`` / ``left`` / ``right``
angles, plus a ``front`` candidate of the main look.
"""
from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

from . import bibles, room_spec, style_lora
from .bibles import BibleError

PANO_LORA = "maestro_pano.safetensors"
PANO_TRIGGER = "maestro_pano"
PANO_RESOLUTION = "2048x1024"
PANO_SIZE = (2048, 1024)
VIEW_FOV = 70.0
VIEW_SIZE = (1280, 720)
# Horizontal field of view the approved plate is assumed to cover when it is projected onto the canvas.
PLATE_FOV = VIEW_FOV
# Yaw of each angle from the plate's own direction (front = 0, clockwise positive).
VIEW_YAWS = {"front": 0.0, "right": 90.0, "reverse": 180.0, "left": -90.0}
# The outpaint route: FLUX.2 Klein 9B base + nomadoor/flux-2-klein-9B-360-erp-outpaint-lora (trained on ~1000
# CC0 panoramas, rank 16). Its workflow: euler 20 steps, CFG 5, LoRA 0.9, the canvas as reference latent.
OUTPAINT_MODEL = "flux2_klein_base_9b"
OUTPAINT_LORA = "flux-2-klein-9B-360-erp-outpaint-lora_V1.safetensors"
OUTPAINT_PROMPT = ("Fill the green spaces according to the image. Outpaint as a seamless 360 equirectangular "
                   "panorama (2:1). Keep the horizon level and match the left and right edges.")
OUTPAINT_NEGATIVE = "text, worst quality, blurry, ugly"
OUTPAINT_STEPS = 20
OUTPAINT_GUIDANCE = 5.0
OUTPAINT_SCALE = "0.9"
GREEN = (0, 255, 0)
MODES = ("auto", "outpaint", "text")
_LIVE: set[str] = set()
_SLOT = threading.Semaphore(1)
_ensure_pipeline: Callable[[], Any] | None = None
_lora_name: Callable[[], str] | None = None


class PanoramaError(RuntimeError):
    pass


def pano_path(bible_id: str, element_id: str) -> Path:
    return bibles._folder(bible_id) / f"pano_{bibles._slug(element_id, 40)}.png"


def lora_file() -> str:
    return str(_lora_name() if _lora_name else "") or PANO_LORA


def lora_available() -> bool:
    return os.path.isfile(os.path.join(style_lora._registry_dir(), lora_file()))


def _outpaint_registry_dir() -> str:
    import wgp
    return wgp.get_lora_dir(OUTPAINT_MODEL)


def outpaint_available() -> bool:
    try:
        return os.path.isfile(os.path.join(_outpaint_registry_dir(), OUTPAINT_LORA))
    except Exception:  # noqa: BLE001 - no model registry (tests)
        return False


def available_modes() -> dict[str, bool]:
    return {"outpaint": outpaint_available(), "text": lora_available()}


def choose_mode(mode: str, *, has_plate: bool, modes: Mapping[str, bool] | None = None) -> str:
    """The way the panorama is painted: ``auto`` takes the outpaint when the place has a plate and the LoRA is there."""
    modes = available_modes() if modes is None else modes
    mode = str(mode or "auto")
    if mode not in MODES:
        raise BibleError(f"Unknown panorama mode {mode!r}")
    if mode == "auto":
        mode = "outpaint" if has_plate and modes.get("outpaint") else "text"
    if mode == "outpaint":
        if not modes.get("outpaint"):
            raise BibleError(f"The 360 outpaint LoRA ({OUTPAINT_LORA}) is not in the {OUTPAINT_MODEL} LoRA folder")
        if not has_plate:
            raise BibleError("The outpaint needs an approved plate of the place: approve its main look first")
    elif not modes.get("text"):
        raise BibleError(f"The panorama LoRA ({lora_file()}) is not in the LoRA registry: train it first "
                         "(Director → LoRA trainings)")
    return mode


def outpaint_prompt(element: Mapping[str, Any], spec: Mapping[str, Any] | None, *, state: str = "") -> str:
    """The outpaint's prompt: the LoRA's trigger, then a short word of what the rest of the room holds."""
    parts = [OUTPAINT_PROMPT, f"The place is {element.get('name', 'a room')}, empty, nobody in it, no text."]
    walls = (spec or {}).get("walls") or {}
    if spec and spec.get("summary"):
        parts.append(str(spec["summary"]))
    elif element.get("description"):
        parts.append(room_spec._clean(element.get("description"), 400))
    for key, label in (("left", "To the left: "), ("right", "To the right: "), ("back", "Behind the camera: ")):
        if walls.get(key):
            parts.append(f"{label}{walls[key]}")
    if spec and spec.get("materials"):
        parts.append(f"Materials: {spec['materials']}")
    if state:
        parts.append(f"State: {room_spec._clean(state, 200)}")
    return " ".join(p.rstrip(".") + "." for p in parts if p)


def panorama_prompt(element: Mapping[str, Any], spec: Mapping[str, Any] | None, *, look: str = "",
                    trigger: str = PANO_TRIGGER, state: str = "") -> str:
    """The prompt of the place's equirectangular panorama from its room sheet (or its description)."""
    walls = (spec or {}).get("walls") or {}
    parts = [f"{trigger}, equirectangular 360 degree panorama of {element.get('name', 'the place')}, empty, "
             "nobody in it, every seat unoccupied, no text, the camera at eye level in the middle of the room."]
    if walls.get("front"):
        parts.append(f"Centre of the panorama (straight ahead): {walls['front']}")
    if walls.get("right"):
        parts.append(f"Right of centre: {walls['right']}")
    if walls.get("left"):
        parts.append(f"Left of centre: {walls['left']}")
    if walls.get("back"):
        parts.append(f"At both edges of the panorama (behind the camera): {walls['back']}")
    if spec:
        for key, label in (("summary", ""), ("materials", "Materials: "), ("light", "Light: ")):
            if spec.get(key):
                parts.append(f"{label}{spec[key]}")
        if spec.get("palette"):
            parts.append("Palette: " + ", ".join(spec["palette"]) + ".")
    else:
        parts.append(f"Place: {room_spec._clean(element.get('description'), 900)}")
        if element.get("notes"):
            parts.append(f"Layout: {room_spec._clean(element.get('notes'), 400)}")
    if state:
        parts.append(f"State: {room_spec._clean(state, 300)}")
    if look:
        parts.append(look)
    return " ".join(p.rstrip(".") + "." for p in parts if p)


def crop_view(panorama, yaw_deg: float, *, pitch_deg: float = 0.0, fov_deg: float = VIEW_FOV,
              size: tuple[int, int] = VIEW_SIZE):
    """A pinhole view of an equirectangular panorama (PIL image), looking at ``yaw_deg`` (clockwise from the centre)."""
    import numpy as np
    from PIL import Image

    source = np.asarray(panorama.convert("RGB"), dtype=np.float32)
    height, width = source.shape[:2]
    out_w, out_h = size
    focal = (out_w / 2) / math.tan(math.radians(fov_deg) / 2)
    xs = (np.arange(out_w, dtype=np.float32) - out_w / 2 + 0.5) / focal
    ys = (out_h / 2 - np.arange(out_h, dtype=np.float32) - 0.5) / focal
    x, y = np.meshgrid(xs, ys)
    z = np.ones_like(x)
    # Pitch (around the x axis), then yaw (around the vertical axis).
    pitch, yaw = math.radians(pitch_deg), math.radians(yaw_deg)
    y2 = y * math.cos(pitch) - z * math.sin(pitch)
    z2 = y * math.sin(pitch) + z * math.cos(pitch)
    x3 = x * math.cos(yaw) + z2 * math.sin(yaw)
    z3 = -x * math.sin(yaw) + z2 * math.cos(yaw)
    norm = np.sqrt(x3 * x3 + y2 * y2 + z3 * z3)
    lon = np.arctan2(x3, z3)            # -pi..pi, 0 at the panorama's centre
    lat = np.arcsin(np.clip(y2 / norm, -1, 1))
    px = (lon / (2 * math.pi) + 0.5) * width - 0.5
    py = (0.5 - lat / math.pi) * height - 0.5
    x0 = np.floor(px).astype(np.int64)
    y0 = np.floor(py).astype(np.int64)
    fx = (px - x0)[..., None]
    fy = (py - y0)[..., None]
    x1 = (x0 + 1) % width
    x0 = x0 % width
    y0c = np.clip(y0, 0, height - 1)
    y1c = np.clip(y0 + 1, 0, height - 1)
    top = source[y0c, x0] * (1 - fx) + source[y0c, x1] * fx
    bottom = source[y1c, x0] * (1 - fx) + source[y1c, x1] * fx
    view = top * (1 - fy) + bottom * fy
    return Image.fromarray(np.clip(view + 0.5, 0, 255).astype(np.uint8))


def project_view(image, yaw_deg: float, *, pitch_deg: float = 0.0, fov_deg: float = PLATE_FOV,
                 size: tuple[int, int] = PANO_SIZE):
    """Paint a pinhole image onto an equirectangular canvas, looking at ``yaw_deg``; returns (RGB array, mask array).

    The inverse of ``crop_view``: every canvas pixel whose direction falls inside the view's frustum samples the
    image (bilinear), the others stay unpainted (mask 0).
    """
    import numpy as np

    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    img_h, img_w = source.shape[:2]
    width, height = size
    focal = (img_w / 2) / math.tan(math.radians(fov_deg) / 2)
    lon = ((np.arange(width, dtype=np.float32) + 0.5) / width - 0.5) * 2 * math.pi
    lat = (0.5 - (np.arange(height, dtype=np.float32) + 0.5) / height) * math.pi
    lon, lat = np.meshgrid(lon, lat)
    # Direction of each canvas pixel, then the view camera's inverse rotation (undo yaw, then pitch).
    dx = np.cos(lat) * np.sin(lon)
    dy = np.sin(lat)
    dz = np.cos(lat) * np.cos(lon)
    pitch, yaw = math.radians(pitch_deg), math.radians(yaw_deg)
    x2 = dx * math.cos(yaw) - dz * math.sin(yaw)
    z2 = dx * math.sin(yaw) + dz * math.cos(yaw)
    y3 = dy * math.cos(pitch) + z2 * math.sin(pitch)
    z3 = -dy * math.sin(pitch) + z2 * math.cos(pitch)
    front = z3 > 1e-6
    safe_z = np.where(front, z3, 1.0)
    px = x2 / safe_z * focal + img_w / 2 - 0.5
    py = img_h / 2 - y3 / safe_z * focal - 0.5
    inside = front & (px >= 0) & (px <= img_w - 1) & (py >= 0) & (py <= img_h - 1)
    x0 = np.clip(np.floor(px), 0, img_w - 2).astype(np.int64)
    y0 = np.clip(np.floor(py), 0, img_h - 2).astype(np.int64)
    fx = np.clip(px - x0, 0, 1)[..., None]
    fy = np.clip(py - y0, 0, 1)[..., None]
    top = source[y0, x0] * (1 - fx) + source[y0, x0 + 1] * fx
    bottom = source[y0 + 1, x0] * (1 - fx) + source[y0 + 1, x0 + 1] * fx
    view = top * (1 - fy) + bottom * fy
    canvas = np.where(inside[..., None], view, 0.0)
    return np.clip(canvas + 0.5, 0, 255).astype(np.uint8), inside.astype(np.uint8)


def outpaint_canvas(plate, *, fov_deg: float = PLATE_FOV, size: tuple[int, int] = PANO_SIZE):
    """The outpaint LoRA's control image: the plate projected at the canvas centre, green everywhere else."""
    import numpy as np
    from PIL import Image

    painted, mask = project_view(plate, 0.0, fov_deg=fov_deg, size=size)
    green = np.empty_like(painted)
    green[...] = GREEN
    return Image.fromarray(np.where(mask[..., None] > 0, painted, green))


def _director():
    if _ensure_pipeline is not None:
        _ensure_pipeline()
    from services import director_pipeline
    return director_pipeline


def start_panorama(bible_id: str, element_id: str, variant_id: str, *, submit: Callable[..., list[str]] | None = None,
                   views: list[str] | None = None, wait: bool = False, with_style: bool = True,
                   mode: str = "auto") -> dict:
    """Render the place's panorama in the background, then its angle crops as candidates; returns the bible."""
    bible = bibles.get_bible(bible_id)
    if not bible:
        raise BibleError("Unknown bible")
    kind, element = bibles.find_element(bible, element_id)
    if kind != "location":
        raise BibleError("Panoramas are for places")
    variant = bibles.find_variant(element, variant_id)
    plate = bibles.image_path(bible_id, bibles.approved_image(element, variant_id) or "")
    mode = choose_mode(mode, has_plate=bool(plate and Path(plate).is_file()))
    key = f"{bible_id}/{element_id}/{variant_id}"
    if key in _LIVE:
        raise BibleError("This place's panorama is being rendered")
    chosen = [v for v in (views or list(bibles.LOCATION_VIEWS)) if v in bibles.LOCATION_VIEWS]
    slots: dict[str, str] = {}
    for view in chosen:
        bible, slot = bibles.start_render(bible_id, element_id, variant_id, view)
        slots[view] = slot
    bible, front_slot = bibles.start_render(bible_id, element_id, variant_id, "")
    slots["front"] = front_slot
    _LIVE.add(key)
    thread = threading.Thread(target=_run, args=(bible_id, element_id, variant_id, slots, submit, with_style, mode),
                              name=f"panorama-{element_id}", daemon=True)
    thread.start()
    if wait:
        thread.join()
        return bibles.get_bible(bible_id)
    return bible


def text_job(bible: Mapping[str, Any], element: Mapping[str, Any], variant: Mapping[str, Any],
             spec: Mapping[str, Any] | None, *, with_style: bool, pipeline) -> dict:
    """The Qwen text-to-panorama job (panorama LoRA, the style LoRA on top when the bible renders with it)."""
    look = str((bible.get("world") or {}).get("look") or "").strip()
    prompt = panorama_prompt(element, spec, look=look, state=variant.get("description") or "")
    loras, mults = [lora_file()], ["1.00"]
    style = style_lora.bible_render_lora(bible, "location", prompt) if with_style else None
    if style:
        loras += style["activated_loras"]
        mults.append(style["loras_multipliers"])
        prompt = f"{prompt} {style['prompt_suffix']}".strip()
    preset = pipeline.DIRECTOR_IMAGE_SAMPLERS.get(style_lora.IMAGE_MODEL, {})
    return {
        "model_type": style_lora.IMAGE_MODEL, "prompt": prompt, "image_mode": 1, "image_prompt_type": "",
        "video_prompt_type": "", "generation_mode": "image",
        "num_inference_steps": preset.get("num_inference_steps", 6),
        "guidance_scale": preset.get("guidance_scale", 1.0), "resolution": PANO_RESOLUTION, "seed": -1,
        "settings_version": 2.52, "repeat_generation": 1, "negative_prompt": "", "video_length": 1,
        "activated_loras": loras, "loras_multipliers": " ".join(mults),
        **pipeline.director_image_job_fields(preset), "_director_cost_stage": "panorama",
    }


def outpaint_job(element: Mapping[str, Any], variant: Mapping[str, Any], spec: Mapping[str, Any] | None,
                 canvas: str, *, pipeline) -> dict:
    """The Klein outpaint job: the green canvas with the plate as the reference, the LoRA's prompt."""
    return {
        "model_type": OUTPAINT_MODEL, "prompt": outpaint_prompt(element, spec, state=variant.get("description") or ""),
        "image_mode": 1, "image_prompt_type": "", "generation_mode": "image",
        "video_prompt_type": pipeline._director_image_prompt_type(OUTPAINT_MODEL, [canvas]),
        "image_refs": [canvas], "image_refs_relative_size": 100,
        "num_inference_steps": OUTPAINT_STEPS, "guidance_scale": OUTPAINT_GUIDANCE, "resolution": PANO_RESOLUTION,
        "seed": -1, "settings_version": 2.52, "repeat_generation": 1, "negative_prompt": OUTPAINT_NEGATIVE,
        "video_length": 1, "activated_loras": [OUTPAINT_LORA], "loras_multipliers": OUTPAINT_SCALE,
        "_director_cost_stage": "panorama",
    }


def _run(bible_id: str, element_id: str, variant_id: str, slots: Mapping[str, str], submit, with_style: bool,
         mode: str = "text") -> None:
    from PIL import Image

    key = f"{bible_id}/{element_id}/{variant_id}"
    try:
        with _SLOT:
            for slot in slots.values():
                bibles.mark_rendering(bible_id, element_id, variant_id, slot)
            bible = bibles.get_bible(bible_id) or {}
            _, element = bibles.find_element(bible, element_id)
            variant = bibles.find_variant(element, variant_id)
            plate = bibles.image_path(bible_id, bibles.approved_image(element, variant_id) or "")
            analyst = globals().get("style_analyst_backend")
            spec = room_spec.ensure_spec(bible_id, element, str(plate), backend=analyst() if callable(analyst) else None) \
                if plate else room_spec.load_spec(bible_id, element_id)
            pipeline = _director()
            out_dir = bibles._folder(bible_id) / "_panorama"
            out_dir.mkdir(parents=True, exist_ok=True)
            if mode == "outpaint":
                with Image.open(plate) as source:
                    canvas = outpaint_canvas(source.convert("RGB"))
                canvas_path = out_dir / "canvas.png"
                canvas.save(canvas_path)
                params = outpaint_job(element, variant, spec, str(canvas_path), pipeline=pipeline)
            else:
                params = text_job(bible, element, variant, spec, with_style=with_style, pipeline=pipeline)
            outputs = (submit or pipeline._submit_and_wait)(params, timeout_s=1200, out_dir=str(out_dir))
            if not outputs or not (out_dir / outputs[0]).is_file():
                raise PanoramaError("The panorama render produced no image")
            with Image.open(out_dir / outputs[0]) as source:
                panorama = source.convert("RGB")
            panorama.save(pano_path(bible_id, element_id))
            for view, slot in slots.items():
                crop = crop_view(panorama, VIEW_YAWS[view])
                target = out_dir / f"{view}.png"
                crop.save(target)
                bibles.finish_render(bible_id, element_id, variant_id, slot, source=str(target))
            print(f"[Panorama] {bible_id}/{element_id}: {len(slots)} angles cut from "
                  f"{pano_path(bible_id, element_id).name} ({mode})")
    except Exception as error:  # noqa: BLE001 - recorded on the slots
        print(f"[Panorama] {bible_id}/{element_id}: {error}")
        for slot in slots.values():
            try:
                bibles.finish_render(bible_id, element_id, variant_id, slot, error=str(error)[:300])
            except Exception:  # noqa: BLE001
                pass
    finally:
        _LIVE.discard(key)


def register_routes(api: Any, *, ensure_pipeline: Callable[[], Any] | None = None,
                    lora_name: Callable[[], str] | None = None) -> None:
    global _ensure_pipeline, _lora_name
    _ensure_pipeline = ensure_pipeline
    _lora_name = lora_name
    base = "/api/v1/director/bibles/{bible_id}/elements/{element_id}/variants/{variant_id}/panorama"

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (BibleError, PanoramaError) as error:
            raise HTTPException(status_code=400, detail=str(error))

    @api.post(base)
    async def bible_panorama_render(bible_id: str, element_id: str, variant_id: str, request: Request):
        body = {}
        if request.headers.get("content-length", "0") != "0":
            try:
                body = await request.json()
            except ValueError:
                body = {}
        body = body if isinstance(body, dict) else {}
        views = [str(v) for v in body.get("views") or []] if isinstance(body.get("views"), list) else None
        bible = call(start_panorama, bible_id, element_id, variant_id, views=views,
                     with_style=body.get("with_style") is not False, mode=str(body.get("mode") or "auto"))
        return {"bible": bible}

    @api.get(base)
    def bible_panorama_get(bible_id: str, element_id: str, variant_id: str):
        path = pano_path(bible_id, element_id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="No panorama yet")
        return FileResponse(str(path))

    @api.get(base + "/available")
    def bible_panorama_available(bible_id: str, element_id: str, variant_id: str):
        modes = available_modes()
        return {"lora": lora_file(), "outpaint_lora": OUTPAINT_LORA, "modes": modes, "available": any(modes.values()),
                "panorama": pano_path(bible_id, element_id).is_file()}
