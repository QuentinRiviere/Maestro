"""A place's 360° panorama and the angles cut from it.

Edited from its plate, a place's other angles come back as the plate; painted
from text, their volumes drift. A panorama fixes the geometry: one
equirectangular image of the room (the panorama LoRA trained on Poly Haven's
CC0 HDRIs, ``lora_training``), then every angle is a perspective crop of it
(``crop_view``: yaw per wall, any yaw the scene lock asks for later), so the
walls, windows and furniture sit where one picture put them.

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
VIEW_FOV = 70.0
VIEW_SIZE = (1280, 720)
# Yaw of each angle from the plate's own direction (front = 0, clockwise positive).
VIEW_YAWS = {"front": 0.0, "right": 90.0, "reverse": 180.0, "left": -90.0}
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


def _director():
    if _ensure_pipeline is not None:
        _ensure_pipeline()
    from services import director_pipeline
    return director_pipeline


def start_panorama(bible_id: str, element_id: str, variant_id: str, *, submit: Callable[..., list[str]] | None = None,
                   views: list[str] | None = None, wait: bool = False, with_style: bool = True) -> dict:
    """Render the place's panorama in the background, then its angle crops as candidates; returns the bible."""
    bible = bibles.get_bible(bible_id)
    if not bible:
        raise BibleError("Unknown bible")
    kind, element = bibles.find_element(bible, element_id)
    if kind != "location":
        raise BibleError("Panoramas are for places")
    variant = bibles.find_variant(element, variant_id)
    if not lora_available():
        raise BibleError(f"The panorama LoRA ({lora_file()}) is not in the LoRA registry: train it first "
                         "(Director → LoRA trainings)")
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
    thread = threading.Thread(target=_run, args=(bible_id, element_id, variant_id, slots, submit, with_style),
                              name=f"panorama-{element_id}", daemon=True)
    thread.start()
    if wait:
        thread.join()
        return bibles.get_bible(bible_id)
    return bible


def _run(bible_id: str, element_id: str, variant_id: str, slots: Mapping[str, str], submit, with_style: bool) -> None:
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
            look = str((bible.get("world") or {}).get("look") or "").strip()
            prompt = panorama_prompt(element, spec, look=look, state=variant.get("description") or "")
            loras, mults = [lora_file()], ["1.00"]
            style = style_lora.bible_render_lora(bible, "location", prompt) if with_style else None
            if style:
                loras += style["activated_loras"]
                mults.append(style["loras_multipliers"])
                prompt = f"{prompt} {style['prompt_suffix']}".strip()
            pipeline = _director()
            preset = pipeline.DIRECTOR_IMAGE_SAMPLERS.get(style_lora.IMAGE_MODEL, {})
            params = {
                "model_type": style_lora.IMAGE_MODEL, "prompt": prompt, "image_mode": 1, "image_prompt_type": "",
                "video_prompt_type": "", "generation_mode": "image",
                "num_inference_steps": preset.get("num_inference_steps", 6),
                "guidance_scale": preset.get("guidance_scale", 1.0), "resolution": PANO_RESOLUTION, "seed": -1,
                "settings_version": 2.52, "repeat_generation": 1, "negative_prompt": "", "video_length": 1,
                "activated_loras": loras, "loras_multipliers": " ".join(mults),
                **pipeline.director_image_job_fields(preset), "_director_cost_stage": "panorama",
            }
            out_dir = bibles._folder(bible_id) / "_panorama"
            out_dir.mkdir(parents=True, exist_ok=True)
            outputs = (submit or pipeline._submit_and_wait)(params, timeout_s=900, out_dir=str(out_dir))
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
            print(f"[Panorama] {bible_id}/{element_id}: {len(slots)} angles cut from {pano_path(bible_id, element_id).name}")
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
                     with_style=body.get("with_style") is not False)
        return {"bible": bible}

    @api.get(base)
    def bible_panorama_get(bible_id: str, element_id: str, variant_id: str):
        path = pano_path(bible_id, element_id)
        if not path.is_file():
            raise HTTPException(status_code=404, detail="No panorama yet")
        return FileResponse(str(path))

    @api.get(base + "/available")
    def bible_panorama_available(bible_id: str, element_id: str, variant_id: str):
        return {"lora": lora_file(), "available": lora_available(),
                "panorama": pano_path(bible_id, element_id).is_file()}
