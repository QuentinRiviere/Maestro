"""Director hook: a blockout guide image per storyboard shot (off by default).

``director_blockout_guide`` picks the guide: ``off`` (default: nothing is
rendered, no job changes), ``depth`` or ``semantic``. When on, each scene shot
(a shot with a scene-locked blocking map) is rendered from the location's 3D
layout, its people and its camera; the depth, semantic and normal images are
written next to the shot's images and the shot record keeps their names and
the camera. An image model that takes a control image (Qwen Image 2.1) gets
the chosen guide as its raw control image (``image_guide``, ``V``); every
model gets the set's screen sides in words ("the window on frame left").
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Optional

import numpy as np

from services.director.blocking_map import main_characters, ots_foreground, parse_blocking_map
from services.director.blockout.layout import default_layout, normalize_layout
from services.director.blockout.render import Frame, render
from services.director.blockout.scene import build_scene, infer_postures, shot_camera

log = logging.getLogger(__name__)

GUIDE_MODES = ("off", "depth", "semantic")
# Long side of the rendered guide; the image model resizes its control image.
GUIDE_LONG_SIDE = 768
# Image models whose job takes a raw control image (``image_guide`` with ``V``).
CONTROL_MODEL_PREFIXES = ("qwen_image_21",)
# Set classes worth naming in the prompt, with the share of the frame they need.
_NAMED_CLASSES = ("window", "door", "counter", "shelf", "sofa", "bed", "cabinet")
_MIN_NAMED_COVERAGE = 0.01
_REFERENCE_NUMBER = re.compile(r"\b(reference image|Picture|image) (\d+)\b|<image(\d+)>")


def guide_mode(params: Optional[dict]) -> str:
    """The run's guide mode; anything unknown stays off."""
    value = (params or {}).get("director_blockout_guide", "off")
    if value is True:
        return "depth"
    text = str(value if value is not None else "off").strip().lower()
    if text in ("", "false", "0", "no", "none", "disabled"):
        return "off"
    if text not in GUIDE_MODES:
        log.warning("Unknown director_blockout_guide value %r; the blockout guide stays off", value)
        return "off"
    return text


def control_supported(image_model: str) -> bool:
    """Whether the image model's job takes a raw control image."""
    model = str(image_model or "").lower()
    return model.startswith(CONTROL_MODEL_PREFIXES)


def guide_size(resolution: Any) -> tuple[int, int]:
    """The guide's (width, height): the shot's aspect with a long side of ``GUIDE_LONG_SIDE``."""
    match = re.match(r"\s*(\d+)\s*[x×]\s*(\d+)", str(resolution or ""))
    width, height = (int(match.group(1)), int(match.group(2))) if match else (1280, 720)
    scale = GUIDE_LONG_SIDE / max(width, height, 1)
    return max(8, round(width * scale / 8) * 8), max(8, round(height * scale / 8) * 8)


def location_layout(params: dict, location: str) -> tuple[dict, str]:
    """(layout, source) of a place: ``params`` override, bible layout, its description, or the default room."""
    key = str(location or "").strip().casefold()
    overrides = params.get("director_location_layouts")
    if isinstance(overrides, dict):
        for name, raw in overrides.items():
            if str(name).strip().casefold() == key and normalize_layout(raw):
                return normalize_layout(raw), "params"
    cast = params.get("_director_cast") if isinstance(params.get("_director_cast"), dict) else {}
    for entry in cast.get("locations") or []:
        if not isinstance(entry, dict) or str(entry.get("name") or "").strip().casefold() != key:
            continue
        layout = normalize_layout(entry.get("layout")) if entry.get("layout") else None
        if layout:
            return layout, "bible"
        text = " ".join(str(entry.get(field) or "") for field in ("description", "notes")).strip()
        if text:
            return default_layout(text), "description"
    return default_layout(""), "default"


def scene_character_slots(clip_plans: list[dict]) -> dict[str, dict[str, int]]:
    """Per continuity group, each person's semantic slot: the scene's most present person is slot 0."""
    groups: dict[str, list[Optional[dict]]] = {}
    for plan in clip_plans:
        group = str((plan or {}).get("_director_continuity_group") or "")
        if group:
            groups.setdefault(group, []).append(parse_blocking_map((plan or {}).get("_director_blocking_map")))
    return {group: {name: slot for slot, name in enumerate(main_characters(maps))}
            for group, maps in groups.items()}


def _shot_text(plan: dict) -> str:
    return " ".join(str(plan.get(key) or "") for key in (
        "_director_opening_blocking", "image_prompt", "_director_environment"))


def _sides(frame: Frame, scene) -> dict[str, str]:
    sides = {}
    for name, info in frame.class_coverage(scene).items():
        if name in _NAMED_CLASSES and info["coverage"] >= _MIN_NAMED_COVERAGE:
            x = info["center_x"]
            sides[name] = "frame left" if x < 0.4 else "frame right" if x > 0.6 else "frame center"
    return sides


def render_shot_guide(plan: dict, params: dict, *, out_dir: str, name: str, resolution: Any,
                      slots: Optional[dict] = None) -> Optional[dict]:
    """Render one scene shot's guide images; the shot record's ``blockout_guide`` entry, or None.

    ``plan`` is the scene-locked plan (``scene_locked_plan``). Returns None
    when the run's mode is off, or the shot has no usable blocking map (an
    insert, a shot outside a scene). ``name`` prefixes the written files.
    """
    mode = guide_mode(params)
    if mode == "off" or not plan.get("_director_continuity_group"):
        return None
    blocking = parse_blocking_map(plan.get("_director_blocking_map"))
    if not blocking:
        return None
    started = time.perf_counter()
    layout, source = location_layout(params, str(plan.get("_director_location") or ""))
    names = [s["name"] for s in [*blocking["subjects"], *blocking.get("offscreen", [])]]
    scene = build_scene(layout, blocking, character_slots=slots,
                        postures=infer_postures(names, _shot_text(plan)))
    foreground = ots_foreground(blocking, plan.get("_director_camera_framing") or "")
    camera = shot_camera(scene, blocking, angle=str(plan.get("_director_camera_angle") or ""),
                         foreground=foreground)
    width, height = guide_size(resolution)
    frame = render(scene, camera, width, height)

    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name))
    files = {}
    for kind, pixels in (("depth", frame.depth_image()), ("semantic", frame.semantic_image()),
                         ("normal", frame.normal_image())):
        filename = f"{stem}_blockout_{kind}.png"
        Image.fromarray(np.ascontiguousarray(pixels)).save(os.path.join(out_dir, filename))
        files[kind] = filename
    near, far = frame.depth_range()
    return {
        "mode": mode,
        "image": files[mode],
        "files": files,
        "size": [width, height],
        "layout_source": source,
        "camera": camera.to_dict(width, height),
        "depth_range_m": [round(near, 3), round(far, 3)],
        "people": {p.name: {"slot": p.slot, "posture": p.posture} for p in scene.people},
        "foreground": foreground,
        "set_sides": _sides(frame, scene),
        "render_ms": round((time.perf_counter() - started) * 1000, 1),
    }


def set_sides_line(record: Optional[dict]) -> str:
    """The set's screen sides in words ("" when nothing worth naming is in frame)."""
    sides = (record or {}).get("set_sides") or {}
    if not sides:
        return ""
    return "Set layout: " + ", ".join(f"the {name} on {side}" for name, side in sides.items()) + "."


def control_role_text(mode: str) -> str:
    """What the control image is, for the editor (it is reference image 1)."""
    if mode == "semantic":
        return ("Reference image 1 is a colour-coded layout of this exact camera view: grey walls, light grey "
                "windows, green doors, dark brown floor, flat colours for the furniture and one flat colour per "
                "person. Keep where every wall, window, door, piece of furniture and person is in the frame; "
                "do not draw its flat colours.")
    return ("Reference image 1 is a depth map of this exact camera view (lighter is closer). Keep its "
            "composition: the camera position, where and how large each person is, the furniture and the walls; "
            "do not draw the depth map.")


def shift_reference_numbers(prompt: str, by: int = 1) -> str:
    """Renumber "reference image N" / "Picture N" / "<imageN>" by ``by``: the control image takes slot 1."""
    def bump(match: re.Match) -> str:
        if match.group(3):
            return f"<image{int(match.group(3)) + by}>"
        return f"{match.group(1)} {int(match.group(2)) + by}"

    return _REFERENCE_NUMBER.sub(bump, prompt or "")


def guide_prompt(prompt: str, record: Optional[dict], *, control: bool) -> str:
    """The shot prompt with the guide's words: the set's sides, and the control image's role when it is sent."""
    if not record:
        return prompt
    text = prompt or ""
    if control:
        text = f"{control_role_text(record.get('mode', 'depth'))} {shift_reference_numbers(text)}"
    sides = set_sides_line(record)
    if sides and sides not in text:
        text = f"{text.rstrip()} {sides}"
    return text


def apply_control(gen_params: dict, guide_path: str) -> dict:
    """An image job with ``guide_path`` as its raw control image (``image_guide``, ``V``).

    The control image takes the editor's first slot, so the canvas mode
    (``K``) is dropped and references stay plain references (``I``).
    """
    if not guide_path:
        return gen_params
    prompt_type = str(gen_params.get("video_prompt_type") or "").replace("K", "").replace("V", "")
    return {**gen_params, "image_guide": guide_path, "video_prompt_type": "V" + prompt_type}
