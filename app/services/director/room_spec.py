"""A place's room sheet: what each wall holds, written by the analyst from the plate.

An image editor given the plate copies its composition, so a place's other
angles (reverse, left wall, right wall) are rendered from text alone. To
keep them one room, the analyst reads the approved plate once and writes a
sheet: materials, colours, light, and the furniture and fixtures of each
wall, including what the plate's own camera has behind it (inferred from the
description and the layout notes). The sheet is a sidecar of the bible
(``room_<element>.json``); ``wall_prompt`` turns it into the prompt of one
angle.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from . import bibles
from .style_reference import StyleReferenceError, _ask, _resolve_backend, data_uri_image, extract_json

WALLS = ("front", "back", "left", "right")
# The wall a camera looks at, per angle, from the plate's point of view: the
# plate looks at the front wall.
VIEW_WALLS = {"reverse": "back", "left": "left", "right": "right"}
# Where the camera stands and what frames the edges, per angle.
VIEW_CAMERAS = {
    "reverse": ("the camera stands against the front wall (the wall the main picture looks at) and looks straight "
                "at the back wall, the one that was behind the main picture's camera", "left", "right"),
    "left": ("the camera stands against the right wall and looks straight at the left wall", "back", "front"),
    "right": ("the camera stands against the left wall and looks straight at the right wall", "front", "back"),
}
MAX_TEXT = 700

ROOM_SYSTEM_PROMPT = """You are a production designer documenting a film set from one photograph of it.
The photograph looks at the set's FRONT wall. Describe the set as a room with four walls so that other camera angles of the same room can be painted from your sheet alone.
Answer with one JSON object only:
{"summary": "one sentence: what kind of place, its period and mood",
 "materials": "floor, walls, ceiling: materials and colours, as seen",
 "light": "the light sources, their colour temperature and direction, time of day",
 "palette": ["3 to 5 dominant colours"],
 "walls": {
   "front": "everything on the wall the photograph looks at: windows, doors, furniture, fixtures, objects, from left to right",
   "left": "what the left wall holds, from the far end (next to the front wall) to the near end",
   "right": "what the right wall holds, from the far end (next to the front wall) to the near end",
   "back": "what the wall behind the camera most plausibly holds: entrance, counter, shelves... consistent with the place"
 }}
Rules: what the photograph shows is fact; what it does not show (parts of the side walls, the back wall) you infer from the place's description and layout, keeping the same materials and period. Name concrete furniture and fixtures, with their colours. Never describe people. No text outside the JSON."""


class RoomSpecError(RuntimeError):
    pass


def spec_path(bible_id: str, element_id: str) -> Path:
    return bibles._folder(bible_id) / f"room_{bibles._slug(element_id, 40)}.json"


def load_spec(bible_id: str, element_id: str) -> dict | None:
    try:
        data = json.loads(spec_path(bible_id, element_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and isinstance(data.get("walls"), dict) else None


def save_spec(bible_id: str, element_id: str, spec: Mapping[str, Any]) -> None:
    spec_path(bible_id, element_id).write_text(json.dumps(spec, indent=1), encoding="utf-8")


def _clean(value: Any, limit: int = MAX_TEXT) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def parse_spec(text: str) -> dict:
    raw = extract_json(text)
    if not isinstance(raw, dict) or not isinstance(raw.get("walls"), dict):
        raise RoomSpecError("The room sheet is not a JSON object with walls")
    walls = {wall: _clean(raw["walls"].get(wall)) for wall in WALLS}
    if sum(1 for value in walls.values() if value) < 3:
        raise RoomSpecError("The room sheet names fewer than three walls")
    palette = raw.get("palette") if isinstance(raw.get("palette"), list) else []
    return {
        "summary": _clean(raw.get("summary"), 300),
        "materials": _clean(raw.get("materials")),
        "light": _clean(raw.get("light")),
        "palette": [_clean(c, 40) for c in palette if _clean(c, 40)][:6],
        "walls": walls,
        "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def describe_room(image_path: str, element: Mapping[str, Any], *, backend: Any = None, fal_key: str | None = None,
                  model: str = "", ask: Callable[..., tuple[str, Any]] = _ask, http: Any = None) -> dict:
    """The room sheet of a place from its plate (one analyst call)."""
    ask, fal_key, model, local_backend = _resolve_backend(backend, fal_key, model or "analyst", ask)
    prompt = (f"Place: {element.get('name', '')}. Description: {_clean(element.get('description'), 1200)}"
              + (f" Layout notes: {_clean(element.get('notes'), 600)}" if element.get("notes") else "")
              + " Write the room sheet of this set from the photograph.")
    media = [image_path] if local_backend else [data_uri_image(image_path)]
    owns_http = http is None
    if owns_http:
        import httpx
        http = httpx.Client(timeout=180)
    try:
        text, _cost = ask(http, fal_key, model, ROOM_SYSTEM_PROMPT, prompt, media, time.sleep, max_tokens=3000,
                          timeout=300)
    except StyleReferenceError as error:
        raise RoomSpecError(str(error)) from error
    finally:
        if owns_http:
            http.close()
    return parse_spec(text)


def ensure_spec(bible_id: str, element: Mapping[str, Any], image_path: str, *, backend: Any = None,
                fal_key: str | None = None, model: str = "") -> dict | None:
    """The stored sheet of the place, written now when missing (None when the analyst is unavailable)."""
    spec = load_spec(bible_id, element["id"])
    if spec:
        return spec
    try:
        spec = describe_room(image_path, element, backend=backend, fal_key=fal_key, model=model)
    except (RoomSpecError, StyleReferenceError) as error:
        print(f"[Room sheet] {bible_id}/{element.get('id')}: {error}")
        return None
    save_spec(bible_id, element["id"], spec)
    return spec


def wall_prompt(view: str, element: Mapping[str, Any], spec: Mapping[str, Any] | None, *, state: str = "") -> str:
    """The text-only prompt of one angle of the place: the wall it looks at fills the frame, nobody in it."""
    if view not in VIEW_CAMERAS:
        raise RoomSpecError(f"Unknown angle {view}")
    camera, left_edge, right_edge = VIEW_CAMERAS[view]
    wall = VIEW_WALLS[view]
    walls = (spec or {}).get("walls") or {}
    parts = [f"Interior of {element.get('name', 'the place')}, empty, nobody in it, every seat unoccupied, "
             f"no text overlay. {camera}, which fills the frame."]
    if walls.get(wall):
        parts.append(f"That wall holds: {walls[wall]}")
    if walls.get(left_edge):
        parts.append(f"At the left edge of the frame, the start of the {left_edge} wall: {walls[left_edge]}")
    if walls.get(right_edge):
        parts.append(f"At the right edge of the frame, the start of the {right_edge} wall: {walls[right_edge]}")
    if spec:
        if spec.get("materials"):
            parts.append(f"Materials: {spec['materials']}")
        if spec.get("light"):
            parts.append(f"Light: {spec['light']}")
        if spec.get("palette"):
            parts.append("Palette: " + ", ".join(spec["palette"]) + ".")
    else:
        parts.append(f"Place: {_clean(element.get('description'), 900)}")
        if element.get("notes"):
            parts.append(f"Layout: {_clean(element.get('notes'), 400)}")
    if state:
        parts.append(f"State: {_clean(state, 300)}")
    parts.append("The same room as the main picture of this place: same architecture, period, materials, colours "
                 "and light, seen from this other side.")
    return " ".join(p.rstrip(".") + "." for p in parts if p)
