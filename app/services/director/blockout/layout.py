"""The rough 3D layout of a place: its room box, wall openings and furniture.

A bible location stores one layout (``layout`` on the location element). The
blockout renderer builds the set from it and puts the storyboard's people and
camera in it, so every shot of a scene sees the same window on the same wall.

Coordinates follow the storyboard's floor plan (see ``blocking_map``): the
plan is seen from above with the scene's master camera at the bottom looking
at the back wall. ``x`` runs from the left wall (0) to the right wall (1),
``y`` from the back wall (0) to the front wall (1). Furniture is placed in
these normalized floor coordinates, so a blocking prop ("table" at 0.5, 0.5)
and a layout table share one place; sizes are in meters.

Openings sit on a wall at ``center`` (0..1 along the wall: left to right for
the back and front walls, back to front for the left and right walls), with a
width, a sill height (``bottom``) and a height in meters.
"""
from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Callable
from typing import Any, Optional

LAYOUT_VERSION = 1
WALLS = ("back", "front", "left", "right")
OPENING_KINDS = ("window", "door")
FURNITURE_CLASSES = ("table", "counter", "shelf", "chair", "sofa", "bed", "cabinet", "plant", "lamp", "prop")
SHAPES = ("box", "cylinder")
MAX_OPENINGS = 8
MAX_FURNITURE = 16

# Room size limits and the default room (meters).
ROOM_LIMITS = {"width": (1.5, 40.0), "depth": (1.5, 40.0), "height": (2.0, 12.0)}
DEFAULT_ROOM = {"width": 6.0, "depth": 5.0, "height": 2.8}
# Default opening geometry per kind: (width, bottom, height) in meters.
OPENING_DEFAULTS = {"window": (1.6, 0.9, 1.3), "door": (0.9, 0.0, 2.1)}
# Default furniture size per class: (width, depth, height, shape) in meters.
FURNITURE_DEFAULTS = {
    "table": (0.8, 0.8, 0.75, "cylinder"),
    "counter": (0.6, 2.4, 1.05, "box"),
    "shelf": (1.6, 0.35, 1.9, "box"),
    "chair": (0.45, 0.45, 0.9, "box"),
    "sofa": (1.9, 0.85, 0.85, "box"),
    "bed": (1.6, 2.0, 0.55, "box"),
    "cabinet": (1.0, 0.5, 0.9, "box"),
    "plant": (0.45, 0.45, 1.2, "cylinder"),
    "lamp": (0.35, 0.35, 1.6, "cylinder"),
    "prop": (0.4, 0.4, 0.4, "box"),
}
# Words that name each furniture class, in prop names and place descriptions.
CLASS_WORDS = {
    "table": ("table", "desk"),
    "counter": ("counter", "bar", "espresso bar", "reception", "kitchen island", "island"),
    "shelf": ("shelf", "shelves", "bookcase", "bookshelf", "rack"),
    "chair": ("chair", "stool", "armchair", "bench"),
    "sofa": ("sofa", "couch", "settee"),
    "bed": ("bed",),
    "cabinet": ("cabinet", "dresser", "wardrobe", "sideboard", "fridge", "refrigerator"),
    "plant": ("plant", "tree", "planter"),
    "lamp": ("lamp", "floor lamp"),
}


class LayoutError(ValueError):
    pass


def _number(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return min(high, max(low, number))


def _text(value: Any, limit: int = 40) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def furniture_class(name: Any) -> str:
    """The furniture class a name or a free class text belongs to ("prop" when none matches)."""
    text = f" {str(name or '').lower()} "
    if text.strip() in FURNITURE_CLASSES:
        return text.strip()
    for cls, words in CLASS_WORDS.items():
        if any(re.search(rf"\b{re.escape(word)}s?\b", text) for word in words):
            return cls
    return "prop"


def wall_length(room: dict, wall: str) -> float:
    return room["width"] if wall in ("back", "front") else room["depth"]


def validate_layout(raw: Any) -> tuple[Optional[dict], list[str]]:
    """(the clean layout or None, what was dropped or changed, in words).

    Missing room sizes take the default room; out-of-range numbers are
    clamped; openings on unknown walls and furniture outside the floor are
    dropped. Returns None only when ``raw`` is not a layout object.
    """
    problems: list[str] = []
    if not isinstance(raw, dict):
        return None, ["the layout is not an object"]
    raw_room = raw.get("room") if isinstance(raw.get("room"), dict) else {}
    if not raw_room:
        problems.append("no room size: the default room is used")
    room = {}
    for key, (low, high) in ROOM_LIMITS.items():
        value = _number(raw_room.get(key), DEFAULT_ROOM[key], low, high)
        if raw_room.get(key) is not None and value != _number(raw_room.get(key), value, -1e9, 1e9):
            problems.append(f"room {key} clamped to {value:g} m")
        room[key] = round(value, 3)

    openings = []
    for index, item in enumerate(raw.get("openings") if isinstance(raw.get("openings"), list) else []):
        if not isinstance(item, dict):
            problems.append(f"opening {index + 1} is not an object")
            continue
        wall = str(item.get("wall") or "").strip().lower()
        kind = str(item.get("kind") or item.get("type") or "window").strip().lower()
        if wall not in WALLS:
            problems.append(f"opening {index + 1}: unknown wall {wall!r}")
            continue
        if kind not in OPENING_KINDS:
            problems.append(f"opening {index + 1}: unknown kind {kind!r}")
            continue
        length = wall_length(room, wall)
        default_width, default_bottom, default_height = OPENING_DEFAULTS[kind]
        width = _number(item.get("width"), min(default_width, length), 0.2, length)
        bottom = _number(item.get("bottom"), default_bottom, 0.0, room["height"] - 0.2)
        height = _number(item.get("height"), default_height, 0.2, room["height"] - bottom)
        half = width / 2 / length
        center = _number(item.get("center"), 0.5, half, 1.0 - half)
        openings.append({"wall": wall, "kind": kind, "center": round(center, 4), "width": round(width, 3),
                         "bottom": round(bottom, 3), "height": round(height, 3),
                         "name": _text(item.get("name")) or kind})
    if len(openings) > MAX_OPENINGS:
        problems.append(f"only the first {MAX_OPENINGS} openings are kept")
    openings = openings[:MAX_OPENINGS]
    for wall in WALLS:
        spans = sorted((o["center"] * wall_length(room, wall) - o["width"] / 2,
                        o["center"] * wall_length(room, wall) + o["width"] / 2, o["name"])
                       for o in openings if o["wall"] == wall)
        for (_s1, e1, n1), (s2, _e2, n2) in zip(spans, spans[1:]):
            if s2 < e1 - 1e-6:
                problems.append(f"openings {n1!r} and {n2!r} overlap on the {wall} wall")

    furniture = []
    for index, item in enumerate(raw.get("furniture") if isinstance(raw.get("furniture"), list) else []):
        if not isinstance(item, dict):
            problems.append(f"furniture {index + 1} is not an object")
            continue
        name = _text(item.get("name")) or _text(item.get("class")) or f"item {index + 1}"
        cls = str(item.get("class") or "").strip().lower()
        cls = cls if cls in FURNITURE_CLASSES else furniture_class(cls or name)
        try:
            x, y = float(item.get("x")), float(item.get("y"))
        except (TypeError, ValueError):
            problems.append(f"furniture {name!r}: no floor position")
            continue
        if not (math.isfinite(x) and math.isfinite(y) and -0.05 <= x <= 1.05 and -0.05 <= y <= 1.05):
            problems.append(f"furniture {name!r}: outside the floor")
            continue
        width, depth, height, shape = FURNITURE_DEFAULTS[cls]
        shape = str(item.get("shape") or shape).strip().lower()
        furniture.append({
            "name": name, "class": cls, "shape": shape if shape in SHAPES else "box",
            "x": round(min(1.0, max(0.0, x)), 4), "y": round(min(1.0, max(0.0, y)), 4),
            "width": round(_number(item.get("width"), width, 0.05, room["width"]), 3),
            "depth": round(_number(item.get("depth"), depth, 0.05, room["depth"]), 3),
            "height": round(_number(item.get("height"), height, 0.05, room["height"]), 3),
            "rotation_deg": round(_number(item.get("rotation_deg"), 0.0, -360.0, 360.0) % 360.0, 2),
        })
    if len(furniture) > MAX_FURNITURE:
        problems.append(f"only the first {MAX_FURNITURE} pieces of furniture are kept")
    return {"version": LAYOUT_VERSION, "room": room, "openings": openings,
            "furniture": furniture[:MAX_FURNITURE]}, problems


def normalize_layout(raw: Any) -> Optional[dict]:
    """The clean layout of ``raw`` (see ``validate_layout``), or None when it is not one."""
    layout, _problems = validate_layout(raw)
    return layout


# ── a first layout from the place's words ──────────────────────────────────

_SIDE_WORDS = {"left": ("left",), "right": ("right",), "back": ("back", "rear", "far", "behind"),
               "front": ("front", "street side", "entrance side")}


def _side_near(text: str, word: str) -> Optional[str]:
    """The wall named next to ``word`` in ``text`` ("window on the left", "left window")."""
    for match in re.finditer(rf"\b{word}s?\b", text):
        window = text[max(0, match.start() - 30): match.end() + 40]
        for wall, words in _SIDE_WORDS.items():
            if any(re.search(rf"\b{w}\b", window) for w in words):
                return wall
    return None


def default_layout(description: str = "") -> dict:
    """A first layout from a place's description, by keywords (no LLM).

    The place's main image is the master view, so it looks at the back wall:
    a window that the text does not place goes there, a door on the right
    wall near the front, a counter along the right wall, a table in the middle.
    It is a starting point for the user (or the LLM suggestion) to correct.
    """
    text = " " + re.sub(r"\s+", " ", str(description or "").lower()) + " "
    room = dict(DEFAULT_ROOM)
    if re.search(r"\b(small|tiny|narrow|cramped|cozy|cosy)\b", text):
        room.update(width=4.0, depth=3.6)
    elif re.search(r"\b(large|spacious|huge|hall|warehouse|loft|open[- ]plan)\b", text):
        room.update(width=10.0, depth=8.0)
    if re.search(r"\b(high ceiling|double[- ]height|loft|warehouse|cathedral)\b", text):
        room["height"] = 4.0
    openings: list[dict] = []
    furniture: list[dict] = []
    if re.search(r"\bwindows?\b|\bglass front\b|\bstorefront\b|\bshop window\b", text):
        wall = _side_near(text, "window") or "back"
        big = bool(re.search(r"floor[- ]to[- ]ceiling|large windows?|big windows?|picture window|glass front|"
                             r"storefront|shop window|bay window|panoramic", text))
        length = wall_length(room, wall)
        opening = {"wall": wall, "kind": "window", "center": 0.5, "name": "window"}
        if big:
            opening.update(width=round(length * 0.7, 2), bottom=0.3, height=round(room["height"] - 0.6, 2))
        openings.append(opening)
    if re.search(r"\b(door|doorway|entrance|entry)\b", text):
        wall = _side_near(text, "door") or "right"
        if wall in ("left", "right"):
            openings.append({"wall": wall, "kind": "door", "center": 0.8, "name": "door"})
        else:
            openings.append({"wall": wall, "kind": "door", "center": 0.15, "name": "door"})
    taken = {o["wall"] for o in openings}
    if re.search(r"\b(counter|bar|espresso bar|reception|kitchen island)\b", text):
        side = "left" if "right" in taken and "left" not in taken else "right"
        furniture.append({"name": "counter", "class": "counter", "x": 0.9 if side == "right" else 0.1,
                          "y": 0.45, "depth": round(min(room["depth"] * 0.5, 3.0), 2), "width": 0.6})
    if re.search(r"\b(shelf|shelves|bookcase|bookshelf)\b", text):
        furniture.append({"name": "shelf", "class": "shelf", "x": 0.5, "y": 0.04})
    if re.search(r"\b(sofa|couch|settee)\b", text):
        furniture.append({"name": "sofa", "class": "sofa", "x": 0.3, "y": 0.25})
    if re.search(r"\bbed\b", text):
        furniture.append({"name": "bed", "class": "bed", "x": 0.5, "y": 0.3})
    if re.search(r"\b(table|desk)s?\b", text):
        furniture.append({"name": "table", "class": "table", "x": 0.5, "y": 0.5})
    return normalize_layout({"room": room, "openings": openings, "furniture": furniture})


# ── a first layout from the LLM ────────────────────────────────────────────

SUGGEST_SYSTEM = """You turn the description of a film set into its rough 3D layout, for a blockout renderer.
The room is seen from above with the main camera at the front looking at the BACK wall: "left" and "right" are
the walls on that camera's left and right. Floor positions are normalized: x from the left wall (0) to the right
wall (1), y from the back wall (0) to the front wall (1). Sizes are meters. Put each window and door on the wall
the description implies; when it does not say, a window goes on the back wall.
Return ONLY JSON:
{"room": {"width": 6, "depth": 5, "height": 2.8},
 "openings": [{"wall": "back|front|left|right", "kind": "window|door", "center": 0.5, "width": 1.6,
               "bottom": 0.9, "height": 1.3, "name": "street window"}],
 "furniture": [{"name": "espresso bar", "class": "table|counter|shelf|chair|sofa|bed|cabinet|plant|lamp|prop",
                "shape": "box|cylinder", "x": 0.9, "y": 0.45, "width": 0.6, "depth": 2.4, "height": 1.05,
                "rotation_deg": 0}]}
At most 8 openings and 16 pieces of furniture; only what matters for the staging (no small objects)."""


def _json_object(text: str) -> Optional[dict]:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def suggest_layout(name: str, description: str, notes: str = "", *,
                   generate: Optional[Callable[..., str]] = None) -> dict:
    """A first layout of a place: ``{"layout", "source": "llm" | "keywords", "problems"}``.

    ``generate`` is the LLM call (``llm_service.generate``'s signature). Its
    answer is validated; without it, or when it fails, the keyword layout
    (``default_layout``) is returned. Nothing is saved here.
    """
    text = " ".join(part for part in (str(description or ""), str(notes or "")) if part).strip()
    if generate is not None:
        prompt = f"Set: {name}\nDescription: {description}\nGeometry notes: {notes}"
        for _attempt in range(2):
            try:
                answer = generate(prompt=prompt, system_prompt=SUGGEST_SYSTEM, max_new_tokens=1500,
                                  temperature=0.2, enable_thinking=False)
            except Exception as error:  # noqa: BLE001 - the keyword layout is the fallback
                return {"layout": default_layout(text), "source": "keywords",
                        "problems": [f"the LLM failed: {error}"]}
            raw = _json_object(answer)
            layout, problems = validate_layout(raw)
            if layout is not None and (layout["openings"] or layout["furniture"] or raw.get("room")):
                return {"layout": layout, "source": "llm", "problems": problems}
    return {"layout": default_layout(text), "source": "keywords",
            "problems": ["no LLM layout: built from keywords"] if generate is not None else []}


def merge_props(layout: dict, props: list[dict]) -> dict:
    """The layout with the scene's blocking props placed as furniture.

    The storyboard locks one place per prop for the whole scene, and the
    people stand around it, so a prop of a furniture class wins over the
    layout: the nearest piece of that class moves to the prop's place, or a
    default piece is added there. Props of no furniture class are ignored
    (cups and phones are too small for a blockout).
    """
    merged = copy.deepcopy(layout)
    moved: set[int] = set()
    for prop in props or []:
        cls = furniture_class(prop.get("name"))
        if cls == "prop":
            continue
        try:
            x, y = float(prop["x"]), float(prop["y"])
        except (KeyError, TypeError, ValueError):
            continue
        candidates = [(math.hypot(item["x"] - x, item["y"] - y), index)
                      for index, item in enumerate(merged["furniture"])
                      if item["class"] == cls and index not in moved]
        if candidates:
            _distance, index = min(candidates)
            merged["furniture"][index].update(x=round(x, 4), y=round(y, 4))
            moved.add(index)
        elif len(merged["furniture"]) < MAX_FURNITURE:
            width, depth, height, shape = FURNITURE_DEFAULTS[cls]
            merged["furniture"].append({"name": _text(prop.get("name")) or cls, "class": cls, "shape": shape,
                                        "x": round(x, 4), "y": round(y, 4), "width": width, "depth": depth,
                                        "height": height, "rotation_deg": 0.0})
            moved.add(len(merged["furniture"]) - 1)
    return merged
