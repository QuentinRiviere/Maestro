"""Per-shot blocking: where the camera stands and where each person is.

The planner writes a small top-view floor plan per shot (normalized 0..1:
x from the left wall to the right wall as the scene's main camera sees it,
y from the back wall to the front). The storyboard renders it as a flat
diagram, so the image editor takes the framing from geometry and only the
surfaces and light from the scene's photos. Geometry apart from texture.
"""
from __future__ import annotations

import math
from typing import Any, Optional

FRAMINGS = ("wide", "medium", "close-up")
# Horizontal field of view drawn for each framing, in degrees.
_FOV = {"wide": 70.0, "medium": 42.0, "close-up": 22.0}
_MAX_SUBJECTS = 6


def normalize_framing(text: Any) -> str:
    """Map free framing text ("medium close-up", "establishing") onto wide, medium or close-up."""
    value = str(text or "").lower()
    if any(word in value for word in ("close", "insert", "ecu", "detail")):
        return "close-up"
    if any(word in value for word in ("wide", "establishing", "long", "full", "master")):
        return "wide"
    return "medium"


def _coordinate(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    # A little outside the room is a camera against a wall; far outside is noise.
    if not math.isfinite(number) or not -0.5 <= number <= 1.5:
        return None
    return min(1.0, max(0.0, number))


def _angle(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number % 360.0 if math.isfinite(number) else None


def parse_blocking_map(raw: Any, fallback_framing: Any = "") -> Optional[dict]:
    """A clean blocking map, or None when the camera or every subject is unusable."""
    if not isinstance(raw, dict):
        return None
    camera = raw.get("camera")
    if not isinstance(camera, dict):
        return None
    x, y, facing = _coordinate(camera.get("x")), _coordinate(camera.get("y")), _angle(camera.get("facing_deg"))
    if x is None or y is None or facing is None:
        return None
    subjects = []
    for item in raw.get("subjects") or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()[:40]
        sx, sy = _coordinate(item.get("x")), _coordinate(item.get("y"))
        if not name or sx is None or sy is None:
            continue
        subjects.append({"name": name, "x": sx, "y": sy, "facing_deg": _angle(item.get("facing_deg"))})
    if not subjects:
        return None
    return {
        "camera": {"x": x, "y": y, "facing_deg": facing,
                   "framing": normalize_framing(camera.get("framing") or fallback_framing)},
        "subjects": subjects[:_MAX_SUBJECTS],
    }


def _to_camera(blocking: dict, x: float, y: float) -> tuple[float, float]:
    """(lateral, depth) of a floor point seen from the camera: lateral > 0 is frame right."""
    camera = blocking["camera"]
    theta = math.radians(camera["facing_deg"])
    dx, dy = x - camera["x"], y - camera["y"]
    # Facing 0 looks toward the back wall (-y); 90 toward the right wall (+x).
    return dx * math.cos(theta) + dy * math.sin(theta), dx * math.sin(theta) - dy * math.cos(theta)


def _facing_words(blocking: dict, facing: Optional[float]) -> str:
    if facing is None:
        return ""
    relative = (facing - blocking["camera"]["facing_deg"]) % 360.0
    if relative <= 45 or relative >= 315:
        return "back to camera"
    if 135 <= relative <= 225:
        return "facing camera"
    return "facing frame right" if relative < 180 else "facing frame left"


def framing_line(blocking: Optional[dict]) -> str:
    """The diagram in words, for editors that do not read images."""
    if not blocking:
        return ""
    placed = []
    for subject in blocking["subjects"]:
        lateral, depth = _to_camera(blocking, subject["x"], subject["y"])
        if depth > 0.02:
            placed.append((lateral, depth, subject))
    framing = blocking["camera"]["framing"]
    if not placed:
        return f"Framing: {framing} shot."
    placed.sort(key=lambda item: item[0])
    people = []
    for _lateral, _depth, subject in placed:
        facing = _facing_words(blocking, subject["facing_deg"])
        people.append(f"{subject['name']} ({facing})" if facing else subject["name"])
    order = " In frame, left to right: " + ", ".join(people) + "." if len(people) > 1 else (
        f" In frame: {people[0]}.")
    return f"Framing: {framing} shot.{order}"


def render_blocking_map(blocking: dict, path: str, size: int = 768) -> str:
    """Draw the shot's top view with the camera at the bottom looking up.

    Turning the plan so the camera always faces up the page makes the page's
    left the frame's left. Returns the written path.
    """
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default(size=max(14, size // 32))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()

    corners = [_to_camera(blocking, x, y) for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))]
    people = [(_to_camera(blocking, s["x"], s["y"]), s) for s in blocking["subjects"]]
    # Fit the camera and the people, not the whole room: a close-up then reads
    # as a camera standing close, and the walls only frame the shot.
    points = [(0.0, 0.0)] + [point for point, _ in people]
    min_l, max_l = min(p[0] for p in points), max(p[0] for p in points)
    min_d, max_d = min(p[1] for p in points), max(p[1] for p in points)
    margin = size * 0.18
    scale = (size - 2 * margin) / max(max_l - min_l, max_d - min_d, 0.25)
    offset_x = (size - (max_l - min_l) * scale) / 2
    offset_y = (size - (max_d - min_d) * scale) / 2

    def to_px(lateral: float, depth: float) -> tuple[float, float]:
        return offset_x + (lateral - min_l) * scale, size - offset_y - (depth - min_d) * scale

    draw.polygon([to_px(*c) for c in corners], outline=(90, 90, 90), width=3)

    framing = blocking["camera"]["framing"]
    fov = _FOV.get(framing, _FOV["medium"])
    cam = to_px(0.0, 0.0)
    reach = max([d for (_, d), _ in people if d > 0] or [0.5]) * 1.25 * scale
    box = [cam[0] - reach, cam[1] - reach, cam[0] + reach, cam[1] + reach]
    draw.pieslice(box, 270 - fov / 2, 270 + fov / 2, fill=(255, 236, 190), outline=(200, 140, 30), width=2)

    body = size * 0.022
    draw.rectangle([cam[0] - body * 1.4, cam[1] - body, cam[0] + body * 1.4, cam[1] + body], fill=(30, 30, 30))
    draw.polygon([(cam[0] - body * 0.8, cam[1] - body), (cam[0] + body * 0.8, cam[1] - body),
                  (cam[0], cam[1] - body * 2)], fill=(30, 30, 30))
    draw.text((cam[0], cam[1] + body * 2.2), f"CAMERA ({framing})", fill=(30, 30, 30), font=font, anchor="mt")
    draw.text((cam[0] - body * 4, cam[1]), "L", fill=(30, 30, 30), font=font, anchor="rm")
    draw.text((cam[0] + body * 4, cam[1]), "R", fill=(30, 30, 30), font=font, anchor="lm")

    radius = size * 0.035
    for (lateral, depth), subject in people:
        cx, cy = to_px(lateral, depth)
        draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=(70, 120, 200),
                     outline=(20, 40, 90), width=3)
        if subject["facing_deg"] is not None:
            relative = math.radians(subject["facing_deg"] - blocking["camera"]["facing_deg"])
            # Relative 0 looks away from the camera, i.e. up the page.
            ux, uy = math.sin(relative), -math.cos(relative)
            tip = (cx + ux * radius * 2.4, cy + uy * radius * 2.4)
            draw.line([(cx + ux * radius, cy + uy * radius), tip], fill=(200, 40, 40), width=4)
            side = (-uy, ux)
            draw.polygon([tip, (tip[0] - ux * 12 + side[0] * 7, tip[1] - uy * 12 + side[1] * 7),
                          (tip[0] - ux * 12 - side[0] * 7, tip[1] - uy * 12 - side[1] * 7)], fill=(200, 40, 40))
        draw.text((cx, cy + radius + 4), subject["name"], fill=(20, 20, 20), font=font, anchor="mt")

    image.save(path, format="PNG")
    return path
