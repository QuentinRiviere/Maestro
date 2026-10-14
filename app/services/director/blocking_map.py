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
# Share of the half field of view a subject may use, so its disc stays in the wedge.
_FIT = 0.8
# Nearest a subject may stand in front of the lens, in room widths.
_MIN_DEPTH = 0.06
# Farthest a close-up camera stands from what it frames.
_CLOSE_UP_REACH = 0.3
# Where a camera with no usable position stands from its subjects.
_DEFAULT_DISTANCE = {"wide": 0.6, "medium": 0.35, "close-up": 0.18}
# A camera may stand outside the room (a tight shot of people far apart is taken
# from far back); farther still is noise.
_CAMERA_RANGE = (-10.0, 11.0)


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


def _camera_coordinate(value: Any) -> Optional[float]:
    """A camera may stand outside the room (behind a wall it shoots through)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and _CAMERA_RANGE[0] <= number <= _CAMERA_RANGE[1] else None


def parse_blocking_map(raw: Any, fallback_framing: Any = "") -> Optional[dict]:
    """A clean blocking map, or None when the camera or every subject is unusable.

    The camera is then aimed at the people it frames (see ``aim_camera``):
    planners often leave the example facing in place, which would draw the
    subject of a close-up outside the field of view.
    """
    if not isinstance(raw, dict):
        return None
    camera = raw.get("camera")
    if not isinstance(camera, dict):
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
    blocking = {
        "camera": {"x": _camera_coordinate(camera.get("x")), "y": _camera_coordinate(camera.get("y")),
                   "facing_deg": _angle(camera.get("facing_deg")),
                   "framing": normalize_framing(camera.get("framing") or fallback_framing)},
        "subjects": subjects[:_MAX_SUBJECTS],
    }
    return aim_camera(blocking)


def subjects_in_view(blocking: dict, fit: float = 1.0) -> bool:
    """Whether every subject stands in front of the camera, inside its field of view."""
    half = math.radians(_FOV.get(blocking["camera"]["framing"], _FOV["medium"]) / 2 * fit)
    for subject in blocking["subjects"]:
        lateral, depth = _to_camera(blocking, subject["x"], subject["y"])
        if depth < _MIN_DEPTH - 1e-9 or abs(lateral) > depth * math.tan(half) + 1e-9:
            return False
    return True


def aim_camera(blocking: dict) -> dict:
    """Point the camera at the subjects it frames and keep them inside its field of view.

    A camera whose facing already frames everyone (and, for a close-up, stands
    close) is kept. Otherwise it turns toward the subjects' centroid from where
    it stands, then slides along that line: back until everyone fits the
    wedge, or in for a close-up planned too far away. A missing position puts
    the camera in front of the subjects, toward the front of the room.
    """
    camera = blocking["camera"]
    subjects = blocking["subjects"]
    framing = camera["framing"]
    cx = sum(s["x"] for s in subjects) / len(subjects)
    cy = sum(s["y"] for s in subjects) / len(subjects)
    placed = camera["x"] is not None and camera["y"] is not None
    distance = math.hypot(camera["x"] - cx, camera["y"] - cy) if placed else 0.0
    # Unit vector from the subjects to the camera; the camera looks back along it.
    if placed and distance > 1e-6:
        ux, uy = (camera["x"] - cx) / distance, (camera["y"] - cy) / distance
    else:
        ux, uy = 0.0, 1.0
        distance = _DEFAULT_DISTANCE.get(framing, _DEFAULT_DISTANCE["medium"])
    tan_half = math.tan(math.radians(_FOV.get(framing, _FOV["medium"]) / 2 * _FIT))
    needed = 0.0
    for subject in subjects:
        ox, oy = subject["x"] - cx, subject["y"] - cy
        toward = ox * ux + oy * uy  # how far the subject stands toward the camera
        across = abs(ox * -uy + oy * ux)
        needed = max(needed, toward + _MIN_DEPTH, toward + across / tan_half)
    reach = max(_CLOSE_UP_REACH, needed) + 1e-3
    if (placed and camera["facing_deg"] is not None and (framing != "close-up" or distance <= reach)
            and subjects_in_view(blocking, _FIT * 1.01)):
        return blocking

    if framing == "close-up":
        distance = min(distance, _CLOSE_UP_REACH)
    distance = max(distance, needed)
    camera = {**camera, "x": round(cx + ux * distance, 4), "y": round(cy + uy * distance, 4),
              # Facing 0 looks toward -y and 90 toward +x: the direction (-ux, -uy).
              "facing_deg": round(math.degrees(math.atan2(-ux, uy)) % 360.0, 2)}
    aimed = {**blocking, "camera": camera}
    if not subjects_in_view(aimed, _FIT * 1.01):
        # Rounding pushed an edge subject out: step back a little more.
        scale = 1.05
        camera.update(x=round(cx + ux * distance * scale, 4), y=round(cy + uy * distance * scale, 4))
    return aimed


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


def _angle_from_camera(blocking: dict, facing: float) -> float:
    """0 when a person faces the camera, 180 when they turn their back to it."""
    relative = (facing - blocking["camera"]["facing_deg"]) % 360.0
    return abs(((relative - 180.0) + 180.0) % 360.0 - 180.0)


def subject_view(blocking: Optional[dict], name: str) -> str:
    """The bible angle that shows ``name`` as this shot sees them ("" for the look's own image).

    Editors copy a reference's framing and pose, so a wide shot of a person
    facing the camera keeps the full-figure look image, closer shots take
    the face, and a person turned three quarters, in profile or away gets
    that angle instead of a front-facing pose.
    """
    if not blocking:
        return ""
    wanted = str(name or "").strip().casefold()
    subject = next((s for s in blocking["subjects"] if str(s.get("name") or "").strip().casefold() == wanted), None)
    if subject is None or subject.get("facing_deg") is None:
        return ""
    angle = _angle_from_camera(blocking, subject["facing_deg"])
    if angle < 30:
        # A full-figure look image makes a full-figure shot: closer shots take the face.
        return "" if blocking["camera"]["framing"] == "wide" else "face_front"
    if angle < 65:
        return "face_three_quarter"
    if angle < 125:
        return "face_profile"
    return "back"


def gaze_targets(blocking: Optional[dict]) -> dict[str, str]:
    """Who looks at whom: the person each subject faces, within 45 degrees."""
    if not blocking:
        return {}
    targets: dict[str, str] = {}
    for subject in blocking["subjects"]:
        if subject.get("facing_deg") is None:
            continue
        theta = math.radians(subject["facing_deg"])
        fx, fy = math.sin(theta), -math.cos(theta)
        best, best_angle = None, 45.0
        for other in blocking["subjects"]:
            if other is subject:
                continue
            dx, dy = other["x"] - subject["x"], other["y"] - subject["y"]
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            angle = math.degrees(math.acos(max(-1.0, min(1.0, (fx * dx + fy * dy) / length))))
            if angle < best_angle:
                best, best_angle = other, angle
        if best is not None:
            targets[subject["name"]] = best["name"]
    return targets


def staging_line(blocking: Optional[dict]) -> str:
    """The layout in words plus who is in frame, once each, and where they look.

    Image editors given a portrait add a second copy of the person or turn
    them to the lens; naming the count and the eyelines prevents both.
    """
    line = framing_line(blocking)
    if not line:
        return ""
    placed = [s for s in blocking["subjects"] if _to_camera(blocking, s["x"], s["y"])[1] > 0.02]
    if placed:
        count = len(placed)
        line += (f" Exactly {count} {'person' if count == 1 else 'people'} in frame, each shown once"
                 " and never duplicated.")
    names = {s["name"] for s in placed}
    looks = [f"{who} looks at {whom}" for who, whom in gaze_targets(blocking).items() if who in names]
    if looks:
        line += " " + "; ".join(looks) + "."
    return line + " Nobody looks into the camera."


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
