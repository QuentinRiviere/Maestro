"""Per-shot blocking: where the camera stands and where each person is.

The planner writes a small top-view floor plan per shot (normalized 0..1:
x from the left wall to the right wall as the scene's main camera sees it,
y from the back wall to the front). The storyboard renders it as a flat
diagram, so the image editor takes the framing from geometry and only the
surfaces and light from the scene's photos. Geometry apart from texture.
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional

FRAMINGS = ("wide", "medium", "close-up")
# Horizontal field of view drawn for each framing, in degrees.
_FOV = {"wide": 70.0, "medium": 42.0, "close-up": 22.0}
_MAX_SUBJECTS = 6
_MAX_PROPS = 8
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
    subjects = _points(raw.get("subjects"), facing=True, flags=True)
    if not subjects:
        return None
    blocking = {
        "camera": {"x": _camera_coordinate(camera.get("x")), "y": _camera_coordinate(camera.get("y")),
                   "facing_deg": _angle(camera.get("facing_deg")),
                   "framing": normalize_framing(camera.get("framing") or fallback_framing)},
        "subjects": subjects[:_MAX_SUBJECTS],
    }
    if _flag(camera.get("crosses_line")):
        blocking["camera"]["crosses_line"] = True
    # Set pieces, and people of the scene who are out of this shot (for eyelines).
    for key, facing in (("props", False), ("offscreen", True)):
        points = _points(raw.get(key), facing=facing)
        if points:
            blocking[key] = points[:_MAX_PROPS]
    return aim_camera(blocking)


def _flag(value: Any) -> bool:
    return value is True or str(value or "").strip().lower() in ("true", "yes", "1")


def _points(raw: Any, *, facing: bool, flags: bool = False) -> list[dict]:
    """Named floor points (people or props); unnamed or off-floor ones are dropped."""
    points = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()[:40]
        px, py = _coordinate(item.get("x")), _coordinate(item.get("y"))
        if not name or px is None or py is None:
            continue
        point = {"name": name, "x": px, "y": py}
        if facing:
            point["facing_deg"] = _angle(item.get("facing_deg"))
        if flags and (_flag(item.get("moves")) or _flag(item.get("moved"))):
            # The action moved this person since the scene's previous shot.
            point["moves"] = True
        points.append(point)
    return points


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


def framing_line(blocking: Optional[dict], foreground: str = "", faces: bool = False) -> str:
    """The diagram in words, for editors that do not read images.

    ``foreground`` names the person whose back fills an over-the-shoulder
    shot (see ``ots_foreground``): they are described as a cropped,
    faceless shoulder rather than listed with the others. ``faces`` is set
    when the shot's framing asks to see the faces (see ``faces_wanted``):
    a person the blocking turns away is then turned three quarters instead.
    """
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
    # Perspective: a point's place across the frame is its lateral offset over its depth.
    placed.sort(key=lambda item: item[0] / item[1])
    back = _name_key(foreground)
    people = []
    for _lateral, _depth, subject in placed:
        if back and _name_key(subject["name"]) == back:
            continue
        facing = _facing_words(blocking, subject["facing_deg"])
        if faces and facing == "back to camera":
            facing = "turned three quarters away, face visible"
        people.append(f"{subject['name']} ({facing})" if facing else subject["name"])
    order = ""
    if len(people) > 1:
        order = " In frame, left to right: " + ", ".join(people) + "."
    elif people:
        order = f" In frame: {people[0]}."
    if back and len(people) < len(placed):
        name = next(s["name"] for _l, _d, s in placed if _name_key(s["name"]) == back)
        order += " " + ots_foreground_line(name, _foreground_side(blocking, name))
    return f"Framing: {framing} shot.{order}"


def _angle_from_camera(blocking: dict, facing: float) -> float:
    """0 when a person faces the camera, 180 when they turn their back to it."""
    relative = (facing - blocking["camera"]["facing_deg"]) % 360.0
    return abs(((relative - 180.0) + 180.0) % 360.0 - 180.0)


def subject_view(blocking: Optional[dict], name: str, faces: bool = False) -> str:
    """The bible angle that shows ``name`` as this shot sees them ("" for the look's own image).

    Editors copy a reference's framing and pose, so a wide shot of a person
    facing the camera keeps the full-figure look image, closer shots take
    the face, and a person turned three quarters, in profile or away gets
    that angle instead of a front-facing pose. With ``faces`` (the framing
    asks to see the faces) a person turned away gets three quarters, not the back.
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
    return "face_three_quarter" if faces else "back"


_OTS_RE = re.compile(r"over[\s-]*(?:the[\s-]*)?shoulders?|\bo\.?t\.?s\b", re.IGNORECASE)
# Framing words that ask to see the faces, and words that ask for a back view.
_FACES_RE = re.compile(r"\bfaces?\b|\beyes\b|\breactions?\b|\bexpressions?\b", re.IGNORECASE)
_BACK_VIEW_RE = re.compile(r"\bback\b|\bbehind\b|silhouett|walk(?:s|ing)? away", re.IGNORECASE)
# A person turned at least this far from the camera shows it their back.
_BACK_ANGLE = 135.0


def is_over_the_shoulder(framing: Any) -> bool:
    """Whether a framing text names an over-the-shoulder shot ("OTS", "over-the-shoulder")."""
    return bool(_OTS_RE.search(str(framing or "")))


def ots_foreground(blocking: Optional[dict], framing: Any = "") -> str:
    """The person whose back fills the foreground of an over-the-shoulder shot ("" when none).

    An OTS framing names one; a medium or close shot whose nearest person
    has their back to the camera (and someone else in frame) is one too.
    That person must not get a front portrait as reference: the editor
    redraws the face and the shot shows them twice.
    """
    if not blocking:
        return ""
    placed = []
    for subject in blocking["subjects"]:
        _lateral, depth = _to_camera(blocking, subject["x"], subject["y"])
        if depth > 0.02:
            placed.append((depth, subject))
    if len(placed) < 2 or (faces_wanted(framing) and not is_over_the_shoulder(framing)):
        return ""
    placed.sort(key=lambda item: item[0])
    nearest = placed[0][1]
    facing = nearest.get("facing_deg")
    back = facing is not None and _angle_from_camera(blocking, facing) >= _BACK_ANGLE
    if is_over_the_shoulder(framing) or (back and blocking["camera"]["framing"] != "wide"):
        return nearest["name"]
    return ""


def faces_wanted(framing: Any) -> bool:
    """Whether a framing text asks to see the faces ("both faces", "reaction") and not a back view."""
    text = str(framing or "")
    return bool(_FACES_RE.search(text)) and not _BACK_VIEW_RE.search(text) and not is_over_the_shoulder(text)


def face_the_camera(blocking: Optional[dict], framing: Any = "") -> Optional[dict]:
    """A close or face shot of one person, with the camera moved in front of them when it stood behind.

    Planners place the camera behind a lone person for a close-up or a
    reaction: the image then shows the back of a head while the shot's text
    describes the face. A framing that asks for a back view (or an OTS) keeps
    its camera. The camera keeps its distance and turns to face the person.
    """
    if not blocking:
        return blocking
    text = str(framing or "")
    if is_over_the_shoulder(text) or _BACK_VIEW_RE.search(text):
        return blocking
    if not (faces_wanted(text) or "close" in text.lower()):
        return blocking
    placed = [s for s in blocking["subjects"] if _to_camera(blocking, s["x"], s["y"])[1] > 0.02]
    if len(placed) != 1 or placed[0].get("facing_deg") is None:
        return blocking
    person = placed[0]
    if _angle_from_camera(blocking, person["facing_deg"]) < _BACK_ANGLE:
        return blocking
    camera = blocking["camera"]
    distance = math.hypot(camera["x"] - person["x"], camera["y"] - person["y"])
    theta = math.radians(person["facing_deg"])
    # Facing 0 looks toward -y and 90 toward +x.
    moved = {**camera, "x": round(person["x"] + math.sin(theta) * distance, 4),
             "y": round(person["y"] - math.cos(theta) * distance, 4),
             "facing_deg": round((person["facing_deg"] + 180.0) % 360.0, 2)}
    return {**blocking, "camera": moved}


def ots_foreground_line(name: str, side: str) -> str:
    """The OTS foreground person in words: their back, cropped, no face."""
    side = side if side in ("left", "right") else "left"
    return (f"Foreground: the out-of-focus back of {name}'s head and shoulder in the {side} third, "
            "cropped by the frame edge, face not visible.")


def _foreground_side(blocking: dict, name: str) -> str:
    sides = screen_sides(blocking)
    side = sides.get(name, "left")
    if side == "center":
        others = [s for n, s in sides.items() if n != name and s in ("left", "right")]
        side = "right" if others and others[0] == "left" else "left"
    return side


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
        # People of the scene out of this shot still draw a look (a reverse's eyeline).
        for other in [*blocking["subjects"], *blocking.get("offscreen", [])]:
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


def insert_line(framing: str, names: list[str]) -> str:
    """An insert shows hands and objects only: a face in it turns it into a portrait."""
    whose = f" of {' and '.join(names)}" if names else ""
    return (f"Framing: {framing} shot. Only the hands{whose} and the objects they touch fill the frame, "
            "seen from close; no face, no head and no shoulders in frame.")


def staging_line(blocking: Optional[dict], framing: Any = "") -> str:
    """The layout in words plus who is in frame, once each, and where they look.

    Image editors given a portrait add a second copy of the person or turn
    them to the lens; naming the count and the eyelines prevents both.
    ``framing`` is the shot's framing text: an over-the-shoulder shot
    describes its foreground person by their back only (``ots_foreground``).
    """
    foreground = ots_foreground(blocking, framing)
    line = framing_line(blocking, foreground, faces=faces_wanted(framing))
    if not line:
        return ""
    placed = [s for s in blocking["subjects"] if _to_camera(blocking, s["x"], s["y"])[1] > 0.02]
    if placed:
        count = len(placed)
        line += (f" Exactly {count} {'person' if count == 1 else 'people'} in frame, each shown once"
                 " and never duplicated.")
        if foreground:
            line += f" {foreground} is seen only from behind."
    names = {s["name"] for s in placed}
    people = {p["name"]: p for p in [*blocking["subjects"], *blocking.get("offscreen", [])]}
    looks = []
    for who, whom in gaze_targets(blocking).items():
        if who not in names or who == foreground:
            continue
        if whom in names:
            looks.append(f"{who} looks at {whom}")
            continue
        # The person looked at is out of frame: the eyeline gives the screen direction.
        side = screen_side(blocking, people[whom]["x"], people[whom]["y"])
        toward = "just past the camera" if side == "center" else f"toward frame {side}"
        looks.append(f"{who} looks {toward} at {whom}, who is out of frame")
    if looks:
        line += " " + "; ".join(looks) + "."
    set_line = props_line(blocking)
    if set_line:
        line += " " + set_line
    return line + " Nobody looks into the camera."


def screen_side(blocking: dict, x: float, y: float) -> str:
    """The side of the frame a floor point is on, seen from this shot's camera: left, center or right."""
    lateral, _depth = _to_camera(blocking, x, y)
    if abs(lateral) < 0.02:
        return "center"
    return "right" if lateral > 0 else "left"


def screen_sides(blocking: Optional[dict]) -> dict[str, str]:
    """Each person of the scene (in this shot or out of it) and the frame side they are on."""
    if not blocking:
        return {}
    return {p["name"]: screen_side(blocking, p["x"], p["y"])
            for p in [*blocking["subjects"], *blocking.get("offscreen", [])]}


def props_line(blocking: Optional[dict]) -> str:
    """Where the scene's set pieces stand in this frame ("" when none is in view)."""
    if not blocking or not blocking.get("props"):
        return ""
    half = math.tan(math.radians(_FOV.get(blocking["camera"]["framing"], _FOV["medium"]) / 2))
    placed = []
    for prop in blocking["props"]:
        lateral, depth = _to_camera(blocking, prop["x"], prop["y"])
        if depth <= 0.02:
            continue
        ratio = lateral / (depth * half)
        if abs(ratio) > 1.0:
            continue
        where = "frame left" if ratio < -0.33 else "frame right" if ratio > 0.33 else "frame center"
        placed.append(f"the {prop['name']} {where}")
    return f"Set: {', '.join(placed)}." if placed else ""


# ── Scene lock: one floor plan per scene ───────────────────────────────────
#
# The planner writes a blocking map per shot, and each one drifts a little
# from the last: two people face to face become side by side, the reverse
# angle swaps who is on the left. A scene keeps ONE floor plan instead
# (where each person stands and faces, the set pieces, the master camera and
# its side of the action line); later shots only move the camera.

# A person this far (room widths) from their place in the scene layout contradicts it
# and is put back (planner drift).
_POSITION_DRIFT = 0.12
# A person farther than this from their place has moved: the layout follows the shot.
_MOVE_DRIFT = 0.2
_FOLLOWED_MOVE = "as the shot places them"
# Notes ending like this follow the shot; any other note is a correction.
_FOLLOWED = ("as planned", _FOLLOWED_MOVE)
# A person turned this far (degrees) from their facing in the layout contradicts it.
_FACING_DRIFT = 60.0
# A camera this close to the action line stands on it and is moved off it, to the scene's side.
_ON_LINE = 0.04
_LINE_OFFSET = 0.06
# A camera turned this far (degrees) from the master's facing looks at the opposite wall.
_REVERSE_ANGLE = 120.0


def _name_key(name: Any) -> str:
    return str(name or "").strip().casefold()


def _angle_between(a: float, b: float) -> float:
    difference = abs(a - b) % 360.0
    return min(difference, 360.0 - difference)


def _find(points: list[dict], name: Any) -> Optional[dict]:
    key = _name_key(name)
    return next((p for p in points if _name_key(p["name"]) == key), None)


def scene_floor_layout(first: Optional[dict], scene_map: Any = None, main: Optional[list[str]] = None
                       ) -> Optional[dict]:
    """The scene's floor plan, from a structured scene map or else the scene's first blocking map.

    Returns ``{"people": [{name, x, y, facing_deg}], "props": [{name, x, y}],
    "camera": {x, y, facing_deg}, "line": [name, name], "side": -1 | 0 | 1}``:
    ``camera`` is the master's (the view the place's main image shows),
    ``line`` the two main characters the 180-degree line runs through and
    ``side`` the side of that line the camera keeps (0 until a shot sets it).
    ``main`` names the main characters, most important first.
    """
    people: list[dict] = []
    props: list[dict] = []
    camera = None
    if isinstance(scene_map, dict):
        people = _points(scene_map.get("subjects") or scene_map.get("people"), facing=True)[:_MAX_SUBJECTS]
        props = _points(scene_map.get("props"), facing=False)[:_MAX_PROPS]
        raw_camera = scene_map.get("camera") if isinstance(scene_map.get("camera"), dict) else {}
        cx, cy = _camera_coordinate(raw_camera.get("x")), _camera_coordinate(raw_camera.get("y"))
        facing = _angle(raw_camera.get("facing_deg"))
        if cx is not None and cy is not None and facing is not None:
            camera = {"x": cx, "y": cy, "facing_deg": facing}
    if first:
        for subject in first["subjects"]:
            if _find(people, subject["name"]) is None:
                people.append({k: subject.get(k) for k in ("name", "x", "y", "facing_deg")})
        for prop in first.get("props", []):
            if _find(props, prop["name"]) is None:
                props.append(dict(prop))
        if camera is None:
            camera = {k: first["camera"][k] for k in ("x", "y", "facing_deg")}
    if not people or camera is None:
        return None
    layout = {"people": people, "props": props, "camera": camera, "line": [], "side": 0}
    _set_line(layout, main)
    return layout


def _set_line(layout: dict, main: Optional[list[str]] = None) -> None:
    """The action line runs through the two main characters of the layout; the master camera picks the side."""
    names = [p["name"] for p in (_find(layout["people"], n) for n in main or []) if p]
    names += [p["name"] for p in layout["people"] if p["name"] not in names]
    if len(names) < 2 or layout["line"]:
        return
    layout["line"] = names[:2]
    distance = _line_distance(layout, layout["camera"]["x"], layout["camera"]["y"])
    layout["side"] = 0 if distance is None or abs(distance) < _ON_LINE else (1 if distance > 0 else -1)


def _line_distance(layout: dict, x: float, y: float) -> Optional[float]:
    """Signed distance of a floor point from the action line (positive on its left, walking from A to B)."""
    if len(layout["line"]) < 2:
        return None
    a, b = (_find(layout["people"], name) for name in layout["line"])
    if a is None or b is None:
        return None
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return None
    return (dx * (y - a["y"]) - dy * (x - a["x"])) / length


def _line_normal(layout: dict) -> tuple[float, float]:
    a, b = (_find(layout["people"], name) for name in layout["line"])
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    length = math.hypot(dx, dy)
    # Rotating A->B by -90 degrees points to the positive side of _line_distance.
    return -dy / length, dx / length


def _reflect_facing(layout: dict, facing: float) -> float:
    """A facing mirrored across the action line."""
    a, b = (_find(layout["people"], name) for name in layout["line"])
    dx, dy = b["x"] - a["x"], b["y"] - a["y"]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    theta = math.radians(facing)
    fx, fy = math.sin(theta), -math.cos(theta)
    dot = fx * ux + fy * uy
    rx, ry = 2 * dot * ux - fx, 2 * dot * uy - fy
    return round(math.degrees(math.atan2(rx, -ry)) % 360.0, 2)


def lock_shot_blocking(layout: dict, blocking: dict) -> tuple[dict, list[str]]:
    """This shot's blocking on the scene's floor plan; returns it and what was changed, in words.

    People keep their place and facing from the layout (only the camera
    moves) unless the shot marks them ``moves`` or puts them farther than
    ``_MOVE_DRIFT`` from their place (the action moved them; the layout
    follows this shot). Small drift is put back. A person the layout does not know yet joins it where
    the shot puts them. The camera keeps the scene's side of the action line
    unless it is marked ``crosses_line`` (the scene then stays on the new
    side). ``layout`` is updated in place.
    """
    notes: list[str] = []
    moved = False
    subjects = [dict(s) for s in blocking["subjects"]]
    before = (sum(s["x"] for s in subjects) / len(subjects), sum(s["y"] for s in subjects) / len(subjects))
    for subject in subjects:
        known = _find(layout["people"], subject["name"])
        if known is None:
            layout["people"].append({k: subject.get(k) for k in ("name", "x", "y", "facing_deg")})
            continue
        if subject.get("moves"):
            known.update(x=subject["x"], y=subject["y"])
            if subject.get("facing_deg") is not None:
                known["facing_deg"] = subject["facing_deg"]
            notes.append(f"{subject['name']} moves to a new place, as planned")
            moved = True
            continue
        drift = math.hypot(subject["x"] - known["x"], subject["y"] - known["y"])
        if drift > _MOVE_DRIFT:
            # Too far to be drift: the shot moved them. Forcing the old place
            # would contradict the shot's prose (duplicates, teleports).
            known.update(x=subject["x"], y=subject["y"])
            if subject.get("facing_deg") is not None:
                known["facing_deg"] = subject["facing_deg"]
            notes.append(f"{subject['name']} moves to a new place, {_FOLLOWED_MOVE}")
            moved = True
            continue
        if drift > _POSITION_DRIFT:
            notes.append(f"{subject['name']} kept at the scene position ({known['x']:.2f}, {known['y']:.2f}) "
                         f"instead of ({subject['x']:.2f}, {subject['y']:.2f})")
        subject.update(x=known["x"], y=known["y"])
        if known.get("facing_deg") is None:
            known["facing_deg"] = subject.get("facing_deg")
        elif subject.get("facing_deg") is None:
            subject["facing_deg"] = known["facing_deg"]
        elif _angle_between(subject["facing_deg"], known["facing_deg"]) > _FACING_DRIFT:
            notes.append(f"{subject['name']} kept facing {known['facing_deg']:.0f} degrees "
                         f"instead of {subject['facing_deg']:.0f}")
            subject["facing_deg"] = known["facing_deg"]
    _set_line(layout)

    # The camera keeps where it stood relative to the people it frames.
    after = (sum(s["x"] for s in subjects) / len(subjects), sum(s["y"] for s in subjects) / len(subjects))
    camera = dict(blocking["camera"])
    camera.update(x=round(camera["x"] + after[0] - before[0], 4), y=round(camera["y"] + after[1] - before[1], 4))

    distance = _line_distance(layout, camera["x"], camera["y"])
    if distance is not None:
        side = 0 if abs(distance) < _ON_LINE else (1 if distance > 0 else -1)
        target = distance
        if layout["side"] == 0 or (moved and side != 0):
            # A planned move draws a new line: this shot's camera sets its side.
            layout["side"] = side
        elif side == -layout["side"] and camera.get("crosses_line"):
            layout["side"] = side
            notes.append("the camera crosses the action line, as planned")
        elif side == -layout["side"]:
            target = -distance
            camera["facing_deg"] = _reflect_facing(layout, camera["facing_deg"])
            notes.append("camera moved back to the scene's side of the action line (180-degree rule)")
        elif side == 0:
            target = layout["side"] * _LINE_OFFSET
        if target != distance:
            nx, ny = _line_normal(layout)
            camera.update(x=round(camera["x"] + nx * (target - distance), 4),
                          y=round(camera["y"] + ny * (target - distance), 4))

    for prop in blocking.get("props", []):
        if _find(layout["props"], prop["name"]) is None:
            layout["props"].append(dict(prop))
    in_shot = {_name_key(s["name"]) for s in subjects}
    locked = {**blocking, "camera": camera, "subjects": subjects}
    offscreen = [dict(p) for p in layout["people"] if _name_key(p["name"]) not in in_shot]
    locked.pop("offscreen", None)
    if offscreen:
        locked["offscreen"] = offscreen
    if layout["props"]:
        locked["props"] = [dict(p) for p in layout["props"]]
    return aim_camera(locked), notes


def camera_faces_reverse(layout: Optional[dict], blocking: Optional[dict]) -> bool:
    """Whether this shot's camera looks at the wall opposite the one the master (and the place's image) sees."""
    if not layout or not blocking or blocking["camera"].get("facing_deg") is None:
        return False
    return _angle_between(blocking["camera"]["facing_deg"], layout["camera"]["facing_deg"]) > _REVERSE_ANGLE


def main_characters(blockings: list[Optional[dict]]) -> list[str]:
    """The scene's people, most often in shot first (first appearance breaks ties)."""
    counts: dict[str, list] = {}
    for index, blocking in enumerate(blockings):
        for subject in (blocking or {}).get("subjects", []):
            entry = counts.setdefault(_name_key(subject["name"]), [0, index, subject["name"]])
            entry[0] += 1
    return [name for _count, _first, name in sorted(counts.values(), key=lambda e: (-e[0], e[1]))]


def lock_scene_blockings(raw_maps: list[Any], *, scene_map: Any = None) -> list[dict]:
    """Every shot of one scene on the scene's single floor plan.

    Returns one entry per shot, in order: ``{"blocking": the locked map or
    None, "reverse": the camera faces the opposite wall, "notes": [what the
    lock changed or followed], "corrected": the shot's own map contradicted
    the layout}``. The layout comes from ``scene_map`` when it is a
    structured map, else from the first shot with a usable blocking map.
    """
    blockings = [parse_blocking_map(raw) for raw in raw_maps]
    main = main_characters(blockings)
    layout = None
    results = []
    for blocking in blockings:
        if layout is None and blocking is not None:
            layout = scene_floor_layout(blocking, scene_map, main)
        if blocking is None or layout is None:
            results.append({"blocking": blocking, "reverse": False, "notes": [], "corrected": False})
            continue
        locked, notes = lock_shot_blocking(layout, blocking)
        results.append({"blocking": locked, "reverse": camera_faces_reverse(layout, locked), "notes": notes,
                        # Planned moves, far moves and crossings are followed, not corrected.
                        "corrected": any(not note.endswith(_FOLLOWED) for note in notes)})
    return results


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
