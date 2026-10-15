"""The 3D set of one shot: the room of a location layout, its furniture, the people and the camera.

World frame (meters, Y up): X runs from the left wall (0) to the right wall
(room width), Z from the back wall (0) to the front wall (room depth), as the
scene's master camera sees the room. A floor-plan point ``(x, y)`` of the
storyboard (normalized, see ``blocking_map``) is ``(x * width, y * depth)``
on the floor. A floor-plan facing of 0 degrees looks at the back wall (-Z),
90 at the right wall (+X); facings are stretched with the room, so a person
facing another on the plan faces them in 3D too.

The camera stands where the scene-locked blocking puts it (the storyboard
already keeps it on its side of the action line) and looks along its facing;
its height and tilt come from the framing, the angle words and the people in
frame. The horizontal field of view is the blocking diagram's per framing, so
everyone the diagram fits in the wedge is in the rendered frame.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np

from services.director.blocking_map import _FOV as FRAMING_FOV
from services.director.blocking_map import normalize_framing
from services.director.blockout.layout import DEFAULT_ROOM, merge_props, normalize_layout

# Semantic classes of the blockout and their label ids.
SET_CLASSES = ("void", "wall", "floor", "ceiling", "window", "door", "table", "counter", "shelf", "chair",
               "sofa", "bed", "cabinet", "plant", "lamp", "prop")
LABELS = {name: index for index, name in enumerate(SET_CLASSES)}
# People take the labels from CHARACTER_BASE on, one per character slot.
CHARACTER_BASE = 32
MAX_CHARACTERS = 6

# Mannequin heights (meters): head centers and the vertical spans of the body boxes.
STANDING_HEAD = 1.6
SEATED_HEAD = 1.22
HEAD_RADIUS = 0.11
# A camera closer than this to a person's axis stands inside them and steps back.
_BODY_CLEARANCE = 0.34
# Sentences that seat or stand people.
_SIT = re.compile(r"\b(sit|sits|sitting|seated|sat|perched|kneel\w*|lying|lies|lie)\b", re.IGNORECASE)
_STAND = re.compile(r"\b(stand|stands|standing|stood|walk\w*|enter\w*|leans? against)\b", re.IGNORECASE)


def character_label(slot: int) -> int:
    return CHARACTER_BASE + max(0, min(MAX_CHARACTERS - 1, int(slot)))


@dataclass(frozen=True)
class Box:
    """An oriented box: ``axes`` rows are its unit axes, ``half`` its half sizes along them."""
    center: tuple
    axes: tuple
    half: tuple
    label: int


@dataclass(frozen=True)
class Cylinder:
    """A vertical cylinder between heights ``y0`` and ``y1``."""
    x: float
    z: float
    radius: float
    y0: float
    y1: float
    label: int


@dataclass(frozen=True)
class Sphere:
    center: tuple
    radius: float
    label: int


@dataclass
class Person:
    name: str
    slot: int
    x: float
    z: float
    facing: tuple  # unit (fx, fz) on the floor
    posture: str  # "standing" | "seated"

    @property
    def head_y(self) -> float:
        return SEATED_HEAD if self.posture == "seated" else STANDING_HEAD


@dataclass
class Scene:
    room: dict
    openings: list
    primitives: list
    people: list
    layout: dict

    def person(self, name: str) -> Optional[Person]:
        key = str(name or "").strip().casefold()
        return next((p for p in self.people if p.name.strip().casefold() == key), None)


@dataclass(frozen=True)
class Camera:
    position: tuple
    target: tuple
    hfov_deg: float
    roll_deg: float = 0.0
    framing: str = "medium"

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(forward, right, up) unit vectors; right is frame right."""
        forward = np.subtract(self.target, self.position).astype(float)
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 1.0, 0.0])
        if abs(float(forward @ world_up)) > 0.999:
            world_up = np.array([0.0, 0.0, -1.0])
        right = np.cross(forward, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        if self.roll_deg:
            angle = math.radians(self.roll_deg)
            right, up = (right * math.cos(angle) + up * math.sin(angle),
                         up * math.cos(angle) - right * math.sin(angle))
        return forward, right, up

    def vfov_deg(self, width: int, height: int) -> float:
        return math.degrees(2 * math.atan(math.tan(math.radians(self.hfov_deg) / 2) * height / width))

    def focal_mm(self, width: int, height: int) -> float:
        """Full-frame equivalent focal length: the frame's long side is the sensor's 36 mm."""
        long_fov = self.hfov_deg if width >= height else self.vfov_deg(width, height)
        return 18.0 / math.tan(math.radians(long_fov) / 2)

    def to_dict(self, width: int = 0, height: int = 0) -> dict:
        data = {"position": [round(float(v), 3) for v in self.position],
                "target": [round(float(v), 3) for v in self.target],
                "hfov_deg": round(self.hfov_deg, 2), "roll_deg": round(self.roll_deg, 2), "framing": self.framing}
        if width and height:
            data["focal_mm"] = round(self.focal_mm(width, height), 1)
        return data


# ── primitives ─────────────────────────────────────────────────────────────


def _yaw_axes(fx: float, fz: float) -> tuple:
    """Axes of a box turned to face the floor direction (fx, fz): lateral, up, facing."""
    length = math.hypot(fx, fz) or 1.0
    fx, fz = fx / length, fz / length
    return ((-fz, 0.0, fx), (0.0, 1.0, 0.0), (fx, 0.0, fz))


def _box(center, axes, half, label) -> Box:
    return Box(tuple(float(v) for v in center), tuple(tuple(float(v) for v in a) for a in axes),
               tuple(float(v) for v in half), int(label))


def furniture_primitives(item: dict, room: dict) -> list:
    """One piece of layout furniture as a primitive."""
    label = LABELS.get(item["class"], LABELS["prop"])
    x, z = item["x"] * room["width"], item["y"] * room["depth"]
    if item.get("shape") == "cylinder":
        radius = max(item["width"], item["depth"]) / 2
        if item["class"] == "table":
            # A round table: its top, and one stem.
            top = min(0.05, item["height"] / 4)
            return [Cylinder(x, z, radius, item["height"] - top, item["height"], label),
                    Cylinder(x, z, min(0.06, radius / 3), 0.0, item["height"] - top, label)]
        return [Cylinder(x, z, radius, 0.0, item["height"], label)]
    angle = math.radians(item.get("rotation_deg") or 0.0)
    # rotation 0 keeps width along X and depth along Z.
    axes = ((math.cos(angle), 0.0, -math.sin(angle)), (0.0, 1.0, 0.0), (math.sin(angle), 0.0, math.cos(angle)))
    return [_box((x, item["height"] / 2, z), axes, (item["width"] / 2, item["height"] / 2, item["depth"] / 2),
                 label)]


def mannequin_primitives(person: Person) -> list:
    """A rough body: legs, torso with shoulders, head; a seated one gets a chair."""
    fx, fz = person.facing
    axes = _yaw_axes(fx, fz)
    label = character_label(person.slot)
    x, z = person.x, person.z

    def at(forward: float) -> tuple[float, float]:
        return x + fx * forward, z + fz * forward

    parts: list = []
    if person.posture == "seated":
        cx, cz = at(-0.08)
        parts.append(_box((cx, 0.22, cz), axes, (0.22, 0.22, 0.22), LABELS["chair"]))
        tx, tz = at(0.12)
        parts.append(_box((tx, 0.52, tz), axes, (0.17, 0.08, 0.2), label))  # thighs
        lx, lz = at(0.3)
        parts.append(_box((lx, 0.24, lz), axes, (0.15, 0.24, 0.07), label))  # lower legs
        bx, bz = at(-0.06)
        parts.append(_box((bx, 0.84, bz), axes, (0.2, 0.26, 0.12), label))  # torso
        parts.append(_box((bx, 1.04, bz), axes, (0.23, 0.06, 0.12), label))  # shoulders
        parts.append(Sphere((bx, SEATED_HEAD, bz), HEAD_RADIUS, label))
    else:
        parts.append(_box((x, 0.43, z), axes, (0.16, 0.43, 0.1), label))  # legs
        parts.append(_box((x, 1.15, z), axes, (0.2, 0.29, 0.12), label))  # torso
        parts.append(_box((x, 1.41, z), axes, (0.23, 0.06, 0.12), label))  # shoulders
        parts.append(Sphere((x, STANDING_HEAD, z), HEAD_RADIUS, label))
    return parts


# ── people from the blocking ───────────────────────────────────────────────


def _floor_direction(facing_deg: Optional[float], room: dict) -> tuple[float, float]:
    """A plan facing as a unit floor direction in meters (stretched with the room)."""
    theta = math.radians(facing_deg or 0.0)
    fx, fz = math.sin(theta) * room["width"], -math.cos(theta) * room["depth"]
    length = math.hypot(fx, fz) or 1.0
    return fx / length, fz / length


def infer_postures(names: list[str], text: str) -> dict[str, str]:
    """Seated or standing per person, from the shot's words.

    A sentence naming a person decides for them; otherwise the whole text
    does (seated when it only sits people), else standing.
    """
    text = str(text or "")
    sentences = re.split(r"(?<=[.!?;])\s+", text)
    overall = "seated" if _SIT.search(text) and not _STAND.search(text) else "standing"
    postures = {}
    for name in names:
        key = str(name or "").strip().casefold()
        first = key.split()[0] if key else ""
        mentions = [s for s in sentences if key and (key in s.casefold() or (first and first in s.casefold()))]
        sits = any(_SIT.search(s) for s in mentions)
        stands = any(_STAND.search(s) for s in mentions)
        postures[name] = "seated" if sits and not stands else "standing" if stands and not sits else overall
    return postures


def build_scene(layout: Any, blocking: dict, *, character_slots: Optional[dict] = None,
                postures: Optional[dict] = None) -> Scene:
    """The shot's 3D set: the layout's room and furniture (with the blocking's props), and its people.

    ``blocking`` is a parsed (ideally scene-locked) blocking map. People in
    frame and the scene's people out of frame (``offscreen``) are all placed:
    a reverse shot still has the partner's shoulder where the plan puts it.
    ``character_slots`` maps a name to its semantic slot (stable across the
    scene); unknown names take the next free slot.
    """
    clean = normalize_layout(layout) or normalize_layout({"room": DEFAULT_ROOM})
    clean = merge_props(clean, blocking.get("props") or [])
    room = clean["room"]
    primitives: list = []
    for item in clean["furniture"]:
        primitives.extend(furniture_primitives(item, room))
    slots = {str(k).strip().casefold(): int(v) for k, v in (character_slots or {}).items()}
    postures = {str(k).strip().casefold(): v for k, v in (postures or {}).items()}
    people: list[Person] = []
    for point in [*blocking.get("subjects", []), *blocking.get("offscreen", [])]:
        key = str(point.get("name") or "").strip().casefold()
        if not key or any(p.name.strip().casefold() == key for p in people):
            continue
        if key not in slots:
            used = set(slots.values())
            slots[key] = next((s for s in range(MAX_CHARACTERS) if s not in used), MAX_CHARACTERS - 1)
        person = Person(name=str(point["name"]), slot=slots[key], x=point["x"] * room["width"],
                        z=point["y"] * room["depth"], facing=_floor_direction(point.get("facing_deg"), room),
                        posture="seated" if postures.get(key) == "seated" else "standing")
        people.append(person)
        primitives.extend(mannequin_primitives(person))
    return Scene(room=room, openings=clean["openings"], primitives=primitives, people=people, layout=clean)


# ── the camera ─────────────────────────────────────────────────────────────


def _angle_kind(text: str) -> str:
    value = str(text or "").lower()
    if re.search(r"overhead|bird|top[- ]down|aerial|god'?s[- ]eye", value):
        return "overhead"
    if re.search(r"\bhigh\b", value):
        return "high"
    if re.search(r"\blow\b|worm", value):
        return "low"
    return "eye"


def shot_camera(scene: Scene, blocking: dict, *, angle: str = "", foreground: str = "") -> Camera:
    """The shot's camera: the blocking's place and facing, a height and tilt from the framing and angle.

    ``foreground`` names the over-the-shoulder person (see ``ots_foreground``):
    the camera frames the others and stands at that person's head height.
    """
    room = scene.room
    raw = blocking["camera"]
    framing = normalize_framing(raw.get("framing"))
    cx, cz = raw["x"] * room["width"], raw["y"] * room["depth"]
    fx, fz = _floor_direction(raw.get("facing_deg"), room)

    framed = [p for p in scene.people
              if any(str(s.get("name") or "").strip().casefold() == p.name.strip().casefold()
                     for s in blocking.get("subjects", []))]
    back = scene.person(foreground) if foreground else None
    subjects = [p for p in framed if p is not back] or framed
    ahead = [p for p in subjects if (p.x - cx) * fx + (p.z - cz) * fz > 0.05] or subjects
    if ahead:
        head = sum(p.head_y for p in ahead) / len(ahead)
        distance = max(0.4, sum((p.x - cx) * fx + (p.z - cz) * fz for p in ahead) / len(ahead))
    else:
        head, distance = STANDING_HEAD, 2.0

    # Where the lens points, in height: the face for a close-up, the chest up for a medium, the body for a wide.
    target_y = {"close-up": head, "medium": head - 0.15, "wide": head * 0.6}[framing]
    eye = {"close-up": head, "medium": head, "wide": max(1.45, head)}[framing]
    if back is not None:
        eye = max(eye, back.head_y + 0.02)
    kind = _angle_kind(angle)
    if kind == "high":
        eye += 0.9 if framing == "wide" else 0.6
    elif kind == "low":
        eye = max(0.35, eye - (0.9 if framing == "wide" else 0.6))
    elif kind == "overhead":
        eye = room["height"] - 0.15
    eye = min(eye, room["height"] - 0.1)
    roll = 12.0 if re.search(r"dutch|canted|tilted", str(angle or "").lower()) else 0.0

    # Step back out of anyone the camera stands in (a tight OTS put on the person's axis).
    for _ in range(20):
        inside = [p for p in scene.people
                  if math.hypot(p.x - cx, p.z - cz) < _BODY_CLEARANCE and eye < p.head_y + 0.25]
        if not inside:
            break
        cx, cz = cx - fx * 0.1, cz - fz * 0.1
        distance += 0.1

    if kind == "overhead":
        target = (cx + fx * max(distance, 0.5), 0.0 if framing == "wide" else target_y - 0.3,
                  cz + fz * max(distance, 0.5))
    else:
        target = (cx + fx * distance, target_y, cz + fz * distance)
    return Camera(position=(cx, eye, cz), target=target, hfov_deg=FRAMING_FOV.get(framing, FRAMING_FOV["medium"]),
                  roll_deg=roll, framing=framing)


# ── camera moves (driving sequences) ───────────────────────────────────────

INTENSITY = {"static": 0.0, "subtle": 0.5, "moderate": 1.0, "dynamic": 1.6}


def _ease(t: float) -> float:
    return t * t * (3 - 2 * t)


def _rotate_y(vector: np.ndarray, degrees: float) -> np.ndarray:
    """Rotate a vector about the vertical axis; positive turns toward frame right for a camera."""
    angle = math.radians(degrees)
    x, y, z = vector
    # Turning right from forward (0, 0, -1) goes toward +X.
    return np.array([x * math.cos(angle) - z * math.sin(angle), y, x * math.sin(angle) + z * math.cos(angle)])


def parse_movement(movement: Any) -> list[tuple[str, float]]:
    """The camera moves a movement text names: [(kind, sign)], e.g. "slow push-in, pan left"."""
    text = str(movement or "").lower()
    if not text or re.search(r"\b(static|locked[- ]off|locked|fixed|none|still)\b", text):
        return []
    moves: list[tuple[str, float]] = []
    patterns = (
        (r"push[- ]?in|dolly[- ]?in|track(ing)?[- ]?in|move[s]? in|creep[s]? in", ("dolly", 1.0)),
        (r"pull[- ]?(out|back)|dolly[- ]?(out|back)|track(ing)?[- ]?(out|back)|move[s]? back", ("dolly", -1.0)),
        (r"zoom[- ]?in", ("zoom", 1.0)),
        (r"zoom[- ]?out", ("zoom", -1.0)),
        (r"pan[s]?[- ]?(to the )?left", ("pan", -1.0)),
        (r"pan[s]?[- ]?(to the )?right", ("pan", 1.0)),
        (r"tilt[s]?[- ]?up", ("tilt", 1.0)),
        (r"tilt[s]?[- ]?down", ("tilt", -1.0)),
        (r"(truck|track|slide|crab)[s]?[- ]?(to the )?left", ("truck", -1.0)),
        (r"(truck|track|slide|crab)[s]?[- ]?(to the )?right", ("truck", 1.0)),
        (r"(arc|orbit)[s]?[- ]?(to the )?left", ("orbit", -1.0)),
        (r"(arc|orbit|circle)[s]?[- ]?(to the )?right|\barc\b|\borbit\b", ("orbit", 1.0)),
        (r"(crane|boom|pedestal|jib)[s]?[- ]?up|rise[s]?", ("crane", 1.0)),
        (r"(crane|boom|pedestal|jib)[s]?[- ]?down|descend", ("crane", -1.0)),
        (r"handheld|hand-held|shaky", ("handheld", 1.0)),
    )
    for pattern, move in patterns:
        if re.search(pattern, text) and move[0] not in {kind for kind, _ in moves}:
            moves.append(move)
    return moves


def camera_path(camera: Camera, movement: Any, *, intensity: Any = "moderate", frames: int = 16) -> list[Camera]:
    """The camera at ``frames`` evenly spaced times of the shot's move (eased), first frame = ``camera``.

    A static shot (or an unknown move) repeats the camera. Distances scale
    with the intensity words of the plan (subtle, moderate, dynamic).
    """
    frames = max(1, int(frames))
    moves = parse_movement(movement)
    scale = INTENSITY.get(str(intensity or "moderate").strip().lower(), 1.0)
    position = np.array(camera.position, float)
    target = np.array(camera.target, float)
    forward, right, _up = camera.basis()
    reach = float(np.linalg.norm(target - position))
    path = []
    for index in range(frames):
        t = _ease(index / (frames - 1)) if frames > 1 else 0.0
        amount = t * scale
        p, g, fov, roll = position.copy(), target.copy(), camera.hfov_deg, camera.roll_deg
        for kind, sign in moves:
            if kind == "dolly":
                step = forward * sign * amount * min(0.35 * reach, 1.2)
                p, g = p + step, g + step
            elif kind == "zoom":
                fov = fov * (1 - 0.35 * sign * amount)
            elif kind == "pan":
                g = p + _rotate_y(g - p, 18.0 * sign * amount)
            elif kind == "tilt":
                g = g + np.array([0.0, 1.0, 0.0]) * sign * amount * 0.25 * reach
            elif kind == "truck":
                flat = np.array([right[0], 0.0, right[2]])
                flat /= np.linalg.norm(flat) or 1.0
                step = flat * sign * amount * 0.6
                p, g = p + step, g + step
            elif kind == "orbit":
                p = g + _rotate_y(p - g, -22.0 * sign * amount)
            elif kind == "crane":
                p = p + np.array([0.0, 1.0, 0.0]) * sign * amount * 0.6
            elif kind == "handheld":
                wobble = 0.015 * scale
                p = p + right * wobble * math.sin(index * 1.7) + np.array([0, 1.0, 0]) * wobble * math.sin(index * 2.3)
                roll = roll + 0.8 * scale * math.sin(index * 1.1)
        path.append(replace(camera, position=tuple(float(v) for v in p), target=tuple(float(v) for v in g),
                            hfov_deg=float(min(110.0, max(5.0, fov))), roll_deg=float(roll)))
    return path
