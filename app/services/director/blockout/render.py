"""A CPU ray caster for blockout scenes: depth, semantic colours, normals and edges per frame.

Every pixel casts one ray (no antialiasing, so semantic colours stay exact)
against the room's six one-sided planes and each primitive; numpy does the
whole frame at once. A 512-pixel frame of a furnished room with two people
renders in well under a second on one CPU core.

The room's planes are visible from inside only: a camera standing outside
the room (a tight shot taken from behind a wall) sees through that wall, as
on a film set with a wall pulled out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from services.director.blockout.scene import (
    CHARACTER_BASE, LABELS, MAX_CHARACTERS, SET_CLASSES, Box, Camera, Cylinder, Scene, Sphere,
)

# Set classes follow the ADE20K palette (what segmentation control models were
# trained on); people get one colour per character slot, the first being ADE20K's person.
PALETTE = {
    "void": (0, 0, 0),
    "wall": (120, 120, 120),
    "floor": (80, 50, 50),
    "ceiling": (120, 120, 80),
    "window": (230, 230, 230),
    "door": (8, 255, 51),
    "table": (255, 6, 82),
    "counter": (235, 12, 255),
    "shelf": (255, 7, 71),
    "chair": (204, 70, 3),
    "sofa": (11, 102, 255),
    "bed": (204, 5, 255),
    "cabinet": (224, 5, 255),
    "plant": (204, 255, 4),
    "lamp": (224, 255, 8),
    "prop": (0, 255, 20),
}
CHARACTER_COLORS = ((150, 5, 61), (0, 190, 255), (255, 140, 0), (120, 255, 60), (170, 80, 255), (255, 90, 190))
_EPS = 1e-4


def palette_table() -> np.ndarray:
    """Label id -> RGB, for every label the renderer writes."""
    table = np.zeros((CHARACTER_BASE + MAX_CHARACTERS, 3), np.uint8)
    for name, color in PALETTE.items():
        table[LABELS[name]] = color
    for slot, color in enumerate(CHARACTER_COLORS[:MAX_CHARACTERS]):
        table[CHARACTER_BASE + slot] = color
    return table


def label_name(label: int, scene: Optional[Scene] = None) -> str:
    """A label id in words: a set class, or ``character_<slot>`` (the person's name when ``scene`` is given)."""
    if label >= CHARACTER_BASE:
        slot = label - CHARACTER_BASE
        person = next((p for p in (scene.people if scene else []) if p.slot == slot), None)
        return person.name if person else f"character_{slot + 1}"
    return SET_CLASSES[label] if 0 <= label < len(SET_CLASSES) else "void"


@dataclass
class Frame:
    """One rendered frame: per-pixel z-depth (meters, inf = nothing), label ids and world normals."""
    camera: Camera
    depth: np.ndarray
    labels: np.ndarray
    normals: np.ndarray

    @property
    def size(self) -> tuple[int, int]:
        return self.depth.shape[1], self.depth.shape[0]

    def depth_range(self) -> tuple[float, float]:
        finite = self.depth[np.isfinite(self.depth)]
        if finite.size == 0:
            return 0.1, 10.0
        return float(finite.min()), float(np.percentile(finite, 99.5))

    def depth_image(self, near: Optional[float] = None, far: Optional[float] = None) -> np.ndarray:
        """8-bit RGB depth, lighter is closer (the depth-control convention); nothing is black."""
        low, high = self.depth_range()
        near = low if near is None else near
        far = high if far is None else far
        span = max(far - near, 1e-3)
        value = np.where(np.isfinite(self.depth), 1.0 - np.clip((self.depth - near) / span, 0.0, 1.0), 0.0)
        gray = np.round(np.power(value, 0.8) * 255).astype(np.uint8)
        return np.repeat(gray[..., None], 3, axis=2)

    def semantic_image(self) -> np.ndarray:
        return palette_table()[self.labels]

    def normal_image(self) -> np.ndarray:
        """Camera-space normals as RGB: x frame right, y up, z toward the camera (0.5 grey = none)."""
        forward, right, up = self.camera.basis()
        cam = np.stack([self.normals @ right, self.normals @ up, -(self.normals @ forward)], axis=-1)
        cam[self.labels == LABELS["void"]] = 0.0
        return np.round((cam * 0.5 + 0.5) * 255).astype(np.uint8)

    def edge_image(self) -> np.ndarray:
        """White lines where the class, the depth or the surface direction changes."""
        labels, depth, normals = self.labels, np.where(np.isfinite(self.depth), self.depth, 1e3), self.normals
        edges = np.zeros(labels.shape, bool)
        for axis in (0, 1):
            a = [slice(None), slice(None)]
            b = [slice(None), slice(None)]
            a[axis], b[axis] = slice(1, None), slice(None, -1)
            a, b = tuple(a), tuple(b)
            jump = (labels[a] != labels[b])
            jump |= np.abs(depth[a] - depth[b]) > 0.04 * np.minimum(depth[a], depth[b])
            jump |= (normals[a] * normals[b]).sum(-1) < 0.8
            edges[a] |= jump
        return np.repeat((edges * 255).astype(np.uint8)[..., None], 3, axis=2)

    def class_coverage(self, scene: Optional[Scene] = None) -> dict[str, dict]:
        """Each visible class or person: share of the frame and its horizontal centre (0 left .. 1 right)."""
        total = self.labels.size
        width = self.labels.shape[1]
        columns = np.broadcast_to(np.arange(width)[None, :], self.labels.shape)
        visible = {}
        for label in np.unique(self.labels):
            if label == LABELS["void"]:
                continue
            mask = self.labels == label
            count = int(mask.sum())
            visible[label_name(int(label), scene)] = {
                "coverage": round(count / total, 4),
                "center_x": round(float(columns[mask].mean() + 0.5) / width, 4),
            }
        return visible


# ── ray casting ────────────────────────────────────────────────────────────


def camera_rays(camera: Camera, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """(origin, unit ray directions HxWx3) through pixel centres."""
    forward, right, up = camera.basis()
    tan_x = np.tan(np.radians(camera.hfov_deg) / 2)
    tan_y = tan_x * height / width
    xs = ((np.arange(width) + 0.5) / width * 2 - 1) * tan_x
    ys = (1 - (np.arange(height) + 0.5) / height * 2) * tan_y
    grid_x, grid_y = np.meshgrid(xs, ys)
    directions = forward + grid_x[..., None] * right + grid_y[..., None] * up
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return np.asarray(camera.position, float), directions


def _safe(d: np.ndarray) -> np.ndarray:
    return np.where(np.abs(d) < 1e-12, np.where(d < 0, -1e-12, 1e-12), d)


def _hit_box(origin: np.ndarray, rays: np.ndarray, box: Box) -> tuple[np.ndarray, np.ndarray]:
    """(t, world normal) of an oriented box; t is inf where missed."""
    axes = np.asarray(box.axes, float)
    half = np.asarray(box.half, float)
    local_origin = axes @ (origin - np.asarray(box.center, float))
    local = _safe(rays @ axes.T)
    t1 = (-half - local_origin) / local
    t2 = (half - local_origin) / local
    near = np.minimum(t1, t2)
    far = np.maximum(t1, t2)
    t_near = near.max(-1)
    t_far = far.min(-1)
    hit = (t_far >= t_near) & (t_near > _EPS)
    t = np.where(hit, t_near, np.inf)
    face = near.argmax(-1)
    sign = -np.sign(np.take_along_axis(local, face[..., None], -1))[..., 0]
    normals = axes[face] * sign[..., None]
    return t, normals


def _hit_cylinder(origin: np.ndarray, rays: np.ndarray, cyl: Cylinder) -> tuple[np.ndarray, np.ndarray]:
    ox, oy, oz = origin[0] - cyl.x, origin[1], origin[2] - cyl.z
    dx, dy, dz = rays[..., 0], rays[..., 1], rays[..., 2]
    a = np.maximum(dx * dx + dz * dz, 1e-12)
    b = 2 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - cyl.radius ** 2
    disc = b * b - 4 * a * c
    root = np.sqrt(np.maximum(disc, 0.0))
    t_side = (-b - root) / (2 * a)
    y_side = oy + t_side * dy
    side = np.where((disc >= 0) & (t_side > _EPS) & (y_side >= cyl.y0) & (y_side <= cyl.y1), t_side, np.inf)
    caps = np.full(dy.shape, np.inf)
    cap_up = np.zeros(dy.shape, bool)
    for level, facing_up in ((cyl.y1, True), (cyl.y0, False)):
        t_cap = (level - oy) / _safe(dy)
        px, pz = ox + t_cap * dx, oz + t_cap * dz
        from_side = (dy < 0) if facing_up else (dy > 0)
        ok = from_side & (t_cap > _EPS) & (px * px + pz * pz <= cyl.radius ** 2) & (t_cap < caps)
        caps = np.where(ok, t_cap, caps)
        cap_up = np.where(ok, facing_up, cap_up)
    t = np.minimum(side, caps)
    hit_x = ox + t_side * dx
    hit_z = oz + t_side * dz
    side_normal = np.stack([hit_x / cyl.radius, np.zeros_like(hit_x), hit_z / cyl.radius], -1)
    cap_normal = np.zeros(rays.shape)
    cap_normal[..., 1] = np.where(cap_up, 1.0, -1.0)
    normals = np.where((caps < side)[..., None], cap_normal, side_normal)
    return t, normals


def _hit_sphere(origin: np.ndarray, rays: np.ndarray, sphere: Sphere) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(sphere.center, float)
    oc = origin - center
    b = rays @ oc
    c = oc @ oc - sphere.radius ** 2
    disc = b * b - c
    t = -b - np.sqrt(np.maximum(disc, 0.0))
    t = np.where((disc >= 0) & (t > _EPS), t, np.inf)
    points = origin + rays * np.where(np.isfinite(t), t, 0.0)[..., None]
    return t, (points - center) / sphere.radius


def _room_hits(origin: np.ndarray, rays: np.ndarray, scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(t, label, normal) of the nearest room plane seen from inside: floor, ceiling, walls with their openings."""
    width, depth, height = scene.room["width"], scene.room["depth"], scene.room["height"]
    best_t = np.full(rays.shape[:2], np.inf)
    best_label = np.zeros(rays.shape[:2], np.int16)
    best_normal = np.zeros(rays.shape)
    # (axis, plane value, inward normal sign, label, wall name)
    planes = ((1, 0.0, 1.0, LABELS["floor"], ""), (1, height, -1.0, LABELS["ceiling"], ""),
              (2, 0.0, 1.0, LABELS["wall"], "back"), (2, depth, -1.0, LABELS["wall"], "front"),
              (0, 0.0, 1.0, LABELS["wall"], "left"), (0, width, -1.0, LABELS["wall"], "right"))
    limits = (width, height, depth)
    for axis, value, inward, label, wall in planes:
        d = rays[..., axis]
        # Visible from inside only: the ray must run against the inward normal.
        facing = d * inward < 0
        t = (value - origin[axis]) / _safe(d)
        points = origin + rays * np.where(np.isfinite(t), t, 0.0)[..., None]
        inside = facing & (t > _EPS)
        for other in range(3):
            if other != axis:
                inside &= (points[..., other] >= -1e-6) & (points[..., other] <= limits[other] + 1e-6)
        t = np.where(inside, t, np.inf)
        labels = np.full(d.shape, label, np.int16)
        for opening in scene.openings if wall else ():
            if opening["wall"] != wall:
                continue
            along = points[..., 0] if wall in ("back", "front") else points[..., 2]
            length = width if wall in ("back", "front") else depth
            middle = opening["center"] * length
            in_opening = ((np.abs(along - middle) <= opening["width"] / 2)
                          & (points[..., 1] >= opening["bottom"])
                          & (points[..., 1] <= opening["bottom"] + opening["height"]))
            labels = np.where(in_opening, LABELS[opening["kind"]], labels)
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        best_label = np.where(closer, labels, best_label)
        normal = np.zeros(3)
        normal[axis] = inward
        best_normal = np.where(closer[..., None], normal, best_normal)
    return best_t, best_label, best_normal


def render(scene: Scene, camera: Camera, width: int = 512, height: int = 288) -> Frame:
    """Ray-cast one frame of ``scene`` from ``camera``."""
    width, height = max(8, int(width)), max(8, int(height))
    origin, rays = camera_rays(camera, width, height)
    best_t, labels, normals = _room_hits(origin, rays, scene)
    for primitive in scene.primitives:
        if isinstance(primitive, Box):
            t, n = _hit_box(origin, rays, primitive)
        elif isinstance(primitive, Cylinder):
            t, n = _hit_cylinder(origin, rays, primitive)
        else:
            t, n = _hit_sphere(origin, rays, primitive)
        closer = t < best_t
        if not closer.any():
            continue
        best_t = np.where(closer, t, best_t)
        labels = np.where(closer, primitive.label, labels)
        normals = np.where(closer[..., None], n, normals)
    forward, _right, _up = camera.basis()
    depth = best_t * (rays @ forward)
    labels = np.where(np.isfinite(best_t), labels, LABELS["void"]).astype(np.uint8)
    return Frame(camera=camera, depth=depth.astype(np.float32), labels=labels, normals=normals.astype(np.float32))


def render_sequence(scene: Scene, cameras: list[Camera], width: int = 512, height: int = 288) -> list[Frame]:
    """One frame per camera of a move (see ``camera_path``)."""
    return [render(scene, camera, width, height) for camera in cameras]


def sequence_depth_range(frames: list[Frame]) -> tuple[float, float]:
    """One depth range for every frame of a sequence, so a driving video does not flicker."""
    ranges = [frame.depth_range() for frame in frames] or [(0.1, 10.0)]
    return min(r[0] for r in ranges), max(r[1] for r in ranges)
