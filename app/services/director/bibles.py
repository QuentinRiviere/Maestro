"""Bibles: the approved people, places, props and look of a series, reused across projects.

A Director cast belongs to one project: its images live in that project's
output folder and the next episode casts again from scratch, so faces, sets
and the visual style drift between episodes. A bible keeps them once: the
characters, locations and props (descriptions and reference images), the
style contract and the world rules. Every episode of a series, or every
mini-scene of a short film, starts from the same bible.

Elements have looks (variants): the default one, and others such as an
outfit, a wet version or an open window. Each variant holds candidate images
and at most one approved image, which is its reference. Approved images seed
the next ones: a new outfit is edited from the approved default sheet, a new
place from the approved places it belongs with.

A bible is one folder: ``settings/director_bibles/<id>/bible.json`` and its
image files, so it does not depend on any project's output folder. Applying a
bible to a project copies the chosen entries into the project's cast; the
project records the bible id and revision it was made from.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import secrets
import shutil
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

BIBLES_DIR = Path(__file__).resolve().parents[2] / "settings" / "director_bibles"
BIBLE_FILE = "bible.json"
KINDS = {"character": "characters", "location": "locations", "prop": "props"}
# Text fields of each kind, with their length limit.
FIELDS = {
    "character": {"role": 120, "descriptor": 80, "description": 700, "wardrobe": 400,
                  "personality": 500, "arc": 500},
    "location": {"description": 700, "notes": 600},
    "prop": {"descriptor": 80, "description": 700, "notes": 600},
}
# A series keeps more people, places and props than one project casts.
MAX_ELEMENTS = 24
MAX_VARIANTS = 12
MAX_IMAGES = 32
DEFAULT_VARIANT = "default"
# Angles of a character's look, edited from its approved image: together they
# make the turnaround sheet a project uses as the character's reference.
VIEWS = {"face_front": "Face", "face_three_quarter": "Face 3/4", "face_profile": "Profile", "back": "Back"}
# Angles of a place: the same empty room seen from its opposite wall, for shots
# whose camera looks back at what the place's main image has behind its camera.
LOCATION_VIEWS = {"reverse": "Reverse"}
ALL_VIEWS = {**VIEWS, **LOCATION_VIEWS}
KIND_VIEWS = {"character": VIEWS, "location": LOCATION_VIEWS}
IMAGE_STATES = ("queued", "rendering", "candidate", "approved", "failed")
IN_PROGRESS = ("queued", "rendering")
# Render slots owned by this process. A slot still queued or rendering that
# is not in it was cut by a restart.
_LIVE_SLOTS: set[str] = set()
WORLD_FIELDS = ("logline", "story", "stakes", "twists", "tone", "narrative_style", "look", "rules", "forbidden")
WORLD_LIMITS = {"story": 4000, "twists": 3000}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
# Reference videos of a series (staging, editing and rhythm to learn from);
# see ``style_reference``. They are stored next to the bible's images.
VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv")
MAX_STYLE_REFERENCES = 10
STYLE_REFERENCE_GRAPHICS = ("ignore", "include")
STYLE_REFERENCE_STATES = ("queued", "analysing", "done", "failed")
STYLE_FRAMES_STATES = ("idle", "queued", "running", "done", "failed")
# Analyses owned by this process. A reference still queued or analysing that
# is not in it was cut by a restart.
_LIVE_ANALYSES: set[str] = set()
# Style LoRAs trained from the references' frames (see ``style_lora``).
MAX_STYLE_LORAS = 5
STYLE_LORA_STATES = ("queued", "preparing", "training", "sampling", "done", "failed")
STYLE_LORA_LIVE = ("queued", "preparing", "training", "sampling")
# Trainings owned by this process; a record still under way that is not in
# it was cut by a restart (a training is hours of GPU: it is not re-queued).
_LIVE_TRAININGS: set[str] = set()
_LOCK = threading.RLock()
_SLUG = re.compile(r"[^a-z0-9]+")
_ID = re.compile(r"[a-z0-9_]{1,40}")
_KEY = re.compile(r"[a-z0-9_-]{1,40}")


class BibleError(ValueError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _folder(bible_id: str) -> Path:
    if not _ID.fullmatch(bible_id or ""):
        raise BibleError("Invalid bible id")
    return BIBLES_DIR / bible_id


def _text(value: Any, limit: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def _slug(value: str, limit: int = 24) -> str:
    return _SLUG.sub("_", str(value or "").lower()).strip("_")[:limit]


def _key(value: Any, fallback: str) -> str:
    value = str(value or "").strip().lower()
    return value if _KEY.fullmatch(value) else fallback


# ── normalization ──────────────────────────────────────────────────────────


def _images(raw: Any, folder: Path) -> list[dict]:
    """Candidate images of one variant; files must exist, one approved at most (the last)."""
    images: list[dict] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        status = item.get("status") if item.get("status") in IMAGE_STATES else "candidate"
        file = os.path.basename(str(item.get("file") or ""))
        error = _text(item.get("error"), 300)
        if status in IN_PROGRESS and str(item.get("id") or "") not in _LIVE_SLOTS:
            status, error = "failed", "Interrupted"
        if status in ("candidate", "approved") and not (file and (folder / file).is_file()):
            continue
        images.append({
            "id": _key(item.get("id"), secrets.token_hex(4)),
            "file": file or None,
            "status": status,
            "error": error,
            "created_at": str(item.get("created_at") or _now()),
            "view": item.get("view") if item.get("view") in ALL_VIEWS else "",
        })
    for view in ("", *ALL_VIEWS):
        approved = [i for i, image in enumerate(images) if image["status"] == "approved" and image["view"] == view]
        for index in approved[:-1]:
            images[index]["status"] = "candidate"
    return images[-MAX_IMAGES:] if len(images) > MAX_IMAGES else images


def _has_approved(variant: dict, view: str = "") -> bool:
    return any(i["status"] == "approved" and i.get("view", "") == view for i in variant["images"])


def _variants(raw: dict, folder: Path) -> list[dict]:
    variants = []
    seen = set()
    for item in raw.get("variants") if isinstance(raw.get("variants"), list) else []:
        if not isinstance(item, dict):
            continue
        variant_id = _key(item.get("id"), _slug(item.get("label") or "", 32) or secrets.token_hex(3))
        if variant_id in seen:
            continue
        seen.add(variant_id)
        variants.append({
            "id": variant_id,
            "label": _text(item.get("label"), 60) or ("Default" if variant_id == DEFAULT_VARIANT else variant_id),
            "description": _text(item.get("description"), 600),
            "images": _images(item.get("images"), folder),
        })
    if DEFAULT_VARIANT not in seen:
        # A step-1 entry (or a project cast) carries one approved image.
        legacy = os.path.basename(str(raw.get("image") or ""))
        images = [{"file": legacy, "status": "approved"}] if legacy else []
        variants.insert(0, {"id": DEFAULT_VARIANT, "label": "Default", "description": "",
                            "images": _images(images, folder)})
    else:
        variants.sort(key=lambda v: v["id"] != DEFAULT_VARIANT)
    return variants[:MAX_VARIANTS]


def _element(raw: Any, kind: str, folder: Path) -> dict | None:
    if not isinstance(raw, dict):
        return None
    name = _text(raw.get("name"), 60)
    if not name:
        return None
    element = {
        "id": _key(raw.get("id"), f"{kind[:4]}_{_slug(name, 20) or 'x'}_{secrets.token_hex(2)}"),
        "kind": kind,
        "name": name,
        "tag": "@" + (_slug(str(raw.get("tag") or "").lstrip("@"), 32) or _slug(name, 32) or kind),
    }
    for field, limit in FIELDS[kind].items():
        element[field] = _text(raw.get(field), limit)
    if kind == "character":
        element["voice_character_id"] = _text(raw.get("voice_character_id"), 80)
    if kind == "location" and "layout" in raw:
        # The place's rough 3D layout for the blockout (see blockout.layout);
        # null clears it, a missing key keeps the stored one (see keep_layouts).
        from services.director.blockout.layout import normalize_layout

        element["layout"] = normalize_layout(raw.get("layout")) if raw.get("layout") else None
    element["variants"] = _variants(raw, folder)
    element["image"] = approved_image(element)
    return element


def keep_layouts(bible: dict, stored: dict) -> None:
    """Give each location saved without a ``layout`` key the layout it had in ``stored``.

    Clients that do not edit layouts (the bible assistant, older UIs) send
    places without one; their layout is kept rather than dropped.
    """
    before = {e["id"]: e for e in stored.get("locations") or []}
    for element in bible.get("locations") or []:
        if "layout" not in element:
            element["layout"] = (before.get(element["id"]) or {}).get("layout")


def _elements(raw: Any, kind: str, folder: Path) -> list[dict]:
    elements, names, ids = [], set(), set()
    for item in raw if isinstance(raw, list) else []:
        element = _element(item, kind, folder)
        if not element or element["name"].casefold() in names or element["id"] in ids:
            continue
        names.add(element["name"].casefold())
        ids.add(element["id"])
        elements.append(element)
    return elements[:MAX_ELEMENTS]


def _world(raw: Any) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    return {field: _text(raw.get(field), WORLD_LIMITS.get(field, 2000)) for field in WORLD_FIELDS}


def _style_references(raw: Any, folder: Path) -> list[dict]:
    """Reference videos of the bible; their files must exist. Analysis results are kept as stored."""
    references, ids = [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        file = os.path.basename(str(item.get("file") or ""))
        if not file.lower().endswith(VIDEO_SUFFIXES) or not (folder / file).is_file():
            continue
        ref_id = _key(item.get("id"), "ref_" + secrets.token_hex(4))
        if ref_id in ids:
            continue
        ids.add(ref_id)
        status = item.get("status") if item.get("status") in STYLE_REFERENCE_STATES else "failed"
        error = _text(item.get("error"), 400)
        if status in ("queued", "analysing") and ref_id not in _LIVE_ANALYSES:
            status, error = "failed", "Interrupted"
        tags = item.get("scene_tags") if isinstance(item.get("scene_tags"), list) else []
        cost = item.get("cost_usd")
        references.append({
            "id": ref_id,
            "label": _text(item.get("label"), 80) or _text(item.get("source_filename"), 80) or ref_id,
            "file": file,
            "source_filename": _text(item.get("source_filename"), 160),
            "created_at": str(item.get("created_at") or _now()),
            "updated_at": str(item.get("updated_at") or item.get("created_at") or _now()),
            "status": status,
            "error": error,
            "progress": _text(item.get("progress"), 200),
            "enabled": item.get("enabled") is not False,
            "scene_tags": list(dict.fromkeys(t for t in (_slug(str(tag), 24) for tag in tags) if t))[:8],
            "stats": copy.deepcopy(item["stats"]) if isinstance(item.get("stats"), dict) else None,
            # The per-shot breakdown lives in a sidecar file (see ``style_reference``).
            "described_shots": described if isinstance(described := item.get("described_shots"), int)
            and not isinstance(described, bool) and described > 0 else 0,
            "grammar": copy.deepcopy(item["grammar"]) if isinstance(item.get("grammar"), dict) else None,
            "summary": str(item.get("summary") or "")[:12000],
            # What was uploaded before the 480p proxy replaced it (style_reference.ensure_proxy).
            "source": copy.deepcopy(item["source"]) if isinstance(item.get("source"), dict) else None,
            # Whether the analyst reads captions, logos and other on-screen graphics as part of the staging.
            "graphics": item.get("graphics") if item.get("graphics") in STYLE_REFERENCE_GRAPHICS else "ignore",
            # Label and tags the analysis may set (true until the user types their own).
            "label_auto": item.get("label_auto") is True,
            "tags_auto": item.get("tags_auto") is True,
            # The reverse-prompt pass over the shots' first and last frames (style_reference.start_frames).
            "frames_status": ("failed" if item.get("frames_status") in ("queued", "running")
                              and ref_id not in _LIVE_ANALYSES else item.get("frames_status"))
            if item.get("frames_status") in STYLE_FRAMES_STATES else "idle",
            "frames_count": count if isinstance(count := item.get("frames_count"), int)
            and not isinstance(count, bool) and count > 0 else 0,
            "frames_progress": _text(item.get("frames_progress"), 200),
            "frames_error": _text(item.get("frames_error"), 400),
            "camera_profile": copy.deepcopy(item["camera_profile"]) if isinstance(item.get("camera_profile"), dict)
            and item["camera_profile"] else None,
            "cost_usd": cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
            "model": _text(item.get("model"), 120),
        })
    return references[:MAX_STYLE_REFERENCES]


def _style_grammar(raw: Any) -> dict | None:
    """The grammar consolidated from the bible's references (style_reference.start_merge), as stored."""
    if not isinstance(raw, dict) or not raw:
        return None
    grammar = raw.get("grammar")
    sources = [{"id": str(s.get("id") or ""), "updated_at": str(s.get("updated_at") or ""),
                "scene_tags": [str(t) for t in (s.get("scene_tags") or [])][:8]}
               for s in raw.get("sources") or [] if isinstance(s, dict)]
    cost = raw.get("cost_usd")
    return {
        "status": _text(raw.get("status"), 20) or "failed",
        "sources": sources[:MAX_STYLE_REFERENCES],
        "grammar": copy.deepcopy(grammar) if isinstance(grammar, dict) and grammar else None,
        "summary": str(raw.get("summary") or "")[:12000],
        "model": _text(raw.get("model"), 120),
        "cost_usd": cost if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
        "updated_at": _text(raw.get("updated_at"), 40),
        "error": _text(raw.get("error"), 400),
    }


def _style_loras(raw: Any) -> list[dict]:
    """The style LoRAs of the bible (style_lora.create_lora); a training cut by a restart reads as failed."""
    loras, ids = [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        lora_id = _key(item.get("id"), "lora_" + secrets.token_hex(4))
        if lora_id in ids:
            continue
        ids.add(lora_id)
        status = item.get("status") if item.get("status") in STYLE_LORA_STATES else "failed"
        error = _text(item.get("error"), 400)
        if status in STYLE_LORA_LIVE and lora_id not in _LIVE_TRAININGS:
            # A sample render cut by a restart keeps the trained adapter.
            if status == "sampling" and item.get("file"):
                status, error = "done", "The sample render was interrupted"
            else:
                status, error = "failed", "Interrupted"
        scale = item.get("scale")
        config = item.get("config") if isinstance(item.get("config"), dict) else {}
        loras.append({
            "id": lora_id,
            "name": _text(item.get("name"), 60) or lora_id,
            # The adapter's file name in the LoRA registry (loras/qwen21) once trained.
            "file": os.path.basename(_text(item.get("file"), 160)),
            "trigger": _text(item.get("trigger"), 40),
            "status": status,
            "progress": _text(item.get("progress"), 200),
            "step": step if isinstance(step := item.get("step"), int) and not isinstance(step, bool) else 0,
            "total_steps": total if isinstance(total := item.get("total_steps"), int)
            and not isinstance(total, bool) else 0,
            "error": error,
            "enabled": item.get("enabled") is True and status == "done",
            "scale": round(float(scale), 2) if isinstance(scale, (int, float)) and not isinstance(scale, bool) else 0.4,
            "config": copy.deepcopy(config),
            "dataset": copy.deepcopy(item["dataset"]) if isinstance(item.get("dataset"), dict) else None,
            "samples": [os.path.basename(str(s)) for s in item.get("samples") or [] if s][:6],
            "samples_note": _text(item.get("samples_note"), 80),
            "job_id": _text(item.get("job_id"), 40),
            "model": _text(item.get("model"), 120),
            "training": copy.deepcopy(item["training"]) if isinstance(item.get("training"), dict) else None,
            "created_at": str(item.get("created_at") or _now()),
            "updated_at": str(item.get("updated_at") or item.get("created_at") or _now()),
        })
    return loras[:MAX_STYLE_LORAS]


def normalize_bible(raw: Any, folder: Path) -> dict:
    """The stored fields of a bible, bounded; ids and timestamps are kept as given."""
    raw = raw if isinstance(raw, dict) else {}
    name = _text(raw.get("name"), 80)
    if not name:
        raise BibleError("A bible needs a name")
    style = raw.get("style")
    bible = {
        "id": str(raw.get("id") or ""),
        "name": name,
        "description": _text(raw.get("description"), 600),
        "style": copy.deepcopy(style) if isinstance(style, dict) and style else None,
        "world": _world(raw.get("world")),
    }
    for kind, plural in KINDS.items():
        bible[plural] = _elements(raw.get(plural), kind, folder)
    bible["style_references"] = _style_references(raw.get("style_references"), folder)
    bible["style_grammar"] = _style_grammar(raw.get("style_grammar"))
    bible["style_loras"] = _style_loras(raw.get("style_loras"))
    return bible


def missing_images(bible: dict, *, every_look: bool = False, element_id: str = "") -> list[tuple[dict, dict]]:
    """(element, look) pairs to render, default looks first.

    Looks with no approved image and no render under way; with
    ``every_look``, every look without a render under way (a new candidate
    next to its approved image). ``element_id`` keeps one element's looks.
    """
    pending = []
    skip = IN_PROGRESS if every_look else ("approved", *IN_PROGRESS)
    for plural in KINDS.values():
        for element in bible.get(plural) or []:
            if element_id and element.get("id") != element_id:
                continue
            for variant in element.get("variants") or []:
                images = [i for i in variant.get("images") or [] if not i.get("view")]
                if not any(i["status"] in skip for i in images):
                    pending.append((element, variant))
    return sorted(pending, key=lambda pair: pair[1]["id"] != DEFAULT_VARIANT)


def approved_image(element: dict, variant_id: str = DEFAULT_VARIANT) -> str | None:
    """The approved image of a variant (of the default one when it has none)."""
    for wanted in dict.fromkeys([variant_id, DEFAULT_VARIANT]):
        variant = next((v for v in element.get("variants") or [] if v["id"] == wanted), None)
        image = next((i["file"] for i in (variant or {}).get("images") or []
                      if i["status"] == "approved" and not i.get("view")), None)
        if image:
            return image
    return None


# ── storage ────────────────────────────────────────────────────────────────


def _read(bible_id: str) -> dict | None:
    try:
        return json.loads((_folder(bible_id) / BIBLE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, BibleError):
        return None


def _write(bible: dict) -> None:
    folder = _folder(bible["id"])
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / BIBLE_FILE
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(bible, indent=1, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)


def _stamp(bible: dict, previous: dict) -> dict:
    now = _now()
    bible["revision"] = int(previous.get("revision") or 0) + 1
    bible["created_at"] = previous.get("created_at") or now
    bible["updated_at"] = now
    return bible


def _new_id(name: str) -> str:
    base = _slug(name, 32) or "bible"
    candidate, suffix = base, 2
    while (BIBLES_DIR / candidate).exists():
        candidate = f"{base[:32]}_{suffix}"
        suffix += 1
    return candidate


def list_bibles() -> list[dict]:
    """Summaries, most recently updated first."""
    if not BIBLES_DIR.is_dir():
        return []
    summaries = []
    for folder in BIBLES_DIR.iterdir():
        bible = _read(folder.name) if folder.is_dir() else None
        if bible:
            summaries.append({
                "id": bible["id"], "name": bible["name"], "description": bible.get("description", ""),
                "revision": bible.get("revision", 1), "updated_at": bible.get("updated_at", ""),
                **{plural: [e["name"] for e in bible.get(plural) or []] for plural in KINDS.values()},
                "style_references": len(bible.get("style_references") or []),
            })
    return sorted(summaries, key=lambda item: item["updated_at"], reverse=True)


def get_bible(bible_id: str) -> dict | None:
    # Read under the write lock: on a bind-mounted data folder a read racing
    # a save can miss the file, and parallel renders save often.
    with _LOCK:
        bible = _read(bible_id)
    if bible is None:
        return None
    # Bibles saved before props and variants existed read as the current shape.
    normalized = normalize_bible(bible, _folder(bible_id))
    for key in ("id", "revision", "created_at", "updated_at"):
        normalized[key] = bible.get(key)
    return normalized


def save_bible(raw: dict, *, bible_id: str | None = None) -> dict:
    """Create (no id) or update (given id) a bible's text; every save is a new revision.

    Images belong to the bible, not to the client's copy: an update keeps
    each element's stored images per variant (matched by element and variant
    id), so a render finishing during an edit is never lost. New elements and
    variants start without images.
    """
    with _LOCK:
        if bible_id is None:
            BIBLES_DIR.mkdir(parents=True, exist_ok=True)
            bible_id = _new_id(_text((raw or {}).get("name"), 80))
            previous: dict = {}
            folder = _folder(bible_id)
            folder.mkdir(parents=True, exist_ok=True)
            bible = normalize_bible(raw, folder)
            for plural in KINDS.values():
                for element in bible[plural]:
                    for variant in element["variants"]:
                        variant["images"] = []
                    element["image"] = None
            bible["style_references"] = []
            bible["style_grammar"] = None
        else:
            previous = get_bible(bible_id) or {}
            if not previous:
                raise BibleError("Unknown bible")
            folder = _folder(bible_id)
            bible = normalize_bible(raw, folder)
            stored = {e["id"]: e for plural in KINDS.values() for e in previous.get(plural) or []}
            for plural in KINDS.values():
                for element in bible[plural]:
                    before = {v["id"]: v for v in (stored.get(element["id"]) or {}).get("variants") or []}
                    for variant in element["variants"]:
                        variant["images"] = copy.deepcopy((before.get(variant["id"]) or {}).get("images") or [])
                    element["image"] = approved_image(element)
            keep_layouts(bible, previous)
            # Reference videos and their analyses have their own endpoints.
            bible["style_references"] = copy.deepcopy(previous.get("style_references") or [])
            bible["style_grammar"] = copy.deepcopy(previous.get("style_grammar"))
        bible["id"] = bible_id
        _write(_stamp(bible, previous))
    return bible


def delete_bible(bible_id: str) -> bool:
    with _LOCK:
        folder = _folder(bible_id)
        if not (folder / BIBLE_FILE).is_file():
            return False
        shutil.rmtree(folder)
        return True


def video_path(bible_id: str, name: str) -> Path | None:
    """The file of one reference video of a bible, or None when the name is not one."""
    name = os.path.basename(name or "")
    if not name.lower().endswith(VIDEO_SUFFIXES):
        return None
    path = _folder(bible_id) / name
    return path if path.is_file() else None


def image_path(bible_id: str, name: str) -> Path | None:
    """The file of one bible image, or None when the name is not one of its images."""
    name = os.path.basename(name or "")
    if not name.lower().endswith(IMAGE_SUFFIXES):
        return None
    path = _folder(bible_id) / name
    return path if path.is_file() else None


# ── elements, variants and images ──────────────────────────────────────────


def find_element(bible: dict, element_id: str) -> tuple[str, dict]:
    for kind, plural in KINDS.items():
        for element in bible.get(plural) or []:
            if element["id"] == element_id:
                return kind, element
    raise BibleError("Unknown element")


def find_variant(element: dict, variant_id: str) -> dict:
    variant = next((v for v in element["variants"] if v["id"] == variant_id), None)
    if variant is None:
        raise BibleError("Unknown variant")
    return variant


def _mutate(bible_id: str, change: Callable[[dict], Any]) -> tuple[dict, Any]:
    """Apply ``change`` to the stored bible and save it as a new revision."""
    with _LOCK:
        bible = get_bible(bible_id)
        if not bible:
            raise BibleError("Unknown bible")
        previous = dict(bible)
        result = change(bible)
        for plural in KINDS.values():
            for element in bible[plural]:
                element["image"] = approved_image(element)
        _write(_stamp(bible, previous))
    return bible, result


def _import_image(folder: Path, source: str | os.PathLike, prefix: str) -> str | None:
    source = str(source or "")
    if not source or not os.path.isfile(source):
        return None
    suffix = os.path.splitext(source)[1].lower()
    if suffix not in IMAGE_SUFFIXES:
        return None
    name = f"{prefix}_{int(time.time() * 1000)}_{secrets.token_hex(2)}{suffix}"
    shutil.copyfile(source, folder / name)
    return name


def _image_prefix(element: dict, variant_id: str) -> str:
    return "_".join(filter(None, [element["kind"][:4], _slug(element["name"], 20),
                                  None if variant_id == DEFAULT_VARIANT else _slug(variant_id, 16)]))


def add_image(bible_id: str, element_id: str, variant_id: str, source: str) -> dict:
    """Add a file (an upload) as a candidate; the first image of a variant is approved."""
    def change(bible: dict) -> None:
        _, element = find_element(bible, element_id)
        variant = find_variant(element, variant_id)
        name = _import_image(_folder(bible_id), source, _image_prefix(element, variant_id))
        if not name:
            raise BibleError("Not an image file")
        variant["images"].append({"id": secrets.token_hex(4), "file": name,
                                  "status": "candidate" if _has_approved(variant) else "approved",
                                  "error": "", "created_at": _now(), "view": ""})
    return _mutate(bible_id, change)[0]


def start_render(bible_id: str, element_id: str, variant_id: str, view: str = "") -> tuple[dict, str]:
    """Reserve a rendering slot in a variant (of one of its angles); returns the bible and the slot id."""
    def change(bible: dict) -> str:
        kind, element = find_element(bible, element_id)
        variant = find_variant(element, variant_id)
        if view and view not in KIND_VIEWS.get(kind, {}):
            raise BibleError("Unknown angle")
        slot = secrets.token_hex(4)
        _LIVE_SLOTS.add(slot)
        variant["images"].append({"id": slot, "file": None, "status": "queued", "error": "",
                                  "created_at": _now(), "view": view})
        return slot
    return _mutate(bible_id, change)


def retry_render(bible_id: str, element_id: str, variant_id: str, image_id: str) -> tuple[dict, str]:
    """Queue a failed slot again in place; returns the bible and the slot's angle ("" for the look)."""
    def change(bible: dict) -> str:
        _, element = find_element(bible, element_id)
        variant = find_variant(element, variant_id)
        image = next((i for i in variant["images"] if i["id"] == image_id), None)
        if image is None:
            raise BibleError("Unknown image")
        if image["status"] != "failed":
            raise BibleError("Only a failed render can be retried")
        _LIVE_SLOTS.add(image_id)
        image.update(status="queued", error="", created_at=_now())
        return image.get("view", "")
    return _mutate(bible_id, change)


def mark_rendering(bible_id: str, element_id: str, variant_id: str, slot: str) -> None:
    """A queued slot starts rendering."""
    def change(bible: dict) -> None:
        _, element = find_element(bible, element_id)
        image = next((i for i in find_variant(element, variant_id)["images"] if i["id"] == slot), None)
        if image is not None and image["status"] == "queued":
            image["status"] = "rendering"
    _mutate(bible_id, change)


def slot_file(bible_id: str, element_id: str, slot: str) -> str | None:
    """The image a finished render slot produced (None while rendering or after a failure)."""
    bible = get_bible(bible_id) or {}
    try:
        _, element = find_element(bible, element_id)
    except BibleError:
        return None
    for variant in element["variants"]:
        for image in variant["images"]:
            if image["id"] == slot and image["file"]:
                return image["file"]
    return None


def active_renders() -> list[dict]:
    """Queued and running bible renders, oldest first, for the generation queue."""
    renders = []
    for summary in list_bibles():
        bible = get_bible(summary["id"]) or {}
        for plural in KINDS.values():
            for element in bible.get(plural) or []:
                for variant in element["variants"]:
                    for image in variant["images"]:
                        if image["status"] in IN_PROGRESS:
                            look = variant["label"] + (f" · {ALL_VIEWS[image['view']]}" if image.get("view") else "")
                            renders.append({
                                "bible_id": bible["id"], "bible_name": bible["name"], "kind": element["kind"],
                                "element": element["name"], "look": look, "slot": image["id"],
                                "status": image["status"], "created_at": image["created_at"],
                            })
    return sorted(renders, key=lambda render: (render["status"] != "rendering", render["created_at"]))


def finish_render(bible_id: str, element_id: str, variant_id: str, slot: str, *,
                  source: str | None = None, error: str = "") -> dict:
    """Turn a rendering slot into a candidate (the first image is approved) or a failure."""
    def change(bible: dict) -> None:
        _LIVE_SLOTS.discard(slot)
        _, element = find_element(bible, element_id)
        variant = find_variant(element, variant_id)
        image = next((i for i in variant["images"] if i["id"] == slot), None)
        if image is None:
            return
        name = _import_image(_folder(bible_id), source, _image_prefix(element, variant_id)) if source else None
        if not name:
            image.update(status="failed", error=_text(error or "The render produced no image", 300))
            return
        image.update(file=name, status="candidate" if _has_approved(variant, image.get("view", "")) else "approved",
                     error="")
    return _mutate(bible_id, change)[0]


def approve_image(bible_id: str, element_id: str, variant_id: str, image_id: str) -> dict:
    def change(bible: dict) -> None:
        _, element = find_element(bible, element_id)
        variant = find_variant(element, variant_id)
        chosen = next((i for i in variant["images"] if i["id"] == image_id and i["file"]), None)
        if chosen is None:
            raise BibleError("Unknown image")
        for image in variant["images"]:
            if image["status"] == "approved" and image.get("view", "") == chosen.get("view", ""):
                image["status"] = "candidate"
        chosen["status"] = "approved"
    return _mutate(bible_id, change)[0]


def unapprove_image(bible_id: str, element_id: str, variant_id: str, image_id: str) -> dict:
    """Turn an approved image back into a candidate: the look has no reference until one is approved."""
    def change(bible: dict) -> None:
        _, element = find_element(bible, element_id)
        variant = find_variant(element, variant_id)
        chosen = next((i for i in variant["images"] if i["id"] == image_id), None)
        if chosen is None:
            raise BibleError("Unknown image")
        if chosen["status"] == "approved":
            chosen["status"] = "candidate"
    return _mutate(bible_id, change)[0]


def _image_files(bible: dict) -> set[str]:
    return {image["file"] for plural in KINDS.values() for element in bible.get(plural) or []
            for variant in element.get("variants") or [] for image in variant.get("images") or []
            if image.get("file")}


def delete_image(bible_id: str, element_id: str, variant_id: str, image_id: str) -> dict:
    """Drop one image and its file; projects keep their own copies of the images they use."""
    def change(bible: dict) -> str:
        _, element = find_element(bible, element_id)
        variant = find_variant(element, variant_id)
        dropped = next((i for i in variant["images"] if i["id"] == image_id), None)
        if dropped is None:
            raise BibleError("Unknown image")
        variant["images"] = [i for i in variant["images"] if i["id"] != image_id]
        return dropped.get("file") or ""
    with _LOCK:
        bible, name = _mutate(bible_id, change)
        if name and name not in _image_files(bible):
            path = image_path(bible_id, name)
            if path is not None:
                path.unlink(missing_ok=True)
    return bible


def reference_paths(bible: dict, element: dict, variant: dict, chain_from: list[str] | None = None,
                    base_image: str | None = None) -> list[Path]:
    """Approved images a new render of ``variant`` is edited from, most important first.

    A non-default look starts from the element's approved default (same person
    or place, changed as the variant says); ``chain_from`` names elements whose
    approved default the new image must match (the same building, the same
    world). Without either, the render starts from text alone. ``base_image``
    (a file of the bible) replaces the approved default, e.g. the default
    just rendered in the same batch.
    """
    folder = _folder(bible["id"])
    files: list[str] = []
    if variant["id"] != DEFAULT_VARIANT:
        files.append(os.path.basename(base_image or "") or approved_image(element) or "")
    for element_id in chain_from or []:
        try:
            _, other = find_element(bible, element_id)
        except BibleError:
            continue
        if other["id"] != element["id"]:
            files.append(approved_image(other) or "")
    paths = []
    for name in dict.fromkeys(f for f in files if f):
        if (folder / name).is_file():
            paths.append(folder / name)
    return paths


def approved_views(variant: dict, views: dict[str, str] = VIEWS) -> dict[str, str]:
    """The approved image of each angle of a look, in ``views`` order (a character's angles by default)."""
    found = {i["view"]: i["file"] for i in variant.get("images") or []
             if i.get("view") and i["status"] == "approved" and i["file"]}
    return {view: found[view] for view in views if view in found}


def compose_turnaround(folder: Path, main: str, views: dict[str, str], target: str) -> bool:
    """One sheet: the full figure and the back side by side, the face angles stacked on the right."""
    from PIL import Image

    height = 1920
    try:
        def tall(name: str) -> Image.Image:
            with Image.open(folder / name) as image:
                image = image.convert("RGB")
                return image.resize((max(1, round(image.width * height / image.height)), height))
        panels = [tall(main)] + ([tall(views["back"])] if "back" in views else [])
        faces = []
        for view in ("face_front", "face_three_quarter", "face_profile"):
            if view in views:
                with Image.open(folder / views[view]) as image:
                    faces.append(image.convert("RGB"))
        side = height // 3
        width = sum(p.width for p in panels) + (side if faces else 0)
        sheet = Image.new("RGB", (width, height), (214, 214, 214))
        x = 0
        for panel in panels:
            sheet.paste(panel, (x, 0))
            x += panel.width
        for row, face in enumerate(faces):
            # Square crop around the centre, then one third of the sheet height.
            edge = min(face.width, face.height)
            left, top = (face.width - edge) // 2, (face.height - edge) // 2
            sheet.paste(face.crop((left, top, left + edge, top + edge)).resize((side, side)), (x, row * side))
        sheet.save(target, "PNG")
        return True
    except (OSError, ValueError):
        return False


# ── projects ───────────────────────────────────────────────────────────────


def add_cast(cast: dict, image_dirs: list[str], *, bible_id: str | None = None, name: str = "",
             style: dict | None = None) -> dict:
    """Save a project's cast (and style) into a new or an existing bible.

    ``image_dirs`` are the folders the cast images are looked up in (the
    project's output folder first). An entry with the name of an element of
    the same kind updates its text and adds its image as a candidate of the
    default look (approved when the look has none); other entries become new
    elements. The style is only set when the bible has none yet.
    """
    if bible_id:
        if not get_bible(bible_id):
            raise BibleError("Unknown bible")
    else:
        bible_id = save_bible({"name": name or "Untitled bible", "style": style})["id"]

    def change(bible: dict) -> None:
        folder = _folder(bible_id)
        for kind, plural in (("character", "characters"), ("location", "locations")):
            by_name = {e["name"].casefold(): e for e in bible[plural]}
            for entry in (cast or {}).get(plural) or []:
                incoming = _element({k: v for k, v in entry.items() if k not in ("image", "variants")}, kind, folder)
                if incoming is None:
                    continue
                element = by_name.get(incoming["name"].casefold())
                if element is None:
                    if len(bible[plural]) >= MAX_ELEMENTS:
                        continue
                    element = incoming
                    bible[plural].append(element)
                    by_name[element["name"].casefold()] = element
                else:
                    for field in FIELDS[kind]:
                        element[field] = incoming[field] or element[field]
                image = os.path.basename(str(entry.get("image") or ""))
                source = next((os.path.join(d, image) for d in image_dirs
                               if d and image and os.path.isfile(os.path.join(d, image))), "")
                stored = _import_image(folder, source, _image_prefix(element, DEFAULT_VARIANT))
                if stored:
                    variant = find_variant(element, DEFAULT_VARIANT)
                    variant["images"].append({"id": secrets.token_hex(4), "file": stored,
                                              "status": "candidate" if _has_approved(variant) else "approved",
                                              "error": "", "created_at": _now(), "view": ""})
        if style and not bible.get("style"):
            bible["style"] = copy.deepcopy(style)
    return _mutate(bible_id, change)[0]


def _chosen(elements: list[dict], names: Any) -> list[dict]:
    if not isinstance(names, list) or not names:
        return list(elements)
    wanted = {str(n).strip().casefold() for n in names if str(n).strip()}
    return [e for e in elements if e["name"].casefold() in wanted or e["id"] in wanted or e["tag"] in wanted]


def _look(element: dict, variants: Any) -> dict:
    """The variant an episode asked for (by element name, id or tag), else the default."""
    variants = variants if isinstance(variants, dict) else {}
    wanted = next((str(variants[key]) for key in (element["name"], element["id"], element["tag"])
                   if key in variants), DEFAULT_VARIANT)
    wanted = wanted.strip().lower()
    return next((v for v in element["variants"] if v["id"] == wanted or v["label"].lower() == wanted),
                element["variants"][0])


def project_cast(bible: dict, out_dir: str, *, characters: Any = None, locations: Any = None,
                 props: Any = None, variants: Any = None) -> dict:
    """The cast of one project made from a bible, its images copied into ``out_dir``.

    The Director keeps cast images as file names in the project's output
    folder; copying them keeps every cast path (planning, review, saved
    projects, Open & Edit) unchanged. ``characters``, ``locations`` and
    ``props`` pick elements (by name, id or tag; none picks all);
    ``variants`` maps an element to the look this project uses. A
    character's look replaces its wardrobe; a place's or a prop's look is
    added to its description. Props only feed shot references (by name).
    A character keeps its ``voice_character_id`` (the library character
    whose voice speaks its lines).
    """
    folder = _folder(bible["id"])
    os.makedirs(out_dir, exist_ok=True)
    cast: dict[str, list] = {"characters": [], "locations": [], "props": []}
    for kind, plural, names in (("character", "characters", characters), ("location", "locations", locations),
                                ("prop", "props", props)):
        for element in _chosen(bible.get(plural) or [], names):
            look = _look(element, variants)
            entry = {field: element.get(field, "") for field in ("name", "role", "descriptor", "description",
                                                                 "wardrobe", "notes") if field in element}
            entry.update(bible_element=element["id"], bible_variant=look["id"], tag=element["tag"])
            # The linked library voice is resolved at render time, so a voice
            # re-recorded in the library reaches the next render of the project.
            if kind == "character" and element.get("voice_character_id"):
                entry["voice_character_id"] = element["voice_character_id"]
            if kind == "location" and element.get("layout"):
                entry["layout"] = copy.deepcopy(element["layout"])
            if look["id"] != DEFAULT_VARIANT and look["description"]:
                if kind == "character":
                    entry["wardrobe"] = look["description"]
                else:
                    entry["description"] = f"{element['description']} In this scene: {look['description']}"
            image = approved_image(element, look["id"])
            # A character's look with approved angles goes to the project as one turnaround sheet.
            variant = next((v for v in element["variants"] if v["id"] == look["id"]), None)
            views = (approved_views(variant, KIND_VIEWS[kind])
                     if kind in KIND_VIEWS and variant and _has_approved(variant) else {})
            entry["image"] = None
            if kind == "character" and image and (folder / image).is_file() and views:
                stem = hashlib.sha1("|".join([image, *views.values()]).encode()).hexdigest()[:10]
                copied = f"bible_{bible['id']}_sheet_{stem}.png"
                target = os.path.join(out_dir, copied)
                if os.path.isfile(target) or compose_turnaround(folder, image, views, target):
                    entry["image"] = copied
            if image and (folder / image).is_file():
                copied = f"bible_{bible['id']}_{image}"
                target = os.path.join(out_dir, copied)
                if not os.path.isfile(target):
                    shutil.copyfile(folder / image, target)
                if entry["image"]:
                    # Image editors copy a sheet's layout: shot images take the look's single image.
                    entry["portrait"] = copied
                else:
                    entry["image"] = copied
            if views:
                # Each approved angle, for shots that see the person (or the place) from that side.
                entry["views"] = {}
                for view, name in views.items():
                    if not (folder / name).is_file():
                        continue
                    copied = f"bible_{bible['id']}_{name}"
                    target = os.path.join(out_dir, copied)
                    if not os.path.isfile(target):
                        shutil.copyfile(folder / name, target)
                    entry["views"][view] = copied
            cast[plural].append(entry)
    return cast


def world_block(bible: dict, *, props: Any = None) -> str:
    """The bible's world rules and recurring props as a planner block ("" when it has none).

    Props are described word for word, so every shot that shows one names
    the same object (``props`` picks them like ``project_cast`` does).
    """
    world = bible.get("world") or {}
    lines = [
        f"{label}: {world[field]}"
        for field, label in (("logline", "Series premise"), ("story", "Series story"),
                             ("stakes", "Stakes"), ("twists", "Possible twists (only when this episode's brief uses one)"),
                             ("tone", "Tone"), ("narrative_style", "Narrative style"),
                             ("look", "Look (every image and video prompt is written in it)"),
                             ("rules", "World rules"), ("forbidden", "Never"))
        if world.get(field)
    ]
    chosen = _chosen(bible.get("props") or [], props)
    if chosen:
        lines.append("Recurring props (describe each exactly like this whenever it appears):")
        lines += [
            f"- {p['name']}: {p['description']}" + (f" ({p['notes']})" if p.get("notes") else "")
            for p in chosen
        ]
    if not lines:
        return ""
    return f"Series bible \"{bible['name']}\" (every episode follows it):\n" + "\n".join(lines)
