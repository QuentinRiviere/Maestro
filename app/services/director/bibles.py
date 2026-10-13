"""Bibles: the approved people, places and look of a series, reused across projects.

A Director cast belongs to one project: its images live in that project's
output folder and the next episode casts again from scratch, so faces, sets
and the visual style drift between episodes. A bible keeps them once: the
characters and locations (descriptions and reference images), the style
contract and the world rules. Every episode of a series, or every mini-scene
of a short film, starts from the same bible.

A bible is one folder: ``settings/director_bibles/<id>/bible.json`` and its
image files, so it does not depend on any project's output folder. Applying a
bible to a project copies the chosen entries into the project's cast; the
project records the bible id and revision it was made from.
"""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
import time
from pathlib import Path
from typing import Any

from services.director import casting

BIBLES_DIR = Path(__file__).resolve().parents[2] / "settings" / "director_bibles"
BIBLE_FILE = "bible.json"
# A series keeps more people and places than one project casts.
MAX_CHARACTERS = 24
MAX_LOCATIONS = 24
WORLD_FIELDS = ("logline", "tone", "rules", "forbidden")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
_LOCK = threading.Lock()
_SLUG = re.compile(r"[^a-z0-9]+")
_ID = re.compile(r"[a-z0-9_]{1,40}")


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


def _entries(raw: Any, kind: str, limit: int) -> list[dict]:
    """Bible entries in the cast shape, names unique (the first one wins)."""
    plural = "characters" if kind == "character" else "locations"
    normalized = casting.normalize_cast({plural: raw or []}, max_characters=limit, max_locations=limit)[plural]
    seen, entries = set(), []
    for entry in normalized:
        key = entry["name"].casefold()
        if key not in seen:
            seen.add(key)
            entries.append(entry)
    return entries


def _world(raw: Any) -> dict:
    raw = raw if isinstance(raw, dict) else {}
    return {field: _text(raw.get(field), 2000) for field in WORLD_FIELDS}


def normalize_bible(raw: Any) -> dict:
    """The stored fields of a bible, bounded; ids and timestamps are kept as given."""
    raw = raw if isinstance(raw, dict) else {}
    name = _text(raw.get("name"), 80)
    if not name:
        raise BibleError("A bible needs a name")
    style = raw.get("style")
    return {
        "id": str(raw.get("id") or ""),
        "name": name,
        "description": _text(raw.get("description"), 600),
        "style": copy.deepcopy(style) if isinstance(style, dict) and style else None,
        "world": _world(raw.get("world")),
        "characters": _entries(raw.get("characters"), "character", MAX_CHARACTERS),
        "locations": _entries(raw.get("locations"), "location", MAX_LOCATIONS),
    }


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


def _drop_missing_images(bible: dict) -> None:
    """An entry only names an image its bible folder holds."""
    folder = _folder(bible["id"])
    for plural in ("characters", "locations"):
        for entry in bible[plural]:
            image = os.path.basename(entry.get("image") or "")
            entry["image"] = image if image and (folder / image).is_file() else None


def _new_id(name: str) -> str:
    base = _SLUG.sub("_", name.lower()).strip("_")[:32] or "bible"
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
                "characters": [c["name"] for c in bible.get("characters") or []],
                "locations": [l["name"] for l in bible.get("locations") or []],
            })
    return sorted(summaries, key=lambda item: item["updated_at"], reverse=True)


def get_bible(bible_id: str) -> dict | None:
    return _read(bible_id)


def save_bible(raw: dict, *, bible_id: str | None = None) -> dict:
    """Create (no id) or overwrite (given id) a bible's fields; every save is a new revision."""
    bible = normalize_bible(raw)
    with _LOCK:
        if bible_id is None:
            BIBLES_DIR.mkdir(parents=True, exist_ok=True)
            bible["id"] = _new_id(bible["name"])
            previous: dict = {}
        else:
            previous = _read(bible_id) or {}
            if not previous:
                raise BibleError("Unknown bible")
            bible["id"] = bible_id
        now = _now()
        bible["revision"] = int(previous.get("revision") or 0) + 1
        bible["created_at"] = previous.get("created_at") or now
        bible["updated_at"] = now
        _folder(bible["id"]).mkdir(parents=True, exist_ok=True)
        _drop_missing_images(bible)
        _write(bible)
    return bible


def delete_bible(bible_id: str) -> bool:
    with _LOCK:
        folder = _folder(bible_id)
        if not (folder / BIBLE_FILE).is_file():
            return False
        shutil.rmtree(folder)
        return True


def image_path(bible_id: str, name: str) -> Path | None:
    """The file of one bible image, or None when the name is not one of its images."""
    name = os.path.basename(name or "")
    if not name.lower().endswith(IMAGE_SUFFIXES):
        return None
    path = _folder(bible_id) / name
    return path if path.is_file() else None


def _import_image(folder: Path, source: str, prefix: str) -> str | None:
    if not source or not os.path.isfile(source):
        return None
    suffix = os.path.splitext(source)[1].lower()
    if suffix not in IMAGE_SUFFIXES:
        return None
    name = f"{prefix}_{int(time.time() * 1000)}{suffix}"
    shutil.copyfile(source, folder / name)
    return name


def add_cast(cast: dict, image_dirs: list[str], *, bible_id: str | None = None, name: str = "",
             style: dict | None = None) -> dict:
    """Save a project's cast (and style) into a new or an existing bible.

    ``image_dirs`` are the folders the cast images are looked up in (the
    project's output folder first). Entries of an existing bible with the same
    name are replaced; the others are kept. The style is only set when the
    bible has none yet.
    """
    cast = casting.normalize_cast(cast, max_characters=MAX_CHARACTERS, max_locations=MAX_LOCATIONS)
    if bible_id:
        bible = get_bible(bible_id)
        if not bible:
            raise BibleError("Unknown bible")
    else:
        bible = save_bible({"name": name or "Untitled bible", "style": style})
    folder = _folder(bible["id"])
    for plural, kind in (("characters", "char"), ("locations", "loc")):
        merged = {entry["name"].casefold(): entry for entry in bible.get(plural) or []}
        for entry in cast[plural]:
            entry = dict(entry)
            image = os.path.basename(entry.get("image") or "")
            source = next((os.path.join(d, image) for d in image_dirs if d and image
                           and os.path.isfile(os.path.join(d, image))), "")
            slug = _SLUG.sub("_", entry["name"].lower()).strip("_")[:24] or kind
            entry["image"] = _import_image(folder, source, f"{kind}_{slug}")
            merged[entry["name"].casefold()] = entry
        bible[plural] = list(merged.values())
    if style and not bible.get("style"):
        bible["style"] = style
    return save_bible(bible, bible_id=bible["id"])


def _chosen(entries: list[dict], names: Any) -> list[dict]:
    if not isinstance(names, list) or not names:
        return list(entries)
    wanted = {str(n).strip().casefold() for n in names if str(n).strip()}
    return [entry for entry in entries if entry["name"].casefold() in wanted]


def project_cast(bible: dict, out_dir: str, *, characters: Any = None, locations: Any = None) -> dict:
    """The cast of one project made from a bible, its images copied into ``out_dir``.

    The Director keeps cast images as file names in the project's output
    folder; copying them keeps every cast path (planning, review, saved
    projects, Open & Edit) unchanged. ``characters`` and ``locations`` pick
    entries by name; none picks every entry.
    """
    folder = _folder(bible["id"])
    os.makedirs(out_dir, exist_ok=True)
    cast: dict[str, list] = {"characters": [], "locations": []}
    for plural, names in (("characters", characters), ("locations", locations)):
        for entry in _chosen(bible.get(plural) or [], names):
            entry = dict(entry)
            image = entry.get("image")
            if image and (folder / image).is_file():
                copied = f"bible_{bible['id']}_r{bible.get('revision', 1)}_{image}"
                target = os.path.join(out_dir, copied)
                if not os.path.isfile(target):
                    shutil.copyfile(folder / image, target)
                entry["image"] = copied
            else:
                entry["image"] = None
            cast[plural].append(entry)
    return cast


def world_block(bible: dict) -> str:
    """The bible's world rules as a planner block ("" when it has none)."""
    world = bible.get("world") or {}
    lines = [
        f"{label}: {world[field]}"
        for field, label in (("logline", "Series premise"), ("tone", "Tone"),
                             ("rules", "World rules"), ("forbidden", "Never"))
        if world.get(field)
    ]
    if not lines:
        return ""
    return f"Series bible \"{bible['name']}\" (every episode follows it):\n" + "\n".join(lines)
