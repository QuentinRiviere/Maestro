"""User annotations on generated outputs, stored in each output's ``.meta.json``.

An annotation is a free-text note plus short tags ("hand morph", "keep"),
kept under the sidecar's ``annotation`` key so it travels with the file
(moves, workspace copies) and shows up in gallery search. An empty note with
no tags removes the key.
"""
from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Mapping, Optional

MAX_NOTE_LENGTH = 4000
MAX_TAGS = 20
MAX_TAG_LENGTH = 40

_lock = threading.Lock()


def normalize(body: Mapping[str, Any] | None) -> dict:
    """``{"note", "tags"}`` cleaned: trimmed note, unique non-empty tags in order.

    Raises ValueError when the note or the tags are too long or malformed.
    """
    body = body or {}
    note = body.get("note")
    if note is None:
        note = ""
    if not isinstance(note, str):
        raise ValueError("note must be a string")
    note = note.strip()
    if len(note) > MAX_NOTE_LENGTH:
        raise ValueError(f"note is longer than {MAX_NOTE_LENGTH} characters")
    raw_tags = body.get("tags") or []
    if isinstance(raw_tags, str):
        raw_tags = raw_tags.split(",")
    if not isinstance(raw_tags, (list, tuple)):
        raise ValueError("tags must be a list of strings")
    tags: list[str] = []
    seen: set[str] = set()
    for raw in raw_tags:
        if not isinstance(raw, str):
            raise ValueError("tags must be a list of strings")
        tag = " ".join(raw.split())
        if not tag or tag.casefold() in seen:
            continue
        if len(tag) > MAX_TAG_LENGTH:
            raise ValueError(f"tag {tag[:20]!r}… is longer than {MAX_TAG_LENGTH} characters")
        seen.add(tag.casefold())
        tags.append(tag)
    if len(tags) > MAX_TAGS:
        raise ValueError(f"at most {MAX_TAGS} tags")
    return {"note": note, "tags": tags}


def write(meta_path: str, body: Mapping[str, Any] | None) -> Optional[dict]:
    """Store the annotation in ``meta_path``; the stored record, or None once cleared.

    Creates a minimal sidecar when there is none (the gallery then still reads
    the settings embedded in the media file). Raises ValueError on a bad body
    and OSError when the sidecar cannot be read or written.
    """
    annotation = normalize(body)
    with _lock:
        sidecar: dict[str, Any] = {}
        if os.path.isfile(meta_path):
            with open(meta_path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                sidecar = loaded
        else:
            sidecar["created_at"] = time.time()
        if annotation["note"] or annotation["tags"]:
            record = {**annotation, "updated_at": time.time()}
            sidecar["annotation"] = record
        else:
            record = None
            if "annotation" not in sidecar:
                return None
            sidecar.pop("annotation")
        temporary = f"{meta_path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(sidecar, handle, indent=2, default=str)
            os.replace(temporary, meta_path)
        except OSError:
            try:
                os.remove(temporary)
            except OSError:
                pass
            raise
    return record

