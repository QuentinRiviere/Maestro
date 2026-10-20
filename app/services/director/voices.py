"""Character voices: references blended into one profile, reused by every cover.

A voice is one to four uploaded references (5 to 30 s each, sung rather
than spoken when possible) with a weight each, optionally tagged with the
bible character it belongs to. The worker builds its *profile*
(``services.seedvc_runner``: the references' CAMPPlus timbre vectors
blended by weight plus a prompt made of a slice of each) the first time a
cover uses it, and the controller keeps that ``profile.npz`` here so the
voice never drifts between covers. Changing the references or weights
drops the profile; the next cover rebuilds it.

Records live in ``settings/director_voices.json``; files under
``settings/voices/<id>/`` (``ref_<n>.<ext>``, ``profile.npz``).
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from . import style_lora
from .bibles import BibleError

MAX_REFERENCES = 4
MAX_UPLOAD_BYTES = 120 * 1024 * 1024
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus")
MODES = ("sing", "rap")
PROFILE_FILE = "profile.npz"
_LOCK = threading.RLock()
_settings_dir: Callable[[], str] | None = None


def _dir() -> Path:
    base = Path(_settings_dir()) if _settings_dir else Path("settings")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _records_path() -> Path:
    return _dir() / "director_voices.json"


def voices_dir() -> Path:
    folder = _dir() / "voices"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def voice_dir(voice_id: str) -> Path:
    folder = voices_dir() / voice_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def list_voices() -> list[dict]:
    with _LOCK:
        try:
            data = json.loads(_records_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = []
        return [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []


def _write(records: list[dict]) -> None:
    with _LOCK:
        tmp = _records_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(records, indent=1), encoding="utf-8")
        os.replace(tmp, _records_path())


def get_voice(voice_id: str) -> dict:
    for record in list_voices():
        if record["id"] == voice_id:
            return record
    raise BibleError("Unknown voice")


def _set(voice_id: str, **fields: Any) -> dict:
    with _LOCK:
        records = list_voices()
        for record in records:
            if record["id"] == voice_id:
                record.update(fields, updated_at=_now())
                _write(records)
                return record
    raise BibleError("Unknown voice")


def normalise_weights(weights: Any, count: int) -> list[float]:
    from services.voice_cover import voice_weights
    return voice_weights([""] * count, weights)


def create_voice(name: str, references: list[str], weights: Any = None, *, mode: str = "sing",
                 bible_id: str = "", character_id: str = "") -> dict:
    """Register a voice over reference files already under ``voice_dir``."""
    references = [str(p) for p in references if p]
    if not references:
        raise BibleError("A voice needs at least one reference")
    if len(references) > MAX_REFERENCES:
        raise BibleError(f"At most {MAX_REFERENCES} references per voice")
    for path in references:
        if not os.path.isfile(path):
            raise BibleError(f"The reference {os.path.basename(path)} is missing")
        if not path.lower().endswith(AUDIO_EXTENSIONS):
            raise BibleError(f"References must be audio files ({', '.join(AUDIO_EXTENSIONS)})")
    mode = mode if mode in MODES else "sing"
    values = normalise_weights(weights, len(references))
    record = {
        "id": "voice_" + secrets.token_hex(4), "name": style_lora._short(name, 60) or "voice", "mode": mode,
        "references": [{"file": path, "weight": weight, "name": os.path.basename(path)}
                       for path, weight in zip(references, values)],
        "profile": "", "bible_id": bible_id, "character_id": character_id, "covers": 0,
        "created_at": _now(), "updated_at": _now(),
    }
    with _LOCK:
        records = list_voices()
        records.append(record)
        _write(records)
    return record


def update_voice(voice_id: str, fields: Mapping[str, Any]) -> dict:
    """Rename, retag, change the mode or the weights; new weights drop the profile."""
    record = get_voice(voice_id)
    changes: dict[str, Any] = {}
    if "name" in fields:
        changes["name"] = style_lora._short(fields["name"], 60) or record["name"]
    if "mode" in fields and fields["mode"] in MODES:
        changes["mode"] = fields["mode"]
    for key in ("bible_id", "character_id"):
        if key in fields:
            changes[key] = str(fields[key] or "")
    if "weights" in fields:
        values = normalise_weights(fields["weights"], len(record["references"]))
        if values != [r["weight"] for r in record["references"]]:
            changes["references"] = [{**r, "weight": w} for r, w in zip(record["references"], values)]
            changes["profile"] = ""
            _remove_profile(voice_id)
    return _set(voice_id, **changes) if changes else record


def _remove_profile(voice_id: str) -> None:
    path = voices_dir() / voice_id / PROFILE_FILE
    if path.is_file():
        try:
            os.remove(path)
        except OSError:
            pass


def job_voice_params(record: Mapping[str, Any]) -> dict:
    """The worker params that name the voice: its profile when built, its references otherwise."""
    profile = str(record.get("profile") or "")
    if profile and os.path.isfile(profile):
        return {"cover_profile": profile}
    return {"cover_references": [r["file"] for r in record["references"]],
            "cover_weights": [r["weight"] for r in record["references"]]}


def keep_profile(voice_id: str, produced: str) -> dict:
    """Store the profile a cover job produced as the voice's own; returns the voice."""
    record = get_voice(voice_id)
    if record.get("profile") and os.path.isfile(record["profile"]):
        return _set(voice_id, covers=int(record.get("covers") or 0) + 1)
    target = voice_dir(voice_id) / PROFILE_FILE
    shutil.copyfile(produced, target)
    return _set(voice_id, profile=str(target), covers=int(record.get("covers") or 0) + 1)


def delete_voice(voice_id: str) -> list[dict]:
    get_voice(voice_id)
    shutil.rmtree(voices_dir() / voice_id, ignore_errors=True)
    with _LOCK:
        records = [r for r in list_voices() if r["id"] != voice_id]
        _write(records)
    return records


def register_routes(api: Any, *, settings_dir: Callable[[], str]) -> None:
    global _settings_dir
    _settings_dir = settings_dir
    base = "/api/v1/director/voices"

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BibleError as error:
            raise HTTPException(status_code=400, detail=str(error))

    @api.get(base)
    def voices_list():
        return {"voices": list_voices()}

    @api.post(base)
    async def voices_create(request: Request, references: list[UploadFile] = File(...), name: str = Form(""),
                            weights: str = Form(""), mode: str = Form("sing"), bible_id: str = Form(""),
                            character_id: str = Form("")):
        """Upload one to four voice references (with weights, comma separated) and register the voice."""
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Files too large")
        if not references or len(references) > MAX_REFERENCES:
            raise HTTPException(status_code=400, detail=f"Upload 1 to {MAX_REFERENCES} references")
        voice_id = "voice_" + secrets.token_hex(4)
        folder = voice_dir(voice_id)
        paths = []
        try:
            for index, upload in enumerate(references):
                suffix = os.path.splitext(upload.filename or "")[1].lower()
                if suffix not in AUDIO_EXTENSIONS:
                    raise HTTPException(status_code=400,
                                        detail=f"References must be audio files ({', '.join(AUDIO_EXTENSIONS)})")
                dest = folder / f"ref_{index + 1}{suffix}"
                with open(dest, "wb") as handle:
                    shutil.copyfileobj(upload.file, handle)
                paths.append(str(dest))
            values = [float(v) for v in weights.split(",") if v.strip()] if weights.strip() else None
            record = create_voice(name or Path(references[0].filename or "voice").stem, paths, values, mode=mode,
                                  bible_id=bible_id, character_id=character_id)
        except (BibleError, ValueError) as error:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(error))
        except HTTPException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        if record["id"] != voice_id:
            target = voice_dir(record["id"])
            moved = []
            for reference in record["references"]:
                dest = target / Path(reference["file"]).name
                shutil.move(reference["file"], dest)
                moved.append({**reference, "file": str(dest)})
            record = _set(record["id"], references=moved)
            shutil.rmtree(folder, ignore_errors=True)
        return {"voice": record}

    @api.get(base + "/{voice_id}")
    def voice_get(voice_id: str):
        return {"voice": call(get_voice, voice_id)}

    @api.patch(base + "/{voice_id}")
    async def voice_patch(voice_id: str, request: Request):
        try:
            body = await request.json()
        except ValueError:
            body = {}
        return {"voice": call(update_voice, voice_id, body if isinstance(body, dict) else {})}

    @api.get(base + "/{voice_id}/audio/{index}")
    def voice_audio(voice_id: str, index: int):
        record = call(get_voice, voice_id)
        if index < 1 or index > len(record["references"]):
            raise HTTPException(status_code=404, detail="No such reference")
        path = record["references"][index - 1]["file"]
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="Reference file missing")
        return FileResponse(path)

    @api.delete(base + "/{voice_id}")
    def voice_delete(voice_id: str):
        return {"voices": call(delete_voice, voice_id)}
