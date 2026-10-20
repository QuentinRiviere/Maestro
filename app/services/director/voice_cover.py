"""Song covers sung by a character's voice (controller side).

A cover takes a song (a file with vocals, rights cleared by the user) and a
short reference of the voice to sing it with, and runs the worker job in
``services.voice_cover`` (RoFormer separation, Seed-VC singing conversion,
remix). The reference is an uploaded sample, 5 to 30 seconds, sung rather
than spoken when possible; it can be tagged with the bible character whose
voice it is so clips can find it later.

Records live in ``settings/director_covers.json``; every cover's files
under ``settings/covers/<id>/`` (``song.<ext>``, ``reference.<ext>`` and
the job's outputs: ``cover.wav``, ``vocals.wav``, ``source_vocals.wav``,
``instrumental.wav``).
"""
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from . import style_lora
from .bibles import BibleError

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus")
STATES = ("queued", "running", "done", "failed")
OUTPUTS = {"cover": "cover.wav", "vocals": "vocals.wav", "source_vocals": "source_vocals.wav",
           "instrumental": "instrumental.wav"}
TIMEOUT_SEC = 3600
_LOCK = threading.RLock()
_LIVE: set[str] = set()
_SLOT = threading.Semaphore(1)
_settings_dir: Callable[[], str] | None = None
_ensure_pipeline: Callable[[], Any] | None = None
_SAFE = re.compile(r"[^a-z0-9]+")


def _dir() -> Path:
    base = Path(_settings_dir()) if _settings_dir else Path("settings")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _records_path() -> Path:
    return _dir() / "director_covers.json"


def covers_dir() -> Path:
    folder = _dir() / "covers"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def cover_dir(cover_id: str) -> Path:
    folder = covers_dir() / cover_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def list_covers() -> list[dict]:
    with _LOCK:
        try:
            data = json.loads(_records_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = []
        records = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
        for record in records:
            if record.get("status") in ("queued", "running") and record["id"] not in _LIVE:
                record["status"], record["error"] = "failed", "Interrupted"
        return records


def _write(records: list[dict]) -> None:
    with _LOCK:
        tmp = _records_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(records, indent=1), encoding="utf-8")
        os.replace(tmp, _records_path())


def get_cover(cover_id: str) -> dict:
    for record in list_covers():
        if record["id"] == cover_id:
            return record
    raise BibleError("Unknown cover")


def _set(cover_id: str, **fields: Any) -> dict:
    with _LOCK:
        records = list_covers()
        for record in records:
            if record["id"] == cover_id:
                record.update(fields, updated_at=_now())
                _write(records)
                return record
    raise BibleError("Unknown cover")


def cover_settings(raw: Mapping[str, Any] | None) -> dict:
    from services.voice_cover import cover_config
    return cover_config(raw)


def create_cover(name: str, song: str, reference: str, raw: Mapping[str, Any] | None = None, *,
                 bible_id: str = "", character_id: str = "") -> dict:
    """Register a cover over files already under ``cover_dir``; ``song`` and ``reference`` are their paths."""
    for label, path in (("song", song), ("reference", reference)):
        if not path or not os.path.isfile(path):
            raise BibleError(f"The {label} file is missing")
        if not path.lower().endswith(AUDIO_EXTENSIONS):
            raise BibleError(f"The {label} must be an audio file ({', '.join(AUDIO_EXTENSIONS)})")
    settings = cover_settings({**dict(raw or {}), "name": name})
    record = {
        "id": "cover_" + secrets.token_hex(4), "name": style_lora._short(name, 60) or "cover",
        "status": "queued", "progress": "Waiting", "error": "", "config": settings,
        "song": song, "reference": reference, "bible_id": bible_id, "character_id": character_id,
        "outputs": {}, "job_id": "", "seconds": None, "created_at": _now(), "updated_at": _now(),
    }
    with _LOCK:
        records = list_covers()
        records.append(record)
        _write(records)
    return record


def start_cover(cover_id: str, *, submit: Callable[..., list[str]] | None = None,
                jobs: Mapping[str, Any] | None = None, wait: bool = False) -> dict:
    record = get_cover(cover_id)
    if cover_id in _LIVE:
        raise BibleError("This cover is running")
    _LIVE.add(cover_id)
    try:
        record = _set(cover_id, status="queued", progress="Waiting for the worker", error="", outputs={})
    except Exception:
        _LIVE.discard(cover_id)
        raise
    thread = threading.Thread(target=_run, args=(cover_id, submit, jobs), name=f"cover-{cover_id}", daemon=True)
    thread.start()
    if wait:
        thread.join()
        return get_cover(cover_id)
    return record


def _director():
    if _ensure_pipeline is not None:
        _ensure_pipeline()
    from services import director_pipeline
    return director_pipeline


def _run(cover_id: str, submit, jobs) -> None:
    started = time.time()
    try:
        with _SLOT:
            if cover_id not in _LIVE:
                return
            record = get_cover(cover_id)
            submit = submit or _director()._submit_and_wait
            jobs = jobs if jobs is not None else _director()._jobs
            job_id = uuid.uuid4().hex[:8]
            folder = cover_dir(cover_id)
            params = {
                "model_type": style_lora.IMAGE_MODEL, "prompt": "", "image_mode": 1, "generation_mode": "image",
                "voice_cover": {**record["config"], "name": cover_id}, "cover_song": record["song"],
                "cover_reference": record["reference"], "_director_cost_stage": "voice_cover",
            }
            _set(cover_id, status="running", job_id=job_id, progress="Submitting to the worker")
            done = threading.Event()
            follower = threading.Thread(
                target=style_lora._follow_generic, args=(job_id, jobs, done, lambda **f: _set(cover_id, **f)),
                daemon=True)
            follower.start()
            try:
                outputs = submit(params, timeout_s=TIMEOUT_SEC, out_dir=str(folder), job_id=job_id)
            finally:
                done.set()
                follower.join(timeout=5)
            made = {key: name for key, name in OUTPUTS.items() if name in (outputs or []) and (folder / name).is_file()}
            if "cover" not in made:
                job = jobs.get(job_id) or {}
                raise BibleError("Cancelled" if job.get("status") == "cancelled" else "The worker returned no cover")
            _set(cover_id, status="done", progress="", outputs=made, seconds=round(time.time() - started))
            print(f"[Voice cover] {cover_id}: {made['cover']} in {round(time.time() - started)} s")
    except BibleError as error:
        try:
            _set(cover_id, status="failed", error=str(error)[:400], progress="")
        except Exception:  # noqa: BLE001
            pass
    except Exception as error:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        try:
            _set(cover_id, status="failed", error=str(error)[:400], progress="")
        except Exception:  # noqa: BLE001
            pass
    finally:
        _LIVE.discard(cover_id)


def delete_cover(cover_id: str) -> list[dict]:
    if cover_id in _LIVE:
        raise BibleError("This cover is running; cancel it first")
    get_cover(cover_id)
    shutil.rmtree(covers_dir() / cover_id, ignore_errors=True)
    with _LOCK:
        records = [r for r in list_covers() if r["id"] != cover_id]
        _write(records)
    return records


def audio_path(cover_id: str, which: str) -> Path:
    record = get_cover(cover_id)
    if which == "song":
        path = Path(record["song"])
    elif which == "reference":
        path = Path(record["reference"])
    elif which in OUTPUTS:
        path = cover_dir(cover_id) / OUTPUTS[which]
    else:
        raise BibleError("Unknown audio")
    if not path.is_file():
        raise BibleError("No such file yet")
    return path


def register_routes(api: Any, *, settings_dir: Callable[[], str], ensure_pipeline: Callable[[], Any] | None = None) -> None:
    global _settings_dir, _ensure_pipeline
    _settings_dir = settings_dir
    _ensure_pipeline = ensure_pipeline
    base = "/api/v1/director/covers"

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BibleError as error:
            raise HTTPException(status_code=400, detail=str(error))

    def save_upload(folder: Path, upload: UploadFile, stem: str) -> str:
        suffix = os.path.splitext(upload.filename or "")[1].lower()
        if suffix not in AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"{stem}: upload an audio file ({', '.join(AUDIO_EXTENSIONS)})")
        dest = folder / f"{stem}{suffix}"
        with open(dest, "wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        return str(dest)

    @api.get(base)
    def covers_list():
        return {"covers": list_covers()}

    @api.post(base)
    async def covers_create(request: Request, song: UploadFile = File(...), reference: UploadFile = File(...),
                            name: str = Form(""), steps: int = Form(30), cfg_rate: float = Form(0.7),
                            semitone_shift: int = Form(0), auto_f0: bool = Form(False),
                            vocal_gain_db: float = Form(0.0), bible_id: str = Form(""),
                            character_id: str = Form(""), run: bool = Form(True)):
        """Upload a song and a voice reference, then sing the song with that voice."""
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Files too large")
        cover_id = "cover_" + secrets.token_hex(4)
        folder = cover_dir(cover_id)
        try:
            song_path = save_upload(folder, song, "song")
            reference_path = save_upload(folder, reference, "reference")
            record = create_cover(name or Path(song.filename or "cover").stem, song_path, reference_path,
                                  {"steps": steps, "cfg_rate": cfg_rate, "semitone_shift": semitone_shift,
                                   "auto_f0": auto_f0, "vocal_gain_db": vocal_gain_db},
                                  bible_id=bible_id, character_id=character_id)
        except BibleError as error:
            shutil.rmtree(folder, ignore_errors=True)
            raise HTTPException(status_code=400, detail=str(error))
        except HTTPException:
            shutil.rmtree(folder, ignore_errors=True)
            raise
        # The record's own folder is named after its id: move the files there.
        if record["id"] != cover_id:
            target = cover_dir(record["id"])
            for key in ("song", "reference"):
                moved = target / Path(record[key]).name
                shutil.move(record[key], moved)
                record[key] = str(moved)
            _set(record["id"], song=record["song"], reference=record["reference"])
            shutil.rmtree(folder, ignore_errors=True)
        if run:
            record = call(start_cover, record["id"])
        return {"cover": record}

    @api.get(base + "/{cover_id}")
    def cover_get(cover_id: str):
        return {"cover": call(get_cover, cover_id)}

    @api.post(base + "/{cover_id}/run")
    def cover_run(cover_id: str):
        return {"cover": call(start_cover, cover_id)}

    @api.post(base + "/{cover_id}/cancel")
    def cover_cancel(cover_id: str):
        record = call(get_cover, cover_id)
        if cover_id in _LIVE and record.get("job_id"):
            from services.director_pipeline import _jobs
            from services.job_lifecycle import request_cancel
            job = (_jobs or {}).get(record["job_id"])
            if job is not None:
                request_cancel(job)
        return {"cover": call(get_cover, cover_id)}

    @api.get(base + "/{cover_id}/audio/{which}")
    def cover_audio(cover_id: str, which: str):
        return FileResponse(str(call(audio_path, cover_id, which)))

    @api.delete(base + "/{cover_id}")
    def cover_delete(cover_id: str):
        return {"covers": call(delete_cover, cover_id)}
