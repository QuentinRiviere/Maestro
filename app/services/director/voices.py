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
import uuid
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
TRAIN_TIMEOUT_SEC = 6 * 3600
_LOCK = threading.RLock()
_TRAINING: set[str] = set()
_SLOT = threading.Semaphore(1)
_ensure_pipeline: Callable[[], Any] | None = None
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


def _rvc_state(record: dict) -> dict:
    rvc = dict(record.get("rvc") or {})
    if rvc.get("status") in ("queued", "running") and record["id"] not in _TRAINING:
        rvc["status"], rvc["error"] = "failed", "Interrupted"
    return rvc


def list_voices() -> list[dict]:
    with _LOCK:
        try:
            data = json.loads(_records_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = []
        records = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
        for record in records:
            if record.get("rvc"):
                record["rvc"] = _rvc_state(record)
        return records


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




def rvc_ready(record: Mapping[str, Any]) -> bool:
    """True when the voice carries a trained RVC model on disk."""
    rvc = record.get("rvc") or {}
    return rvc.get("status") == "done" and bool(rvc.get("model")) and os.path.isfile(rvc["model"])


def resolve_engine(record: Mapping[str, Any] | None, engine: str = "auto") -> str:
    """``auto`` takes the trained RVC model when the voice has one, Seed-VC otherwise."""
    if engine == "rvc":
        if not (record and rvc_ready(record)):
            raise BibleError("This voice has no trained RVC model yet")
        return "rvc"
    if engine == "seedvc":
        return "seedvc"
    return "rvc" if record and rvc_ready(record) else "seedvc"


def job_voice_params(record: Mapping[str, Any], engine: str = "seedvc") -> dict:
    """The worker params that name the voice: its RVC pair, its profile when built, its references otherwise."""
    if engine == "rvc" and rvc_ready(record):
        rvc = record["rvc"]
        # The references ride along for the judge's target timbre; the conversion itself uses the model.
        return {"cover_rvc_model": rvc["model"], "cover_rvc_index": rvc.get("index") or "",
                "cover_references": [r["file"] for r in record["references"]],
                "cover_weights": [r["weight"] for r in record["references"]]}
    profile = str(record.get("profile") or "")
    if profile and os.path.isfile(profile):
        return {"cover_profile": profile}
    return {"cover_references": [r["file"] for r in record["references"]],
            "cover_weights": [r["weight"] for r in record["references"]]}


QUIET_PEAK = 0.1


def has_voice(path: str, *, peak: float = QUIET_PEAK) -> bool:
    """False for a stem that is all but silent (a song the separator found no vocals in)."""
    try:
        import numpy as np
        import soundfile as sf

        data, _ = sf.read(path, dtype="float32", always_2d=True)
        return bool(data.size) and float(np.abs(data).max()) >= peak
    except Exception:  # noqa: BLE001 - unreadable files are not training material either
        return False


def training_dataset(record: Mapping[str, Any]) -> list[str]:
    """The recordings an RVC training learns from: the references and the vocals Seed-VC sang for this voice."""
    from . import voice_cover

    files = [r["file"] for r in record["references"] if os.path.isfile(r["file"])]
    for cover in voice_cover.list_covers():
        if cover.get("voice_id") != record["id"] or cover.get("status") != "done":
            continue
        if (cover.get("config") or {}).get("engine", "seedvc") != "seedvc":
            continue
        vocals = voice_cover.cover_dir(cover["id"]) / voice_cover.OUTPUTS["vocals"]
        if vocals.is_file() and has_voice(str(vocals)):
            files.append(str(vocals))
    return files


def _director():
    if _ensure_pipeline is not None:
        _ensure_pipeline()
    from services import director_pipeline
    return director_pipeline


def start_training(voice_id: str, raw: Mapping[str, Any] | None = None, *,
                   submit: Callable[..., list[str]] | None = None, jobs: Mapping[str, Any] | None = None,
                   wait: bool = False) -> dict:
    """Train the voice's RVC model on the worker (``voice_train`` job) from its references and sung covers."""
    from services.voice_train import train_config

    record = get_voice(voice_id)
    if voice_id in _TRAINING:
        raise BibleError("This voice is training")
    dataset = training_dataset(record)
    if not dataset:
        raise BibleError("Nothing to train on: the voice needs references or sung covers")
    config = train_config({**dict(raw or {}), "name": record["id"]})
    _TRAINING.add(voice_id)
    try:
        _set(voice_id, rvc={"status": "queued", "progress": "Waiting for the worker", "error": "", "config": config,
                            "dataset": len(dataset), "model": "", "index": "", "job_id": "", "seconds": None,
                            "started_at": _now()})
    except Exception:
        _TRAINING.discard(voice_id)
        raise
    thread = threading.Thread(target=_train, args=(voice_id, dataset, submit, jobs), name=f"voice-{voice_id}",
                              daemon=True)
    thread.start()
    if wait:
        thread.join()
    return get_voice(voice_id)


def _set_rvc(voice_id: str, **fields: Any) -> dict:
    record = get_voice(voice_id)
    return _set(voice_id, rvc={**(record.get("rvc") or {}), **fields})


def _train(voice_id: str, dataset: list[str], submit, jobs) -> None:
    started = time.time()
    try:
        with _SLOT:
            if voice_id not in _TRAINING:
                return
            record = get_voice(voice_id)
            submit = submit or _director()._submit_and_wait
            jobs = jobs if jobs is not None else _director()._jobs
            job_id = uuid.uuid4().hex[:8]
            folder = voice_dir(voice_id)
            params = {
                "model_type": style_lora.IMAGE_MODEL, "prompt": "", "image_mode": 1, "generation_mode": "image",
                "voice_train": record["rvc"]["config"], "train_dataset": dataset, "_director_cost_stage": "voice_train",
            }
            _set_rvc(voice_id, status="running", job_id=job_id, progress="Submitting to the worker")
            done = threading.Event()
            follower = threading.Thread(
                target=style_lora._follow_generic, args=(job_id, jobs, done, lambda **f: _set_rvc(voice_id, **f)),
                daemon=True)
            follower.start()
            try:
                outputs = submit(params, timeout_s=TRAIN_TIMEOUT_SEC, out_dir=str(folder), job_id=job_id)
            finally:
                done.set()
                follower.join(timeout=5)
            model = next((folder / n for n in (outputs or []) if n.endswith(".pth") and (folder / n).is_file()), None)
            index = next((folder / n for n in (outputs or []) if n.endswith(".index") and (folder / n).is_file()), None)
            if model is None:
                job = jobs.get(job_id) or {}
                raise BibleError("Cancelled" if job.get("status") == "cancelled" else "The worker returned no model")
            _set_rvc(voice_id, status="done", progress="", model=str(model), index=str(index) if index else "",
                     seconds=round(time.time() - started))
            print(f"[Voice training] {voice_id}: {model.name} in {round(time.time() - started)} s")
    except BibleError as error:
        try:
            _set_rvc(voice_id, status="failed", error=str(error)[:400], progress="")
        except Exception:  # noqa: BLE001
            pass
    except Exception as error:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        try:
            _set_rvc(voice_id, status="failed", error=str(error)[:400], progress="")
        except Exception:  # noqa: BLE001
            pass
    finally:
        _TRAINING.discard(voice_id)


def cancel_training(voice_id: str) -> dict:
    record = get_voice(voice_id)
    if voice_id not in _TRAINING:
        raise BibleError("This voice is not training")
    _TRAINING.discard(voice_id)
    job_id = (record.get("rvc") or {}).get("job_id")
    if job_id:
        from services.job_lifecycle import request_cancel
        job = _director()._jobs.get(job_id)
        if job:
            request_cancel(job)
    return _set_rvc(voice_id, status="failed", error="Cancelled", progress="")


def keep_profile(voice_id: str, produced: str) -> dict:
    """Store the profile a cover job produced as the voice's own; returns the voice."""
    record = get_voice(voice_id)
    if record.get("profile") and os.path.isfile(record["profile"]):
        return _set(voice_id, covers=int(record.get("covers") or 0) + 1)
    target = voice_dir(voice_id) / PROFILE_FILE
    shutil.copyfile(produced, target)
    return _set(voice_id, profile=str(target), covers=int(record.get("covers") or 0) + 1)


def delete_voice(voice_id: str) -> list[dict]:
    if voice_id in _TRAINING:
        raise BibleError("This voice is training; cancel it first")
    get_voice(voice_id)
    shutil.rmtree(voices_dir() / voice_id, ignore_errors=True)
    with _LOCK:
        records = [r for r in list_voices() if r["id"] != voice_id]
        _write(records)
    return records


def register_routes(api: Any, *, settings_dir: Callable[[], str], ensure_pipeline: Callable[[], Any] | None = None) -> None:
    global _settings_dir, _ensure_pipeline
    _settings_dir = settings_dir
    _ensure_pipeline = ensure_pipeline
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

    @api.post(base + "/{voice_id}/train")
    async def voice_train(voice_id: str, request: Request):
        """Train the voice's RVC model (epochs, batch_size, sample_rate in the JSON body)."""
        try:
            body = await request.json()
        except ValueError:
            body = {}
        return {"voice": call(start_training, voice_id, body if isinstance(body, dict) else {})}

    @api.post(base + "/{voice_id}/train/cancel")
    def voice_train_cancel(voice_id: str):
        return {"voice": call(cancel_training, voice_id)}

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
