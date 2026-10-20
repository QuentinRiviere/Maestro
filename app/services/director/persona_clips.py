"""Persona clips: one fixed character, a signature gesture, one place per clip, silent and vertical (controller side).

The short-form format of the AI personas (one recurring character seen in a
different place every clip, always doing the same move, no speech): a clip
takes a bible character, a place and a gesture, and runs two worker jobs
under ``settings/persona_clips/<id>/``:

1. ``start.png``: the Director's image editor paints the character
   (reference 1) at the start of the gesture inside the place (reference 2),
   framed for the orientation, with the bible look.
2. ``clip.mp4``: LTX-2.5 image-to-video performs the gesture in one take.
   The audio request is "A" without a soundtrack, which WanGP fills with a
   silent track (the model's silent-movie mode), so the mouth stays shut and
   the clip carries no sound. ``audio: "model"`` lets LTX invent a
   soundtrack instead.

Records live in ``settings/director_persona_clips.json`` (``status``,
``progress``, ``error``, ``settings``, ``outputs``, ``job_id``, ``prompt``).
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

from . import bibles, cover_video, style_lora, voice_cover
from .bibles import BibleError

DEFAULT_SECONDS = 6.0
AUDIO_MODES = ("silent", "model")
OUTPUTS = {"start": "start.png", "clip": "clip.mp4"}
_LOCK = threading.RLock()
_LIVE: set[str] = set()
_SLOT = threading.Semaphore(1)
_settings_dir: Callable[[], str] | None = None


def _dir() -> Path:
    base = Path(_settings_dir()) if _settings_dir else Path("settings")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _records_path() -> Path:
    return _dir() / "director_persona_clips.json"


def clips_dir() -> Path:
    folder = _dir() / "persona_clips"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def clip_dir(clip_id: str) -> Path:
    folder = clips_dir() / clip_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def list_clips() -> list[dict]:
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


def get_clip(clip_id: str) -> dict:
    for record in list_clips():
        if record["id"] == clip_id:
            return record
    raise BibleError("Unknown clip")


def _set(clip_id: str, **fields: Any) -> dict:
    with _LOCK:
        records = list_clips()
        for record in records:
            if record["id"] == clip_id:
                record.update(fields, updated_at=voice_cover._now())
                _write(records)
                return record
    raise BibleError("Unknown clip")


def clip_settings(raw: Mapping[str, Any] | None) -> dict:
    raw = dict(raw or {})
    gesture = str(raw.get("gesture") or "").strip()[:500]
    if not gesture:
        raise BibleError("The clip needs the persona's gesture")
    audio = str(raw.get("audio") or "silent").strip().lower()
    if audio not in AUDIO_MODES:
        raise BibleError(f"Unknown audio mode (one of {', '.join(AUDIO_MODES)})")
    base = cover_video.video_settings({**raw, "seconds": raw.get("seconds") or DEFAULT_SECONDS, "start_sec": 0})
    if base["frames"] > cover_video.MAX_FRAMES:
        raise BibleError(f"A persona clip fits one window: at most {cover_video.seconds_for(cover_video.MAX_FRAMES):.0f} s")
    return {
        "character_id": base["character_id"], "location_id": base["location_id"], "gesture": gesture,
        "orientation": base["orientation"], "frames": base["frames"], "seconds": base["seconds"],
        "seed": base["seed"], "model_type": base["model_type"], "audio": audio,
        "prompt": base["prompt"], "start_prompt": base["start_prompt"],
        "place": str(raw.get("place") or "").strip()[:300],
    }


def start_prompt(character: Mapping[str, Any], location: Mapping[str, Any] | None, settings: Mapping[str, Any],
                 look: str = "") -> str:
    framing = "vertical 9:16 medium shot" if settings["orientation"] == "portrait" else "16:9 medium shot"
    if location:
        place = f" inside the exact place of reference 2 ({location.get('name') or 'the place'})"
    elif settings.get("place"):
        place = f" in {settings['place']}"
    else:
        place = ""
    text = (f"{character.get('name') or 'The character'} from reference 1, same face, hair and clothes, stands{place} "
            f"at the first beat of this move: {settings['gesture']}. Looking at the camera, {framing}, natural light, "
            "photographic, no text.")
    return f"{text} {look}".strip()


def clip_prompt(character: Mapping[str, Any], settings: Mapping[str, Any]) -> str:
    name = character.get("name") or "The character"
    body = settings.get("prompt") or (
        f"{name} performs the signature move in one continuous take: {settings['gesture']}. Facing the camera, "
        "steady handheld framing, the place unchanged, no camera cut, no text.")
    silence = " Silent clip: nobody speaks, the mouth stays closed." if settings.get("audio") == "silent" else ""
    return f"{body}{silence}"


def clip_job(prompt: str, start_image: str, settings: Mapping[str, Any]) -> dict:
    """LTX-2.5 image-to-video; "A" without a soundtrack makes WanGP condition on silence."""
    job = {
        "model_type": settings["model_type"], "prompt": prompt, "image_mode": 0, "image_prompt_type": "S",
        "image_start": start_image, "generation_mode": "video", "num_inference_steps": 8, "guidance_scale": 1.0,
        "input_video_strength": 1.0, "resolution": cover_video.ORIENTATIONS[settings["orientation"]][0],
        "video_length": int(settings["frames"]), "sliding_window_size": int(settings["frames"]),
        "seed": int(settings.get("seed", -1)), "settings_version": 2.52, "repeat_generation": 1, "negative_prompt": "",
        "activated_loras": [], "loras_multipliers": "", "_director_cost_stage": "persona_clip",
    }
    job["audio_prompt_type"] = "A" if settings.get("audio") == "silent" else ""
    return job


def create_clip(name: str, raw: Mapping[str, Any] | None, *, bible_id: str, start_image: str = "") -> dict:
    settings = clip_settings(raw)
    bible = bibles.get_bible(bible_id)
    if not bible:
        raise BibleError("Unknown bible")
    if not settings["character_id"]:
        raise BibleError("The clip needs a character")
    kind, character = bibles.find_element(bible, settings["character_id"])
    if kind != "character":
        raise BibleError("The persona must be a character")
    if not start_image and not style_lora.element_view(bible_id, character, ""):
        raise BibleError("The character needs an approved image")
    if settings["location_id"]:
        kind, location = bibles.find_element(bible, settings["location_id"])
        if kind != "location" or (not start_image and not style_lora.element_view(bible_id, location, "")):
            raise BibleError("The place needs an approved image")
    if start_image and not os.path.isfile(start_image):
        raise BibleError("The start image is missing")
    record = {
        "id": "clip_" + secrets.token_hex(4), "name": style_lora._short(name, 60) or character.get("name") or "clip",
        "status": "queued", "progress": "Waiting", "error": "", "settings": settings, "bible_id": bible_id,
        "outputs": {}, "job_id": "", "prompt": "", "seconds": None,
        "created_at": voice_cover._now(), "updated_at": voice_cover._now(),
    }
    if start_image:
        dest = clip_dir(record["id"]) / OUTPUTS["start"]
        if os.path.abspath(start_image) != os.path.abspath(str(dest)):
            shutil.copyfile(start_image, dest)
        record["outputs"] = {"start": OUTPUTS["start"]}
    with _LOCK:
        records = list_clips()
        records.append(record)
        _write(records)
    return record


def start_clip(clip_id: str, *, submit: Callable[..., list[str]] | None = None,
               jobs: Mapping[str, Any] | None = None, wait: bool = False) -> dict:
    record = get_clip(clip_id)
    if clip_id in _LIVE:
        raise BibleError("This clip is running")
    _LIVE.add(clip_id)
    try:
        outputs = {k: v for k, v in (record.get("outputs") or {}).items() if k == "start"}
        _set(clip_id, status="queued", progress="Waiting for the worker", error="", outputs=outputs, job_id="")
    except Exception:
        _LIVE.discard(clip_id)
        raise
    if wait:
        _run(clip_id, submit, jobs)
    else:
        threading.Thread(target=_run, args=(clip_id, submit, jobs), daemon=True).start()
    return get_clip(clip_id)


def _run(clip_id: str, submit, jobs) -> None:
    started = time.time()
    try:
        with _SLOT:
            if clip_id not in _LIVE:
                return
            record = get_clip(clip_id)
            settings = record["settings"]
            pipeline = voice_cover._director()
            submit = submit or pipeline._submit_and_wait
            jobs = jobs if jobs is not None else pipeline._jobs
            folder = clip_dir(clip_id)
            made = dict(record.get("outputs") or {})
            update = lambda **f: _set(clip_id, **f)  # noqa: E731
            bible = bibles.get_bible(record["bible_id"]) or {}
            character = bibles.find_element(bible, settings["character_id"])[1]
            location = bibles.find_element(bible, settings["location_id"])[1] if settings.get("location_id") else None
            if "start" not in made or not (folder / made["start"]).is_file():
                refs = [style_lora.element_view(record["bible_id"], character, "")]
                if location:
                    refs.append(style_lora.element_view(record["bible_id"], location, ""))
                look = str((bible.get("world") or {}).get("look") or "").strip()
                prompt = settings.get("start_prompt") or start_prompt(character, location, settings, look)
                params = cover_video.start_job(prompt, refs, settings["orientation"], pipeline, settings.get("seed", -1))
                params["_director_cost_stage"] = "persona_clip_start"
                outputs = cover_video.run_stage(submit, jobs, params, folder, "Start image", update)
                shutil.move(str(folder / outputs[0]), str(folder / OUTPUTS["start"]))
                made["start"] = OUTPUTS["start"]
                _set(clip_id, outputs=made, progress="Start image ready")
            prompt = clip_prompt(character, settings)
            outputs = cover_video.run_stage(submit, jobs, clip_job(prompt, str(folder / made["start"]), settings),
                                            folder, "Clip", update)
            shutil.move(str(folder / outputs[0]), str(folder / OUTPUTS["clip"]))
            made["clip"] = OUTPUTS["clip"]
            _set(clip_id, status="done", progress="", outputs=made, prompt=prompt,
                 seconds=round(time.time() - started))
            print(f"[Persona clip] {clip_id}: {OUTPUTS['clip']} in {round(time.time() - started)} s")
    except BibleError as error:
        try:
            _set(clip_id, status="failed", error=str(error)[:400], progress="")
        except Exception:  # noqa: BLE001
            pass
    except Exception as error:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        try:
            _set(clip_id, status="failed", error=str(error)[:400], progress="")
        except Exception:  # noqa: BLE001
            pass
    finally:
        _LIVE.discard(clip_id)


def cancel_clip(clip_id: str) -> dict:
    record = get_clip(clip_id)
    if clip_id not in _LIVE:
        raise BibleError("This clip is not running")
    _LIVE.discard(clip_id)
    if record.get("job_id"):
        from services.job_lifecycle import request_cancel
        job = voice_cover._director()._jobs.get(record["job_id"])
        if job:
            request_cancel(job)
    return _set(clip_id, status="failed", error="Cancelled", progress="")


def delete_clip(clip_id: str) -> list[dict]:
    if clip_id in _LIVE:
        raise BibleError("This clip is running; cancel it first")
    get_clip(clip_id)
    shutil.rmtree(clips_dir() / clip_id, ignore_errors=True)
    with _LOCK:
        records = [r for r in list_clips() if r["id"] != clip_id]
        _write(records)
    return records


def file_path(clip_id: str, which: str) -> Path:
    record = get_clip(clip_id)
    outputs = record.get("outputs") or {}
    if which not in OUTPUTS or which not in outputs:
        raise BibleError("No such file")
    path = clips_dir() / clip_id / outputs[which]
    if not path.is_file():
        raise BibleError("The file is missing")
    return path


def register_routes(api: Any, *, settings_dir: Callable[[], str]) -> None:
    global _settings_dir
    _settings_dir = settings_dir
    base = "/api/v1/director/persona-clips"

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BibleError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @api.get(base)
    def clips_list(bible_id: str = ""):
        records = list_clips()
        return {"clips": [r for r in records if not bible_id or r.get("bible_id") == bible_id]}

    @api.post(base)
    async def clip_create(request: Request, bible_id: str = Form(...), character_id: str = Form(...),
                          gesture: str = Form(...), location_id: str = Form(""), place: str = Form(""),
                          name: str = Form(""), orientation: str = Form("portrait"),
                          seconds: float = Form(DEFAULT_SECONDS), seed: int = Form(-1), audio: str = Form("silent"),
                          prompt: str = Form(""), start_prompt: str = Form(""),
                          model_type: str = Form(cover_video.VIDEO_MODEL), run: bool = Form(True),
                          start_image: UploadFile | None = File(None)):
        raw = {"character_id": character_id, "location_id": location_id, "gesture": gesture, "place": place,
               "orientation": orientation, "seconds": seconds, "seed": seed, "audio": audio, "prompt": prompt,
               "start_prompt": start_prompt, "model_type": model_type}
        uploaded = ""
        if start_image is not None and start_image.filename:
            suffix = os.path.splitext(start_image.filename)[1].lower()
            if suffix not in cover_video.IMAGE_SUFFIXES:
                raise HTTPException(status_code=400, detail="The start image must be a png, jpg or webp")
            data = await start_image.read()
            if len(data) > voice_cover.MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=400, detail="The start image is too large")
            import io
            from PIL import Image
            uploaded = str(clips_dir() / f"upload_{secrets.token_hex(4)}.png")
            Image.open(io.BytesIO(data)).convert("RGB").save(uploaded)
        try:
            record = call(create_clip, name, raw, bible_id=bible_id, start_image=uploaded)
        finally:
            if uploaded and os.path.isfile(uploaded):
                os.remove(uploaded)
        if run:
            record = call(start_clip, record["id"])
        return {"clip": record}

    @api.get(base + "/{clip_id}")
    def clip_get(clip_id: str):
        return {"clip": call(get_clip, clip_id)}

    @api.post(base + "/{clip_id}/run")
    def clip_run(clip_id: str):
        return {"clip": call(start_clip, clip_id)}

    @api.post(base + "/{clip_id}/cancel")
    def clip_cancel(clip_id: str):
        return {"clip": call(cancel_clip, clip_id)}

    @api.get(base + "/{clip_id}/file/{which}")
    def clip_file(clip_id: str, which: str):
        return FileResponse(str(call(file_path, clip_id, which)))

    @api.delete(base + "/{clip_id}")
    def clip_delete(clip_id: str):
        return {"clips": call(delete_clip, clip_id)}
