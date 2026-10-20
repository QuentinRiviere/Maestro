"""AI personas: one fixed character, a signature move, a new place every clip (controller side).

The short-form persona format of late 2026 (Jean Phil and his copies): a
character whose look never changes, who never speaks, performs the same move
in public places that get more absurd clip after clip, with a song laid over
the silent video. A persona here is that identity: a name and handle, a
look in words (or a bible character), the move, a home world, the places to
visit. Its portrait is painted once by the Director's image editor (or taken
from the bible character), every clip reuses it as the identity reference
(``services.director.persona_clips``), and a persona can join a bible as a
character so episodes and covers can use it too.

Records live in ``settings/director_personas.json``, the portrait under
``settings/personas/<id>/portrait.png``. Each clip of a persona can be judged
by the analyst (the vision model that writes room sheets): identity match to
the portrait, readability of the move and of the place, artefacts, hook.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from . import bibles, cover_video, style_lora, voice_cover
from .bibles import BibleError

FORMATS = {
    "silent_move": "Silent clip: the same signature move in a new place every time, a song laid over it.",
    "diary": "Talks to the camera about their day (needs a voice; not built yet).",
}
PORTRAIT = "portrait.png"
MAX_PLACES = 40
_LOCK = threading.RLock()
_LIVE: set[str] = set()
_settings_dir: Callable[[], str] | None = None
style_analyst_backend: Callable[[], Any] | None = None

EVALUATION_SYSTEM = (
    "You judge short vertical AI persona clips for social media. You receive the persona's portrait (first image) "
    "and a sheet of frames from one clip in time order (second image), plus the move and place the clip was meant "
    "to show. Answer with one JSON object only: {\"identity\": 1-5 (same face, hair and outfit as the portrait in "
    "every frame), \"move\": 1-5 (the described move is clearly performed), \"place\": 1-5 (the place reads as "
    "described), \"artefacts\": 1-5 (5 = no AI artefacts: extra fingers, melting, flicker, text), \"hook\": 1-5 "
    "(would a viewer stop scrolling: absurdity, contrast, deadpan), \"verdict\": one sentence, \"fix\": one short "
    "suggestion}. Be strict: 5 is rare."
)


def _dir() -> Path:
    base = Path(_settings_dir()) if _settings_dir else Path("settings")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _records_path() -> Path:
    return _dir() / "director_personas.json"


def personas_dir() -> Path:
    folder = _dir() / "personas"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def persona_dir(persona_id: str) -> Path:
    folder = personas_dir() / persona_id
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def list_personas() -> list[dict]:
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


def get_persona(persona_id: str) -> dict:
    for record in list_personas():
        if record["id"] == persona_id:
            return record
    raise BibleError("Unknown persona")


def _set(persona_id: str, **fields: Any) -> dict:
    with _LOCK:
        records = list_personas()
        for record in records:
            if record["id"] == persona_id:
                record.update(fields, updated_at=voice_cover._now())
                _write(records)
                return record
    raise BibleError("Unknown persona")


def _places(raw: Any) -> list[str]:
    if isinstance(raw, str):
        raw = raw.replace("\r", "").split("\n")
    out = []
    for item in raw if isinstance(raw, list) else []:
        text = str(item or "").strip()[:200]
        if text and text not in out:
            out.append(text)
    return out[:MAX_PLACES]


def persona_fields(raw: Mapping[str, Any] | None) -> dict:
    raw = dict(raw or {})
    fmt = str(raw.get("format") or "silent_move").strip()
    if fmt not in FORMATS:
        raise BibleError(f"Unknown format (one of {', '.join(FORMATS)})")
    orientation = str(raw.get("orientation") or "portrait").strip().lower()
    if orientation not in cover_video.ORIENTATIONS:
        raise BibleError("Unknown orientation")
    audio = str(raw.get("audio") or "silent").strip().lower()
    try:
        seconds = float(raw.get("seconds") or 6.0)
    except (TypeError, ValueError):
        seconds = 6.0
    try:
        speed = float(raw.get("speed") if raw.get("speed") not in (None, "") else 1.5)
    except (TypeError, ValueError):
        speed = 1.5
    speed = max(1.0, min(3.0, speed))
    seconds = max(1.0, min(cover_video.seconds_for(cover_video.MAX_FRAMES) / speed, seconds))
    handle = str(raw.get("handle") or "").strip().lstrip("@")[:40]
    return {
        "handle": handle,
        "look": str(raw.get("look") or "").strip()[:1200],
        "move": str(raw.get("move") or "").strip()[:500],
        "format": fmt,
        "world": str(raw.get("world") or "").strip()[:600],
        "music": str(raw.get("music") or "").strip()[:300],
        "places": _places(raw.get("places")),
        "orientation": orientation,
        "seconds": round(seconds, 2),
        "speed": speed,
        "audio": audio if audio in ("silent", "model") else "silent",
    }


def portrait_prompt(record: Mapping[str, Any]) -> str:
    """The editor prompt for the identity portrait: full body, neutral place, the look word for word."""
    framing = "vertical 9:16" if record["orientation"] == "portrait" else "16:9"
    return (f"{record['name']}: {record['look']}. Full body, standing, facing the camera with a deadpan expression, "
            f"on an empty city sidewalk at golden hour, {framing} photograph, 35 mm lens, natural light, sharp, "
            "no text, no watermark.")


def portrait_path(record: Mapping[str, Any]) -> str:
    path = personas_dir() / record["id"] / PORTRAIT
    return str(path) if path.is_file() else ""


def create_persona(name: str, raw: Mapping[str, Any] | None, *, bible_id: str = "", character_id: str = "",
                   portrait: str = "", submit: Callable[..., list[str]] | None = None,
                   jobs: Mapping[str, Any] | None = None, wait: bool = False) -> dict:
    """Register a persona; its portrait comes from the bible character, an upload, or the editor (a job)."""
    fields = persona_fields(raw)
    name = style_lora._short(name, 60)
    if not name:
        raise BibleError("The persona needs a name")
    if not fields["move"]:
        raise BibleError("The persona needs its signature move")
    source = ""
    if character_id:
        bible = bibles.get_bible(bible_id)
        if not bible:
            raise BibleError("Unknown bible")
        kind, character = bibles.find_element(bible, character_id)
        if kind != "character":
            raise BibleError("The persona must come from a character")
        source = style_lora.element_view(bible_id, character, "") or ""
        if not source:
            raise BibleError("The character needs an approved image")
        fields["look"] = fields["look"] or str(character.get("description") or "")[:1200]
    elif portrait:
        if not os.path.isfile(portrait):
            raise BibleError("The portrait file is missing")
        source = portrait
    elif not fields["look"]:
        raise BibleError("Describe the persona's look, or pick a bible character, or upload a portrait")
    record = {
        "id": "persona_" + secrets.token_hex(4), "name": name, **fields,
        "bible_id": bible_id, "character_id": character_id,
        "status": "done" if source else "queued", "progress": "" if source else "Waiting for the worker",
        "error": "", "portrait": "", "job_id": "", "clips": 0,
        "created_at": voice_cover._now(), "updated_at": voice_cover._now(),
    }
    if source:
        target = persona_dir(record["id"]) / PORTRAIT
        from PIL import Image
        with Image.open(source) as image:
            image.convert("RGB").save(target)
        record["portrait"] = str(target)
    with _LOCK:
        records = list_personas()
        records.append(record)
        _write(records)
    if not source:
        start_portrait(record["id"], submit=submit, jobs=jobs, wait=wait)
    return get_persona(record["id"])


def start_portrait(persona_id: str, *, submit: Callable[..., list[str]] | None = None,
                   jobs: Mapping[str, Any] | None = None, wait: bool = False) -> dict:
    record = get_persona(persona_id)
    if persona_id in _LIVE:
        raise BibleError("This persona's portrait is being painted")
    if not record.get("look"):
        raise BibleError("The persona has no look to paint")
    _LIVE.add(persona_id)
    _set(persona_id, status="queued", progress="Waiting for the worker", error="", job_id="")
    thread = threading.Thread(target=_paint, args=(persona_id, submit, jobs), name=f"persona-{persona_id}",
                              daemon=True)
    thread.start()
    if wait:
        thread.join()
    return get_persona(persona_id)


def _paint(persona_id: str, submit, jobs) -> None:
    try:
        record = get_persona(persona_id)
        pipeline = voice_cover._director()
        submit = submit or pipeline._submit_and_wait
        jobs = jobs if jobs is not None else pipeline._jobs
        folder = persona_dir(persona_id)
        params = cover_video.start_job(portrait_prompt(record), [], record["orientation"], pipeline, -1)
        params["_director_cost_stage"] = "persona_portrait"
        outputs = cover_video.run_stage(submit, jobs, params, folder, "Portrait", lambda **f: _set(persona_id, **f))
        shutil.move(str(folder / outputs[0]), str(folder / PORTRAIT))
        _set(persona_id, status="done", progress="", portrait=str(folder / PORTRAIT))
    except BibleError as error:
        _set(persona_id, status="failed", error=str(error)[:400], progress="")
    except Exception as error:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        _set(persona_id, status="failed", error=str(error)[:400], progress="")
    finally:
        _LIVE.discard(persona_id)


def update_persona(persona_id: str, fields: Mapping[str, Any]) -> dict:
    record = get_persona(persona_id)
    changes: dict[str, Any] = {}
    if "name" in fields:
        changes["name"] = style_lora._short(fields["name"], 60) or record["name"]
    merged = persona_fields({**{k: record.get(k) for k in persona_fields({})}, **fields})
    for key, value in merged.items():
        if key in fields or key == "places" and "places" in fields:
            changes[key] = value
    return _set(persona_id, **changes) if changes else record


def series(persona_id: str, places: list[str] | None = None, *, count: int = 0, seconds: float | None = None,
           submit: Callable[..., list[str]] | None = None, jobs: Mapping[str, Any] | None = None,
           wait: bool = False) -> list[dict]:
    """Queue one clip per place (the persona's own list when none is given, ``count`` of them at most)."""
    from . import persona_clips

    record = get_persona(persona_id)
    if not portrait_path(record):
        raise BibleError("The persona has no portrait yet")
    wanted = _places(places) or list(record.get("places") or [])
    if count:
        wanted = wanted[:count]
    if not wanted:
        raise BibleError("Name the places to visit")
    made = []
    for place in wanted:
        clip = persona_clips.create_clip(f"{record['name']} · {place[:40]}", {
            "persona_id": persona_id, "gesture": record["move"], "place": place,
            "orientation": record["orientation"], "seconds": seconds or record["seconds"], "audio": record["audio"],
            "speed": record.get("speed", 1.5),
        }, bible_id=record.get("bible_id") or "")
        made.append(persona_clips.start_clip(clip["id"], submit=submit, jobs=jobs, wait=wait))
    _set(persona_id, clips=int(record.get("clips") or 0) + len(made))
    return made


def frame_sheet(video: str, output: str, *, frames: int = 8, width: int = 240) -> str:
    """A contact sheet of ``frames`` stills in time order, one row (ffmpeg)."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", video],
                           capture_output=True, text=True, check=True)
    duration = max(0.2, float(probe.stdout.strip() or 1.0))
    rate = frames / duration
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-frames:v", "1",
                    "-vf", f"fps={rate:.4f},scale={width}:-1,tile={frames}x1", output],
                   capture_output=True, text=True, check=True)
    return output


def evaluate_clip(clip_id: str, *, backend: Any = None, ask: Callable[..., Any] | None = None) -> dict:
    """Score one clip of a persona with the analyst; the scores are kept on the clip record."""
    from . import persona_clips
    from .room_spec import _ask, _resolve_backend, extract_json
    from .style_reference import StyleReferenceError, data_uri_image

    clip = persona_clips.get_clip(clip_id)
    settings = clip.get("settings") or {}
    persona = get_persona(settings.get("persona_id") or "") if settings.get("persona_id") else None
    video = persona_clips.file_path(clip_id, "clip")
    portrait = portrait_path(persona) if persona else str(persona_clips.file_path(clip_id, "start"))
    if backend is None and callable(style_analyst_backend):
        backend = style_analyst_backend()
    ask, fal_key, model, local_backend = _resolve_backend(backend, None, "analyst", ask or _ask)
    work = tempfile.mkdtemp(prefix="persona_eval_")
    try:
        sheet = frame_sheet(str(video), os.path.join(work, "sheet.jpg"))
        media = [portrait, sheet] if local_backend else [data_uri_image(portrait), data_uri_image(sheet)]
        prompt = (f"Persona: {persona['name'] if persona else settings.get('character_id', '')}. "
                  f"Move meant: {settings.get('gesture', '')}. Place meant: {settings.get('place') or 'a bible place'}. "
                  f"Clip length {settings.get('seconds')} s, {settings.get('frames')} frames, the sheet reads left to right.")
        import httpx
        with httpx.Client(timeout=180) as http:
            try:
                text, _cost = ask(http, fal_key, model, EVALUATION_SYSTEM, prompt, media, time.sleep,
                                  max_tokens=600, timeout=240)
            except StyleReferenceError as error:
                raise BibleError(str(error)) from error
    finally:
        shutil.rmtree(work, ignore_errors=True)
    raw = extract_json(text)
    if not isinstance(raw, dict):
        raise BibleError("The analyst did not answer with scores")
    scores = {}
    for key in ("identity", "move", "place", "artefacts", "hook"):
        try:
            scores[key] = max(1, min(5, int(round(float(raw.get(key))))))
        except (TypeError, ValueError):
            scores[key] = None
    known = [v for v in scores.values() if v is not None]
    evaluation = {**scores, "overall": round(sum(known) / len(known), 2) if known else None,
                  "verdict": str(raw.get("verdict") or "")[:300], "fix": str(raw.get("fix") or "")[:300],
                  "model": model, "evaluated_at": voice_cover._now()}
    persona_clips._set(clip_id, evaluation=evaluation)
    return evaluation


def evaluate_persona(persona_id: str) -> list[dict]:
    """Judge every finished, unjudged clip of the persona; returns the clips."""
    from . import persona_clips

    out = []
    for clip in persona_clips.list_clips():
        if (clip.get("settings") or {}).get("persona_id") != persona_id or clip.get("status") != "done":
            continue
        if not clip.get("evaluation"):
            try:
                evaluate_clip(clip["id"])
            except BibleError as error:
                persona_clips._set(clip["id"], evaluation={"error": str(error)[:300]})
        out.append(persona_clips.get_clip(clip["id"]))
    return out


def add_to_bible(persona_id: str, bible_id: str) -> dict:
    """Add the persona to a bible as a character whose approved image is the portrait; returns the bible."""
    record = get_persona(persona_id)
    portrait = portrait_path(record)
    if not portrait:
        raise BibleError("The persona has no portrait yet")
    bible = bibles.get_bible(bible_id)
    if not bible:
        raise BibleError("Unknown bible")
    element_id = ""

    def change(bible: dict) -> None:
        nonlocal element_id
        element = bibles._element({
            "name": record["name"], "tag": record.get("handle") or record["name"],
            "role": "Persona", "description": record.get("look") or "", "wardrobe": "",
            "personality": f"Signature move: {record['move']}. {record.get('world') or ''}".strip(),
        }, "character", bibles._folder(bible_id))
        if element is None:
            raise BibleError("The persona cannot become a character")
        if any(e["name"].casefold() == element["name"].casefold() for e in bible["characters"]):
            raise BibleError("The bible already has a character with this name")
        bible["characters"].append(element)
        element_id = element["id"]

    bibles._mutate(bible_id, change)
    bible = bibles.add_image(bible_id, element_id, bibles.DEFAULT_VARIANT, portrait)
    _set(persona_id, bible_id=bible_id, character_id=element_id)
    return bible


def delete_persona(persona_id: str) -> list[dict]:
    if persona_id in _LIVE:
        raise BibleError("This persona's portrait is being painted")
    get_persona(persona_id)
    shutil.rmtree(personas_dir() / persona_id, ignore_errors=True)
    with _LOCK:
        records = [r for r in list_personas() if r["id"] != persona_id]
        _write(records)
    return records


def register_routes(api: Any, *, settings_dir: Callable[[], str]) -> None:
    global _settings_dir
    _settings_dir = settings_dir
    base = "/api/v1/director/personas"

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BibleError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @api.get(base)
    def personas_list():
        return {"personas": list_personas(), "formats": FORMATS}

    @api.post(base)
    async def persona_create(request: Request, name: str = Form(...), handle: str = Form(""), look: str = Form(""),
                             move: str = Form(...), format: str = Form("silent_move"), world: str = Form(""),
                             music: str = Form(""), places: str = Form(""), orientation: str = Form("portrait"),
                             seconds: float = Form(6.0), speed: float = Form(1.5), audio: str = Form("silent"),
                             bible_id: str = Form(""),
                             character_id: str = Form(""), portrait: UploadFile | None = File(None)):
        raw = {"handle": handle, "look": look, "move": move, "format": format, "world": world, "music": music,
               "places": places, "orientation": orientation, "seconds": seconds, "speed": speed, "audio": audio}
        uploaded = ""
        if portrait is not None and portrait.filename:
            if os.path.splitext(portrait.filename)[1].lower() not in cover_video.IMAGE_SUFFIXES:
                raise HTTPException(status_code=400, detail="The portrait must be a png, jpg or webp")
            data = await portrait.read()
            if len(data) > voice_cover.MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=400, detail="The portrait is too large")
            uploaded = str(personas_dir() / f"upload_{secrets.token_hex(4)}.png")
            import io
            from PIL import Image
            Image.open(io.BytesIO(data)).convert("RGB").save(uploaded)
        try:
            record = call(create_persona, name, raw, bible_id=bible_id, character_id=character_id, portrait=uploaded)
        finally:
            if uploaded and os.path.isfile(uploaded):
                os.remove(uploaded)
        return {"persona": record}

    @api.get(base + "/{persona_id}")
    def persona_get(persona_id: str):
        from . import persona_clips
        record = call(get_persona, persona_id)
        clips = [c for c in persona_clips.list_clips() if (c.get("settings") or {}).get("persona_id") == persona_id]
        return {"persona": record, "clips": clips}

    @api.patch(base + "/{persona_id}")
    async def persona_patch(persona_id: str, request: Request):
        try:
            body = await request.json()
        except ValueError:
            body = {}
        return {"persona": call(update_persona, persona_id, body if isinstance(body, dict) else {})}

    @api.post(base + "/{persona_id}/portrait")
    def persona_portrait(persona_id: str):
        return {"persona": call(start_portrait, persona_id)}

    @api.get(base + "/{persona_id}/portrait")
    def persona_portrait_file(persona_id: str):
        path = portrait_path(call(get_persona, persona_id))
        if not path:
            raise HTTPException(status_code=404, detail="No portrait yet")
        return FileResponse(path)

    @api.post(base + "/{persona_id}/series")
    async def persona_series(persona_id: str, request: Request):
        try:
            body = await request.json()
        except ValueError:
            body = {}
        body = body if isinstance(body, dict) else {}
        try:
            count = int(body.get("count") or 0)
        except (TypeError, ValueError):
            count = 0
        seconds = body.get("seconds")
        return {"clips": call(series, persona_id, _places(body.get("places")), count=count,
                              seconds=float(seconds) if seconds else None)}

    @api.post(base + "/{persona_id}/evaluate")
    def persona_evaluate(persona_id: str):
        return {"clips": call(evaluate_persona, persona_id)}

    @api.post(base + "/{persona_id}/bible")
    async def persona_bible(persona_id: str, request: Request):
        try:
            body = await request.json()
        except ValueError:
            body = {}
        bible_id = str((body or {}).get("bible_id") or "")
        return {"bible": call(add_to_bible, persona_id, bible_id), "persona": call(get_persona, persona_id)}

    @api.delete(base + "/{persona_id}")
    def persona_delete(persona_id: str):
        return {"personas": call(delete_persona, persona_id)}
