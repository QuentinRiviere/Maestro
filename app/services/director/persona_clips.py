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
# LTX-2.5 moves at about two thirds of real pace whatever the prompt says, so a clip is rendered ``speed`` times
# longer than asked and sped up in post (silent clips lose nothing); 1.5 reads as a phone video.
DEFAULT_SPEED = 1.5
MAX_SPEED = 3.0
AUDIO_MODES = ("silent", "model")
OUTPUTS = {"start": "start.png", "clip": "clip.mp4", "raw": "clip_raw.mp4"}
# A shot's framing: how its start image is framed, how the phone moves during it, which part of the move it shows.
SHOTS = {
    "full": ("a full-body shot facing the camera, phone at chest height", "handheld phone framing, slight sway",
             "the whole move"),
    "wide": ("a wide shot filmed from across the way with a phone held at chest height, the character small but "
             "clearly visible in the middle of the scene, passers-by crossing the foreground, the place filling the frame",
             "the phone drifts sideways and pushes in a little, people walk through the frame", "the opening of the move"),
    "side": ("a medium shot from a three-quarter side angle, waist up, phone at eye level, the place blurred behind",
             "handheld with a slight sway, the phone slowly circles a quarter turn around him", "the middle of the move"),
    "low": ("a low angle from knee height looking up, full body, the place towering behind him",
            "handheld low angle with a slight tilt up", "the middle of the move"),
    "close": ("a close-up on the face and hands, shallow depth of field, the whistle and the pointing finger in frame",
              "steady close framing with micro shake and a quick push in at the end", "the finish of the move"),
    "over_shoulder": ("over the shoulder of a bystander in the foreground, the character in focus behind",
                      "the bystander's shoulder sways, the phone peeks past it", "the middle of the move"),
}
# Templates: the shots of a clip and their lengths (seconds, as shown).
TEMPLATES = {
    "single": [("full", None)],
    "trend": [("wide", 4.0), ("side", 4.0), ("close", 3.0)],
    "street": [("over_shoulder", 4.0), ("low", 4.0), ("close", 3.0)],
    "reveal": [("close", 3.0), ("wide", 5.0), ("side", 3.0)],
}
DEFAULT_TEMPLATE = "single"
# Engines: LTX-2.5 (default, 576x1024, 8k+1 frames) or MiniMax H3 image-to-video (480x864 on the Spark, 17n+5 frames,
# the 8-step turbo preset, its own prompt grammar: a shot list in integrated_multimodal_description).
ENGINES = ("ltx", "h3")
# Post-processing the worker applies to every shot (the LTX Director workflow's RIFE + grain idea): RIFE x2 doubles the
# frames before the speed-up so the result stays smooth, a touch of film grain reads as phone footage.
SMOOTH = "rife2"
GRAIN = 0.08
H3_MODEL = "minimax_h3"
H3_TURBO_PRESET = "alibaba-pai-fl2va-pdd-8step"
H3_RESOLUTIONS = {"portrait": "480x864", "landscape": "864x480"}
H3_MIN_FRAMES, H3_MAX_FRAMES, H3_STEP = 124, 345, 17
# The start images read as candid phone stills, not studio portraits.
REALISM = ("candid vertical smartphone video still, natural light, imperfect framing, slight motion blur on passers-by, "
           "nobody looks at him, true-to-life colours, no text, no watermark")
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
    try:
        speed = float(raw.get("speed") if raw.get("speed") not in (None, "") else DEFAULT_SPEED)
    except (TypeError, ValueError):
        speed = DEFAULT_SPEED
    speed = max(1.0, min(MAX_SPEED, speed))
    wanted = float(raw.get("seconds") or DEFAULT_SECONDS)
    template = str(raw.get("template") or DEFAULT_TEMPLATE).strip()
    if template not in TEMPLATES:
        raise BibleError(f"Unknown template (one of {', '.join(TEMPLATES)})")
    engine = str(raw.get("engine") or "ltx").strip().lower()
    if engine not in ENGINES:
        raise BibleError(f"Unknown engine (one of {', '.join(ENGINES)})")
    base = cover_video.video_settings({**raw, "seconds": wanted * speed, "start_sec": 0,
                                       **({"model_type": cover_video.VIDEO_MODEL} if engine == "h3" else {})})
    maximum = H3_MAX_FRAMES if engine == "h3" else cover_video.MAX_FRAMES
    limit = cover_video.seconds_for(maximum) / speed
    shots = []
    for framing, seconds in TEMPLATES[template]:
        shown = float(seconds if seconds is not None else wanted)
        if shown > limit + 1e-6:
            raise BibleError(f"A shot fits one window: at most {limit:.0f} s at speed {speed:g}")
        frames = h3_frames(shown * speed) if engine == "h3" else cover_video.frames_for(shown * speed)
        shots.append({"framing": framing, "seconds": round(shown, 3), "frames": frames,
                      "rendered_seconds": round(cover_video.seconds_for(frames), 3)})
    total = round(sum(shot["seconds"] for shot in shots), 3)
    return {
        "persona_id": str(raw.get("persona_id") or "").strip(),
        "character_id": base["character_id"], "location_id": base["location_id"], "gesture": gesture,
        "orientation": base["orientation"], "frames": shots[0]["frames"], "seconds": total,
        "speed": speed, "rendered_seconds": round(sum(s["rendered_seconds"] for s in shots), 3),
        "template": template, "shots": shots, "engine": engine,
        "smooth": bool(raw.get("smooth", True)) if str(raw.get("smooth", "true")).lower() not in ("false", "0", "") else False,
        "grain": max(0.0, min(0.5, float(raw.get("grain", GRAIN) if raw.get("grain") not in (None, "") else GRAIN))),
        "seed": base["seed"], "model_type": H3_MODEL if engine == "h3" else base["model_type"], "audio": audio,
        "prompt": base["prompt"], "start_prompt": base["start_prompt"],
        "place": str(raw.get("place") or "").strip()[:300],
    }


def h3_frames(seconds: float) -> int:
    """H3's video VAE takes 17n+5 frames at 24 fps, 124 to 345."""
    import math
    wanted = int(math.ceil(max(0.0, seconds) * 24))
    frames = int(math.ceil(max(wanted - 5, 0) / H3_STEP)) * H3_STEP + 5
    return max(H3_MIN_FRAMES, min(H3_MAX_FRAMES, frames))


def start_prompt(character: Mapping[str, Any], location: Mapping[str, Any] | None, settings: Mapping[str, Any],
                 look: str = "", *, framing: str = "full", same_place: bool = False) -> str:
    """The editor prompt of a shot's start image: the character (reference 1) in the place, framed for the shot.

    ``same_place``: reference 2 is the first shot's start image, the same place seen from another angle.
    """
    aspect = "vertical 9:16" if settings["orientation"] == "portrait" else "16:9"
    shot = SHOTS.get(framing) or SHOTS["full"]
    if same_place:
        place = " in the exact same place as reference 2, seen from another angle"
    elif location:
        place = f" inside the exact place of reference 2 ({location.get('name') or 'the place'})"
    elif settings.get("place"):
        place = f" in {settings['place']}"
    else:
        place = ""
    beat = {"the opening of the move": "the first beat", "the middle of the move": "the middle beat",
            "the finish of the move": "the final beat"}.get(shot[2], "the first beat")
    text = (f"{character.get('name') or 'The character'} from reference 1, same face, hair and clothes,{place}, "
            f"at {beat} of this move: {settings['gesture']}. {shot[0]}, {aspect}, {REALISM}.")
    return f"{text} {look}".strip()


# LTX-2.5 drifts into slow motion when the prompt reads calm; the clip has to ask for real-time pace outright.
PACE = ("Real-time speed, brisk natural motion as in a phone video, the whole move completes within the clip, "
        "no slow motion.")
NEGATIVE = "slow motion, slowed down, frozen pose, static, blurry, text, watermark, extra fingers, deformed hands"


def clip_prompt(character: Mapping[str, Any], settings: Mapping[str, Any], *, framing: str = "full") -> str:
    name = character.get("name") or "The character"
    shot = SHOTS.get(framing) or SHOTS["full"]
    body = settings.get("prompt") or (
        f"{name} performs {shot[2]} in one continuous take: {settings['gesture']}. {shot[1]}, smartphone footage, "
        f"passers-by ignore him, the place unchanged, no camera cut, no text. {PACE}")
    silence = " Silent clip: nobody speaks, the mouth stays closed." if settings.get("audio") == "silent" else ""
    return f"{body}{silence}"


def h3_prompt(character: Mapping[str, Any], settings: Mapping[str, Any], *, framing: str = "full") -> str:
    """H3's grammar: the start image referenced at 0 s, one shot in integrated_multimodal_description, no sound."""
    name = character.get("name") or "The character"
    shot = SHOTS.get(framing) or SHOTS["full"]
    aspect = "vertical 9:16" if settings["orientation"] == "portrait" else "16:9"
    seconds = next((s["seconds"] for s in settings.get("shots") or [] if s.get("framing") == framing), settings["seconds"])
    description = (f"[Shot 1] Live-action, {aspect}, handheld smartphone footage, {seconds:g}-second single take. {name} "
                   f"performs {shot[2]}: {settings['gesture']}. {shot[1]}. Passers-by ignore him, the place unchanged, "
                   "real-time pace, no slow motion, nobody speaks.")
    return ("For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
            f"\n\nintegrated_multimodal_description: {description}\n\noverall_soundscape: N/A.\n\nnon_diegetic_music: N/A")


def post_params(settings: Mapping[str, Any]) -> dict:
    """RIFE x2 and film grain on the worker, as the clip settings ask."""
    out: dict = {}
    if settings.get("smooth", True):
        out["temporal_upsampling"] = SMOOTH
    grain = float(settings.get("grain") or 0.0)
    if grain > 0:
        out["film_grain_intensity"] = grain
        out["film_grain_saturation"] = 0.5
    return out


def h3_job(prompt: str, start_image: str, settings: Mapping[str, Any], *, frames: int | None = None) -> dict:
    """MiniMax H3 image-to-video with the official 8-step turbo preset."""
    frames = int(frames or settings["frames"])
    return {**post_params(settings),
        "model_type": H3_MODEL, "prompt": prompt, "image_mode": 0, "image_prompt_type": "S",
        "image_start": start_image, "generation_mode": "video", "num_inference_steps": 8, "guidance_scale": 1.0,
        "input_video_strength": 1.0, "resolution": H3_RESOLUTIONS[settings["orientation"]],
        "video_length": frames, "sliding_window_size": frames, "minimax_h3_turbo_mode": True,
        "minimax_h3_turbo_preset": H3_TURBO_PRESET, "seed": int(settings.get("seed", -1)), "settings_version": 2.52,
        "repeat_generation": 1, "negative_prompt": "", "activated_loras": [], "loras_multipliers": "",
        "_director_cost_stage": "persona_clip",
    }


def clip_job(prompt: str, start_image: str, settings: Mapping[str, Any], *, frames: int | None = None) -> dict:
    """LTX-2.5 image-to-video; "A" without a soundtrack makes WanGP condition on silence."""
    frames = int(frames or settings["frames"])
    job = {**post_params(settings),
        "model_type": settings["model_type"], "prompt": prompt, "image_mode": 0, "image_prompt_type": "S",
        "image_start": start_image, "generation_mode": "video", "num_inference_steps": 8, "guidance_scale": 1.0,
        "input_video_strength": 1.0, "resolution": cover_video.ORIENTATIONS[settings["orientation"]][0],
        "video_length": frames, "sliding_window_size": frames,
        "seed": int(settings.get("seed", -1)), "settings_version": 2.52, "repeat_generation": 1,
        "negative_prompt": NEGATIVE, "activated_loras": [], "loras_multipliers": "", "_director_cost_stage": "persona_clip",
    }
    job["audio_prompt_type"] = "A" if settings.get("audio") == "silent" else ""
    return job


def speed_up(source: str, output: str, speed: float, *, silent: bool = True) -> str:
    """Play the rendered clip ``speed`` times faster (ffmpeg); a kept soundtrack is tempo-shifted to match."""
    import subprocess

    command = ["ffmpeg", "-v", "error", "-y", "-i", source, "-filter:v", f"setpts=PTS/{speed:.4f},fps=24",
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
    if silent:
        command.append("-an")
    else:
        command += ["-filter:a", f"atempo={min(2.0, speed):.4f}", "-c:a", "aac"]
    subprocess.run(command + [output], check=True, capture_output=True, text=True)
    return output


def concat_shots(parts: list[str], output: str) -> str:
    """Join the shots in order into one clip (ffmpeg, re-encoded so cuts are clean)."""
    import subprocess

    listing = Path(output).with_suffix(".txt")
    listing.write_text("".join(f"file '{Path(part).resolve()}'\n" for part in parts), encoding="utf-8")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-r", "24",
                    "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", output],
                   check=True, capture_output=True, text=True)
    listing.unlink(missing_ok=True)
    return output


def create_clip(name: str, raw: Mapping[str, Any] | None, *, bible_id: str = "", start_image: str = "") -> dict:
    """A clip of a bible character (``character_id`` in a bible) or of a persona (``persona_id``, its portrait)."""
    from . import personas

    settings = clip_settings(raw)
    bible = bibles.get_bible(bible_id) if bible_id else None
    if settings["persona_id"]:
        persona = personas.get_persona(settings["persona_id"])
        if not start_image and not personas.portrait_path(persona):
            raise BibleError("The persona has no portrait yet")
        character = {"name": persona["name"]}
    else:
        if not bible:
            raise BibleError("Unknown bible")
        if not settings["character_id"]:
            raise BibleError("The clip needs a character or a persona")
        kind, character = bibles.find_element(bible, settings["character_id"])
        if kind != "character":
            raise BibleError("The persona must be a character")
        if not start_image and not style_lora.element_view(bible_id, character, ""):
            raise BibleError("The character needs an approved image")
    if settings["location_id"]:
        if not bible:
            raise BibleError("A bible place needs a bible")
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
            bible = bibles.get_bible(record["bible_id"]) or {} if record.get("bible_id") else {}
            if settings.get("persona_id"):
                from . import personas
                persona = personas.get_persona(settings["persona_id"])
                character = {"name": persona["name"]}
                identity = personas.portrait_path(persona)
            else:
                character = bibles.find_element(bible, settings["character_id"])[1]
                identity = style_lora.element_view(record["bible_id"], character, "")
            location = bibles.find_element(bible, settings["location_id"])[1] if settings.get("location_id") else None
            look = str((bible.get("world") or {}).get("look") or "").strip()
            shots = settings.get("shots") or [{"framing": "full", "frames": settings["frames"],
                                               "seconds": settings["seconds"]}]
            speed = float(settings.get("speed") or 1.0)
            silent = settings.get("audio") == "silent"
            parts, prompts = [], []
            for index, shot in enumerate(shots, start=1):
                framing = shot.get("framing") or "full"
                start_name = OUTPUTS["start"] if index == 1 else f"start_{index}.png"
                if index == 1 and "start" in made and (folder / made["start"]).is_file():
                    pass
                else:
                    refs = [identity]
                    same_place = False
                    if location:
                        refs.append(style_lora.element_view(record["bible_id"], location, ""))
                    elif index > 1:
                        refs.append(str(folder / OUTPUTS["start"]))
                        same_place = True
                    prompt = (settings.get("start_prompt") if index == 1 and settings.get("start_prompt") else
                              start_prompt(character, location, settings, look, framing=framing, same_place=same_place))
                    params = cover_video.start_job(prompt, refs, settings["orientation"], pipeline, settings.get("seed", -1))
                    params["_director_cost_stage"] = "persona_clip_start"
                    outputs = cover_video.run_stage(submit, jobs, params, folder, f"Start image {index}/{len(shots)}", update)
                    shutil.move(str(folder / outputs[0]), str(folder / start_name))
                    if index == 1:
                        made["start"] = start_name
                        _set(clip_id, outputs=made, progress="Start image ready")
                if settings.get("engine") == "h3":
                    prompt = h3_prompt(character, settings, framing=framing)
                    job = h3_job(prompt, str(folder / start_name), settings, frames=shot.get("frames"))
                else:
                    prompt = clip_prompt(character, settings, framing=framing)
                    job = clip_job(prompt, str(folder / start_name), settings, frames=shot.get("frames"))
                prompts.append(prompt)
                outputs = cover_video.run_stage(submit, jobs, job, folder, f"Shot {index}/{len(shots)}", update)
                raw_name = OUTPUTS["raw"] if len(shots) == 1 else f"shot_{index}_raw.mp4"
                shot_name = f"shot_{index}.mp4"
                if speed > 1.0 or (silent and settings.get("engine") == "h3"):
                    cover_video.keep_last_video(folder, outputs, raw_name)
                    speed_up(str(folder / raw_name), str(folder / shot_name), max(speed, 1.0), silent=silent)
                    if len(shots) == 1:
                        made["raw"] = raw_name
                else:
                    cover_video.keep_last_video(folder, outputs, shot_name)
                parts.append(str(folder / shot_name))
            if len(parts) == 1:
                shutil.move(parts[0], str(folder / OUTPUTS["clip"]))
            else:
                concat_shots(parts, str(folder / OUTPUTS["clip"]))
            made["clip"] = OUTPUTS["clip"]
            prompt = " | ".join(prompts)
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
    def clips_list(bible_id: str = "", persona_id: str = ""):
        records = list_clips()
        return {"clips": [r for r in records if (not bible_id or r.get("bible_id") == bible_id)
                          and (not persona_id or (r.get("settings") or {}).get("persona_id") == persona_id)]}

    @api.post(base)
    async def clip_create(request: Request, bible_id: str = Form(""), character_id: str = Form(""),
                          persona_id: str = Form(""),
                          gesture: str = Form(...), location_id: str = Form(""), place: str = Form(""),
                          name: str = Form(""), orientation: str = Form("portrait"),
                          seconds: float = Form(DEFAULT_SECONDS), speed: float = Form(DEFAULT_SPEED), seed: int = Form(-1),
                          audio: str = Form("silent"), template: str = Form(DEFAULT_TEMPLATE), engine: str = Form("ltx"),
                          smooth: bool = Form(True), grain: float = Form(GRAIN),
                          prompt: str = Form(""), start_prompt: str = Form(""),
                          model_type: str = Form(cover_video.VIDEO_MODEL), run: bool = Form(True),
                          start_image: UploadFile | None = File(None)):
        raw = {"character_id": character_id, "persona_id": persona_id, "location_id": location_id, "gesture": gesture,
               "place": place,
               "orientation": orientation, "seconds": seconds, "speed": speed, "seed": seed, "audio": audio, "prompt": prompt,
               "template": template, "engine": engine, "smooth": smooth, "grain": grain,
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
