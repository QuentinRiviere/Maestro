"""The singing video of a cover: a character at a microphone, lips driven by the converted vocals (controller side).

Two worker jobs, run one after the other under the cover's folder
(``settings/covers/<id>/``):

1. ``start.png``: the Director's image editor paints the bible character
   (reference 1) at a studio microphone in the chosen place (reference 2),
   framed for the clip's orientation. A start image uploaded by the user
   skips this step.
2. ``cover.mp4``: LTX-2.5 image-to-video driven by the cover's audio. The
   full mix (``cover.wav``) is the soundtrack (``audio_guide``) and the
   converted vocal stem (``vocals.wav``) conditions the lips
   (``audio_conditioning_guide``), the recipe the Director's music-video mode
   already uses. One LTX window holds at most ``MAX_FRAMES`` frames, so the
   clip covers ``[start_sec, start_sec + seconds]`` of the song, both stems
   sliced to that range (``clip_mix.wav``, ``clip_vocals.wav``).

The state lives in the cover record under ``video`` (``status``,
``progress``, ``error``, ``settings``, ``outputs``, ``job_id``).
"""
from __future__ import annotations

import math
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from . import bibles, style_lora, voice_cover
from .bibles import BibleError

VIDEO_MODEL = "ltx2_25"
FPS = 24
FRAME_STEP = 8
MIN_FRAMES = 17
# One LTX window; longer videos run as sliding windows that continue each other (overlap 9, 8 frames dropped).
WINDOW_FRAMES = 289
MAX_FRAMES = 345
WINDOW_OVERLAP = 9
WINDOW_DISCARD = 8
MAX_VIDEO_SECONDS = 120.0
DEFAULT_SECONDS = 12.0
MAX_START_SECONDS = 36000.0
ORIENTATIONS = {"portrait": ("576x1024", "720x1280"), "landscape": ("1024x576", "1280x720")}
STATES = ("queued", "running", "done", "failed")
OUTPUTS = {"start": "start.png", "video": "cover.mp4", "clip_mix": "clip_mix.wav", "clip_vocals": "clip_vocals.wav"}
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp")
TIMEOUT_SEC = 3600
# LTX-2.5 lip-syncs a source track only when the prompt says so explicitly (Director music-video contract).
SYNC_CONTRACT = ("SOURCE-AUDIO LIP SYNC: keep the supplied track as the only source of sound. The singer lip-syncs "
                 "the vocal line exactly while it is audible, lips relaxed and closed through instrumental gaps. "
                 "No other person appears.")
_LIVE: set[str] = set()
_SLOT = threading.Semaphore(1)


def frames_for(seconds: float, *, maximum: int = MAX_FRAMES) -> int:
    """The LTX frame count (8k+1) that covers ``seconds`` at 24 fps, at most ``maximum``."""
    wanted = int(math.ceil(max(0.0, float(seconds)) * FPS))
    frames = int(math.ceil(max(wanted - 1, 0) / FRAME_STEP)) * FRAME_STEP + 1
    return max(MIN_FRAMES, min(maximum, frames))


def window_params(frames: int) -> dict:
    """One window when the clip fits; sliding windows of WINDOW_FRAMES that continue each other otherwise."""
    if frames <= MAX_FRAMES:
        return {"sliding_window_size": int(frames)}
    return {"sliding_window_size": WINDOW_FRAMES, "sliding_window_overlap": WINDOW_OVERLAP,
            "sliding_window_discard_last_frames": WINDOW_DISCARD}


def seconds_for(frames: int) -> float:
    return (int(frames) - 1) / FPS


def video_settings(raw: Mapping[str, Any] | None) -> dict:
    raw = dict(raw or {})
    orientation = str(raw.get("orientation") or "portrait").strip().lower()
    if orientation not in ORIENTATIONS:
        raise BibleError(f"Unknown orientation (one of {', '.join(ORIENTATIONS)})")
    try:
        start = max(0.0, min(MAX_START_SECONDS, float(raw.get("start_sec") or 0.0)))
        seconds = float(raw.get("seconds") or DEFAULT_SECONDS)
    except (TypeError, ValueError) as error:
        raise BibleError("start_sec and seconds must be numbers") from error
    if seconds <= 0:
        raise BibleError("seconds must be positive")
    if seconds > MAX_VIDEO_SECONDS:
        raise BibleError(f"seconds must be at most {MAX_VIDEO_SECONDS:.0f}")
    frames = frames_for(seconds, maximum=frames_for(MAX_VIDEO_SECONDS, maximum=10 ** 6))
    try:
        seed = int(raw.get("seed") if raw.get("seed") not in (None, "") else -1)
    except (TypeError, ValueError):
        seed = -1
    model = str(raw.get("model_type") or VIDEO_MODEL).strip()
    if not model.startswith("ltx2_25"):
        raise BibleError("The cover video runs on an LTX-2.5 model")
    return {
        "character_id": str(raw.get("character_id") or "").strip(),
        "location_id": str(raw.get("location_id") or "").strip(),
        "orientation": orientation, "start_sec": round(start, 3), "frames": frames,
        "seconds": round(seconds_for(frames), 3), "seed": seed, "model_type": model,
        "lyrics": str(raw.get("lyrics") or "").strip()[:2000],
        "prompt": str(raw.get("prompt") or "").strip()[:2000],
        "start_prompt": str(raw.get("start_prompt") or "").strip()[:1000],
    }


def start_prompt(character: Mapping[str, Any], location: Mapping[str, Any] | None, look: str = "",
                 orientation: str = "portrait") -> str:
    """The editor prompt for the start image: the character at a microphone in the place."""
    framing = ("vertical 9:16 medium close-up" if orientation == "portrait" else "16:9 medium shot")
    place = (f" inside the exact place of reference 2 ({location.get('name') or 'the set'})" if location
             else " in a small home recording studio")
    text = (f"{character.get('name') or 'The character'} from reference 1, same face, hair and clothes, stands at a "
            f"studio condenser microphone on a stand{place}, about to sing, looking at the camera, {framing}, "
            "warm practical light, photographic, no text.")
    return f"{text} {look}".strip()


def video_prompt(character: Mapping[str, Any], settings: Mapping[str, Any], mode: str = "sing") -> str:
    """The LTX prompt: the performance, the lyrics (quoted, as the Director workflow does) and the sync contract."""
    verb = "raps" if mode == "rap" else "sings"
    name = character.get("name") or "The singer"
    body = settings.get("prompt") or (
        f"{name} {verb} into the studio microphone, facing the camera, steady framing, subtle head and shoulder "
        f"movement with the rhythm, expressive face, the set unchanged, no camera cut.")
    lyrics = (settings.get("lyrics") or "").replace("\n", " / ").strip()
    if lyrics:
        body = f'{body} Lyrics sung: "{lyrics}".'
    return f"{body} {SYNC_CONTRACT}"


def start_job(prompt: str, refs: list[str], orientation: str, pipeline, seed: int = -1) -> dict:
    from services.director_pipeline import DIRECTOR_IMAGE_SAMPLERS, director_image_job_fields

    preset = DIRECTOR_IMAGE_SAMPLERS.get(style_lora.IMAGE_MODEL, {})
    return {
        "model_type": style_lora.IMAGE_MODEL, "prompt": prompt, "image_mode": 1, "image_prompt_type": "",
        "video_prompt_type": pipeline._director_image_prompt_type(style_lora.IMAGE_MODEL, refs) if refs else "",
        "generation_mode": "image", **({"image_refs": refs, "remove_background_images_ref": 0} if refs else {}),
        "num_inference_steps": preset.get("num_inference_steps", 40), "guidance_scale": preset.get("guidance_scale", 4.0),
        "resolution": ORIENTATIONS[orientation][1], "seed": seed, "settings_version": 2.52, "repeat_generation": 1,
        "negative_prompt": "", "video_length": 1, "activated_loras": [], "loras_multipliers": "",
        **(director_image_job_fields(preset) if preset else {"sample_solver": "default"}),
        "_director_cost_stage": "cover_video_start",
    }


def video_job(prompt: str, start_image: str, mix: str, vocals: str, settings: Mapping[str, Any]) -> dict:
    """The LTX-2.5 job: start image, soundtrack = the mix, lips conditioned on the vocal stem."""
    return {
        "model_type": settings["model_type"], "prompt": prompt, "image_mode": 0, "image_prompt_type": "S",
        "image_start": start_image, "generation_mode": "video", "num_inference_steps": 8, "guidance_scale": 1.0,
        "input_video_strength": 1.0, "resolution": ORIENTATIONS[settings["orientation"]][0],
        "video_length": int(settings["frames"]), **window_params(int(settings["frames"])),
        "audio_prompt_type": "A", "audio_guide": mix, "audio_conditioning_guide": vocals, "audio_frame_offset": 0,
        "seed": int(settings.get("seed", -1)), "settings_version": 2.52, "repeat_generation": 1, "negative_prompt": "",
        "activated_loras": [], "loras_multipliers": "", "_director_cost_stage": "cover_video",
    }


def slice_audio(source: str, start_sec: float, seconds: float, output: str) -> str:
    """Cut ``[start, start+seconds]`` of a wav into ``output`` (same rate and channels, padded with silence)."""
    import numpy as np
    import soundfile as sf

    data, rate = sf.read(source, always_2d=True)
    first = int(round(start_sec * rate))
    count = int(round(seconds * rate))
    piece = data[first:first + count]
    if len(piece) < count:
        piece = np.concatenate([piece, np.zeros((count - len(piece), data.shape[1]), dtype=data.dtype)])
    sf.write(output, piece, rate)
    return output


def keep_last_video(folder: Path, outputs: list[str], name: str) -> Path:
    """WanGP saves the video again after every sliding window: the last output is the whole clip, the others go."""
    videos = [n for n in outputs if n.lower().endswith((".mp4", ".webm", ".mov"))] or list(outputs)
    target = folder / name
    shutil.move(str(folder / videos[-1]), str(target))
    for extra in videos[:-1]:
        for path in (folder / extra, (folder / extra).with_suffix(".meta.json")):
            if path.is_file():
                path.unlink()
    for path in (folder / videos[-1]).with_suffix(".meta.json"),:
        if path.is_file():
            path.unlink()
    return target


def _set_video(cover_id: str, **fields: Any) -> dict:
    record = voice_cover.get_cover(cover_id)
    video = dict(record.get("video") or {})
    video.update(fields)
    return voice_cover._set(cover_id, video=video)


def start_video(cover_id: str, raw: Mapping[str, Any] | None, *, start_image: str = "",
                submit: Callable[..., list[str]] | None = None, jobs: Mapping[str, Any] | None = None,
                wait: bool = False) -> dict:
    """Queue the two jobs for a finished cover; ``start_image`` (a path under the cover folder) skips the editor."""
    record = voice_cover.get_cover(cover_id)
    if record.get("status") != "done" or not {"cover", "vocals"} <= set(record.get("outputs") or {}):
        raise BibleError("The cover must be done (mix and vocals) before its video")
    if cover_id in _LIVE:
        raise BibleError("This cover's video is running")
    settings = video_settings(raw)
    settings["character_id"] = settings["character_id"] or record.get("character_id") or ""
    bible_id = str((raw or {}).get("bible_id") or record.get("bible_id") or "")
    if not start_image:
        if not settings["character_id"]:
            raise BibleError("The video needs a character (or a start image)")
        bible = bibles.get_bible(bible_id)
        if not bible:
            raise BibleError("The cover has no bible for its character")
        kind, character = bibles.find_element(bible, settings["character_id"])
        if kind != "character" or not style_lora.element_view(bible_id, character, ""):
            raise BibleError("The character needs an approved image")
        if settings["location_id"]:
            kind, location = bibles.find_element(bible, settings["location_id"])
            if kind != "location" or not style_lora.element_view(bible_id, location, ""):
                raise BibleError("The place needs an approved image")
    elif not os.path.isfile(start_image):
        raise BibleError("The start image is missing")
    _LIVE.add(cover_id)
    try:
        _set_video(cover_id, status="queued", progress="Waiting for the worker", error="", settings=settings,
                   bible_id=bible_id, outputs={"start": OUTPUTS["start"]} if start_image else {}, job_id="",
                   seconds=None, started_at=voice_cover._now())
    except Exception:
        _LIVE.discard(cover_id)
        raise
    if wait:
        _run(cover_id, submit, jobs)
    else:
        threading.Thread(target=_run, args=(cover_id, submit, jobs), daemon=True).start()
    return voice_cover.get_cover(cover_id)


def run_stage(submit, jobs, params: dict, folder: Path, stage: str, update: Callable[..., Any]) -> list[str]:
    """Submit one worker job, mirror its progress through ``update(**fields)``, return its output names."""
    job_id = uuid.uuid4().hex[:8]
    update(status="running", job_id=job_id, progress=f"{stage}: submitting to the worker")
    done = threading.Event()
    follower = threading.Thread(target=style_lora._follow_generic, args=(job_id, jobs, done, update), daemon=True)
    follower.start()
    try:
        outputs = submit(params, timeout_s=TIMEOUT_SEC, out_dir=str(folder), job_id=job_id)
    finally:
        done.set()
        follower.join(timeout=5)
    if not outputs or not (folder / outputs[0]).is_file():
        job = jobs.get(job_id) or {}
        raise BibleError("Cancelled" if job.get("status") == "cancelled" else f"{stage}: the worker returned nothing")
    return outputs


def _submit(cover_id: str, submit, jobs, params: dict, folder: Path, stage: str) -> list[str]:
    return run_stage(submit, jobs, params, folder, stage, lambda **f: _set_video(cover_id, **f))


def _run(cover_id: str, submit, jobs) -> None:
    started = time.time()
    try:
        with _SLOT:
            if cover_id not in _LIVE:
                return
            record = voice_cover.get_cover(cover_id)
            video = record.get("video") or {}
            settings = video["settings"]
            pipeline = voice_cover._director()
            submit = submit or pipeline._submit_and_wait
            jobs = jobs if jobs is not None else pipeline._jobs
            folder = voice_cover.cover_dir(cover_id)
            made = dict(video.get("outputs") or {})
            bible = bibles.get_bible(video.get("bible_id") or "") or {}
            character = {}
            if settings.get("character_id"):
                try:
                    character = bibles.find_element(bible, settings["character_id"])[1]
                except BibleError:
                    character = {"name": settings["character_id"]}
            if "start" not in made:
                location = bibles.find_element(bible, settings["location_id"])[1] if settings.get("location_id") else None
                refs = [style_lora.element_view(video["bible_id"], character, "")]
                if location:
                    refs.append(style_lora.element_view(video["bible_id"], location, ""))
                look = str((bible.get("world") or {}).get("look") or "").strip()
                prompt = settings.get("start_prompt") or start_prompt(character, location, look, settings["orientation"])
                outputs = _submit(cover_id, submit, jobs, start_job(prompt, refs, settings["orientation"], pipeline,
                                                                    settings.get("seed", -1)), folder, "Start image")
                shutil.move(str(folder / outputs[0]), str(folder / OUTPUTS["start"]))
                made["start"] = OUTPUTS["start"]
                _set_video(cover_id, outputs=made, progress="Start image ready")
            mix = slice_audio(str(folder / voice_cover.OUTPUTS["cover"]), settings["start_sec"], settings["seconds"],
                              str(folder / OUTPUTS["clip_mix"]))
            vocals = slice_audio(str(folder / voice_cover.OUTPUTS["vocals"]), settings["start_sec"],
                                 settings["seconds"], str(folder / OUTPUTS["clip_vocals"]))
            made.update(clip_mix=OUTPUTS["clip_mix"], clip_vocals=OUTPUTS["clip_vocals"])
            mode = str((record.get("config") or {}).get("mode") or "sing")
            prompt = video_prompt(character, settings, mode)
            outputs = _submit(cover_id, submit, jobs,
                              video_job(prompt, str(folder / made["start"]), mix, vocals, settings), folder, "Video")
            keep_last_video(folder, outputs, OUTPUTS["video"])
            made["video"] = OUTPUTS["video"]
            _set_video(cover_id, status="done", progress="", outputs=made, prompt=prompt,
                       seconds=round(time.time() - started))
            print(f"[Cover video] {cover_id}: {OUTPUTS['video']} in {round(time.time() - started)} s")
    except BibleError as error:
        try:
            _set_video(cover_id, status="failed", error=str(error)[:400], progress="")
        except Exception:  # noqa: BLE001
            pass
    except Exception as error:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        try:
            _set_video(cover_id, status="failed", error=str(error)[:400], progress="")
        except Exception:  # noqa: BLE001
            pass
    finally:
        _LIVE.discard(cover_id)


def cancel_video(cover_id: str) -> dict:
    record = voice_cover.get_cover(cover_id)
    video = record.get("video") or {}
    if cover_id not in _LIVE:
        raise BibleError("This cover's video is not running")
    _LIVE.discard(cover_id)
    job_id = video.get("job_id")
    if job_id:
        from services.job_lifecycle import request_cancel
        job = voice_cover._director()._jobs.get(job_id)
        if job:
            request_cancel(job)
    return _set_video(cover_id, status="failed", error="Cancelled", progress="")


def video_path(cover_id: str, which: str) -> Path:
    record = voice_cover.get_cover(cover_id)
    outputs = (record.get("video") or {}).get("outputs") or {}
    if which not in OUTPUTS or which not in outputs:
        raise BibleError("No such file")
    path = voice_cover.cover_dir(cover_id) / outputs[which]
    if not path.is_file():
        raise BibleError("The file is missing")
    return path


def register_routes(api: Any) -> None:
    base = "/api/v1/director/covers/{cover_id}/video"

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BibleError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @api.post(base)
    async def cover_video_start(cover_id: str, request: Request, start_image: UploadFile | None = File(None),
                                character_id: str = Form(""), location_id: str = Form(""),
                                orientation: str = Form("portrait"), start_sec: float = Form(0.0),
                                seconds: float = Form(DEFAULT_SECONDS), seed: int = Form(-1),
                                lyrics: str = Form(""), prompt: str = Form(""), start_prompt: str = Form(""),
                                model_type: str = Form(VIDEO_MODEL), bible_id: str = Form("")):
        raw = {"character_id": character_id, "location_id": location_id, "orientation": orientation,
               "start_sec": start_sec, "seconds": seconds, "seed": seed, "lyrics": lyrics, "prompt": prompt,
               "start_prompt": start_prompt, "model_type": model_type, "bible_id": bible_id}
        uploaded = ""
        if start_image is not None and start_image.filename:
            suffix = os.path.splitext(start_image.filename)[1].lower()
            if suffix not in IMAGE_SUFFIXES:
                raise HTTPException(status_code=400, detail="The start image must be a png, jpg or webp")
            folder = call(voice_cover.cover_dir, cover_id)
            data = await start_image.read()
            if len(data) > voice_cover.MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=400, detail="The start image is too large")
            dest = folder / OUTPUTS["start"]
            from PIL import Image
            import io
            Image.open(io.BytesIO(data)).convert("RGB").save(dest)
            uploaded = str(dest)
        return {"cover": call(start_video, cover_id, raw, start_image=uploaded)}

    @api.post(base + "/cancel")
    def cover_video_cancel(cover_id: str):
        return {"cover": call(cancel_video, cover_id)}

    @api.get(base + "/{which}")
    def cover_video_file(cover_id: str, which: str):
        return FileResponse(str(call(video_path, cover_id, which)))
