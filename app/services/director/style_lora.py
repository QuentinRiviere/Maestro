"""Style LoRAs of a bible: trained on the GPU worker from the reference frames.

A bible's analysed reference videos carry, per sampled shot, a first and a
last frame with a reverse prompt (``style_reference.describe_frames``).
This module turns them into a training set, submits a ``lora_training``
job to the worker (``services.style_lora_training``), brings the adapter
back into the controller's ``loras/qwen21`` folder with a registry sidecar,
renders a few sample images with an invented cast (the identity check: the
samples must not look like the reference actors) and lets the Director
attach the enabled LoRA to every shot image of a run that uses the bible.

Captions name the content, not the style: the trigger word carries the
look. Close-ups are left out by default, so faces are under-represented
and the adapter learns the grade, the light and the framing rather than
the people.

Records live on the bible (``style_loras``); the dataset, the samples and
the training sidecar sit in the bible folder under ``stylelora_<id>*``.
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
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

from . import bibles
from .bibles import BibleError

IMAGE_MODEL = "qwen_image_21_7B"
MAX_STYLE_LORAS = bibles.MAX_STYLE_LORAS
CAPTION_MODES = ("content", "full")
TARGETS = ("attention", "all")
CLOSE_UP_SIZES = ("close_up", "extreme_close_up", "medium_close_up", "insert")
MIN_IMAGES = 12
MAX_IMAGES = 2000
DEFAULT_STEPS = 1500
DEFAULT_RANK = 16
DEFAULT_LR = 1e-4
DEFAULT_RESOLUTION = 768
DEFAULT_SCALE = 0.7
SAMPLE_COUNT = 3
SAMPLE_RESOLUTION = "1280x720"
# The identity check: nobody from the references, three registers.
SAMPLE_PROMPTS = (
    "Medium shot of a man in his forties with grey hair and a navy raincoat, standing at a bus stop at dusk, "
    "looking down the empty street, light rain.",
    "Wide shot of a small kitchen in the morning, a woman in her twenties with a red headscarf pouring tea "
    "at the table, window light.",
    "Two-shot of two teenagers in school uniforms on a rooftop, sitting on the ledge, city skyline behind them, "
    "late afternoon.",
)
SHOT_SIZE_WORDS = {
    "extreme_wide": "extreme wide shot", "wide": "wide shot", "medium": "medium shot",
    "medium_close_up": "medium close-up", "close_up": "close-up", "extreme_close_up": "extreme close-up",
    "insert": "insert shot", "ots": "over-the-shoulder shot", "pov": "point-of-view shot", "two_shot": "two-shot",
}
TRAIN_TIMEOUT_SEC = 3 * 3600  # no-progress timeout: a step reports every few seconds
# Trainings owned by this process (bibles reads it to mark the ones a restart cut).
_LIVE_TRAININGS = bibles._LIVE_TRAININGS
_TRAIN_SLOT = threading.Semaphore(1)
_TRIGGER = re.compile(r"[^a-z0-9_]+")


class StyleLoraError(RuntimeError):
    pass


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _short(value: Any, limit: int = 200) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else text[:limit - 1] + "…"


def default_trigger(bible: Mapping[str, Any]) -> str:
    """A trigger word from the bible's name: lower-case, underscores, ``_style`` suffix."""
    base = _TRIGGER.sub("_", str(bible.get("name") or bible.get("id") or "bible").lower()).strip("_")[:24]
    return f"{base or 'bible'}_style"


def normalize_trigger(value: Any, fallback: str) -> str:
    text = _TRIGGER.sub("_", str(value or "").lower()).strip("_")[:40]
    return text or fallback


def training_settings(raw: Mapping[str, Any] | None, bible: Mapping[str, Any]) -> dict:
    """The bounded settings of a training request."""
    raw = dict(raw or {})

    def number(key: str, default: float, low: float, high: float, cast=float):
        try:
            value = cast(raw.get(key, default))
        except (TypeError, ValueError):
            value = cast(default)
        return max(low, min(high, value))

    caption_mode = str(raw.get("caption_mode") or "content")
    targets = str(raw.get("targets") or "attention")
    return {
        "steps": number("steps", DEFAULT_STEPS, 50, 20000, int),
        "rank": number("rank", DEFAULT_RANK, 4, 128, int),
        "learning_rate": number("learning_rate", DEFAULT_LR, 1e-6, 1e-2),
        "resolution": number("resolution", DEFAULT_RESOLUTION, 384, 1280, int),
        "exclude_close_ups": raw.get("exclude_close_ups") is not False,
        "caption_mode": caption_mode if caption_mode in CAPTION_MODES else "content",
        "targets": targets if targets in TARGETS else "attention",
        "max_images": number("max_images", MAX_IMAGES, MIN_IMAGES, MAX_IMAGES, int),
        "trigger": normalize_trigger(raw.get("trigger"), default_trigger(bible)),
        "references": [str(r) for r in raw.get("references") or [] if isinstance(r, str)][:bibles.MAX_STYLE_REFERENCES],
    }


# ── dataset ────────────────────────────────────────────────────────────────


def caption_for(frame: Mapping[str, Any], shot: Mapping[str, Any] | None, trigger: str, mode: str) -> str:
    """The caption of a frame: the trigger word, then the content (or the whole reverse prompt with ``full``)."""
    if mode == "full" and frame.get("prompt"):
        return f"{trigger}, {_short(frame['prompt'], 600)}"
    parts = []
    size = str((frame.get("shot_size") or (shot or {}).get("shot_size") or "")).strip()
    if size:
        parts.append(SHOT_SIZE_WORDS.get(size, size.replace("_", " ")))
    shot = shot or {}
    subject = _short(shot.get("subject"), 160)
    action = _short(shot.get("action"), 160)
    if subject:
        parts.append(subject + (f", {action}" if action else ""))
    elif action:
        parts.append(action)
    elif frame.get("prompt"):
        # No shot description: the reverse prompt's first sentence names the content.
        parts.append(_short(str(frame["prompt"]).split(". ")[0], 200))
    angle = str(shot.get("angle") or "").strip()
    if angle and angle not in ("eye_level", "eye level"):
        parts.append(angle.replace("_", " ") + " angle")
    return ", ".join([trigger] + parts) if parts else trigger


def candidate_frames(bible_id: str, bible: Mapping[str, Any], settings: Mapping[str, Any]) -> dict:
    """The frames the dataset would use and why the others are left out.

    Returns ``{"items": [...], "references": [...], "available", "excluded_close_ups", "excluded_credits"}``;
    each item carries ``ref_id``, ``t``, ``caption`` and the frame.
    """
    from . import style_reference as sr

    wanted = set(settings.get("references") or [])
    items, references = [], []
    close_ups = credits = 0
    for reference in bible.get("style_references") or []:
        if reference.get("status") != "done" or not reference.get("enabled"):
            continue
        if wanted and reference["id"] not in wanted:
            continue
        frames = [f for f in sr.load_frames(bible_id, reference) if f.get("prompt")]
        if not frames:
            continue
        shots = {s.get("index"): s for s in sr.load_shots(bible_id, reference)}
        kept = 0
        for frame in frames:
            shot = shots.get(frame.get("index")) or {}
            if shot.get("credits"):
                credits += 1
                continue
            size = frame.get("shot_size") or shot.get("shot_size") or ""
            if settings.get("exclude_close_ups", True) and size in CLOSE_UP_SIZES:
                close_ups += 1
                continue
            try:
                t = float(frame.get("t"))
            except (TypeError, ValueError):
                continue
            items.append({"ref_id": reference["id"], "t": t, "index": frame.get("index"),
                          "position": frame.get("position"), "shot_size": size,
                          "caption": caption_for(frame, shot, settings["trigger"], settings["caption_mode"])})
            kept += 1
        references.append({"id": reference["id"], "label": reference.get("label"), "frames": len(frames),
                           "used": kept})
    limit = int(settings.get("max_images") or MAX_IMAGES)
    if len(items) > limit:
        step = len(items) / limit
        items = [items[int(i * step)] for i in range(limit)]
    return {"items": items, "references": references, "available": len(items),
            "excluded_close_ups": close_ups, "excluded_credits": credits}


def _lora_prefix(lora_id: str) -> str:
    return f"stylelora_{lora_id}"


def lora_folder(bible_id: str, lora_id: str) -> Path:
    return bibles._folder(bible_id) / _lora_prefix(lora_id)


def build_dataset(bible_id: str, bible: Mapping[str, Any], lora_id: str, settings: Mapping[str, Any], *,
                  progress: Callable[[str], Any] | None = None) -> tuple[str, dict]:
    """Extract the frames and write ``dataset.zip`` (images + manifest) in the LoRA's folder; returns (zip, facts)."""
    from . import style_reference as sr

    found = candidate_frames(bible_id, bible, settings)
    if found["available"] < MIN_IMAGES:
        raise StyleLoraError(f"Only {found['available']} usable frames; a style LoRA needs at least {MIN_IMAGES}. "
                             "Analyse more references or run their frame prompts first.")
    folder = lora_folder(bible_id, lora_id)
    images = folder / "images"
    if images.exists():
        shutil.rmtree(images)
    images.mkdir(parents=True, exist_ok=True)
    references = {r["id"]: r for r in bible.get("style_references") or []}
    manifest = []
    for number, item in enumerate(found["items"], start=1):
        if progress and number % 25 == 1:
            progress(f"Extracting frame {number} of {found['available']}")
        reference = references.get(item["ref_id"])
        video = bibles.video_path(bible_id, (reference or {}).get("file") or "")
        if video is None:
            continue
        dest = images / f"{number:05d}.jpg"
        try:
            sr._extract_still(str(video), item["t"], str(dest), long_side=int(settings["resolution"]))
        except Exception:  # noqa: BLE001 - one bad frame does not stop the set
            continue
        manifest.append({"image": f"images/{dest.name}", "caption": item["caption"], "reference": item["ref_id"],
                         "t": item["t"], "shot_size": item["shot_size"]})
    if len(manifest) < MIN_IMAGES:
        raise StyleLoraError(f"Only {len(manifest)} frames could be extracted")
    (folder / "manifest.json").write_text(json.dumps({"trigger": settings["trigger"], "items": manifest}, indent=1),
                                          encoding="utf-8")
    archive = folder / "dataset.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zipped:
        zipped.write(folder / "manifest.json", "manifest.json")
        for entry in manifest:
            zipped.write(folder / entry["image"], entry["image"])
    facts = {"images": len(manifest), "references": found["references"],
             "excluded_close_ups": found["excluded_close_ups"], "excluded_credits": found["excluded_credits"],
             "by_shot_size": _count(entry["shot_size"] for entry in manifest)}
    return str(archive), facts


def _count(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


# ── records ────────────────────────────────────────────────────────────────


def list_loras(bible_id: str) -> list[dict]:
    bible = bibles.get_bible(bible_id)
    if not bible:
        raise BibleError("Unknown bible")
    return list(bible.get("style_loras") or [])


def get_lora(bible_id: str, lora_id: str) -> dict:
    for lora in list_loras(bible_id):
        if lora["id"] == lora_id:
            return lora
    raise BibleError("Unknown style LoRA")


def _set(bible_id: str, lora_id: str, **fields: Any) -> dict:
    def change(bible: dict) -> dict:
        for lora in bible.get("style_loras") or []:
            if lora["id"] == lora_id:
                lora.update(fields)
                lora["updated_at"] = _now()
                return lora
        raise BibleError("Unknown style LoRA")

    _, lora = bibles._mutate(bible_id, change)
    return lora


def create_lora(bible_id: str, raw: Mapping[str, Any] | None) -> dict:
    """Add a style LoRA record to the bible (not started)."""
    bible = bibles.get_bible(bible_id)
    if not bible:
        raise BibleError("Unknown bible")
    if len(bible.get("style_loras") or []) >= MAX_STYLE_LORAS:
        raise BibleError(f"A bible keeps at most {MAX_STYLE_LORAS} style LoRAs; delete one first")
    settings = training_settings(raw, bible)
    lora_id = "lora_" + secrets.token_hex(4)
    name = _short((raw or {}).get("name"), 60) or f"{settings['trigger']}_{len(bible.get('style_loras') or []) + 1}"
    record = {
        "id": lora_id, "name": name, "file": "", "trigger": settings["trigger"],
        "status": "queued", "progress": "", "step": 0, "total_steps": settings["steps"], "error": "",
        "enabled": False, "scale": DEFAULT_SCALE, "config": settings, "dataset": None, "samples": [],
        "job_id": "", "model": IMAGE_MODEL, "created_at": _now(), "updated_at": _now(), "training": None,
    }

    def change(bible: dict) -> dict:
        bible.setdefault("style_loras", []).append(record)
        return record

    _, created = bibles._mutate(bible_id, change)
    return created


def delete_lora(bible_id: str, lora_id: str) -> list[dict]:
    """Remove the record, its folder and the adapter in the LoRA registry."""
    if lora_id in _LIVE_TRAININGS:
        raise BibleError("This LoRA is training; cancel it first")
    lora = get_lora(bible_id, lora_id)
    shutil.rmtree(lora_folder(bible_id, lora_id), ignore_errors=True)
    if lora.get("file"):
        _remove_registry_file(lora["file"])

    def change(bible: dict) -> None:
        bible["style_loras"] = [l for l in bible.get("style_loras") or [] if l["id"] != lora_id]

    bible, _ = bibles._mutate(bible_id, change)
    return bible["style_loras"]


def update_lora(bible_id: str, lora_id: str, body: Mapping[str, Any]) -> dict:
    fields: dict[str, Any] = {}
    if "enabled" in body:
        fields["enabled"] = bool(body["enabled"])
    if "scale" in body:
        try:
            fields["scale"] = max(0.0, min(1.5, round(float(body["scale"]), 2)))
        except (TypeError, ValueError):
            pass
    if "name" in body and _short(body["name"], 60):
        fields["name"] = _short(body["name"], 60)
    lora = _set(bible_id, lora_id, **fields) if fields else get_lora(bible_id, lora_id)
    if fields.get("enabled"):
        # One enabled LoRA per bible: the runs take the first enabled, done one.
        def change(bible: dict) -> None:
            for other in bible.get("style_loras") or []:
                if other["id"] != lora_id:
                    other["enabled"] = False

        bibles._mutate(bible_id, change)
    return lora


def _registry_dir() -> str:
    import wgp
    return wgp.get_lora_dir(IMAGE_MODEL)


def _remove_registry_file(file: str) -> None:
    folder = _registry_dir()
    stem = os.path.splitext(os.path.basename(file))[0]
    for name in os.listdir(folder) if os.path.isdir(folder) else []:
        if name == os.path.basename(file) or name.startswith(stem + ".") and name.endswith(".json"):
            try:
                os.remove(os.path.join(folder, name))
            except OSError:
                pass


def enabled_lora(bible: Mapping[str, Any], lora_id: str = "") -> dict | None:
    """The trained LoRA a run attaches: the one named, else the first enabled, done one."""
    for lora in bible.get("style_loras") or []:
        if lora.get("status") != "done" or not lora.get("file"):
            continue
        if lora_id and lora["id"] == lora_id:
            return lora
        if not lora_id and lora.get("enabled"):
            return lora
    return None


def registry_sidecar(lora: Mapping[str, Any], bible: Mapping[str, Any]) -> dict:
    """The ``<stem>.civitai.json`` sidecar the LoRA registry reads (trigger words, base model, description)."""
    dataset = lora.get("dataset") or {}
    references = ", ".join(str(r.get("label") or r.get("id")) for r in dataset.get("references") or [])
    return {
        "name": f"{bible.get('name')} style ({lora.get('name')})",
        "baseModel": "Qwen Image 2.1",
        "trainedWords": [lora.get("trigger")] if lora.get("trigger") else [],
        "description": (f"Style LoRA trained by Maestro on {dataset.get('images', 0)} frames of the bible "
                        f"'{bible.get('name')}' ({references}). Close-ups "
                        f"{'left out' if (lora.get('config') or {}).get('exclude_close_ups', True) else 'included'}. "
                        f"Suggested weight {lora.get('scale', DEFAULT_SCALE)}."),
        "recommendedWeights": [lora.get("scale", DEFAULT_SCALE)],
        "downloadedAt": _now(),
        "maestro_style_lora": {"bible": bible.get("id"), "lora": lora.get("id")},
    }


# ── training ───────────────────────────────────────────────────────────────


def start_training(bible_id: str, lora_id: str, *, submit: Callable[..., list[str]] | None = None,
                   jobs: Mapping[str, Any] | None = None, wait: bool = False) -> dict:
    """Queue the training of a record in a background thread; returns the record.

    ``submit`` / ``jobs`` default to the Director's ``_submit_and_wait`` and
    job table (tests pass fakes).
    """
    lora = get_lora(bible_id, lora_id)
    if lora_id in _LIVE_TRAININGS:
        raise BibleError("This LoRA is already training")
    _LIVE_TRAININGS.add(lora_id)
    try:
        lora = _set(bible_id, lora_id, status="queued", progress="Waiting for the trainer", error="", step=0)
    except Exception:
        _LIVE_TRAININGS.discard(lora_id)
        raise
    thread = threading.Thread(target=_run_training, args=(bible_id, lora_id, submit, jobs),
                              name=f"style-lora-{lora_id}", daemon=True)
    thread.start()
    if wait:
        thread.join()
        return get_lora(bible_id, lora_id)
    return lora


# launch.py wires the Director pipeline lazily (its job table, the generation
# runner): the mount passes the function that does it.
_ensure_pipeline: Callable[[], Any] | None = None


def _director():
    if _ensure_pipeline is not None:
        _ensure_pipeline()
    from services import director_pipeline
    return director_pipeline


def _follow(job_id: str, jobs: Mapping[str, Any], bible_id: str, lora_id: str, done: threading.Event) -> None:
    """Mirror the training job's progress onto the record until ``done``."""
    last = None
    while not done.wait(3.0):
        job = jobs.get(job_id) or {}
        state = (job.get("message"), job.get("step"), job.get("progress"))
        if state == last:
            continue
        last = state
        try:
            _set(bible_id, lora_id, progress=_short(job.get("message"), 200), step=int(job.get("step") or 0),
                 total_steps=int(job.get("total_steps") or 0) or None)
        except Exception:  # noqa: BLE001
            pass


def _run_training(bible_id: str, lora_id: str, submit, jobs) -> None:
    try:
        with _TRAIN_SLOT:
            if lora_id not in _LIVE_TRAININGS:
                return
            bible = bibles.get_bible(bible_id) or {}
            lora = _set(bible_id, lora_id, status="preparing", progress="Preparing the dataset")
            settings = lora.get("config") or training_settings({}, bible)
            archive, facts = build_dataset(bible_id, bible, lora_id, settings,
                                           progress=lambda m: _set(bible_id, lora_id, progress=m))
            _set(bible_id, lora_id, dataset=facts, status="training", progress="Submitting to the worker")
            submit = submit or _director()._submit_and_wait
            jobs = jobs if jobs is not None else _director()._jobs
            job_id = uuid.uuid4().hex[:8]
            folder = lora_folder(bible_id, lora_id)
            params = {
                "model_type": IMAGE_MODEL, "prompt": "", "image_mode": 1, "generation_mode": "image",
                "lora_training": {"name": _lora_prefix(lora_id), "trigger": settings["trigger"],
                                  "steps": settings["steps"], "rank": settings["rank"],
                                  "learning_rate": settings["learning_rate"],
                                  "resolution": settings["resolution"], "targets": settings["targets"]},
                "lora_dataset": archive,
                "_director_cost_stage": "lora_training",
            }
            _set(bible_id, lora_id, job_id=job_id)
            done = threading.Event()
            follower = threading.Thread(target=_follow, args=(job_id, jobs, bible_id, lora_id, done), daemon=True)
            follower.start()
            try:
                outputs = submit(params, timeout_s=TRAIN_TIMEOUT_SEC, out_dir=str(folder), job_id=job_id)
            finally:
                done.set()
                follower.join(timeout=5)
            weights = next((n for n in outputs or [] if str(n).endswith(".safetensors")), None)
            if not weights or not (folder / weights).is_file():
                raise StyleLoraError("The worker returned no adapter file")
            training = {}
            sidecar = folder / (os.path.splitext(weights)[0] + ".json")
            if sidecar.is_file():
                try:
                    training = json.loads(sidecar.read_text(encoding="utf-8"))
                except ValueError:
                    training = {}
            file = _install(bible_id, lora_id, weights, folder)
            lora = _set(bible_id, lora_id, file=file, status="sampling", progress="Rendering the sample images",
                        training={k: training.get(k) for k in ("final_loss", "seconds", "items", "losses")})
            samples = _render_samples(bible_id, lora, submit, folder)
            _set(bible_id, lora_id, samples=samples, status="done", progress="", enabled=enabled_lora(
                bibles.get_bible(bible_id) or {}) is None)
            print(f"[Style LoRA] {bible_id}/{lora_id}: {file} trained on {facts['images']} frames, "
                  f"{len(samples)} samples")
    except BibleError:
        pass
    except Exception as error:  # noqa: BLE001 - recorded on the record
        import traceback
        traceback.print_exc()
        print(f"[Style LoRA] {bible_id}/{lora_id}: {error}")
        try:
            _set(bible_id, lora_id, status="failed", error=_short(error, 400), progress="")
        except Exception:  # noqa: BLE001
            pass
    finally:
        _LIVE_TRAININGS.discard(lora_id)


def _install(bible_id: str, lora_id: str, weights: str, folder: Path) -> str:
    """Copy the adapter into the LoRA registry with its sidecar; returns the registry file name."""
    bible = bibles.get_bible(bible_id) or {}
    lora = get_lora(bible_id, lora_id)
    registry = _registry_dir()
    os.makedirs(registry, exist_ok=True)
    name = f"{_lora_prefix(lora_id)}.safetensors"
    shutil.copyfile(folder / weights, os.path.join(registry, name))
    with open(os.path.join(registry, f"{_lora_prefix(lora_id)}.civitai.json"), "w", encoding="utf-8") as handle:
        json.dump(registry_sidecar({**lora, "dataset": lora.get("dataset")}, bible), handle, indent=1)
    return name


def _render_samples(bible_id: str, lora: Mapping[str, Any], submit, folder: Path) -> list[str]:
    """Three images with the LoRA at its scale through the Director's image sampler; names in the bible folder."""
    from services.director_pipeline import DIRECTOR_IMAGE_SAMPLERS, director_image_job_fields

    preset = DIRECTOR_IMAGE_SAMPLERS.get(IMAGE_MODEL, {})
    samples = []
    out_dir = folder / "samples"
    out_dir.mkdir(parents=True, exist_ok=True)
    for number, text in enumerate(SAMPLE_PROMPTS[:SAMPLE_COUNT], start=1):
        params = {
            "model_type": IMAGE_MODEL, "prompt": f"{text} {lora['trigger']}", "image_mode": 1,
            "image_prompt_type": "", "video_prompt_type": "", "generation_mode": "image",
            "num_inference_steps": preset.get("num_inference_steps", 6),
            "guidance_scale": preset.get("guidance_scale", 1.0), "resolution": SAMPLE_RESOLUTION,
            "seed": 1000 + number, "settings_version": 2.52, "repeat_generation": 1, "negative_prompt": "",
            "video_length": 1, "activated_loras": [lora["file"]],
            "loras_multipliers": f"{float(lora.get('scale') or DEFAULT_SCALE):.2f}",
            **director_image_job_fields(preset), "_director_cost_stage": "lora_sample",
        }
        try:
            outputs = submit(params, timeout_s=900, out_dir=str(out_dir))
        except Exception as error:  # noqa: BLE001 - a failed sample is not a failed LoRA
            print(f"[Style LoRA] sample {number} failed: {error}")
            continue
        if outputs and (out_dir / outputs[0]).is_file():
            name = f"{_lora_prefix(lora['id'])}_sample{number}{os.path.splitext(outputs[0])[1]}"
            shutil.move(str(out_dir / outputs[0]), str(bibles._folder(bible_id) / name))
            samples.append(name)
    shutil.rmtree(out_dir, ignore_errors=True)
    return samples


def sample_path(bible_id: str, lora_id: str, number: int) -> Path | None:
    lora = get_lora(bible_id, lora_id)
    for name in lora.get("samples") or []:
        if name.startswith(f"{_lora_prefix(lora_id)}_sample{number}."):
            path = bibles._folder(bible_id) / name
            return path if path.is_file() else None
    return None


# ── routes ─────────────────────────────────────────────────────────────────


def register_routes(api: Any, *, ensure_pipeline: Callable[[], Any] | None = None) -> None:
    global _ensure_pipeline
    _ensure_pipeline = ensure_pipeline
    # fastapi resolves the string annotations (``from __future__ import annotations``) against the module globals.
    base = "/api/v1/director/bibles/{bible_id}/style-loras"

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (BibleError, StyleLoraError) as error:
            raise HTTPException(status_code=400, detail=str(error))

    async def body_of(request: Request) -> dict:
        if request.headers.get("content-length", "0") == "0":
            return {}
        try:
            body = await request.json()
        except ValueError:
            return {}
        return body if isinstance(body, dict) else {}

    @api.get(base)
    def style_loras_list(bible_id: str):
        return {"loras": call(list_loras, bible_id)}

    @api.get(base + "/dataset")
    def style_loras_dataset(bible_id: str, exclude_close_ups: bool = True, caption_mode: str = "content",
                            trigger: str = ""):
        """What a training would use: counts per reference and a few captions, without extracting anything."""
        bible = bibles.get_bible(bible_id)
        if not bible:
            raise HTTPException(status_code=404, detail="Unknown bible")
        settings = training_settings({"exclude_close_ups": exclude_close_ups, "caption_mode": caption_mode,
                                      "trigger": trigger}, bible)
        found = candidate_frames(bible_id, bible, settings)
        return {"available": found["available"], "references": found["references"],
                "excluded_close_ups": found["excluded_close_ups"], "excluded_credits": found["excluded_credits"],
                "minimum": MIN_IMAGES, "trigger": settings["trigger"],
                "captions": [item["caption"] for item in found["items"][:6]]}

    @api.post(base)
    async def style_loras_create(bible_id: str, request: Request):
        body = await body_of(request)
        lora = call(create_lora, bible_id, body)
        if body.get("train", True):
            lora = call(start_training, bible_id, lora["id"])
        return {"lora": lora}

    @api.get(base + "/{lora_id}")
    def style_lora_get(bible_id: str, lora_id: str):
        return {"lora": call(get_lora, bible_id, lora_id)}

    @api.patch(base + "/{lora_id}")
    async def style_lora_patch(bible_id: str, lora_id: str, request: Request):
        return {"lora": call(update_lora, bible_id, lora_id, await body_of(request))}

    @api.delete(base + "/{lora_id}")
    def style_lora_delete(bible_id: str, lora_id: str):
        return {"loras": call(delete_lora, bible_id, lora_id)}

    @api.post(base + "/{lora_id}/train")
    def style_lora_train(bible_id: str, lora_id: str):
        return {"lora": call(start_training, bible_id, lora_id)}

    @api.post(base + "/{lora_id}/cancel")
    def style_lora_cancel(bible_id: str, lora_id: str):
        lora = call(get_lora, bible_id, lora_id)
        job_id = lora.get("job_id")
        if lora_id in _LIVE_TRAININGS and job_id:
            try:
                from services.director_pipeline import _jobs
                from services.job_lifecycle import request_cancel
                job = _jobs.get(job_id)
                if job is not None:
                    request_cancel(job)
            except Exception as error:  # noqa: BLE001
                raise HTTPException(status_code=500, detail=f"Could not cancel: {error}")
        return {"lora": call(get_lora, bible_id, lora_id)}

    @api.get(base + "/{lora_id}/samples/{number}")
    def style_lora_sample(bible_id: str, lora_id: str, number: int):
        path = call(sample_path, bible_id, lora_id, number)
        if path is None:
            raise HTTPException(status_code=404, detail="No such sample")
        return FileResponse(str(path))
