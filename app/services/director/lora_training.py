"""LoRA trainings from an uploaded dataset (not tied to a bible).

A zip of images with a ``manifest.json`` (``{"trigger", "items": [{"image",
"caption"}]}``, the format ``style_lora.build_dataset`` writes) is trained on
the GPU worker like a bible's style LoRA (``services.style_lora_training``)
and lands in the Qwen 2.1 LoRA registry with a sidecar. The first use is the
panorama adapter (``room_pano``), trained once on Poly Haven's CC0 HDRIs.

Records live in ``settings/director_lora_trainings.json``; datasets under
``settings/lora_datasets/``.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

from fastapi import File, Form, HTTPException, Request, UploadFile

from . import style_lora
from .bibles import BibleError

MAX_ZIP_BYTES = 4 * 1024 * 1024 * 1024
STATES = ("queued", "training", "done", "failed")
_LOCK = threading.RLock()
_LIVE: set[str] = set()
_SLOT = threading.Semaphore(1)
_settings_dir: Callable[[], str] | None = None
_ensure_pipeline: Callable[[], Any] | None = None


def _dir() -> Path:
    base = Path(_settings_dir()) if _settings_dir else Path("settings")
    base.mkdir(parents=True, exist_ok=True)
    return base


def _records_path() -> Path:
    return _dir() / "director_lora_trainings.json"


def datasets_dir() -> Path:
    folder = _dir() / "lora_datasets"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def list_trainings() -> list[dict]:
    with _LOCK:
        try:
            data = json.loads(_records_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = []
        records = [r for r in data if isinstance(r, dict)] if isinstance(data, list) else []
        for record in records:
            if record.get("status") in ("queued", "training") and record["id"] not in _LIVE:
                record["status"], record["error"] = "failed", "Interrupted"
        return records


def _write(records: list[dict]) -> None:
    with _LOCK:
        tmp = _records_path().with_suffix(".json.tmp")
        tmp.write_text(json.dumps(records, indent=1), encoding="utf-8")
        os.replace(tmp, _records_path())


def get_training(training_id: str) -> dict:
    for record in list_trainings():
        if record["id"] == training_id:
            return record
    raise BibleError("Unknown training")


def _set(training_id: str, **fields: Any) -> dict:
    with _LOCK:
        records = list_trainings()
        for record in records:
            if record["id"] == training_id:
                record.update(fields, updated_at=_now())
                _write(records)
                return record
    raise BibleError("Unknown training")


def inspect_dataset(path: str) -> dict:
    """Counts of a dataset zip; raises BibleError when it is not one."""
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if "manifest.json" not in names:
                raise BibleError("The zip has no manifest.json")
            manifest = json.loads(archive.read("manifest.json"))
    except zipfile.BadZipFile as error:
        raise BibleError(f"Not a zip file: {error}") from error
    except ValueError as error:
        raise BibleError(f"manifest.json is not JSON: {error}") from error
    items = manifest.get("items") if isinstance(manifest, dict) else manifest
    items = [i for i in items if isinstance(i, dict) and i.get("image") in names] if isinstance(items, list) else []
    if len(items) < style_lora.MIN_IMAGES:
        raise BibleError(f"The dataset has {len(items)} usable images; at least {style_lora.MIN_IMAGES} are needed")
    return {"images": len(items), "trigger": str(manifest.get("trigger") or "") if isinstance(manifest, dict) else "",
            "captions": [str(i.get("caption") or "") for i in items[:3]]}


def create_training(name: str, dataset_zip: str, raw: Mapping[str, Any] | None = None) -> dict:
    """Register a training over an uploaded dataset zip (already under ``datasets_dir``)."""
    facts = inspect_dataset(dataset_zip)
    raw = dict(raw or {})
    settings = style_lora.training_settings({**raw, "trigger": raw.get("trigger") or facts["trigger"] or name},
                                            {"name": name})
    record = {
        "id": "train_" + secrets.token_hex(4), "name": style_lora._short(name, 60) or "lora",
        "trigger": settings["trigger"], "file": "", "status": "queued", "progress": "Waiting", "step": 0,
        "total_steps": settings["steps"], "error": "", "config": settings, "dataset": {"zip": dataset_zip, **facts},
        "job_id": "", "training": None, "created_at": _now(), "updated_at": _now(),
    }
    with _LOCK:
        records = list_trainings()
        records.append(record)
        _write(records)
    return record


def start_training(training_id: str, *, submit: Callable[..., list[str]] | None = None,
                   jobs: Mapping[str, Any] | None = None, wait: bool = False) -> dict:
    record = get_training(training_id)
    if training_id in _LIVE:
        raise BibleError("This training is running")
    _LIVE.add(training_id)
    try:
        record = _set(training_id, status="queued", progress="Waiting for the trainer", error="", step=0)
    except Exception:
        _LIVE.discard(training_id)
        raise
    thread = threading.Thread(target=_run, args=(training_id, submit, jobs), name=f"lora-train-{training_id}",
                              daemon=True)
    thread.start()
    if wait:
        thread.join()
        return get_training(training_id)
    return record


def _director():
    if _ensure_pipeline is not None:
        _ensure_pipeline()
    from services import director_pipeline
    return director_pipeline


def _run(training_id: str, submit, jobs) -> None:
    try:
        with _SLOT:
            if training_id not in _LIVE:
                return
            record = get_training(training_id)
            settings = record["config"]
            archive = record["dataset"]["zip"]
            if not os.path.isfile(archive):
                raise BibleError("The dataset zip is missing")
            submit = submit or _director()._submit_and_wait
            jobs = jobs if jobs is not None else _director()._jobs
            job_id = uuid.uuid4().hex[:8]
            folder = datasets_dir() / training_id
            folder.mkdir(parents=True, exist_ok=True)
            params = {
                "model_type": style_lora.IMAGE_MODEL, "prompt": "", "image_mode": 1, "generation_mode": "image",
                "lora_training": {"name": f"{training_id}_{record['trigger']}", "trigger": settings["trigger"],
                                  "steps": settings["steps"], "rank": settings["rank"],
                                  "learning_rate": settings["learning_rate"], "resolution": settings["resolution"],
                                  "targets": settings["targets"]},
                "lora_dataset": archive, "_director_cost_stage": "lora_training",
            }
            _set(training_id, status="training", job_id=job_id, progress="Submitting to the worker")
            done = threading.Event()
            follower = threading.Thread(
                target=style_lora._follow_generic, args=(job_id, jobs, done, lambda **f: _set(training_id, **f)),
                daemon=True)
            follower.start()
            try:
                outputs = submit(params, timeout_s=style_lora.TRAIN_TIMEOUT_SEC, out_dir=str(folder), job_id=job_id)
            finally:
                done.set()
                follower.join(timeout=5)
            weights = next((n for n in outputs or [] if str(n).endswith(".safetensors")), None)
            if not weights or not (folder / weights).is_file():
                job = jobs.get(job_id) or {}
                raise BibleError("Cancelled" if job.get("status") == "cancelled" else "The worker returned no adapter")
            summary = {}
            sidecar = folder / (os.path.splitext(weights)[0] + ".json")
            if sidecar.is_file():
                try:
                    summary = json.loads(sidecar.read_text(encoding="utf-8"))
                except ValueError:
                    summary = {}
            registry = style_lora._registry_dir()
            os.makedirs(registry, exist_ok=True)
            file = f"{style_lora._TRIGGER.sub('_', record['name'].lower()).strip('_') or training_id}.safetensors"
            shutil.copyfile(folder / weights, os.path.join(registry, file))
            with open(os.path.join(registry, os.path.splitext(file)[0] + ".civitai.json"), "w", encoding="utf-8") as h:
                json.dump({"name": record["name"], "baseModel": "Qwen Image 2.1", "trainedWords": [record["trigger"]],
                           "description": f"LoRA trained by Maestro on {record['dataset']['images']} images "
                                          f"({settings['steps']} steps, rank {settings['rank']}).",
                           "recommendedWeights": [1.0], "downloadedAt": _now(),
                           "maestro_lora_training": training_id}, h, indent=1)
            _set(training_id, file=file, status="done", progress="",
                 training={k: summary.get(k) for k in ("final_loss", "seconds", "items")})
            print(f"[LoRA training] {training_id}: {file} ({record['dataset']['images']} images)")
    except BibleError as error:
        try:
            _set(training_id, status="failed", error=str(error)[:400], progress="")
        except Exception:  # noqa: BLE001
            pass
    except Exception as error:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        try:
            _set(training_id, status="failed", error=str(error)[:400], progress="")
        except Exception:  # noqa: BLE001
            pass
    finally:
        _LIVE.discard(training_id)


def delete_training(training_id: str) -> list[dict]:
    if training_id in _LIVE:
        raise BibleError("This training is running; cancel it first")
    record = get_training(training_id)
    shutil.rmtree(datasets_dir() / training_id, ignore_errors=True)
    zip_path = (record.get("dataset") or {}).get("zip")
    if zip_path and os.path.isfile(zip_path) and Path(zip_path).parent == datasets_dir():
        os.remove(zip_path)
    with _LOCK:
        records = [r for r in list_trainings() if r["id"] != training_id]
        _write(records)
    return records


def register_routes(api: Any, *, settings_dir: Callable[[], str], ensure_pipeline: Callable[[], Any] | None = None) -> None:
    global _settings_dir, _ensure_pipeline
    _settings_dir = settings_dir
    _ensure_pipeline = ensure_pipeline
    base = "/api/v1/director/loras/trainings"

    def call(function, *args, **kwargs):
        try:
            return function(*args, **kwargs)
        except BibleError as error:
            raise HTTPException(status_code=400, detail=str(error))

    @api.get(base)
    def lora_trainings_list():
        return {"trainings": list_trainings()}

    @api.post(base)
    async def lora_trainings_create(request: Request, file: UploadFile = File(...), name: str = Form(""),
                                    trigger: str = Form(""), steps: int = Form(style_lora.DEFAULT_STEPS),
                                    rank: int = Form(style_lora.DEFAULT_RANK),
                                    resolution: int = Form(style_lora.DEFAULT_RESOLUTION),
                                    learning_rate: float = Form(style_lora.DEFAULT_LR),
                                    targets: str = Form("attention"), train: bool = Form(True)):
        """Upload a dataset zip (images + manifest.json) and train a LoRA on it."""
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_ZIP_BYTES:
            raise HTTPException(status_code=413, detail="Dataset too large")
        if not (file.filename or "").lower().endswith(".zip"):
            raise HTTPException(status_code=400, detail="Upload a zip")
        dest = datasets_dir() / f"{style_lora._TRIGGER.sub('_', (name or 'dataset').lower()).strip('_')}_{secrets.token_hex(3)}.zip"
        with open(dest, "wb") as handle:
            shutil.copyfileobj(file.file, handle)
        record = call(create_training, name or Path(file.filename).stem, str(dest),
                      {"trigger": trigger, "steps": steps, "rank": rank, "resolution": resolution,
                       "learning_rate": learning_rate, "targets": targets})
        if train:
            record = call(start_training, record["id"])
        return {"training": record}

    @api.get(base + "/{training_id}")
    def lora_training_get(training_id: str):
        return {"training": call(get_training, training_id)}

    @api.post(base + "/{training_id}/train")
    def lora_training_start(training_id: str):
        return {"training": call(start_training, training_id)}

    @api.post(base + "/{training_id}/cancel")
    def lora_training_cancel(training_id: str):
        record = call(get_training, training_id)
        if training_id in _LIVE and record.get("job_id"):
            from services.director_pipeline import _jobs
            from services.job_lifecycle import request_cancel
            job = (_jobs or {}).get(record["job_id"])
            if job is not None:
                request_cancel(job)
        return {"training": call(get_training, training_id)}

    @api.delete(base + "/{training_id}")
    def lora_training_delete(training_id: str):
        return {"trainings": call(delete_training, training_id)}
