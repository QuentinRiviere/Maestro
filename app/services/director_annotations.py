"""Post-analysis annotations on Director runs.

Reviewers pin short notes on a run, a scene, a shot, a pipeline stage or any
canvas node so the pipeline can be improved afterwards. Notes never influence
generation, so they may be written while a run is still rendering.

Storage: one sidecar JSON per run, ``_director_annotations_<pid>.json``, next
to the run's ``_director_pipeline_<pid>.json``. The running pipeline rewrites
its state file from an in-memory snapshot at every phase boundary; keeping the
notes in their own file means such a save can never drop a note written in the
meantime, without teaching every state writer to merge them back. Writes are
serialized with the same file lock and atomic replace helper as every other
Director state edit.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
import uuid
from typing import Any, Optional

from fastapi import Request
from fastapi.responses import JSONResponse, Response



def _pipeline_module():
    """Director pipeline module, imported lazily: launch.py mounts these routes at
    startup and defers the (heavy) pipeline import to the first request."""
    from services import director_pipeline
    return director_pipeline

ANNOTATION_FILE_PREFIX = "_director_annotations_"
ANNOTATIONS_VERSION = 1

CATEGORIES = (
    "storyboard",
    "image",
    "video_motion",
    "identity",
    "continuity",
    "audio_dialogue",
    "prompt",
    "edit_timing",
    "other",
)
TARGET_KINDS = ("run", "scene", "shot", "stage", "node")
MAX_TEXT_CHARS = 4000
MAX_REF_CHARS = 200
MAX_ANNOTATIONS_PER_RUN = 2000
MAX_TIMECODE_SEC = 24 * 3600
PROMPT_EXCERPT_CHARS = 240
DESCRIPTION_EXCERPT_CHARS = 200

_PID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_AID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class AnnotationValidationError(ValueError):
    """The request body is not a valid annotation."""


class AnnotationNotFoundError(LookupError):
    """The run or the annotation does not exist."""


def valid_pipeline_id(pid: Any) -> bool:
    return isinstance(pid, str) and bool(_PID_RE.match(pid))


def _clean_text(value: Any, field: str, *, limit: int, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise AnnotationValidationError(f"{field} must be text")
    cleaned = value.strip()
    if not cleaned and not allow_empty:
        raise AnnotationValidationError(f"{field} must not be empty")
    if len(cleaned) > limit:
        raise AnnotationValidationError(f"{field} is longer than {limit} characters")
    return cleaned


def _clean_target(value: Any) -> dict:
    if not isinstance(value, dict):
        raise AnnotationValidationError("target must be an object with kind and ref")
    kind = value.get("kind")
    if kind not in TARGET_KINDS:
        raise AnnotationValidationError(
            f"target.kind must be one of: {', '.join(TARGET_KINDS)}"
        )
    ref = value.get("ref", "")
    if isinstance(ref, bool):
        raise AnnotationValidationError("target.ref must be text or a shot index")
    if isinstance(ref, int):
        ref = str(ref)
    if ref is None:
        ref = ""
    ref = _clean_text(ref, "target.ref", limit=MAX_REF_CHARS, allow_empty=True)
    if kind != "run" and not ref:
        raise AnnotationValidationError("target.ref is required for this target kind")
    if kind == "shot" and not ref.isdigit():
        raise AnnotationValidationError("target.ref of a shot must be its index")
    target = {"kind": kind, "ref": ref}
    node = value.get("node")
    if node not in (None, ""):
        target["node"] = _clean_text(node, "target.node", limit=MAX_REF_CHARS)
    return target


def _clean_severity(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnnotationValidationError("severity must be 1, 2 or 3")
    if int(value) != value or not 1 <= int(value) <= 3:
        raise AnnotationValidationError("severity must be 1, 2 or 3")
    return int(value)


def _clean_timecode(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnnotationValidationError("timecode_sec must be a number of seconds")
    number = float(value)
    if not math.isfinite(number) or number < 0 or number > MAX_TIMECODE_SEC:
        raise AnnotationValidationError("timecode_sec is out of range")
    return round(number, 3)


def validate_annotation_fields(body: Any, *, partial: bool) -> dict:
    """Return the cleaned writable fields of a create (``partial=False``) or update body."""
    if not isinstance(body, dict):
        raise AnnotationValidationError("Annotation must be a JSON object")
    cleaned: dict = {}
    if "target" in body or not partial:
        cleaned["target"] = _clean_target(body.get("target"))
    if "category" in body or not partial:
        category = body.get("category", "other")
        if category not in CATEGORIES:
            raise AnnotationValidationError(
                f"category must be one of: {', '.join(CATEGORIES)}"
            )
        cleaned["category"] = category
    if "severity" in body or not partial:
        cleaned["severity"] = _clean_severity(body.get("severity", 2))
    if "text" in body or not partial:
        cleaned["text"] = _clean_text(body.get("text"), "text", limit=MAX_TEXT_CHARS)
    if "timecode_sec" in body:
        cleaned["timecode_sec"] = _clean_timecode(body.get("timecode_sec"))
    elif not partial:
        cleaned["timecode_sec"] = None
    if "resolved" in body or not partial:
        resolved = body.get("resolved", False)
        if not isinstance(resolved, bool):
            raise AnnotationValidationError("resolved must be true or false")
        cleaned["resolved"] = resolved
    if partial and not cleaned:
        raise AnnotationValidationError("No annotation field was supplied")
    return cleaned


# ── Storage ────────────────────────────────────────────────────────────────

def _run_dir(out_dir: str, pid: str) -> Optional[str]:
    """Folder holding the run's state file, or the live run's folder before its first save."""
    state_file = _pipeline_module()._find_pipeline_file(out_dir, pid)
    if state_file:
        return os.path.dirname(state_file)
    live = _pipeline_module().get_pipeline(pid)
    if live:
        return live.get("out_dir") or out_dir
    return None


def annotations_path(out_dir: str, pid: str) -> Optional[str]:
    if not valid_pipeline_id(pid):
        return None
    run_dir = _run_dir(out_dir, pid)
    if run_dir is None:
        return None
    return os.path.join(run_dir, f"{ANNOTATION_FILE_PREFIX}{pid}.json")


def _read_file_unlocked(path: str) -> list[dict]:
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return []
    items = data.get("annotations") if isinstance(data, dict) else None
    return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []


def _write_file_unlocked(path: str, pid: str, annotations: list[dict]) -> None:
    _pipeline_module()._write_pipeline_json_unlocked(path, {
        "version": ANNOTATIONS_VERSION,
        "pipeline_id": pid,
        "annotations": annotations,
    })


def _require_path(out_dir: str, pid: str) -> str:
    path = annotations_path(out_dir, pid)
    if path is None:
        raise AnnotationNotFoundError("Pipeline not found")
    return path


def list_annotations(out_dir: str, pid: str) -> list[dict]:
    path = _require_path(out_dir, pid)
    with _pipeline_module()._pipeline_file_lock:
        return _read_file_unlocked(path)


def create_annotation(out_dir: str, pid: str, body: Any) -> dict:
    fields = validate_annotation_fields(body, partial=False)
    path = _require_path(out_dir, pid)
    now = time.time()
    annotation = {
        "id": uuid.uuid4().hex[:10],
        **fields,
        "created_at": now,
        "updated_at": now,
    }
    with _pipeline_module()._pipeline_file_lock:
        annotations = _read_file_unlocked(path)
        if len(annotations) >= MAX_ANNOTATIONS_PER_RUN:
            raise AnnotationValidationError(
                f"This run already has {MAX_ANNOTATIONS_PER_RUN} annotations"
            )
        annotations.append(annotation)
        _write_file_unlocked(path, pid, annotations)
    return annotation


def update_annotation(out_dir: str, pid: str, aid: str, body: Any) -> dict:
    fields = validate_annotation_fields(body, partial=True)
    path = _require_path(out_dir, pid)
    if not isinstance(aid, str) or not _AID_RE.match(aid):
        raise AnnotationNotFoundError("Annotation not found")
    with _pipeline_module()._pipeline_file_lock:
        annotations = _read_file_unlocked(path)
        for annotation in annotations:
            if annotation.get("id") == aid:
                annotation.update(fields)
                annotation["updated_at"] = time.time()
                _write_file_unlocked(path, pid, annotations)
                return annotation
    raise AnnotationNotFoundError("Annotation not found")


def delete_annotation(out_dir: str, pid: str, aid: str) -> None:
    path = _require_path(out_dir, pid)
    with _pipeline_module()._pipeline_file_lock:
        annotations = _read_file_unlocked(path)
        kept = [item for item in annotations if item.get("id") != aid]
        if len(kept) == len(annotations):
            raise AnnotationNotFoundError("Annotation not found")
        _write_file_unlocked(path, pid, kept)


def remove_annotations_file(run_dir: str, pid: str) -> None:
    """Drop a deleted run's notes; called with the pipeline file lock held."""
    path = os.path.join(run_dir, f"{ANNOTATION_FILE_PREFIX}{pid}.json")
    try:
        os.remove(path)
    except OSError:
        pass


# ── Aggregation for post-analysis ──────────────────────────────────────────

def _excerpt(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _shot_prompt_excerpt(state: dict, target: dict) -> Optional[str]:
    if target.get("kind") != "shot":
        return None
    try:
        index = int(target.get("ref", ""))
    except ValueError:
        return None
    clips = state.get("clips") or []
    if not 0 <= index < len(clips) or not isinstance(clips[index], dict):
        return None
    clip = clips[index]
    prompt = clip.get("video_prompt") or clip.get("image_prompt") or ""
    return _excerpt(prompt, PROMPT_EXCERPT_CHARS) or None


def aggregate_annotations(out_dir: str) -> list[dict]:
    """Every run's notes with enough context to analyse them without the canvas."""
    rows: list[dict] = []
    if not os.path.isdir(out_dir):
        return rows
    scan_dirs = [out_dir] + [
        os.path.join(out_dir, name) for name in sorted(os.listdir(out_dir))
        if os.path.isdir(os.path.join(out_dir, name))
    ]
    for scan_dir in scan_dirs:
        try:
            names = sorted(os.listdir(scan_dir))
        except OSError:
            continue
        for name in names:
            if not (name.startswith(ANNOTATION_FILE_PREFIX) and name.endswith(".json")):
                continue
            pid = name[len(ANNOTATION_FILE_PREFIX):-len(".json")]
            if not valid_pipeline_id(pid):
                continue
            state_path = os.path.join(scan_dir, f"{_pipeline_module()._PIPELINE_FILE_PREFIX}{pid}.json")
            with _pipeline_module()._pipeline_file_lock:
                annotations = _read_file_unlocked(os.path.join(scan_dir, name))
                state: dict = {}
                if os.path.isfile(state_path):
                    try:
                        with open(state_path, "r", encoding="utf-8") as handle:
                            loaded = json.load(handle)
                        state = loaded if isinstance(loaded, dict) else {}
                    except (OSError, ValueError):
                        state = {}
            description = _excerpt(state.get("scene_description"), DESCRIPTION_EXCERPT_CHARS)
            workspace = "default" if scan_dir == out_dir else os.path.basename(scan_dir)
            for annotation in annotations:
                target = annotation.get("target") if isinstance(annotation.get("target"), dict) else {}
                rows.append({
                    "pid": pid,
                    "workspace": workspace,
                    "run_description": description,
                    "run_status": state.get("status"),
                    "run_created_at": state.get("created_at"),
                    "id": annotation.get("id"),
                    "target": target,
                    "category": annotation.get("category"),
                    "severity": annotation.get("severity"),
                    "text": annotation.get("text"),
                    "timecode_sec": annotation.get("timecode_sec"),
                    "resolved": bool(annotation.get("resolved")),
                    "created_at": annotation.get("created_at"),
                    "updated_at": annotation.get("updated_at"),
                    "shot_prompt_excerpt": _shot_prompt_excerpt(state, target),
                })
    rows.sort(key=lambda row: row.get("created_at") or 0, reverse=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    by_category: dict[str, int] = {}
    by_severity: dict[str, int] = {}
    open_count = 0
    for row in rows:
        by_category[str(row.get("category"))] = by_category.get(str(row.get("category")), 0) + 1
        by_severity[str(row.get("severity"))] = by_severity.get(str(row.get("severity")), 0) + 1
        if not row.get("resolved"):
            open_count += 1
    return {
        "total": len(rows),
        "open": open_count,
        "runs": len({row.get("pid") for row in rows}),
        "by_category": by_category,
        "by_severity": by_severity,
    }


# ── HTTP routes ────────────────────────────────────────────────────────────

def register_routes(api, *, get_base_dir) -> None:
    """Mount the annotation endpoints. ``get_base_dir`` returns the output root."""
    def error(message: str, status: int):
        return JSONResponse({"error": message}, status_code=status)

    async def read_body(request: Request):
        try:
            return await request.json()
        except Exception:
            raise AnnotationValidationError("Request body must be valid JSON")

    @api.get("/api/v1/director/annotations")
    def list_all_annotations(include_resolved: bool = True):
        rows = aggregate_annotations(get_base_dir())
        if not include_resolved:
            rows = [row for row in rows if not row.get("resolved")]
        return {"annotations": rows, "summary": summarize(rows), "categories": list(CATEGORIES)}

    @api.get("/api/v1/director/annotations/export")
    def export_all_annotations():
        rows = aggregate_annotations(get_base_dir())
        payload = {
            "exported_at": time.time(),
            "categories": list(CATEGORIES),
            "summary": summarize(rows),
            "annotations": rows,
        }
        stamp = time.strftime("%Y%m%d-%H%M%S")
        return Response(
            json.dumps(payload, indent=2, ensure_ascii=False),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="director-annotations-{stamp}.json"'},
        )

    @api.get("/api/v1/director/pipelines/{pid}/annotations")
    def get_run_annotations(pid: str):
        try:
            return {"annotations": list_annotations(get_base_dir(), pid)}
        except AnnotationNotFoundError as exc:
            return error(str(exc), 404)

    @api.post("/api/v1/director/pipelines/{pid}/annotations")
    async def post_run_annotation(pid: str, request: Request):
        try:
            body = await read_body(request)
            return {"annotation": create_annotation(get_base_dir(), pid, body)}
        except AnnotationValidationError as exc:
            return error(str(exc), 400)
        except AnnotationNotFoundError as exc:
            return error(str(exc), 404)

    @api.put("/api/v1/director/pipelines/{pid}/annotations/{aid}")
    async def put_run_annotation(pid: str, aid: str, request: Request):
        try:
            body = await read_body(request)
            return {"annotation": update_annotation(get_base_dir(), pid, aid, body)}
        except AnnotationValidationError as exc:
            return error(str(exc), 400)
        except AnnotationNotFoundError as exc:
            return error(str(exc), 404)

    @api.delete("/api/v1/director/pipelines/{pid}/annotations/{aid}")
    def delete_run_annotation(pid: str, aid: str):
        try:
            delete_annotation(get_base_dir(), pid, aid)
        except AnnotationNotFoundError as exc:
            return error(str(exc), 404)
        return {"status": "ok"}
