"""Video fetch service: downloads a public video (YouTube and whatever yt-dlp supports) at 480p.

The Maestro controller asks for a URL, polls the job, then streams the file
into a bible as a style reference. Everything stays on this host: jobs in
memory, files under DATA_DIR (one folder per job), removed after TTL_HOURS.

  POST   /api/fetch            {"url": "...", "max_height": 480}   -> job
  GET    /api/fetch             -> {"jobs": [...]}
  GET    /api/fetch/{id}        -> job
  GET    /api/fetch/{id}/file   -> the video
  DELETE /api/fetch/{id}        -> removes the job and its folder
  GET    /healthz

Every /api call needs ``Authorization: Bearer $VIDEO_FETCH_TOKEN``.
"""
from __future__ import annotations

import os
import secrets
import shutil
import threading
import time
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

DATA_DIR = os.environ.get("DATA_DIR", "/data")
TOKEN = os.environ.get("VIDEO_FETCH_TOKEN", "")
TTL_HOURS = float(os.environ.get("TTL_HOURS", "48"))
MAX_JOBS = 200
DEFAULT_HEIGHT = 480
# Tolerated hosts: a URL from anywhere else is refused before yt-dlp sees it.
ALLOWED_HOSTS = tuple(h.strip().lower() for h in os.environ.get(
    "ALLOWED_HOSTS", "youtube.com,youtu.be,vimeo.com,dailymotion.com").split(",") if h.strip())

_LOCK = threading.Lock()
_JOBS: dict[str, dict] = {}
_SLOT = threading.Semaphore(int(os.environ.get("PARALLEL", "1")))


class FetchRequest(BaseModel):
    url: str = Field(min_length=8, max_length=2000)
    max_height: int = Field(default=DEFAULT_HEIGHT, ge=144, le=2160)


def _now() -> float:
    return time.time()


def _host_allowed(url: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    host = parsed.hostname.lower()
    return any(host == allowed or host.endswith("." + allowed) for allowed in ALLOWED_HOSTS)


def _public(job: dict) -> dict:
    return {key: job.get(key) for key in ("id", "url", "status", "progress", "title", "duration_sec", "bytes",
                                          "filename", "error", "created_at", "updated_at", "width", "height")}


def _set(job_id: str, **fields: Any) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            job.update(fields, updated_at=_now())


def _run(job_id: str) -> None:
    import yt_dlp

    with _LOCK:
        job = dict(_JOBS[job_id])
    folder = os.path.join(DATA_DIR, job_id)
    os.makedirs(folder, exist_ok=True)

    def hook(data: dict) -> None:
        if data.get("status") == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
            done = data.get("downloaded_bytes") or 0
            percent = f"{100 * done / total:.0f}%" if total else f"{done / 1e6:.0f} MB"
            _set(job_id, progress=f"Downloading {percent}")
        elif data.get("status") == "finished":
            _set(job_id, progress="Merging")

    options = {
        "format": "bv*+ba/b",
        "format_sort": [f"res:{job['max_height']}", "vcodec:h264", "acodec:m4a"],
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(folder, "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
        "js_runtimes": {"node": {}},
        "retries": 3,
        "socket_timeout": 30,
        "max_filesize": 4 * 1024 * 1024 * 1024,
    }
    with _SLOT:
        _set(job_id, status="downloading", progress="Starting")
        try:
            with yt_dlp.YoutubeDL(options) as ydl:
                info = ydl.extract_info(job["url"], download=True)
            if info.get("entries"):
                info = next((e for e in info["entries"] if e), info)
            files = [name for name in os.listdir(folder) if not name.endswith((".part", ".ytdl"))]
            if not files:
                raise RuntimeError("yt-dlp produced no file")
            filename = max(files, key=lambda name: os.path.getsize(os.path.join(folder, name)))
            path = os.path.join(folder, filename)
            _set(job_id, status="done", progress="", filename=filename, bytes=os.path.getsize(path),
                 title=str(info.get("title") or "")[:200], duration_sec=info.get("duration"),
                 width=info.get("width"), height=info.get("height"))
        except Exception as error:  # noqa: BLE001 - recorded on the job
            _set(job_id, status="failed", progress="", error=str(error)[:400])


def _cleanup() -> None:
    while True:
        cutoff = _now() - TTL_HOURS * 3600
        with _LOCK:
            stale = [job_id for job_id, job in _JOBS.items()
                     if job["created_at"] < cutoff and job["status"] in ("done", "failed")]
            for job_id in stale:
                _JOBS.pop(job_id, None)
        for job_id in stale:
            shutil.rmtree(os.path.join(DATA_DIR, job_id), ignore_errors=True)
        if os.path.isdir(DATA_DIR):
            for name in os.listdir(DATA_DIR):  # folders of a previous run
                path = os.path.join(DATA_DIR, name)
                with _LOCK:
                    known = name in _JOBS
                if not known and os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
        time.sleep(600)


def auth(request: Request) -> None:
    if not TOKEN:
        raise HTTPException(status_code=503, detail="VIDEO_FETCH_TOKEN is not set")
    header = request.headers.get("authorization", "")
    given = header[7:] if header.lower().startswith("bearer ") else ""
    if not given or not secrets.compare_digest(given, TOKEN):
        raise HTTPException(status_code=401, detail="Unauthorized")


def main_app() -> FastAPI:
    app = FastAPI(title="Video fetch", docs_url=None, redoc_url=None, openapi_url=None)
    os.makedirs(DATA_DIR, exist_ok=True)
    threading.Thread(target=_cleanup, name="video-fetch-cleanup", daemon=True).start()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    @app.post("/api/fetch", dependencies=[Depends(auth)])
    def fetch(body: FetchRequest) -> dict:
        if not _host_allowed(body.url):
            raise HTTPException(status_code=400, detail="This host is not allowed")
        with _LOCK:
            if len(_JOBS) >= MAX_JOBS:
                raise HTTPException(status_code=429, detail="Too many jobs; wait for the cleanup")
            job_id = secrets.token_hex(8)
            _JOBS[job_id] = {"id": job_id, "url": body.url, "max_height": body.max_height, "status": "queued",
                             "progress": "Waiting", "title": "", "duration_sec": None, "bytes": None,
                             "filename": "", "error": "", "created_at": _now(), "updated_at": _now(),
                             "width": None, "height": None}
            job = dict(_JOBS[job_id])
        threading.Thread(target=_run, args=(job_id,), name=f"video-fetch-{job_id}", daemon=True).start()
        return {"job": _public(job)}

    @app.get("/api/fetch", dependencies=[Depends(auth)])
    def list_jobs() -> dict:
        with _LOCK:
            return {"jobs": [_public(j) for j in sorted(_JOBS.values(), key=lambda j: j["created_at"])]}

    @app.get("/api/fetch/{job_id}", dependencies=[Depends(auth)])
    def get_job(job_id: str) -> dict:
        with _LOCK:
            job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return {"job": _public(job)}

    @app.get("/api/fetch/{job_id}/file", dependencies=[Depends(auth)])
    def get_file(job_id: str) -> FileResponse:
        with _LOCK:
            job = dict(_JOBS.get(job_id) or {})
        if not job or job.get("status") != "done":
            raise HTTPException(status_code=404, detail="No file for this job")
        path = os.path.join(DATA_DIR, job_id, os.path.basename(job["filename"]))
        if not os.path.isfile(path):
            raise HTTPException(status_code=410, detail="The file was removed")
        return FileResponse(path, media_type="video/mp4", filename=job["filename"])

    @app.delete("/api/fetch/{job_id}", dependencies=[Depends(auth)])
    def delete_job(job_id: str) -> dict:
        with _LOCK:
            job = _JOBS.pop(job_id, None)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        shutil.rmtree(os.path.join(DATA_DIR, job_id), ignore_errors=True)
        return {"deleted": job_id}

    return app
