"""Model Context Protocol endpoint for driving Maestro from an AI agent.

Stateless Streamable HTTP transport: every JSON-RPC message is a POST to
``/mcp`` answered with ``application/json`` (no SSE stream, no session id),
which the MCP specification allows. It is implemented directly instead of
through the ``mcp`` SDK because the SDK releases with Streamable HTTP need a
newer pydantic than the one Maestro pins.

Tools call Maestro's own REST API over loopback, so they get the validation,
queueing and remote-worker routing of the web UI. They expose a small,
agent-friendly surface instead of the 200+ raw routes (most of which read
untyped JSON bodies and so have no usable OpenAPI request schema).
"""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

log = logging.getLogger(__name__)

SERVER_NAME = "maestro"
SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
MAX_WAIT_SECONDS = 55
_TERMINAL = {"completed", "failed", "cancelled"}
_PIPELINE_TERMINAL = {"completed", "failed", "cancelled", "stopped", "error"}
DIRECTOR_PIPELINE_TYPES = ("short_film_story", "short_film_audio", "music_video", "viral_video")
VIRAL_PLATFORMS = ("general", "tiktok", "reels", "shorts")
VIRAL_STYLES = ("cinematic", "meme", "ugc", "direct_address", "reaction", "tutorial")

INSTRUCTIONS = """\
Maestro generates video, images and audio (MiniMax H3, LTX, Wan, Flux, ...).

Single clip (Studio): list_models -> optional upload_media for reference
images/audio -> optional enhance_prompt -> generate -> get_job with
wait_seconds until completed -> output URLs.

Multi-shot project (Director): director_start with a brief (scene_description)
-> director_status with wait_seconds. The pipeline plans shots with the LLM,
renders start images, then videos, and joins them. If auto_mode is false it
pauses for review; call director_continue to proceed.

Generations may run on a remote GPU worker that boots on demand (several
minutes the first time). remote_worker reports and controls it; starting it
bills the GPU. Video renders take minutes: poll, never resubmit."""


class ToolError(Exception):
    """A tool failure reported to the agent as an isError result."""


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], Awaitable[Any]]
    read_only: bool = False
    destructive: bool = False

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": self.input_schema,
            "annotations": {"readOnlyHint": self.read_only, "destructiveHint": self.destructive},
        }


def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": properties, "additionalProperties": False}
    if required:
        schema["required"] = required
    return schema


_STR = {"type": "string"}
_STR_LIST = {"type": "array", "items": {"type": "string"}}
_WAIT = {"type": "number", "minimum": 0, "maximum": MAX_WAIT_SECONDS,
         "description": f"Block up to this many seconds (max {MAX_WAIT_SECONDS}) until it finishes"}


class MaestroMcp:
    """JSON-RPC dispatcher plus the Maestro tool set."""

    def __init__(
        self,
        call_api: Callable[..., Awaitable[httpx.Response]],
        *,
        version: str = "1.0.0",
        public_base_url: Callable[[], str] = lambda: "",
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
        fetch_url: Callable[[str], Awaitable[tuple[bytes, str]]] | None = None,
    ) -> None:
        self._call_api = call_api
        self._version = version
        self._public_base_url = public_base_url
        self._sleep = sleep
        self._fetch_url = fetch_url or _fetch_url
        self.tools = {tool.name: tool for tool in self._build_tools()}

    # ── JSON-RPC ───────────────────────────────────────────────────────────

    async def handle(self, message: Any) -> dict[str, Any] | None:
        """Answer one JSON-RPC message; None for notifications."""
        if not isinstance(message, dict) or message.get("jsonrpc") != "2.0":
            return _error(None, -32600, "Invalid Request")
        method = message.get("method")
        msg_id = message.get("id")
        is_notification = "id" not in message
        if not isinstance(method, str):
            return None if is_notification else _error(msg_id, -32600, "Invalid Request")
        params = message.get("params") or {}
        try:
            if method == "initialize":
                result = self._initialize(params)
            elif method == "ping":
                result = {}
            elif method == "tools/list":
                result = {"tools": [tool.descriptor() for tool in self.tools.values()]}
            elif method == "tools/call":
                result = await self._call_tool(params)
            elif method.startswith("notifications/"):
                return None
            else:
                return None if is_notification else _error(msg_id, -32601, f"Method not found: {method}")
        except _RpcError as exc:
            return None if is_notification else _error(msg_id, exc.code, str(exc))
        return None if is_notification else {"jsonrpc": "2.0", "id": msg_id, "result": result}

    def _initialize(self, params: dict[str, Any]) -> dict[str, Any]:
        requested = str(params.get("protocolVersion") or "")
        version = requested if requested in SUPPORTED_PROTOCOL_VERSIONS else SUPPORTED_PROTOCOL_VERSIONS[0]
        return {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "title": "Maestro", "version": self._version},
            "instructions": INSTRUCTIONS,
        }

    async def _call_tool(self, params: dict[str, Any]) -> dict[str, Any]:
        name = params.get("name")
        tool = self.tools.get(name) if isinstance(name, str) else None
        if tool is None:
            raise _RpcError(-32602, f"Unknown tool: {name}")
        arguments = params.get("arguments") or {}
        if not isinstance(arguments, dict):
            raise _RpcError(-32602, "Tool arguments must be an object")
        try:
            value = await tool.handler(arguments)
        except ToolError as exc:
            return {"content": [{"type": "text", "text": str(exc)}], "isError": True}
        except Exception as exc:  # unexpected: still a tool result, never a transport error
            log.exception("MCP tool %s failed", name)
            return {"content": [{"type": "text", "text": f"{name} failed: {exc}"}], "isError": True}
        result: dict[str, Any] = {
            "content": [{"type": "text", "text": json.dumps(value, indent=2, default=str)}],
            "isError": False,
        }
        if isinstance(value, dict):
            result["structuredContent"] = value
        return result

    # ── REST helpers ───────────────────────────────────────────────────────

    async def _api(self, method: str, path: str, **kwargs: Any) -> Any:
        response = await self._call_api(method, path, **kwargs)
        if response.status_code >= 400:
            detail: Any = response.text
            try:
                detail = response.json().get("detail", detail)
            except Exception:
                pass
            raise ToolError(f"Maestro {method} {path} failed (HTTP {response.status_code}): {detail}")
        return response.json() if response.content else None

    def _absolute(self, url: Any) -> Any:
        if not isinstance(url, str) or not url.startswith("/"):
            return url
        base = self._public_base_url().rstrip("/")
        return f"{base}{url}" if base else url

    async def _defaults(self, model_type: str) -> dict[str, Any]:
        return dict(await self._api("GET", f"/api/v1/defaults/{quote(model_type, safe='')}") or {})

    async def _wait(self, fetch: Callable[[], Awaitable[dict[str, Any]]], done: Callable[[dict[str, Any]], bool],
                    wait_seconds: Any) -> dict[str, Any]:
        wait = max(0.0, min(float(wait_seconds or 0), MAX_WAIT_SECONDS))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait
        while True:
            value = await fetch()
            remaining = deadline - loop.time()
            if done(value) or remaining <= 0:
                return value
            await self._sleep(min(3.0, remaining))

    # ── tool table ─────────────────────────────────────────────────────────

    def _build_tools(self) -> list[Tool]:
        return [
            Tool("list_models",
                 "List generation models (model_type ids for generate and director_start). Filter by "
                 "family (e.g. 'ltx', 'minimax', 'wan', 'flux') or a search string.",
                 _schema({"family": _STR, "search": _STR}), self._list_models, read_only=True),
            Tool("get_model_defaults",
                 "Default generation parameters of a model (resolution, video_length, steps, ...). "
                 "Any of them can be overridden through generate's params.",
                 _schema({"model_type": _STR}, ["model_type"]), self._get_model_defaults, read_only=True),
            Tool("upload_media",
                 "Upload a reference image, audio or video from a URL or base64 data. Returns the server "
                 "path to pass to generate params (image_start, image_refs, audio_guide, ...) or to "
                 "director_start (reference_image_path, character_ref_paths, audio_path).",
                 _schema({"url": {"type": "string", "description": "http(s) URL to download"},
                          "data_base64": {"type": "string", "description": "File content, base64"},
                          "filename": {"type": "string", "description": "Name with extension, e.g. ref.png"}}),
                 self._upload_media),
            Tool("enhance_prompt",
                 "Rewrite a prompt with Maestro's model-specific prompt guide (uses the configured LLM).",
                 _schema({"prompt": _STR, "model_type": _STR,
                          "mode": {"type": "string", "enum": ["video", "image", "avatar", "audio"], "default": "video"},
                          "image_paths": _STR_LIST}, ["prompt", "model_type"]),
                 self._enhance_prompt),
            Tool("generate",
                 "Queue one Studio generation. Starts from the model defaults, then applies prompt and "
                 "params. Returns a job_id immediately; follow it with get_job.",
                 _schema({"model_type": _STR, "prompt": _STR,
                          "params": {"type": "object", "description": "Extra or overridden generation parameters"},
                          "hold": {"type": "boolean", "default": False,
                                   "description": "Add to the Studio queue without starting it"}},
                         ["model_type", "prompt"]),
                 self._generate),
            Tool("get_job",
                 "Status of a Studio job with progress and, once completed, output URLs.",
                 _schema({"job_id": _STR, "wait_seconds": _WAIT}, ["job_id"]), self._get_job, read_only=True),
            Tool("list_active_jobs", "Studio jobs that are held, queued or running.",
                 _schema({}), self._list_active_jobs, read_only=True),
            Tool("cancel_job", "Cancel a queued or running Studio job.",
                 _schema({"job_id": _STR}, ["job_id"]), self._cancel_job, destructive=True),
            Tool("list_outputs", "Most recent generated media in the gallery.",
                 _schema({"limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                          "search": _STR}),
                 self._list_outputs, read_only=True),
            Tool("director_start",
                 "Start a Director project: the LLM plans the shots from the brief, then Maestro renders "
                 "start images and videos and joins them. Returns a pipeline_id; follow it with "
                 "director_status. music_video and short_film_audio need audio_path (upload_media first). "
                 "viral_video writes a short hook-first vertical-style piece from the brief.",
                 _schema({
                     "scene_description": {"type": "string", "description": "The brief: story, style, characters"},
                     "pipeline_type": {"type": "string", "enum": list(DIRECTOR_PIPELINE_TYPES),
                                       "default": "short_film_story"},
                     "video_model": {"type": "string", "description": "model_type of the video model"},
                     "image_model": {"type": "string", "description": "model_type for start images "
                                                                     "(default flux2_klein_9b)"},
                     "target_duration": {"type": "number", "description": "Seconds (short films, viral videos)"},
                     "platform": {"type": "string", "enum": list(VIRAL_PLATFORMS),
                                  "description": "viral_video only"},
                     "style": {"type": "string", "enum": list(VIRAL_STYLES), "description": "viral_video only"},
                     "aspect_ratio": {"type": "string", "description": "e.g. 16:9, 9:16, 1:1"},
                     "resolution_preset": {"type": "string", "description": "e.g. 720p, 1080p"},
                     "audio_path": _STR,
                     "reference_image_path": _STR,
                     "character_ref_paths": _STR_LIST,
                     "character_ref_labels": _STR_LIST,
                     "location_ref_paths": _STR_LIST,
                     "location_ref_labels": _STR_LIST,
                     "auto_mode": {"type": "boolean", "default": True,
                                   "description": "False pauses after planning for review (director_continue)"},
                     "params": {"type": "object", "description": "Advanced: raw pipeline params merged last "
                                                                "(video_params, image_params, seamless, ...)"},
                 }, ["scene_description", "video_model"]),
                 self._director_start),
            Tool("director_status",
                 "Phase, progress, planned clips and outputs of a running or saved Director project.",
                 _schema({"pipeline_id": _STR, "wait_seconds": _WAIT}, ["pipeline_id"]),
                 self._director_status, read_only=True),
            Tool("director_continue",
                 "Resume a paused Director project, optionally with edited clip_plans from director_status.",
                 _schema({"pipeline_id": _STR, "clip_plans": {"type": "array", "items": {"type": "object"}}},
                         ["pipeline_id"]),
                 self._director_continue),
            Tool("director_stop", "Stop a running Director project.",
                 _schema({"pipeline_id": _STR}, ["pipeline_id"]), self._director_stop, destructive=True),
            Tool("list_director_projects", "Saved Director projects.",
                 _schema({}), self._list_director_projects, read_only=True),
            Tool("remote_worker",
                 "Remote GPU worker (dstack/RunPod): 'status', 'start' (optionally preloading model_type) or "
                 "'stop'. Starting bills the GPU until it stops (idle TTL or stop). Generations start it "
                 "automatically when needed.",
                 _schema({"action": {"type": "string", "enum": ["status", "start", "stop"]}, "model_type": _STR},
                         ["action"]),
                 self._remote_worker),
        ]

    # ── Studio ─────────────────────────────────────────────────────────────

    async def _list_models(self, args: dict[str, Any]) -> dict[str, Any]:
        data = await self._api("GET", "/api/v1/models")
        family = str(args.get("family") or "").lower()
        search = str(args.get("search") or "").lower()
        models = []
        for model in data.get("models", []):
            haystack = f"{model.get('model_type', '')} {model.get('name', '')}".lower()
            if family and family not in str(model.get("family") or "").lower() and family not in haystack:
                continue
            if search and search not in haystack and search not in str(model.get("description") or "").lower():
                continue
            models.append({
                "model_type": model.get("model_type"),
                "name": model.get("name"),
                "family": model.get("family"),
                "description": str(model.get("description") or "")[:240],
                "image_to_video": model.get("is_i2v"),
                "text_to_video": model.get("is_t2v"),
                "supports_audio_input": model.get("supports_audio_input"),
                "generates_audio": model.get("generates_audio"),
                "supports_ref_images": model.get("supports_ref_images"),
            })
        return {"count": len(models), "models": models}

    async def _get_model_defaults(self, args: dict[str, Any]) -> dict[str, Any]:
        return await self._defaults(_required(args, "model_type"))

    async def _upload_media(self, args: dict[str, Any]) -> dict[str, Any]:
        url, encoded = args.get("url"), args.get("data_base64")
        if bool(url) == bool(encoded):
            raise ToolError("Pass exactly one of url or data_base64.")
        if url:
            if urlparse(str(url)).scheme not in ("http", "https"):
                raise ToolError("url must be http(s).")
            data, content_type = await self._fetch_url(str(url))
            filename = str(args.get("filename") or os.path.basename(urlparse(str(url)).path) or "upload")
        else:
            try:
                data = base64.b64decode(str(encoded), validate=True)
            except Exception as exc:
                raise ToolError(f"data_base64 is not valid base64: {exc}") from exc
            content_type = "application/octet-stream"
            filename = str(args.get("filename") or "upload")
        if len(data) > MAX_UPLOAD_BYTES:
            raise ToolError("File too large (max 500 MB).")
        if not os.path.splitext(filename)[1]:
            raise ToolError("filename needs an extension (e.g. ref.png, voice.wav).")
        result = await self._api("POST", "/api/v1/upload", files={"file": (filename, data, content_type)},
                                 timeout=600)
        result["url"] = self._absolute(result.get("url"))
        return result

    async def _enhance_prompt(self, args: dict[str, Any]) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": _required(args, "prompt"),
            "model_type": _required(args, "model_type"),
            "mode": args.get("mode") or "video",
        }
        if args.get("image_paths"):
            body["image_paths"] = list(args["image_paths"])
        return await self._api("POST", "/api/v1/llm/enhance-prompt", json=body, timeout=600)

    async def _generate(self, args: dict[str, Any]) -> dict[str, Any]:
        model_type = _required(args, "model_type")
        prompt = _required(args, "prompt")
        extra = args.get("params") or {}
        if not isinstance(extra, dict):
            raise ToolError("params must be an object.")
        body = await self._defaults(model_type)
        body.update(extra)
        body.update({
            "model_type": model_type,
            "prompt": prompt,
            "_queue_mode": "held" if args.get("hold") else "now",
        })
        result = await self._api("POST", "/api/v1/generate", json=body, timeout=600)
        return {"job_id": result.get("job_id"), "status": result.get("status"),
                **{k: v for k, v in result.items() if k not in ("job_id", "status")}}

    async def _get_job(self, args: dict[str, Any]) -> dict[str, Any]:
        job_id = _required(args, "job_id")
        job = await self._wait(
            lambda: self._api("GET", f"/api/v1/jobs/{quote(job_id, safe='')}"),
            lambda value: value.get("status") in _TERMINAL,
            args.get("wait_seconds"),
        )
        for item in job.get("outputs") or []:
            item["url"] = self._absolute(item.get("url"))
        if isinstance(job.get("preview"), dict):
            job["preview"]["url"] = self._absolute(job["preview"].get("url"))
        return job

    async def _list_active_jobs(self, args: dict[str, Any]) -> dict[str, Any]:
        data = await self._api("GET", "/api/v1/jobs")
        jobs = data.get("jobs", []) if isinstance(data, dict) else data or []
        return {"jobs": [
            {k: job.get(k) for k in ("job_id", "status", "kind", "progress", "phase", "message")}
            for job in jobs
        ]}

    async def _cancel_job(self, args: dict[str, Any]) -> dict[str, Any]:
        job_id = _required(args, "job_id")
        return await self._api("POST", f"/api/v1/cancel/{quote(job_id, safe='')}") or {"status": "cancel requested"}

    async def _list_outputs(self, args: dict[str, Any]) -> dict[str, Any]:
        query: dict[str, Any] = {"limit": max(1, min(int(args.get("limit") or 20), 100))}
        if args.get("search"):
            query["search"] = str(args["search"])
        data = await self._api("GET", "/api/v1/outputs", params=query)
        return {"outputs": [
            {"name": item.get("name"), "type": item.get("type"), "size": item.get("size"),
             "created_at": item.get("created_at"), "url": self._absolute(item.get("url"))}
            for item in data.get("outputs", [])
        ]}

    # ── Director ───────────────────────────────────────────────────────────

    async def _director_start(self, args: dict[str, Any]) -> dict[str, Any]:
        pipeline_type = str(args.get("pipeline_type") or "short_film_story")
        if pipeline_type not in DIRECTOR_PIPELINE_TYPES:
            raise ToolError(f"pipeline_type must be one of {', '.join(DIRECTOR_PIPELINE_TYPES)}.")
        if pipeline_type in ("music_video", "short_film_audio") and not args.get("audio_path"):
            raise ToolError(f"{pipeline_type} needs audio_path (upload the track with upload_media).")
        skill_fields: dict[str, Any] = {}
        if pipeline_type == "viral_video":
            if args.get("audio_path"):
                raise ToolError("viral_video from MCP is written from the brief; audio-driven viral videos "
                                "need the audio analysis of the Director UI.")
            platform = str(args.get("platform") or "general")
            style = str(args.get("style") or "cinematic")
            if platform not in VIRAL_PLATFORMS or style not in VIRAL_STYLES:
                raise ToolError(f"platform must be one of {', '.join(VIRAL_PLATFORMS)}; "
                                f"style one of {', '.join(VIRAL_STYLES)}.")
            # A viral video is the story workflow with the viral planner.
            pipeline_type = "short_film_story"
            skill_fields = {"director_skill": "viral_video", "platform": platform, "style": style,
                            "target_duration": args.get("target_duration") or 30}
        scene_description = _required(args, "scene_description")
        video_model = _required(args, "video_model")
        image_model = str(args.get("image_model") or "flux2_klein_9b")
        services = await self._api("GET", "/api/v1/services-config") or {}

        body: dict[str, Any] = {
            "pipeline_type": pipeline_type,
            "scene_description": scene_description,
            "auto_mode": bool(args.get("auto_mode", True)),
            "use_director_v2": services.get("use_director_v2", True),
            "llm_model_id": services.get("llm_model_id"),
            "llm_device": services.get("llm_device"),
            "llm_provider": services.get("llm_provider") or "local",
            "video_model": video_model,
            "video_params": await self._defaults(video_model),
            "image_model": image_model,
            "image_params": await self._defaults(image_model),
            "video_loras": {}, "image_loras": {},
            **skill_fields,
        }
        optional = {
            "target_duration": "target_duration",
            "aspect_ratio": "director_aspect_ratio",
            "resolution_preset": "director_resolution_preset",
            "audio_path": "audio_path",
            "reference_image_path": "reference_image_path",
            "character_ref_paths": "character_ref_paths",
            "character_ref_labels": "character_ref_labels",
            "location_ref_paths": "location_ref_paths",
            "location_ref_labels": "location_ref_labels",
        }
        for arg, key in optional.items():
            if args.get(arg) not in (None, "", []):
                body[key] = args[arg]
        extra = args.get("params") or {}
        if not isinstance(extra, dict):
            raise ToolError("params must be an object.")
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(body.get(key), dict):
                body[key] = {**body[key], **value}
            else:
                body[key] = value
        result = await self._api("POST", "/api/v1/director/pipeline/start", json=body, timeout=600)
        return {"pipeline_id": result.get("pipeline_id"), "status": "running"}

    async def _director_status(self, args: dict[str, Any]) -> dict[str, Any]:
        pid = _required(args, "pipeline_id")
        status = await self._wait(
            lambda: self._api("GET", f"/api/v1/director/pipeline/{quote(pid, safe='')}"),
            lambda value: value.get("status") in _PIPELINE_TERMINAL or value.get("status") == "paused",
            args.get("wait_seconds"),
        )
        for key in ("output_files", "final_video", "joined_file"):
            if isinstance(status.get(key), str):
                status[key] = self._absolute(status[key])
        return status

    async def _director_continue(self, args: dict[str, Any]) -> dict[str, Any]:
        pid = _required(args, "pipeline_id")
        body = {"clip_plans": args["clip_plans"]} if args.get("clip_plans") else None
        kwargs: dict[str, Any] = {"json": body} if body else {}
        return await self._api("POST", f"/api/v1/director/pipeline/{quote(pid, safe='')}/continue", **kwargs)

    async def _director_stop(self, args: dict[str, Any]) -> dict[str, Any]:
        pid = _required(args, "pipeline_id")
        return await self._api("POST", f"/api/v1/director/pipeline/{quote(pid, safe='')}/stop")

    async def _list_director_projects(self, args: dict[str, Any]) -> Any:
        return await self._api("GET", "/api/v1/director/pipelines")

    # ── remote worker ──────────────────────────────────────────────────────

    async def _remote_worker(self, args: dict[str, Any]) -> dict[str, Any]:
        action = _required(args, "action")
        if action == "status":
            return await self._api("GET", "/api/v1/remote-worker/status")
        if action == "start":
            body = {"model_type": args["model_type"]} if args.get("model_type") else {}
            return await self._api("POST", "/api/v1/remote-worker/start", json=body, timeout=120)
        if action == "stop":
            return await self._api("POST", "/api/v1/remote-worker/stop", json={}, timeout=120)
        raise ToolError("action must be status, start or stop.")


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _error(msg_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def _required(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ToolError(f"{key} is required.")
    return value.strip()


async def _fetch_url(url: str) -> tuple[bytes, str]:
    async with httpx.AsyncClient(follow_redirects=True, timeout=120) as client:
        async with client.stream("GET", url) as response:
            if response.status_code >= 400:
                raise ToolError(f"Download failed: HTTP {response.status_code}")
            chunks, total = [], 0
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise ToolError("File too large (max 500 MB).")
                chunks.append(chunk)
            content_type = response.headers.get("content-type", "application/octet-stream").split(";")[0]
            return b"".join(chunks), content_type


_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}


def origin_allowed(origin: str | None, environ: dict[str, str] | None = None) -> bool:
    """Reject browser requests from other sites (DNS-rebinding guard).

    Non-browser clients (agents, LiteLLM, Claude Desktop bridges) send no
    Origin header and are accepted; MAESTRO_MCP_ALLOWED_ORIGINS adds hosts.
    """
    if not origin:
        return True
    env = os.environ if environ is None else environ
    extra = {item.strip().lower() for item in env.get("MAESTRO_MCP_ALLOWED_ORIGINS", "").split(",") if item.strip()}
    host = (urlparse(origin).hostname or "").lower()
    return host in _LOCAL_HOSTS or host in extra or origin.lower() in extra


def mount_mcp(api: FastAPI, *, port: Callable[[], int], version: str = "1.0.0") -> MaestroMcp:
    """Expose the MCP endpoint at /mcp, calling this server over loopback.

    ``port`` is read per call: the listening port is only final at startup.
    """
    configured_public = os.environ.get("MAESTRO_PUBLIC_URL", "").rstrip("/")
    seen_base = {"url": ""}

    async def call_api(method: str, path: str, *, timeout: float = 60, **kwargs: Any) -> httpx.Response:
        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port()}", timeout=timeout) as client:
            return await client.request(method, path, **kwargs)

    server = MaestroMcp(call_api, version=version,
                        public_base_url=lambda: configured_public or seen_base["url"])

    @api.post("/mcp", include_in_schema=False)
    async def mcp_post(request: Request) -> Response:
        if not origin_allowed(request.headers.get("origin")):
            return JSONResponse(_error(None, -32600, "Origin not allowed"), status_code=403)
        seen_base["url"] = str(request.base_url).rstrip("/")
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(_error(None, -32700, "Parse error"), status_code=400)
        if isinstance(payload, list):
            replies = [reply for reply in [await server.handle(item) for item in payload] if reply]
            return JSONResponse(replies) if replies else Response(status_code=202)
        reply = await server.handle(payload)
        return JSONResponse(reply) if reply else Response(status_code=202)

    @api.get("/mcp", include_in_schema=False)
    async def mcp_get() -> Response:
        # No server-initiated stream: this endpoint only answers requests.
        return Response(status_code=405, headers={"Allow": "POST"})

    @api.delete("/mcp", include_in_schema=False)
    async def mcp_delete() -> Response:
        return Response(status_code=405, headers={"Allow": "POST"})

    return server
