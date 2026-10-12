"""MCP endpoint tests: JSON-RPC protocol and tools against a fake Maestro API."""
from __future__ import annotations

import asyncio
import base64
import os
import sys
import unittest

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.mcp_server import MaestroMcp, origin_allowed  # noqa: E402


class FakeApi:
    """Records calls; answers from a (method, path) -> response table."""

    def __init__(self):
        self.calls = []
        self.routes = {}

    def on(self, method, path, body=None, status=200):
        self.routes.setdefault((method, path), []).append((status, body))

    async def __call__(self, method, path, **kwargs):
        self.calls.append((method, path, kwargs))
        answers = self.routes.get((method, path))
        if not answers:
            return httpx.Response(404, json={"detail": "not found"})
        status, body = answers.pop(0) if len(answers) > 1 else answers[0]
        return httpx.Response(status, json=body)


def run(coro):
    return asyncio.run(coro)


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.api = FakeApi()
        self.server = MaestroMcp(self.api, public_base_url=lambda: "http://localhost:7860")

    def rpc(self, method, params=None, msg_id=1):
        message = {"jsonrpc": "2.0", "method": method, "id": msg_id}
        if params is not None:
            message["params"] = params
        return run(self.server.handle(message))

    def call(self, name, arguments=None):
        return self.rpc("tools/call", {"name": name, "arguments": arguments or {}})["result"]

    def test_initialize_negotiates_a_supported_version(self):
        result = self.rpc("initialize", {"protocolVersion": "2025-03-26"})["result"]
        self.assertEqual(result["protocolVersion"], "2025-03-26")
        self.assertIn("tools", result["capabilities"])
        newer = self.rpc("initialize", {"protocolVersion": "2099-01-01"})["result"]
        self.assertEqual(newer["protocolVersion"], "2025-06-18")

    def test_notifications_get_no_reply(self):
        self.assertIsNone(run(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"})))

    def test_unknown_method_and_tool(self):
        self.assertEqual(self.rpc("resources/list")["error"]["code"], -32601)
        self.assertEqual(self.rpc("tools/call", {"name": "nope"})["error"]["code"], -32602)

    def test_tools_list_has_schemas_and_hints(self):
        tools = {t["name"]: t for t in self.rpc("tools/list")["result"]["tools"]}
        self.assertIn("generate", tools)
        self.assertIn("director_start", tools)
        self.assertEqual(tools["generate"]["inputSchema"]["required"], ["model_type", "prompt"])
        self.assertTrue(tools["list_models"]["annotations"]["readOnlyHint"])
        self.assertTrue(tools["cancel_job"]["annotations"]["destructiveHint"])

    def test_generate_merges_defaults_params_and_prompt(self):
        self.api.on("GET", "/api/v1/defaults/ltx2", {"resolution": "1280x720", "video_length": 121, "prompt": ""})
        self.api.on("POST", "/api/v1/generate", {"job_id": "j1", "status": "queued"})
        result = self.call("generate", {"model_type": "ltx2", "prompt": "a cat",
                                        "params": {"video_length": 241}})
        self.assertFalse(result["isError"])
        self.assertEqual(result["structuredContent"]["job_id"], "j1")
        body = self.api.calls[-1][2]["json"]
        self.assertEqual(body["video_length"], 241)
        self.assertEqual(body["resolution"], "1280x720")
        self.assertEqual(body["prompt"], "a cat")
        self.assertEqual(body["model_type"], "ltx2")
        self.assertEqual(body["_queue_mode"], "now")

    def test_api_errors_become_tool_errors(self):
        self.api.on("GET", "/api/v1/defaults/bad", {"detail": "Unknown model"}, status=404)
        result = self.call("generate", {"model_type": "bad", "prompt": "x"})
        self.assertTrue(result["isError"])
        self.assertIn("Unknown model", result["content"][0]["text"])

    def test_missing_required_argument(self):
        result = self.call("generate", {"model_type": "ltx2"})
        self.assertTrue(result["isError"])
        self.assertIn("prompt is required", result["content"][0]["text"])

    def test_get_job_waits_until_done_and_absolutizes_urls(self):
        self.api.on("GET", "/api/v1/jobs/j1", {"status": "running", "outputs": []})
        self.api.on("GET", "/api/v1/jobs/j1", {"status": "completed",
                                               "outputs": [{"name": "a.mp4", "url": "/api/v1/file/a.mp4"}]})
        sleeps = []

        async def fake_sleep(seconds):
            sleeps.append(seconds)
        self.server._sleep = fake_sleep
        job = self.call("get_job", {"job_id": "j1", "wait_seconds": 30})["structuredContent"]
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["outputs"][0]["url"], "http://localhost:7860/api/v1/file/a.mp4")
        self.assertEqual(len(sleeps), 1)

    def test_get_job_without_wait_returns_immediately(self):
        self.api.on("GET", "/api/v1/jobs/j1", {"status": "running"})
        job = self.call("get_job", {"job_id": "j1"})["structuredContent"]
        self.assertEqual(job["status"], "running")
        self.assertEqual(len(self.api.calls), 1)

    def test_upload_media_from_base64(self):
        self.api.on("POST", "/api/v1/upload", {"filename": "ab.png", "path": "/data/uploads/ab.png",
                                               "url": "/api/v1/uploads/ab.png"})
        data = base64.b64encode(b"\x89PNG").decode()
        result = self.call("upload_media", {"data_base64": data, "filename": "ref.png"})["structuredContent"]
        self.assertEqual(result["path"], "/data/uploads/ab.png")
        self.assertEqual(result["url"], "http://localhost:7860/api/v1/uploads/ab.png")
        sent = self.api.calls[-1][2]["files"]["file"]
        self.assertEqual(sent[0], "ref.png")
        self.assertEqual(sent[1], b"\x89PNG")

    def test_upload_media_needs_exactly_one_source(self):
        self.assertTrue(self.call("upload_media", {"filename": "a.png"})["isError"])
        self.assertTrue(self.call("upload_media", {"url": "file:///etc/passwd", "filename": "a"})["isError"])

    def test_director_start_builds_pipeline_params(self):
        self.api.on("GET", "/api/v1/services-config", {"llm_provider": "openrouter",
                                                       "llm_model_id": "google/gemini-3.8-flash",
                                                       "use_director_v2": True})
        self.api.on("GET", "/api/v1/defaults/minimax_h3", {"resolution": "1280x704", "num_inference_steps": 20})
        self.api.on("GET", "/api/v1/defaults/flux2_klein_9b", {"resolution": "1024x1024"})
        self.api.on("POST", "/api/v1/director/pipeline/start", {"pipeline_id": "p1"})
        result = self.call("director_start", {
            "scene_description": "A detective in the rain",
            "video_model": "minimax_h3",
            "target_duration": 30,
            "aspect_ratio": "16:9",
            "params": {"video_params": {"num_inference_steps": 8}, "seamless": True},
        })
        self.assertFalse(result["isError"], result)
        body = self.api.calls[-1][2]["json"]
        self.assertEqual(body["pipeline_type"], "short_film_story")
        self.assertEqual(body["llm_provider"], "openrouter")
        self.assertEqual(body["video_params"], {"resolution": "1280x704", "num_inference_steps": 8})
        self.assertEqual(body["image_model"], "flux2_klein_9b")
        self.assertEqual(body["director_aspect_ratio"], "16:9")
        self.assertEqual(body["target_duration"], 30)
        self.assertTrue(body["seamless"])
        self.assertTrue(body["auto_mode"])

    def test_director_music_video_needs_audio(self):
        result = self.call("director_start", {"scene_description": "x", "video_model": "ltx2",
                                              "pipeline_type": "music_video"})
        self.assertTrue(result["isError"])
        self.assertIn("audio_path", result["content"][0]["text"])

    def test_director_status_waits_for_pause(self):
        self.api.on("GET", "/api/v1/director/pipeline/p1", {"status": "running", "phase": "planning"})
        self.api.on("GET", "/api/v1/director/pipeline/p1", {"status": "paused", "phase": "review"})

        async def no_sleep(seconds):
            return None
        self.server._sleep = no_sleep
        status = self.call("director_status", {"pipeline_id": "p1", "wait_seconds": 20})["structuredContent"]
        self.assertEqual(status["status"], "paused")

    def test_list_models_filters(self):
        self.api.on("GET", "/api/v1/models", {"models": [
            {"model_type": "ltx2_22B", "name": "LTX 2.3", "family": "ltx2"},
            {"model_type": "flux2_klein_9b", "name": "Flux 2 Klein", "family": "flux"},
        ]})
        result = self.call("list_models", {"family": "ltx"})["structuredContent"]
        self.assertEqual([m["model_type"] for m in result["models"]], ["ltx2_22B"])


class OriginTests(unittest.TestCase):
    def test_origin_guard(self):
        self.assertTrue(origin_allowed(None))
        self.assertTrue(origin_allowed("http://localhost:5173"))
        self.assertFalse(origin_allowed("https://evil.example"))
        self.assertTrue(origin_allowed("https://claude.ai", {"MAESTRO_MCP_ALLOWED_ORIGINS": "claude.ai"}))


if __name__ == "__main__":
    unittest.main()
