"""ComfyUI backend tests against an in-memory fake server (no network, no GPU)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from inference import BackendRegistry, TaskRequest  # noqa: E402
from inference.comfy import (  # noqa: E402
    ComfyBackend,
    ComfyClient,
    ComfyError,
    ComfyFile,
    ComfyRejectedError,
    GraphValidationError,
    PromptState,
    RecipeInput,
    decide_state,
    validate_graph,
)
from inference.comfy.states import extract_error, extract_outputs  # noqa: E402
from inference.configure import register_configured_backends  # noqa: E402


# ── fake HTTP layer ────────────────────────────────────────────────────────


class FakeResponse:
    def __init__(self, status=200, body=None, *, content=b"", headers=None):
        self.status_code = status
        self._body = body
        self._content = content
        self.headers = headers or {}

    @property
    def ok(self):
        return 200 <= self.status_code < 400 and not self.is_redirect

    @property
    def is_redirect(self):
        return self.status_code in (301, 302, 303, 307, 308)

    @property
    def text(self):
        return json.dumps(self._body) if self._body is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body

    def iter_content(self, chunk_size=1):
        yield self._content

    def close(self):
        pass


class FakeComfyServer:
    """Minimal stateful ComfyUI: one prompt at a time, scripted outcomes."""

    def __init__(self):
        self.calls = []
        self.uploaded = {}
        self.queue = {"queue_running": [], "queue_pending": []}
        self.history = {}
        self.next_ids = ["p1", "p2", "p3"]
        self.submit_response = None
        self.object_info = {"SaveVideo": {}, "Gen": {}, "LoadImage": {}}
        self.files = {}
        self.redirect_view_to = None
        self.on_poll = None  # callable(server) run at each /history read
        self.interrupt_finishes = True

    # requests.Session surface
    def get(self, url, params=None, headers=None, **kwargs):
        route = url.split("/api/", 1)[-1] if "/api/" in url else url
        self.calls.append(("GET", route, dict(headers or {}), params))
        if route == "queue":
            return FakeResponse(body=json.loads(json.dumps(self.queue)))
        if route.startswith("history/"):
            if self.on_poll:
                self.on_poll(self)
            prompt_id = route.split("/", 1)[1]
            entry = self.history.get(prompt_id)
            return FakeResponse(body={prompt_id: entry} if entry is not None else {})
        if route == "object_info":
            return FakeResponse(body=self.object_info)
        if route == "view":
            if self.redirect_view_to and "storage" not in url:
                return FakeResponse(302, headers={"Location": self.redirect_view_to})
            return FakeResponse(content=self.files.get(params["filename"], b""))
        if "storage" in url:
            return FakeResponse(content=b"redirected-bytes")
        return FakeResponse(404, body={"error": "not found"})

    def post(self, url, data=None, files=None, headers=None, **kwargs):
        route = url.split("/api/", 1)[-1]
        body = json.loads(data) if isinstance(data, str) else data
        self.calls.append(("POST", route, dict(headers or {}), body))
        if route == "upload/image":
            requested = files["image"][0]
            name = requested
            if requested in self.uploaded:
                stem, ext = os.path.splitext(requested)
                name = f"{stem} (1){ext}"
            self.uploaded[name] = body
            return FakeResponse(body={"name": name, "subfolder": "", "type": "input"})
        if route == "prompt":
            if self.submit_response is not None:
                return self.submit_response
            prompt_id = self.next_ids.pop(0)
            self.queue["queue_pending"].append([0, prompt_id, {}, {}, []])
            return FakeResponse(body={"prompt_id": prompt_id, "number": 0, "node_errors": {}})
        if route == "queue":
            for prompt_id in body.get("delete", []):
                self.queue["queue_pending"] = [
                    item for item in self.queue["queue_pending"] if item[1] != prompt_id
                ]
            return FakeResponse(body={})
        if route == "interrupt":
            if self.interrupt_finishes:
                prompt_id = body.get("prompt_id")
                self.queue["queue_running"] = []
                self.history[prompt_id] = {
                    "status": {
                        "status_str": "error",
                        "completed": False,
                        "messages": [["execution_interrupted", {"prompt_id": prompt_id}]],
                    },
                    "outputs": {},
                }
            return FakeResponse(body={})
        return FakeResponse(404, body={"error": "not found"})

    # scripted transitions
    def start(self, prompt_id):
        self.queue["queue_pending"] = [i for i in self.queue["queue_pending"] if i[1] != prompt_id]
        self.queue["queue_running"] = [[0, prompt_id, {}, {}, []]]

    def finish(self, prompt_id, filename="clip_00001.mp4", content=b"video-bytes"):
        self.queue["queue_running"] = []
        self.files[filename] = content
        self.history[prompt_id] = {
            "status": {"status_str": "success", "completed": True, "messages": []},
            "outputs": {
                "save": {"videos": [{"filename": filename, "subfolder": "", "type": "output"}]},
                "preview": {"images": [{"filename": "ignored.png", "type": "temp"}]},
            },
        }

    def fail(self, prompt_id):
        self.queue["queue_running"] = []
        self.history[prompt_id] = {
            "status": {
                "status_str": "error",
                "completed": False,
                "messages": [["execution_error", {
                    "node_id": "gen", "node_type": "Gen", "exception_message": "CUDA OOM",
                }]],
            },
            "outputs": {},
        }


class FakeRecipe:
    name = "fake"
    model_patterns = ("fake_*",)
    allowed_node_types = frozenset({"Gen", "SaveVideo", "LoadImage"})
    output_node = "save"

    def __init__(self, inputs=()):
        self._inputs = list(inputs)

    def collect_inputs(self, request):
        return self._inputs

    def build_graph(self, request, uploaded):
        graph = {
            "gen": {"class_type": "Gen", "inputs": {"prompt": request.params.get("prompt")}},
            "save": {"class_type": "SaveVideo", "inputs": {"video": ["gen", 0]}},
        }
        if "ref" in uploaded:
            graph["ref"] = {"class_type": "LoadImage", "inputs": {"image": uploaded["ref"]}}
            graph["gen"]["inputs"]["image"] = ["ref", 0]
        return graph


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def _request(state=None, model_type="fake_model"):
    state = state if state is not None else {"gen": {"file_list": [], "abort": False}}
    return TaskRequest(task={}, params={"model_type": model_type, "prompt": "a cat", "state": state})


# ── pure state decoding ────────────────────────────────────────────────────


class DecideStateTests(unittest.TestCase):
    def test_history_success_and_error(self):
        ok = {"p": {"status": {"status_str": "success"}}}
        self.assertIs(decide_state("p", ok, None).state, PromptState.SUCCEEDED)
        err = {"p": {"status": {"status_str": "error", "messages": []}}}
        self.assertIs(decide_state("p", err, None).state, PromptState.FAILED)

    def test_interrupt_message_means_cancelled(self):
        hist = {"p": {"status": {"status_str": "error", "messages": [["execution_interrupted", {}]]}}}
        self.assertIs(decide_state("p", hist, None).state, PromptState.CANCELLED)

    def test_interrupt_string_inside_payload_dict_is_ignored(self):
        hist = {"p": {"status": {"status_str": "error", "messages": [
            ["execution_error", {"exception_message": "execution_interrupted"}],
        ]}}}
        self.assertIs(decide_state("p", hist, None).state, PromptState.FAILED)

    def test_unknown_status_and_missing_status_keep_running(self):
        weird = {"p": {"status": {"status_str": "constructor"}}}
        decision = decide_state("p", weird, None)
        self.assertIs(decision.state, PromptState.RUNNING)
        self.assertIsNotNone(decision.trace)
        self.assertIs(decide_state("p", {"p": {"status": None}}, None).state, PromptState.RUNNING)

    def test_queue_lookup_uses_index_one_only(self):
        queue = {"queue_running": [[0, "other", {"note": "p"}, {}, ["p"]]], "queue_pending": [[1, "p"]]}
        decision = decide_state("p", {}, queue)
        self.assertIs(decision.state, PromptState.RUNNING)
        self.assertEqual(decision.raw, "queue_pending")
        self.assertIs(decide_state("x", {}, queue).state, PromptState.LOST)

    def test_extract_outputs_reads_only_declared_node(self):
        entry = {"outputs": {
            "save": {"videos": [{"filename": "a.mp4"}], "animated": [True]},
            "preview": {"images": [{"filename": "p.png"}]},
        }}
        self.assertEqual(extract_outputs(entry, "save"), [ComfyFile("a.mp4", "", "output")])
        self.assertEqual(extract_outputs(entry, "missing"), [])

    def test_extract_error_names_the_node(self):
        entry = {"status": {"messages": [["execution_error", {
            "node_id": "7", "node_type": "KSampler", "exception_message": "boom",
        }]]}}
        self.assertIn("KSampler", extract_error(entry))
        self.assertIn("boom", extract_error(entry))


class ValidateGraphTests(unittest.TestCase):
    allowed = frozenset({"Gen", "SaveVideo"})

    def test_valid_graph_passes(self):
        graph = FakeRecipe().build_graph(_request(), {})
        validate_graph(graph, allowed_node_types=self.allowed, output_node="save")

    def test_rejects_disallowed_type_missing_link_and_output(self):
        graph = {
            "gen": {"class_type": "Evil", "inputs": {}},
            "save": {"class_type": "SaveVideo", "inputs": {"video": ["nope", 0]}},
        }
        with self.assertRaises(GraphValidationError) as ctx:
            validate_graph(graph, allowed_node_types=self.allowed, output_node="out")
        message = str(ctx.exception)
        self.assertIn("Evil", message)
        self.assertIn("missing node nope", message)
        self.assertIn("output node out", message)

    def test_rejects_types_absent_from_server(self):
        graph = FakeRecipe().build_graph(_request(), {})
        with self.assertRaises(GraphValidationError):
            validate_graph(
                graph,
                allowed_node_types=self.allowed,
                output_node="save",
                available_node_types=frozenset({"SaveVideo"}),
            )


# ── client ─────────────────────────────────────────────────────────────────


class ComfyClientTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeComfyServer()
        self.client = ComfyClient("http://comfy.test/svc", token="secret", session=self.server)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _local(self, name="ref.png"):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as handle:
            handle.write(b"png")
        return path

    def test_upload_returns_server_name_and_uses_input_root(self):
        path = self._local()
        self.assertEqual(self.client.upload(path, "ref.png"), "ref.png")
        self.assertEqual(self.client.upload(path, "ref.png"), "ref (1).png")
        post = [c for c in self.server.calls if c[1] == "upload/image"][0]
        self.assertEqual(post[3]["subfolder"], "")
        self.assertEqual(post[2]["Authorization"], "Bearer secret")

    def test_submit_rejects_node_errors_even_on_200(self):
        self.server.submit_response = FakeResponse(body={"prompt_id": "x", "node_errors": {"3": "bad"}})
        with self.assertRaises(ComfyRejectedError):
            self.client.submit({"a": {}})

    def test_submit_rejects_http_400(self):
        self.server.submit_response = FakeResponse(400, body={"error": {"type": "prompt_outputs_failed_validation"}})
        with self.assertRaises(ComfyRejectedError):
            self.client.submit({"a": {}})

    def test_poll_reads_queue_before_history(self):
        prompt_id = self.client.submit({"a": {}})
        self.client.poll(prompt_id)
        routes = [c[1] for c in self.server.calls if c[0] == "GET"]
        self.assertEqual(routes[:2], ["queue", f"history/{prompt_id}"])

    def test_poll_does_not_report_lost_when_job_finishes_between_reads(self):
        prompt_id = self.client.submit({"a": {}})
        self.server.start(prompt_id)
        # Finish lands after /queue was read but before /history is read.
        self.server.on_poll = lambda srv: srv.finish(prompt_id)
        self.assertIs(self.client.poll(prompt_id).decision.state, PromptState.SUCCEEDED)

    def test_download_follows_redirect_without_auth(self):
        self.server.redirect_view_to = "https://storage.example/obj"
        dest = os.path.join(self.tmp.name, "out", "clip.mp4")
        self.client.download(ComfyFile("clip.mp4"), dest)
        with open(dest, "rb") as handle:
            self.assertEqual(handle.read(), b"redirected-bytes")
        redirected = [c for c in self.server.calls if "storage" in c[1]][0]
        self.assertNotIn("Authorization", redirected[2])
        self.assertFalse(os.path.exists(dest + ".part"))

    def test_download_rejects_empty_file(self):
        self.server.files["empty.mp4"] = b""
        dest = os.path.join(self.tmp.name, "empty.mp4")
        with self.assertRaises(ComfyError):
            self.client.download(ComfyFile("empty.mp4"), dest)
        self.assertFalse(os.path.exists(dest))

    def test_cancel_pending_deletes_from_queue(self):
        prompt_id = self.client.submit({"a": {}})
        state = self.client.cancel(prompt_id, confirm_interval=0)
        self.assertIs(state, PromptState.LOST)
        self.assertIn(("POST", "queue", {"Authorization": "Bearer secret", "Content-Type": "application/json"},
                       {"delete": [prompt_id]}), self.server.calls)

    def test_cancel_running_interrupts_that_prompt(self):
        prompt_id = self.client.submit({"a": {}})
        self.server.start(prompt_id)
        self.assertIs(self.client.cancel(prompt_id, confirm_interval=0), PromptState.CANCELLED)
        interrupt = [c for c in self.server.calls if c[1] == "interrupt"][0]
        self.assertEqual(interrupt[3], {"prompt_id": prompt_id})

    def test_cancel_without_effect_raises(self):
        prompt_id = self.client.submit({"a": {}})
        self.server.start(prompt_id)
        self.server.interrupt_finishes = False
        with self.assertRaises(ComfyError):
            self.client.cancel(prompt_id, confirm_attempts=2, confirm_interval=0)


# ── backend ────────────────────────────────────────────────────────────────


class ComfyBackendTests(unittest.TestCase):
    def setUp(self):
        self.server = FakeComfyServer()
        self.client = ComfyClient("http://comfy.test/", session=self.server)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = FakeClock()
        self.sent = []

    def _backend(self, recipe=None, **kwargs):
        return ComfyBackend(
            "runpod",
            self.client,
            [recipe or FakeRecipe()],
            save_dir=lambda: self.tmp.name,
            poll_interval=1.0,
            clock=self.clock,
            sleep=self.clock.sleep,
            **kwargs,
        )

    def _send(self, cmd, data):
        self.sent.append((cmd, data))

    def _script(self, steps):
        """Advance the fake server one step per /history read."""
        steps = list(steps)

        def on_poll(server):
            if steps:
                steps.pop(0)(server)

        self.server.on_poll = on_poll

    def test_success_downloads_and_registers_output(self):
        self._script([lambda s: s.start("p1"), lambda s: None, lambda s: s.finish("p1")])
        request = _request()
        self._backend().run_task(request, self._send)

        files = request.state["gen"]["file_list"]
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].startswith(self.tmp.name))
        self.assertTrue(files[0].endswith(".mp4"))
        with open(files[0], "rb") as handle:
            self.assertEqual(handle.read(), b"video-bytes")
        timing = [d for c, d in self.sent if c == "generation_time"]
        self.assertEqual(timing[0]["outputs"], files)
        submitted = [c for c in self.server.calls if c[1] == "prompt"][0][3]["prompt"]
        self.assertEqual(submitted["gen"]["inputs"]["prompt"], "a cat")

    def test_uploads_inputs_and_wires_server_names(self):
        ref = os.path.join(self.tmp.name, "my ref.png")
        with open(ref, "wb") as handle:
            handle.write(b"png")
        self._script([lambda s: s.finish("p1")])
        self._backend(FakeRecipe([RecipeInput("ref", ref)])).run_task(_request(), self._send)
        submitted = [c for c in self.server.calls if c[1] == "prompt"][0][3]["prompt"]
        remote = submitted["ref"]["inputs"]["image"]
        self.assertIn(remote, self.server.uploaded)
        self.assertNotIn(" ", remote)

    def test_remote_failure_raises_with_node_message(self):
        self._script([lambda s: s.fail("p1")])
        with self.assertRaises(ComfyError) as ctx:
            self._backend().run_task(_request(), self._send)
        self.assertIn("CUDA OOM", str(ctx.exception))

    def test_abort_cancels_remote_prompt(self):
        request = _request()

        def start_then_abort(server):
            server.start("p1")
            request.state["gen"]["abort"] = True

        self._script([start_then_abort])
        self._backend().run_task(request, self._send)
        self.assertTrue(any(c[1] == "interrupt" for c in self.server.calls))
        self.assertEqual(request.state["gen"]["file_list"], [])

    def test_lost_prompt_is_resubmitted_once(self):
        def lose(server):
            server.queue["queue_pending"] = []

        self._script([lose, lambda s: s.finish("p2")])
        request = _request()
        self._backend().run_task(request, self._send)
        self.assertEqual(sum(1 for c in self.server.calls if c[1] == "prompt"), 2)
        self.assertEqual(len(request.state["gen"]["file_list"]), 1)

    def test_lost_twice_gives_up(self):
        def lose(server):
            server.queue["queue_pending"] = []

        self._script([lose, lose])
        with self.assertRaises(ComfyError):
            self._backend().run_task(_request(), self._send)

    def test_deadline_cancels_and_raises(self):
        self._script([lambda s: s.start("p1")])
        with self.assertRaises(ComfyError) as ctx:
            self._backend(deadline_seconds=3).run_task(_request(), self._send)
        self.assertIn("exceeded", str(ctx.exception))
        self.assertTrue(any(c[1] == "interrupt" for c in self.server.calls))

    def test_graph_with_node_missing_on_server_is_not_submitted(self):
        self.server.object_info = {"SaveVideo": {}}
        with self.assertRaises(GraphValidationError):
            self._backend().run_task(_request(), self._send)
        self.assertFalse(any(c[1] == "prompt" for c in self.server.calls))

    def test_model_without_recipe_fails_clearly(self):
        with self.assertRaises(ComfyError) as ctx:
            self._backend().run_task(_request(model_type="wan2_2"), self._send)
        self.assertIn("wan2_2", str(ctx.exception))


class ConfigureTests(unittest.TestCase):
    def test_registers_comfy_backends_lazily(self):
        registry = BackendRegistry()
        config = {"inference": {"backends": {
            "runpod": {"type": "comfy", "base_url": "http://x/", "token_env": "MAESTRO_TEST_TOKEN"},
            "broken": {"type": "nope"},
        }}}
        names = register_configured_backends(
            registry, config, comfy_recipes=[FakeRecipe()], save_dir=lambda: "/tmp",
        )
        self.assertEqual(names, ["runpod", "broken"])
        backend = registry.get("runpod")
        self.assertIsInstance(backend, ComfyBackend)
        self.assertEqual(backend.name, "runpod")
        with self.assertRaises(ValueError):
            registry.get("broken")


if __name__ == "__main__":
    unittest.main()
