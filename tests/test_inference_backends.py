"""Engine-free tests for the inference backend seam."""
from __future__ import annotations

import os
import sys
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from inference import (  # noqa: E402
    BackendRegistry,
    InferenceBackend,
    LocalWgpBackend,
    TaskRequest,
    UnknownBackendError,
    resolve_backend_name,
)


class _FakeWgp:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def generate_video(self, task, send_cmd, prompt, plugin_data=None, state=None):
        self.calls.append((task, prompt, plugin_data, state))
        send_cmd("progress", [(1, 2), "Denoising"])
        state["gen"]["file_list"].append("clip.mp4")


class ResolveBackendNameTests(unittest.TestCase):
    def test_defaults_to_local_without_config(self):
        self.assertEqual(resolve_backend_name("ltx2_22B", None), "local")
        self.assertEqual(resolve_backend_name("ltx2_22B", {}), "local")

    def test_first_matching_route_wins(self):
        config = {
            "inference": {
                "default": "local",
                "routes": {"minimax_h3_turbo*": "spark", "minimax_h3*": "comfy"},
            }
        }
        self.assertEqual(resolve_backend_name("minimax_h3_turbo_x", config), "spark")
        self.assertEqual(resolve_backend_name("minimax_h3_frames", config), "comfy")
        self.assertEqual(resolve_backend_name("wan2_2", config), "local")

    def test_custom_default(self):
        config = {"inference": {"default": "comfy"}}
        self.assertEqual(resolve_backend_name("anything", config), "comfy")


class BackendRegistryTests(unittest.TestCase):
    def test_factory_is_lazy_and_cached(self):
        built = []
        registry = BackendRegistry()
        registry.register("local", lambda: built.append(1) or LocalWgpBackend(_FakeWgp()))
        self.assertEqual(built, [])
        first = registry.get("local")
        second = registry.get("local")
        self.assertIs(first, second)
        self.assertEqual(built, [1])

    def test_unknown_backend_fails_loudly(self):
        registry = BackendRegistry()
        registry.register("local", lambda: LocalWgpBackend(_FakeWgp()))
        config = {"inference": {"routes": {"minimax_h3*": "comfy"}}}
        with self.assertRaises(UnknownBackendError):
            registry.for_model("minimax_h3_frames", config)


class LocalWgpBackendTests(unittest.TestCase):
    def test_forwards_only_signature_params_and_plugin_data(self):
        fake = _FakeWgp()
        backend = LocalWgpBackend(fake)
        self.assertIsInstance(backend, InferenceBackend)
        state = {"gen": {"file_list": [], "abort": False}}
        task = {"id": 1, "plugin_data": {"p": 1}}
        params = {
            "model_type": "ltx2_22B",
            "prompt": "a cat",
            "state": state,
            "not_a_generate_video_arg": 42,
        }
        sent = []
        backend.run_task(TaskRequest(task=task, params=params), lambda c, d: sent.append((c, d)))

        self.assertEqual(fake.calls, [(task, "a cat", {"p": 1}, state)])
        self.assertEqual(sent, [("progress", [(1, 2), "Denoising"])])
        self.assertEqual(state["gen"]["file_list"], ["clip.mp4"])


class TaskRequestTests(unittest.TestCase):
    def test_abort_flag_is_read_from_state_gen(self):
        state = {"gen": {"abort": False}}
        request = TaskRequest(task={}, params={"model_type": "m", "state": state})
        self.assertEqual(request.model_type, "m")
        self.assertFalse(request.abort_requested())
        state["gen"]["abort"] = True
        self.assertTrue(request.abort_requested())


if __name__ == "__main__":
    unittest.main()
