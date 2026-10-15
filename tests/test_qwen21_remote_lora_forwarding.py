"""Trace a Qwen Image 2.1 LoRA from the controller to the worker's wgp.resolve_lora_path.

The remote forwarding path was ruled out as the cause of LoRA jobs rendering
identically to LoRA-free jobs: the file reaches the worker's LoRA directory and
the job keeps the bare name that the worker resolves to that file.
"""

import ast
import os
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tests"))

from test_remote_job_forwarder import FakeManager, FakeWorker  # noqa: E402
from services.remote_worker.job_forwarder import RemoteJobForwarder  # noqa: E402

LORA = "qwen21_next_scene_step2500.safetensors"
MODEL = "qwen_image_21_7B"


def load_functions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, "exec"), namespace)
    return namespace


class HTTPException(Exception):
    def __init__(self, status_code, detail):
        super().__init__(detail)


class RemoteQwenLoraForwardingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.controller_lora = os.path.join(self.tmp, "controller", "loras", "qwen21", LORA)
        os.makedirs(os.path.dirname(self.controller_lora))
        with open(self.controller_lora, "wb") as handle:
            handle.write(b"adapter")
        self.worker_lora_dir = os.path.join(self.tmp, "worker", "loras", "qwen21")

    def test_lora_job_reaches_the_worker_lora_directory_and_resolves_there(self):
        worker = FakeWorker()
        worker.final = {"status": "completed", "output_files": [], "files": [], "sidecars": []}
        forwarder = RemoteJobForwarder(FakeManager(), client_factory=lambda url, token: worker,
                                       store_factory=lambda settings: None, poll_interval=0,
                                       sleep=lambda s: None)
        params = {"model_type": MODEL, "prompt": "Next Scene: a cafe", "image_mode": 1,
                  "video_prompt_type": "I", "activated_loras": [LORA], "loras_multipliers": "0.8"}
        resolved = []

        def resolve_lora(model_type, name):
            resolved.append((model_type, name))
            return self.controller_lora

        forwarder.run(params, out_dir=os.path.join(self.tmp, "out"), resolve_lora=resolve_lora,
                      update=lambda **_: None, is_cancelled=lambda: False)

        self.assertEqual(resolved, [(MODEL, LORA)])
        self.assertIn(("lora", LORA, MODEL), worker.uploads)
        self.assertEqual(worker.submitted["activated_loras"], [LORA])
        self.assertEqual(worker.submitted["loras_multipliers"], "0.8")

        # Worker side: where the upload lands, and what wgp resolves the name to.
        fake_wgp = SimpleNamespace(get_model_def=lambda model_type: {"architecture": MODEL},
                                   get_lora_dir=lambda model_type: self.worker_lora_dir)
        launch = load_functions("app/launch.py", ["_worker_file_target"], {
            "os": os, "wgp": fake_wgp, "HTTPException": HTTPException,
            "_SHA256_RE": re.compile(r"^[0-9a-f]{64}$"),
        })
        target = launch["_worker_file_target"]("lora", LORA, "a" * 64, MODEL)
        self.assertEqual(target, os.path.join(self.worker_lora_dir, LORA))
        os.makedirs(self.worker_lora_dir)
        shutil.copy(self.controller_lora, target)

        wgp = load_functions("app/wgp.py", ["resolve_lora_path"], {
            "os": os, "get_lora_search_dirs": lambda model_type: [self.worker_lora_dir],
        })
        self.assertEqual(wgp["resolve_lora_path"](MODEL, worker.submitted["activated_loras"][0]), target)


if __name__ == "__main__":
    unittest.main()
