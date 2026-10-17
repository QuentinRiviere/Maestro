"""docker/spark-stack/model-hub: the unified memory view (host, llama-swap models, the Maestro worker)."""
from __future__ import annotations

import os
import sys
import unittest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "docker", "spark-stack", "model-hub"))
sys.path.insert(0, os.path.join(_ROOT, "app"))

from model_hub import memory  # noqa: E402
from services.remote_worker import worker_control  # noqa: E402

MEMINFO = """MemTotal:       130000000 kB
MemFree:         6000000 kB
MemAvailable:   28000000 kB
Buffers:          100000 kB
SwapTotal:      16000000 kB
SwapFree:        4000000 kB
"""


class MeminfoTests(unittest.TestCase):
    def test_figures_in_gb(self):
        host = memory.parse_meminfo(MEMINFO)
        self.assertEqual(host, {"total_gb": 133.1, "available_gb": 28.7, "used_gb": 104.4,
                                "swap_total_gb": 16.4, "swap_used_gb": 12.3})

    def test_unreadable_is_empty(self):
        self.assertEqual(memory.parse_meminfo(""), {})
        self.assertEqual(memory.parse_meminfo("Garbage: x kB"), {})
        self.assertEqual(memory.read_meminfo("/nonexistent/meminfo"), {})


class WorkerSummaryTests(unittest.TestCase):
    def test_a_state_is_reduced(self):
        state = {"paused": False, "running_job": {"job_id": "7b6d", "phase": "Denoising"}, "queued": [1, 2],
                 "loaded_models": ["ltx2_25", "qwen_image_21_7B"],
                 "memory": {"gpu_free_gb": 26.1, "gpu_total_gb": 121.0, "unified_memory": True, "cuda_free_gb": 3.0},
                 "process_rss_gb": 50.3, "idle_unload_seconds": 1800}
        summary = memory.worker_summary(state)
        self.assertTrue(summary["reachable"])
        self.assertEqual(summary["loaded_models"], ["ltx2_25", "qwen_image_21_7B"])
        self.assertEqual(summary["running_job"], {"id": "7b6d", "phase": "Denoising"})
        self.assertEqual(summary["queued"], 2)
        self.assertEqual(summary["gpu_free_gb"], 26.1)
        self.assertEqual(summary["process_rss_gb"], 50.3)

    def test_unreachable(self):
        self.assertEqual(memory.worker_summary(None, "HTTP 401"), {"reachable": False, "error": "HTTP 401"})


class MemoryViewTests(unittest.TestCase):
    def test_the_worker_entry_carries_its_own_models(self):
        host = memory.parse_meminfo(MEMINFO)
        running = [{"model": "gemma4-judge", "state": "ready", "ttl": 0},
                   {"model": "maestro-worker", "state": "ready", "ttl": 0}]
        worker = memory.worker_summary({"loaded_models": ["ltx2_25"], "memory": {}, "process_rss_gb": 50.0})
        view = memory.memory_view(host, running, lambda m: 22.0 if m == "gemma4-judge" else None,
                                  lambda m: m == "gemma4-judge", worker)
        self.assertTrue(view["llama_swap"]["reachable"])
        judge, worker_entry = view["llama_swap"]["models"]
        self.assertEqual((judge["estimated_gb"], judge["resident"]), (22.0, True))
        self.assertEqual(worker_entry["estimated_gb"], 50.0)
        self.assertEqual(worker_entry["worker"]["loaded_models"], ["ltx2_25"])
        self.assertEqual(view["accounted_gb"], 72.0)
        self.assertEqual(view["other_gb"], 32.4)

    def test_llama_swap_down(self):
        view = memory.memory_view({}, None, lambda m: None, lambda m: False,
                                  memory.worker_summary(None, "x"), "connection refused")
        self.assertFalse(view["llama_swap"]["reachable"])
        self.assertEqual(view["llama_swap"]["error"], "connection refused")
        self.assertIsNone(view["other_gb"])


class WorkerProcessSizeTests(unittest.TestCase):
    def test_rss_from_proc_status(self):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".status", delete=False) as handle:
            handle.write("Name:\tpython3\nVmPeak:\t 60000000 kB\nVmRSS:\t 52428800 kB\n")
        try:
            self.assertEqual(worker_control.process_rss_gb(handle.name), 53.69)
        finally:
            os.unlink(handle.name)
        self.assertIsNone(worker_control.process_rss_gb("/nonexistent/status"))


if __name__ == "__main__":
    unittest.main()
