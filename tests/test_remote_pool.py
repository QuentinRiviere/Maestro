"""Remote worker pool: pods, job spreading, controller slots, parallel bible renders."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.remote_worker import RemoteWorkerManager, RemoteWorkerSettings, RemoteWorkerSettingsStore  # noqa: E402
from services.remote_worker.pool import RemoteWorkerPool, Slots  # noqa: E402
from services.remote_worker.worker_spec import build_fleet_spec, build_run_spec  # noqa: E402


class PoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.primary = RemoteWorkerManager(RemoteWorkerSettingsStore(self.tmp.name))
        self.pool = RemoteWorkerPool(self.primary)

    def test_a_burst_spreads_over_every_pod_and_single_jobs_stay_on_the_first(self):
        self.primary.update_settings({"worker_count": 3})
        pods = [self.pool.acquire() for _ in range(3)]
        self.assertEqual([p.run_name for p in pods], ["maestro-worker", "maestro-worker-2", "maestro-worker-3"])
        for pod in pods:
            pod.job_finished()
        self.assertIs(self.pool.acquire(), self.primary)

    def test_a_ready_pod_is_preferred_over_one_still_to_start(self):
        self.primary.update_settings({"worker_count": 2})
        second = self.pool.pods()[1]
        second._ready_since = 1.0
        self.assertIs(self.pool.acquire(), second)

    def test_extra_pods_follow_the_first_pods_settings(self):
        self.primary.update_settings({"worker_count": 2, "gpu": "80GB.."})
        second = self.pool.pods()[1]
        self.assertEqual(second.settings.gpu, "80GB..")
        second.update_settings({"max_price": 1.5})
        self.assertEqual(self.primary.settings.max_price, 1.5)
        self.assertIs(second.r2_keys, self.primary.r2_keys)

    def test_one_pod_by_default(self):
        self.assertEqual(len(self.pool.pods()), 1)
        self.assertEqual(self.pool.capacity(), 1)
        self.primary.update_settings({"worker_count": 9, "jobs_per_worker": 2})
        self.assertEqual(self.pool.capacity(), 8)  # 4 pods at most

    def test_specs_name_each_pod_and_size_the_fleet(self):
        settings = RemoteWorkerSettings(worker_count=3)
        self.assertEqual(build_run_spec(settings, run_name="maestro-worker-2")["run_name"], "maestro-worker-2")
        self.assertEqual(build_fleet_spec(settings)["configuration"]["nodes"], {"min": 0, "max": 3})

    def test_keeping_models_in_vram_reaches_the_worker(self):
        self.assertNotIn("MAESTRO_KEEP_MODELS_IN_VRAM", build_run_spec(RemoteWorkerSettings())["configuration"]["env"])
        env = build_run_spec(RemoteWorkerSettings(keep_models_in_vram=True))["configuration"]["env"]
        self.assertEqual(env["MAESTRO_KEEP_MODELS_IN_VRAM"], "1")


class SlotsTests(unittest.TestCase):
    def test_capacity_follows_the_callable(self):
        size = [2]
        slots = Slots(lambda: size[0])
        self.assertTrue(slots.acquire(timeout=0.01))
        self.assertTrue(slots.acquire(timeout=0.01))
        self.assertFalse(slots.acquire(timeout=0.01))
        size[0] = 3
        self.assertTrue(slots.acquire(timeout=0.01))
        for _ in range(3):
            slots.release()
        self.assertEqual(slots.taken, 0)


class ParallelBibleRenderTests(unittest.TestCase):
    def test_renders_overlap_up_to_the_parallelism(self):
        from services import director_pipeline as pipeline

        running, peak, lock = [0], [0], threading.Lock()

        def render():
            with lock:
                running[0] += 1
                peak[0] = max(peak[0], running[0])
            time.sleep(0.05)
            with lock:
                running[0] -= 1

        pipeline._run_bible_renders([render] * 6, 3)
        self.assertEqual(peak[0], 3)
        peak[0] = 0
        pipeline._run_bible_renders([render] * 3, 1)
        self.assertEqual(peak[0], 1)

    def test_cloud_models_render_side_by_side_and_local_ones_follow_the_pool(self):
        from services import director_pipeline as pipeline

        self.assertEqual(pipeline._bible_parallelism("cloud_fal_qwen_image_edit_2511"), pipeline.CLOUD_BIBLE_PARALLEL)
        original = pipeline.bible_render_slots
        try:
            pipeline.bible_render_slots = lambda: 6
            self.assertEqual(pipeline._bible_parallelism("qwen_image_edit_2511_20B_fp8_lightning_8step"), 6)
        finally:
            pipeline.bible_render_slots = original
        self.assertEqual(pipeline._bible_parallelism("flux2_klein_9b"), 1)


if __name__ == "__main__":
    unittest.main()
