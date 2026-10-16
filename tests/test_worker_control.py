"""Worker-side pause / hand back / unload control and the bearer token middleware."""
from __future__ import annotations

import os
import sys
import threading
import time
import unittest
from unittest import mock


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.job_lifecycle import generation_slot, try_hand_back, try_start  # noqa: E402
from services.remote_worker import worker_auth  # noqa: E402
from services.remote_worker.worker_auth import BearerTokenMiddleware, authorized  # noqa: E402
from services.remote_worker.worker_control import WorkerBusy, WorkerControl, gpu_memory_stats  # noqa: E402


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


def _job(job_id, status="queued"):
    return {"id": job_id, "status": status, "worker_job": True}


class Harness:
    def __init__(self, **kwargs):
        self.lock = threading.Lock()
        self.jobs = {}
        self.loaded = ["ltx2"]
        self.releases = 0
        self.clock = Clock()

        def release_models():
            self.releases += 1
            released, self.loaded = list(self.loaded), []
            return released

        self.control = WorkerControl(
            self.lock, jobs=lambda: list(self.jobs.values()), release_models=release_models,
            loaded_models=lambda: list(self.loaded), memory=lambda: {"gpu_free_gb": 100.0},
            clock=self.clock, **kwargs,
        )


class HandBackLifecycleTests(unittest.TestCase):
    def test_only_an_unstarted_job_is_handed_back(self):
        queued, running = _job("a"), _job("b", "running")
        self.assertTrue(try_hand_back(queued, message="Handed back"))
        self.assertEqual(queued["status"], "cancelled")
        self.assertTrue(queued["handed_back"])
        self.assertFalse(try_start(queued))
        self.assertFalse(try_hand_back(running))
        self.assertEqual(running["status"], "running")
        self.assertNotIn("handed_back", running)


class WorkerControlTests(unittest.TestCase):
    def setUp(self):
        self.h = Harness()
        self.control = self.h.control

    def test_pause_and_resume_are_idempotent(self):
        self.assertTrue(self.control.pause())
        self.assertFalse(self.control.pause())
        self.assertTrue(self.control.paused)
        self.assertTrue(self.control.resume())
        self.assertFalse(self.control.resume())
        self.assertFalse(self.control.paused)

    def test_start_paused(self):
        self.assertTrue(Harness(start_paused=True).control.paused)

    def test_gate_holds_jobs_while_paused_and_lets_them_run_after_resume(self):
        self.control.pause()
        job = _job("a")
        self.assertFalse(self.control.gate.acquire(timeout=0.05))
        self.assertFalse(self.h.lock.locked())
        started = threading.Event()

        def run():
            with generation_slot(self.control.gate, job, poll_interval=0.01) as acquired:
                if acquired and try_start(job):
                    started.set()

        thread = threading.Thread(target=run)
        thread.start()
        time.sleep(0.1)
        self.assertFalse(started.is_set())
        self.control.resume()
        thread.join(timeout=2)
        self.assertTrue(started.is_set())
        self.assertFalse(self.h.lock.locked())

    def test_unload_queue_hands_back_queued_jobs_only(self):
        self.h.jobs = {"a": _job("a"), "b": _job("b", "running"), "c": _job("c")}
        self.assertEqual(self.control.unload_queue(), ["a", "c"])
        self.assertEqual(self.h.jobs["b"]["status"], "running")
        self.assertTrue(self.h.jobs["a"]["handed_back"])
        self.assertEqual(self.control.queued(), [])

    def test_unload_waits_for_the_running_generation(self):
        self.h.lock.acquire()
        self.h.jobs = {"b": _job("b", "running")}
        with self.assertRaises(WorkerBusy):
            self.control.unload_models(wait=False)
        self.assertEqual(self.h.releases, 0)
        done = threading.Event()
        thread = threading.Thread(target=lambda: (self.control.unload_models(), done.set()))
        thread.start()
        time.sleep(0.1)
        self.assertFalse(done.is_set())  # never interrupts the job
        self.h.jobs["b"]["status"] = "completed"
        self.h.lock.release()
        thread.join(timeout=2)
        self.assertTrue(done.is_set())
        self.assertEqual(self.h.releases, 1)
        self.assertEqual(self.control.loaded_models(), [])

    def test_release_pauses_hands_back_and_unloads(self):
        self.h.jobs = {"a": _job("a")}
        result = self.control.release()
        self.assertTrue(result["paused"])
        self.assertEqual(result["handed_back"], ["a"])
        self.assertEqual(result["released"], ["ltx2"])
        self.assertEqual(result["loaded_models"], [])
        self.assertEqual(self.control.acquire()["paused"], False)

    def test_state(self):
        self.h.jobs = {"a": _job("a"), "b": _job("b", "running")}
        state = self.control.state()
        self.assertEqual(state["running_job"], "b")
        self.assertEqual(state["queued"], ["a"])
        self.assertEqual(state["loaded_models"], ["ltx2"])
        self.assertEqual(state["memory"], {"gpu_free_gb": 100.0})
        self.assertFalse(state["paused"])

    def test_idle_unload(self):
        h = Harness(idle_unload_seconds=600)
        self.assertFalse(h.control.idle_tick())
        h.clock.now += 601
        h.jobs = {"a": _job("a")}
        self.assertFalse(h.control.idle_tick())  # work queued: the clock restarts
        h.jobs = {}
        h.clock.now += 601
        self.assertTrue(h.control.idle_tick())
        self.assertEqual(h.loaded, [])
        h.clock.now += 601
        self.assertFalse(h.control.idle_tick())  # nothing left to unload
        self.assertFalse(self.control.idle_tick())  # off by default


class ControlRouteTests(unittest.TestCase):
    def _client(self, harness, enabled=True):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from services.remote_worker.worker_control import create_router

        app = FastAPI()
        app.include_router(create_router(harness.control, enabled=lambda: enabled))
        return TestClient(app)

    def test_pause_resume_and_probes(self):
        h = Harness()
        client = self._client(h)
        self.assertEqual(client.get("/readyz").status_code, 200)
        for _ in range(2):
            response = client.post("/api/v1/worker/pause")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(response.json()["paused"])
        self.assertEqual(client.get("/readyz").status_code, 503)
        self.assertEqual(client.get("/healthz").status_code, 200)
        self.assertFalse(client.post("/api/v1/worker/acquire").json()["paused"])
        self.assertEqual(client.get("/readyz").status_code, 200)

    def test_unload_without_waiting_is_refused_while_a_job_runs(self):
        h = Harness()
        h.jobs = {"b": _job("b", "running"), "a": _job("a")}
        h.lock.acquire()
        client = self._client(h)
        self.assertEqual(client.post("/api/v1/worker/unload-models", params={"wait": "false"}).status_code, 409)
        response = client.post("/api/v1/worker/release", params={"wait": "false"})
        self.assertEqual(response.status_code, 409)
        # The release still paused the worker and handed the queued job back.
        state = client.get("/api/v1/worker/state").json()
        self.assertTrue(state["paused"])
        self.assertEqual(state["queued"], [])
        self.assertEqual(state["running_job"], "b")
        self.assertEqual(h.loaded, ["ltx2"])
        h.lock.release()
        h.jobs["b"]["status"] = "completed"
        released = client.post("/api/v1/worker/release").json()
        self.assertEqual(released["released"], ["ltx2"])

    def test_unload_queue_route(self):
        h = Harness()
        h.jobs = {"a": _job("a")}
        body = self._client(h).post("/api/v1/worker/unload-queue").json()
        self.assertEqual(body["handed_back"], ["a"])

    def test_control_routes_only_on_a_worker(self):
        client = self._client(Harness(), enabled=False)
        self.assertEqual(client.get("/api/v1/worker/state").status_code, 404)
        self.assertEqual(client.post("/api/v1/worker/release").status_code, 404)
        self.assertEqual(client.get("/healthz").status_code, 200)


class AuthTests(unittest.TestCase):
    def _client(self, token):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        app = FastAPI()

        @app.get("/healthz")
        def healthz():
            return {"status": "ok"}

        @app.get("/api/v1/worker/state")
        def state():
            return {"paused": False}

        @app.post("/api/v1/worker/files")
        def files():
            return {"path": "x"}

        @app.get("/docs-like")
        def other():
            return {}

        if token:
            app.add_middleware(BearerTokenMiddleware, token=token)
        return TestClient(app)

    def test_token_required_when_set(self):
        client = self._client("s3cret-token")
        self.assertEqual(client.get("/api/v1/worker/state").status_code, 401)
        self.assertEqual(client.post("/api/v1/worker/files").status_code, 401)
        self.assertEqual(client.get("/docs-like").status_code, 401)
        wrong = {"Authorization": "Bearer nope"}
        self.assertEqual(client.get("/api/v1/worker/state", headers=wrong).status_code, 401)
        good = {"Authorization": "Bearer s3cret-token"}
        self.assertEqual(client.get("/api/v1/worker/state", headers=good).status_code, 200)
        self.assertEqual(client.post("/api/v1/worker/files", headers=good).status_code, 200)
        self.assertEqual(client.get("/healthz").status_code, 200)

    def test_unchanged_without_token(self):
        client = self._client("")
        self.assertEqual(client.get("/api/v1/worker/state").status_code, 200)

    def test_env_token(self):
        self.assertEqual(worker_auth.worker_token_from_env({"MAESTRO_WORKER_TOKEN": " t "}), "t")
        self.assertEqual(worker_auth.worker_token_from_env({}), "")

    def test_compare_is_constant_time(self):
        with mock.patch.object(worker_auth.hmac, "compare_digest", wraps=worker_auth.hmac.compare_digest) as compare:
            self.assertTrue(authorized("Bearer abc", "abc"))
            self.assertTrue(authorized("bearer abc", "abc"))
            self.assertFalse(authorized("Bearer abd", "abc"))
            self.assertEqual(compare.call_count, 3)
        self.assertFalse(authorized("Basic abc", "abc"))
        self.assertFalse(authorized(None, "abc"))
        self.assertFalse(authorized("Bearer ", ""))


class GpuMemoryStatsTest(unittest.TestCase):
    GB = 10**9

    def test_discrete_gpu_reports_cuda_free(self):
        stats = gpu_memory_stats(30 * self.GB, 80 * self.GB, unified=False, mem_available=200 * self.GB)
        self.assertEqual(stats, {"gpu_free_gb": 30.0, "gpu_total_gb": 80.0, "unified_memory": False})

    def test_unified_memory_counts_page_cache_as_free(self):
        # DGX Spark after loading weights: CUDA sees 21 GB, the kernel can hand out 72 GB.
        stats = gpu_memory_stats(21 * self.GB, 130 * self.GB, unified=True, mem_available=72 * self.GB)
        self.assertEqual(stats["gpu_free_gb"], 72.0)
        self.assertEqual(stats["cuda_free_gb"], 21.0)
        self.assertTrue(stats["unified_memory"])

    def test_unified_memory_capped_at_cuda_total(self):
        stats = gpu_memory_stats(100 * self.GB, 120 * self.GB, unified=True, mem_available=125 * self.GB)
        self.assertEqual(stats["gpu_free_gb"], 120.0)

    def test_unified_memory_without_meminfo_keeps_cuda_free(self):
        stats = gpu_memory_stats(21 * self.GB, 130 * self.GB, unified=True, mem_available=None)
        self.assertEqual(stats["gpu_free_gb"], 21.0)


if __name__ == "__main__":
    unittest.main()
