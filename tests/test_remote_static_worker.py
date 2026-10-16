"""Static remote worker (provider "static"): settings, status, waits, pool, forwarder."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.remote_worker import (  # noqa: E402
    RemoteWorkerError,
    RemoteWorkerManager,
    RemoteWorkerSettings,
    RemoteWorkerSettingsStore,
)
from services.remote_worker.job_forwarder import (  # noqa: E402
    RemoteJobForwarder,
    RemoteWorkerPaused,
    WorkerClient,
    _worker_status_message,
)
from services.remote_worker.pool import RemoteWorkerPool  # noqa: E402
from services.remote_worker.static_worker import StaticWorker, redact  # noqa: E402


TOKEN = "worker-token-0123456789"


def _join_wakes():
    import threading

    for thread in threading.enumerate():
        if thread.name == "maestro_static_worker_wake":
            thread.join(timeout=1)


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class SettingsTests(unittest.TestCase):
    def test_defaults_to_dstack(self):
        settings = RemoteWorkerSettings()
        self.assertEqual(settings.provider, "dstack")
        self.assertFalse(settings.is_static)
        self.assertFalse(settings.configured)

    def test_static_mode_needs_only_a_url_and_masks_its_token(self):
        settings = RemoteWorkerSettings()
        settings.update({"provider": "Static", "static_url": "http://spark:7860", "static_token": TOKEN})
        self.assertTrue(settings.is_static)
        self.assertTrue(settings.configured)
        self.assertFalse(settings.dstack_configured)
        self.assertEqual(settings.worker_token, TOKEN)
        self.assertEqual(settings.pool_size, 1)
        public = settings.public()
        self.assertNotIn(TOKEN, str(public))
        self.assertEqual(public["static_token"], "work…6789")
        self.assertTrue(public["static_token_set"])

    def test_dstack_mode_keeps_the_dstack_token_for_the_worker(self):
        settings = RemoteWorkerSettings()
        settings.update({"token": "dstack-token", "static_token": TOKEN, "worker_count": 3})
        self.assertEqual(settings.worker_token, "dstack-token")
        self.assertEqual(settings.pool_size, 3)

    def test_rejects_an_unknown_provider(self):
        with self.assertRaises(ValueError):
            RemoteWorkerSettings().update({"provider": "kubernetes"})

    def test_masked_token_sent_back_is_kept_and_env_overrides_apply(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = RemoteWorkerManager(RemoteWorkerSettingsStore(tmp), environ={
                "MAESTRO_REMOTE_PROVIDER": "static",
                "MAESTRO_REMOTE_STATIC_URL": "http://100.64.0.2:7860",
            })
            manager.update_settings({"static_token": TOKEN})
            public = manager.update_settings({"static_token": "work…6789", "provider": "dstack"})
            self.assertEqual(manager.settings.static_token, TOKEN)
            self.assertEqual(manager.settings.provider, "static")  # env wins
            self.assertIn("provider", public["env_locked"])
            self.assertIn("static_url", public["env_locked"])


class FakeProbe:
    def __init__(self):
        self.result = (200, {"paused": False, "running_job": None, "queued": [], "loaded_models": []}, "")
        self.calls = []

    def __call__(self, url, token):
        self.calls.append((url, token))
        return self.result


class StaticManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.clock = Clock()

        def no_dstack(settings):
            raise AssertionError("static mode must not call dstack")

        self.manager = RemoteWorkerManager(RemoteWorkerSettingsStore(self.tmp), client_factory=no_dstack,
                                           clock=self.clock, sleep=self.clock.sleep)
        self.probe = FakeProbe()
        self.wakes = []
        self.manager.static_worker = StaticWorker(clock=self.clock, probe=self.probe, wake=self.wakes.append)
        self.manager.update_settings({"provider": "static", "static_url": "http://spark:7860",
                                      "static_token": TOKEN, "static_wake_url": "http://spark:8080/upstream/maestro-worker/readyz"})

    def _state(self, code=200, **body):
        self.probe.result = (code, {"paused": False, "running_job": None, "queued": [], **body}, "")
        self.clock.now += 10  # past the probe cache

    def test_states_from_the_worker_probe(self):
        status = self.manager.status()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["service_url"], "http://spark:7860/")
        self.assertEqual(status["price_per_hour"], 0.0)
        self.assertEqual(self.probe.calls[-1], ("http://spark:7860/", TOKEN))
        self.assertTrue(self.manager.ready)
        self._state(running_job="abc")
        self.assertEqual(self.manager.status()["state"], "busy")
        self.assertTrue(self.manager.ready)
        self._state(paused=True)
        self.assertEqual(self.manager.status()["state"], "paused")
        self.assertFalse(self.manager.ready)
        self.probe.result = (0, None, "connection refused")
        self.clock.now += 10
        status = self.manager.status()
        self.assertEqual(status["state"], "unreachable")
        self.assertIn("connection refused", status["message"])
        self.probe.result = (401, {"detail": "Unauthorized"}, "")
        self.clock.now += 10
        self.assertEqual(self.manager.status()["state"], "unknown")

    def test_probe_results_are_cached_briefly(self):
        self.manager.status()
        self.manager.status()
        self.assertEqual(len(self.probe.calls), 1)
        self.clock.now += 5
        self.manager.status()
        self.assertEqual(len(self.probe.calls), 2)

    def test_start_stop_and_idle_never_touch_the_worker_process(self):
        self.assertEqual(self.manager.start("ltx2")["state"], "ready")
        self.assertEqual(self.manager.stop()["state"], "ready")
        self.clock.now += 10 * 3600
        self.assertFalse(self.manager.idle_tick())
        self.assertEqual(self.manager.logs(), [])

    def test_ensure_ready_waits_through_a_long_pause_and_wakes_the_worker(self):
        self._state(paused=True)
        polls = {"n": 0}

        def on_update(status):
            polls["n"] += 1
            if polls["n"] == 2000:  # ~100 min of polling at 3 s
                self.probe.result = (200, {"paused": False, "running_job": None}, "")

        status = self.manager.ensure_ready("ltx2", on_update=on_update, timeout=600)
        self.assertEqual(status["state"], "ready")
        _join_wakes()
        # Woken repeatedly, but at most once a minute over the ~100 min pause.
        self.assertGreater(len(self.wakes), 1)
        self.assertLessEqual(len(self.wakes), 6000 // 60 + 1)
        self.assertTrue(self.wakes[0].endswith("/upstream/maestro-worker/readyz"))

    def test_ensure_ready_times_out_while_unreachable(self):
        self.probe.result = (0, None, "timed out")
        with self.assertRaises(RemoteWorkerError) as raised:
            self.manager.ensure_ready("ltx2", timeout=60, poll_interval=3)
        self.assertIn("unreachable", str(raised.exception))
        self.assertEqual(self.wakes, [])

    def test_ensure_ready_fails_at_once_on_a_refused_token(self):
        self.probe.result = (403, None, "")
        with self.assertRaises(RemoteWorkerError):
            self.manager.ensure_ready("ltx2", timeout=3600)
        self.assertEqual(len(self.probe.calls), 1)

    def test_pool_waits_for_a_paused_worker_without_failing(self):
        pool = RemoteWorkerPool(self.manager)
        self._state(paused=True)
        messages = []

        def on_update(status):
            messages.append(_worker_status_message(status))
            if len(messages) == 1500:  # well past the 45 min timeout
                self.probe.result = (200, {"paused": False, "running_job": None}, "")

        pod = pool.acquire("ltx2", on_update=on_update, poll_interval=3.0)
        self.assertIs(pod, self.manager)
        self.assertEqual(pod.active_jobs, 1)
        self.assertIn("paused", messages[0])
        _join_wakes()
        self.assertTrue(self.wakes)

    def test_pool_gives_up_on_an_unreachable_worker_after_the_timeout(self):
        pool = RemoteWorkerPool(self.manager)
        self.probe.result = (0, None, "no route to host")
        with self.assertRaises(RemoteWorkerError):
            pool.acquire("ltx2", timeout=60, poll_interval=3.0)


class FakeResponse:
    def __init__(self, status, body=None):
        self.status_code = status
        self._body = body
        self.content = b"x" if body is not None else b""
        self.text = str(body)

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._body

    def close(self):
        pass


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers = []

    def request(self, method, url, **kwargs):
        self.headers.append(kwargs.get("headers"))
        return self.responses.pop(0)


class WorkerClientTests(unittest.TestCase):
    def test_paused_submission_raises_a_retry_later_error(self):
        session = FakeSession([FakeResponse(423, {"detail": "Worker paused"})])
        client = WorkerClient("http://spark:7860", TOKEN, session=session, sleep=lambda s: None)
        with self.assertRaises(RemoteWorkerPaused):
            client.post_json("api/v1/worker/jobs", {"params": {}}, retry=False)
        self.assertEqual(session.headers[0]["Authorization"], f"Bearer {TOKEN}")


class ForwarderManager:
    def __init__(self):
        self.ready_calls = 0
        self.finished = 0

        class S:
            token = "dstack"
            worker_token = TOKEN

        self.settings = S()

    def job_started(self):
        pass

    def job_finished(self):
        self.finished += 1

    def ensure_ready(self, model_type, *, should_abort, on_update):
        self.ready_calls += 1
        return {"state": "ready", "service_url": "http://spark:7860/"}


class HandBackWorker:
    """A worker that refuses or hands back the first submissions."""

    def __init__(self, script):
        self.script = list(script)  # per submission: "paused", "handed_back" or "ok"
        self.submissions = 0
        self.current = None

    def get_json(self, path, *, timeout=30.0, params=None):
        if path == "api/v1/models/downloads/status":
            return {"downloads": {"ltx2": {"status": "completed"}}}
        if path.startswith("api/v1/worker/jobs/"):
            if self.current == "handed_back":
                return {"status": "cancelled", "handed_back": True, "message": "Handed back to the controller"}
            return {"status": "completed", "output_files": ["out.mp4"], "files": ["out.mp4"], "sidecars": []}
        raise AssertionError(path)

    def post_json(self, path, body=None, *, timeout=60.0, retry=True):
        if path == "api/v1/worker/jobs":
            self.submissions += 1
            self.current = self.script.pop(0)
            if self.current == "paused":
                raise RemoteWorkerPaused("The remote worker is paused")
            return {"job_id": f"r{self.submissions}"}
        raise AssertionError(path)

    def download(self, path, dest):
        with open(dest, "wb") as handle:
            handle.write(b"video")


class ForwarderHandBackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.manager = ForwarderManager()
        self.tokens = []
        self.updates = []

    def _run(self, worker, cancelled=lambda: False):
        def client_factory(url, token):
            self.tokens.append(token)
            return worker

        forwarder = RemoteJobForwarder(self.manager, client_factory=client_factory,
                                       store_factory=lambda s: None, poll_interval=0, sleep=lambda s: None)
        return forwarder.run({"model_type": "ltx2", "prompt": "x"}, out_dir=os.path.join(self.tmp, "out"),
                             resolve_lora=lambda m, n: None, update=lambda **u: self.updates.append(u),
                             is_cancelled=cancelled)

    def test_handed_back_and_paused_jobs_are_resubmitted_not_failed(self):
        worker = HandBackWorker(["paused", "handed_back", "ok"])
        result = self._run(worker)
        self.assertEqual(result.output_files, ["out.mp4"])
        self.assertEqual(worker.submissions, 3)
        self.assertEqual(self.manager.ready_calls, 3)
        self.assertEqual(self.manager.finished, 3)
        self.assertEqual(set(self.tokens), {TOKEN})
        self.assertTrue(any("waiting to resubmit" in str(u.get("message")) for u in self.updates))

    def test_a_cancelled_job_is_not_resubmitted(self):
        from services.remote_worker.job_forwarder import RemoteJobCancelled

        state = {"cancel": False}
        worker = HandBackWorker(["handed_back", "ok"])
        original = worker.get_json

        def get_json(path, **kwargs):
            if path.startswith("api/v1/worker/jobs/"):
                state["cancel"] = True
            return original(path, **kwargs)

        worker.get_json = get_json
        with self.assertRaises(RemoteJobCancelled):
            self._run(worker, cancelled=lambda: state["cancel"])
        self.assertEqual(worker.submissions, 1)


class RedactTests(unittest.TestCase):
    def test_wake_url_credentials_never_reach_the_logs(self):
        self.assertEqual(redact("http://user:sk-secret@spark:8080/upstream/x"), "http://***@spark:8080/upstream/x")
        self.assertEqual(redact("http://spark:8080/upstream/x"), "http://spark:8080/upstream/x")


if __name__ == "__main__":
    unittest.main()
