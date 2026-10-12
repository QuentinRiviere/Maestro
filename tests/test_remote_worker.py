"""Remote worker lifecycle tests against a fake dstack server (no network)."""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.remote_worker import (  # noqa: E402
    DstackClient,
    DstackError,
    RemoteWorkerError,
    RemoteWorkerManager,
    RemoteWorkerSettings,
    RemoteWorkerSettingsStore,
)
from services.remote_worker.settings import read_local_dstack_config  # noqa: E402
from services.remote_worker.worker_spec import build_run_spec  # noqa: E402


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code = status
        self._body = body
        self.content = b"" if body is None else json.dumps(body).encode()
        self.text = self.content.decode()

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        return self._body


class FakeDstack:
    """Stateful fake of the few dstack routes Maestro uses."""

    def __init__(self):
        self.calls = []
        self.run = None
        self.token = "tok-123456789abc"
        self.volumes = [{"name": "maestro-models", "status": "active",
                         "configuration": {"region": "EU-SE-1"}}]

    def post(self, url, json=None, headers=None, timeout=None):
        path = url.split("127.0.0.1:3000", 1)[1]
        self.calls.append((path, json, headers))
        if headers.get("Authorization") != f"Bearer {self.token}":
            return FakeResponse(401, {"detail": [{"msg": "bad token"}]})
        if path == "/api/server/get_info":
            return FakeResponse(body={"server_version": "0.20.29"})
        if path.endswith("/fleets/list"):
            return FakeResponse(body=[{"name": "studio-comfy-fleet"}])
        if path.endswith("/volumes/list"):
            return FakeResponse(body=self.volumes)
        if path.endswith("/volumes/create"):
            config = json["configuration"]
            self.volumes.append({"name": config["name"], "status": "submitted", "configuration": config})
            return FakeResponse(body=self.volumes[-1])
        if path.endswith("/runs/get"):
            if self.run is None:
                return FakeResponse(400, {"detail": [{"msg": "Run not found", "code": "resource_not_exists"}]})
            return FakeResponse(body=self.run)
        if path.endswith("/runs/apply"):
            spec = json["plan"]["run_spec"]
            self.run = {"id": "run-1", "status": "submitted", "run_spec": spec,
                        "jobs": [{"job_spec": {"probes": [{"ready_after": 2}]}}],
                        "latest_job_submission": {"id": "sub-1", "status": "submitted", "probes": []}}
            return FakeResponse(body=self.run)
        if path.endswith("/runs/stop"):
            self.run["status"] = "terminating"
            return FakeResponse()
        if path.endswith("/logs/poll"):
            line = base64.b64encode(b"[maestro-worker] building venv").decode()
            return FakeResponse(body={"logs": [{"timestamp": "t1", "log_source": "stdout", "message": line}],
                                      "next_token": None})
        return FakeResponse(404, {"detail": "nope"})

    # transitions
    def to_running(self, *, ready):
        self.run["status"] = "running"
        self.run["latest_job_submission"].update({
            "status": "running",
            "probes": [{"success_streak": 2 if ready else 0}],
            "job_provisioning_data": {"price": 0.59, "region": "EU-SE-1"},
        })
        self.run["service"] = {"url": "/proxy/services/main/maestro-worker/"}

    def finish(self, status="terminated", reason=None):
        self.run["status"] = status
        if reason:
            self.run["latest_job_submission"]["termination_reason"] = reason


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class SettingsTests(unittest.TestCase):
    def test_public_masks_token_and_coerces_types(self):
        settings = RemoteWorkerSettings()
        changed = settings.update({"token": "abcdefghijklmnop", "idle_ttl_minutes": "45",
                                   "max_price": "1.2", "backends": "runpod, vastai", "unknown": 1})
        self.assertEqual(set(changed), {"token", "idle_ttl_minutes", "max_price", "backends"})
        public = settings.public()
        self.assertEqual(public["token"], "abcd…mnop")
        self.assertTrue(public["token_set"])
        self.assertEqual(public["backends"], ["runpod", "vastai"])
        self.assertEqual(public["idle_ttl_minutes"], 45)

    def test_rejects_zero_ttl(self):
        with self.assertRaises(ValueError):
            RemoteWorkerSettings().update({"idle_ttl_minutes": 0})

    def test_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RemoteWorkerSettingsStore(tmp)
            settings = store.load()
            settings.update({"token": "secret-token-xyz", "repo_ref": "feat/x"})
            store.save(settings)
            loaded = store.load()
            self.assertEqual(loaded.token, "secret-token-xyz")
            self.assertEqual(loaded.repo_ref, "feat/x")

    def test_reads_default_project_from_dstack_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.yml")
            with open(path, "w") as handle:
                handle.write("projects:\n- name: other\n  url: http://x\n  token: a\n"
                             "- default: true\n  name: main\n  url: http://127.0.0.1:3000\n  token: b\n")
            self.assertEqual(read_local_dstack_config(path),
                             {"server_url": "http://127.0.0.1:3000", "project": "main", "token": "b"})


class RunSpecTests(unittest.TestCase):
    def test_spec_has_safety_limits_volume_and_probe(self):
        settings = RemoteWorkerSettings(fleet="studio-comfy-fleet", repo_ref="feat/remote")
        spec = build_run_spec(settings)
        config = spec["configuration"]
        self.assertEqual(spec["run_name"], "maestro-worker")
        self.assertEqual(config["type"], "service")
        self.assertEqual(config["max_duration"], "4h")
        self.assertEqual(config["max_price"], 0.8)
        self.assertTrue(config["auth"])
        self.assertEqual(config["volumes"], [{"name": "maestro-models", "path": "/workspace/vol"}])
        self.assertEqual(config["fleets"], ["studio-comfy-fleet"])
        self.assertTrue(config["probes"][0]["url"].startswith("/"))
        self.assertTrue(any("--branch feat/remote" in c for c in config["commands"]))
        self.assertTrue(config["commands"][-1].endswith("remote_worker_boot.sh"))
        self.assertNotIn("repo_id", spec)


class DstackClientTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeDstack()
        self.client = DstackClient("http://127.0.0.1:3000/", "main", self.fake.token, session=self.fake)

    def test_missing_run_is_none(self):
        self.assertIsNone(self.client.get_run("maestro-worker"))

    def test_bad_token_raises(self):
        client = DstackClient("http://127.0.0.1:3000", "main", "wrong", session=self.fake)
        with self.assertRaises(DstackError):
            client.get_info()

    def test_logs_are_base64_decoded(self):
        lines = self.client.poll_logs("maestro-worker", "sub-1")
        self.assertEqual(lines[0]["message"], "[maestro-worker] building venv")

    def test_service_url_is_absolute(self):
        url = self.client.service_base_url({"service": {"url": "/proxy/services/main/w/"}})
        self.assertEqual(url, "http://127.0.0.1:3000/proxy/services/main/w/")


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = FakeDstack()
        self.clock = Clock()
        self.preloads = []
        self.manager = RemoteWorkerManager(
            RemoteWorkerSettingsStore(self.tmp.name),
            client_factory=lambda s: DstackClient(s.server_url, s.project, s.token, session=self.fake),
            clock=self.clock,
            sleep=self.clock.sleep,
        )
        self.manager._preload = lambda url, token, model: self.preloads.append((url, model))
        self.manager.update_settings({"token": self.fake.token})

    def _wait_preload_threads(self):
        import threading
        for thread in threading.enumerate():
            if thread.name == "maestro_remote_worker_preload":
                thread.join(timeout=1)

    def test_not_configured_without_token(self):
        manager = RemoteWorkerManager(RemoteWorkerSettingsStore(self.tmp.name + "/empty"))
        self.assertEqual(manager.status()["state"], "not_configured")
        with self.assertRaises(RemoteWorkerError):
            manager.start()

    def test_masked_token_from_ui_is_ignored(self):
        public = self.manager.update_settings({"token": "tok-…9abc", "project": "main"})
        self.assertTrue(public["token_set"])
        self.assertEqual(self.manager.settings.token, self.fake.token)

    def test_lifecycle_start_boot_ready_preload_stop(self):
        self.assertEqual(self.manager.status()["state"], "stopped")
        self.assertEqual(self.manager.start("minimax_h3_frames")["state"], "starting")
        self.manager.start("minimax_h3_frames")  # idempotent while active
        self.assertEqual(sum(1 for c in self.fake.calls if c[0].endswith("/runs/apply")), 1)

        self.fake.to_running(ready=False)
        self.assertEqual(self.manager.status()["state"], "booting")
        self.fake.to_running(ready=True)
        status = self.manager.status()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["service_url"], "http://127.0.0.1:3000/proxy/services/main/maestro-worker/")
        self.assertEqual(status["price_per_hour"], 0.59)
        self.manager.status()
        self._wait_preload_threads()
        self.assertEqual(self.preloads, [(status["service_url"], "minimax_h3_frames")])

        self.assertEqual(self.manager.stop()["state"], "stopping")
        self.fake.finish()
        self.assertEqual(self.manager.status()["message"], "Stopped by user")

    def test_idle_ttl_stops_ready_worker_but_not_during_jobs(self):
        self.manager.start()
        self.fake.to_running(ready=True)
        self.manager.status()
        self.clock.now += 29 * 60
        self.assertFalse(self.manager.idle_tick())

        self.manager.job_started()
        self.clock.now += 60 * 60
        self.assertFalse(self.manager.idle_tick())
        self.manager.job_finished()

        self.clock.now += 29 * 60
        self.assertFalse(self.manager.idle_tick())
        self.clock.now += 2 * 60
        self.assertTrue(self.manager.idle_tick())
        self.assertTrue(any(c[0].endswith("/runs/stop") for c in self.fake.calls))

    def test_idle_clock_does_not_run_while_booting(self):
        self.manager.start()
        self.fake.to_running(ready=False)
        self.manager.status()
        self.clock.now += 3 * 3600
        self.assertFalse(self.manager.idle_tick())

    def test_ensure_ready_auto_starts_and_waits(self):
        polls = []

        def on_update(status):
            polls.append(status["state"])
            if len(polls) == 2:
                self.fake.to_running(ready=False)
            if len(polls) == 3:
                self.fake.to_running(ready=True)

        status = self.manager.ensure_ready("ltx2_22B", on_update=on_update, poll_interval=5)
        self.assertEqual(status["state"], "ready")
        self.assertEqual(polls, ["stopped", "starting", "booting", "ready"])

    def test_ensure_ready_surfaces_failed_run(self):
        def on_update(status):
            if status["state"] == "starting":
                self.fake.finish("failed", "failed_to_start_due_to_no_capacity")

        with self.assertRaises(RemoteWorkerError) as ctx:
            self.manager.ensure_ready(None, on_update=on_update, poll_interval=5)
        self.assertIn("no_capacity", str(ctx.exception))

    def test_start_refuses_missing_or_inactive_volume(self):
        self.fake.volumes = []
        with self.assertRaises(RemoteWorkerError):
            self.manager.start()
        created = self.manager.create_volume()
        self.assertEqual(created, {"name": "maestro-models", "status": "submitted"})
        create_body = [c[1] for c in self.fake.calls if c[0].endswith("/volumes/create")][0]
        self.assertEqual(create_body["configuration"]["region"], "EU-SE-1")
        self.assertEqual(create_body["configuration"]["size"], "200GB")
        with self.assertRaises(RemoteWorkerError):
            self.manager.start()  # still "submitted"
        with self.assertRaises(RemoteWorkerError):
            self.manager.create_volume()  # already exists
        self.assertFalse(any(c[0].endswith("/runs/apply") for c in self.fake.calls))

    def test_ensure_ready_respects_disabled_auto_start(self):
        self.manager.update_settings({"auto_start": False})
        with self.assertRaises(RemoteWorkerError):
            self.manager.ensure_ready(None)
        self.assertFalse(any(c[0].endswith("/runs/apply") for c in self.fake.calls))


if __name__ == "__main__":
    unittest.main()
