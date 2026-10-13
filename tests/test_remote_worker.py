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
from services.remote_worker.worker_spec import build_run_spec, dstack_secrets  # noqa: E402


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
        self.secrets = {}
        self.fleets = {}
        self.log_lines = [("t1", "[maestro-worker] building venv")]

    def post(self, url, json=None, headers=None, timeout=None):
        path = url.split("127.0.0.1:3000", 1)[1]
        self.calls.append((path, json, headers))
        if headers.get("Authorization") != f"Bearer {self.token}":
            return FakeResponse(401, {"detail": [{"msg": "bad token"}]})
        if path == "/api/server/get_info":
            return FakeResponse(body={"server_version": "0.20.29"})
        if path.endswith("/fleets/list"):
            return FakeResponse(body=[{"name": "studio-comfy-fleet"}])
        if path.endswith("/fleets/apply"):
            conf = json["plan"]["spec"]["configuration"]
            self.fleets[conf["name"]] = conf
            return FakeResponse(body={"name": conf["name"]})
        if path.endswith("/secrets/create_or_update"):
            self.secrets[json["name"]] = json["value"]
            return FakeResponse(body={"name": json["name"]})
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
            start = json.get("start_time")
            logs = [{"timestamp": ts, "log_source": "stdout", "message": base64.b64encode(text.encode()).decode()}
                    for ts, text in self.log_lines if not start or ts > start]
            return FakeResponse(body={"logs": logs, "next_token": None})
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


class EnvOverrideTests(unittest.TestCase):
    def test_env_values_win_are_locked_and_never_stored(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = RemoteWorkerSettingsStore(tmp)
            env = {
                "MAESTRO_REMOTE_SERVER_URL": "http://host.docker.internal:3000",
                "MAESTRO_REMOTE_TOKEN": "env-token-1234567890",
                "MAESTRO_REMOTE_ROUTE_GENERATIONS": "false",
                "MAESTRO_REMOTE_IDLE_TTL_MINUTES": "45",
                "MAESTRO_REMOTE_BACKENDS": "runpod,vastai",
                "UNRELATED": "x",
            }
            manager = RemoteWorkerManager(store, environ=env)
            settings = manager.settings
            self.assertEqual(settings.server_url, "http://host.docker.internal:3000")
            self.assertEqual(settings.token, "env-token-1234567890")
            self.assertFalse(settings.route_generations)
            self.assertEqual(settings.idle_ttl_minutes, 45)
            self.assertEqual(settings.backends, ["runpod", "vastai"])
            public = manager.public_settings()
            self.assertEqual(public["env_locked"], sorted([
                "backends", "idle_ttl_minutes", "route_generations", "server_url", "token",
            ]))
            self.assertNotIn("env-token", json.dumps(public))

            # UI edits to locked fields are ignored; other fields are stored.
            manager.update_settings({"token": "ui-token-abcdef123456", "route_generations": True,
                                     "fleet": "studio-comfy-fleet"})
            self.assertEqual(manager.settings.token, "env-token-1234567890")
            self.assertFalse(manager.settings.route_generations)
            self.assertEqual(manager.settings.fleet, "studio-comfy-fleet")
            with open(os.path.join(tmp, "remote_worker.json")) as handle:
                stored = json.load(handle)
            self.assertEqual(stored["token"], "")
            self.assertEqual(stored["server_url"], "http://127.0.0.1:3000")
            self.assertEqual(stored["fleet"], "studio-comfy-fleet")

    def test_true_strings(self):
        manager = RemoteWorkerManager(
            RemoteWorkerSettingsStore(tempfile.mkdtemp()),
            environ={"MAESTRO_REMOTE_ROUTE_GENERATIONS": "1", "MAESTRO_REMOTE_AUTO_START": "no"},
        )
        self.assertTrue(manager.settings.route_generations)
        self.assertFalse(manager.settings.auto_start)


class RunSpecTests(unittest.TestCase):
    def test_spec_has_safety_limits_probe_and_no_volume(self):
        settings = RemoteWorkerSettings(fleet="studio-comfy-fleet", repo_ref="feat/remote")
        spec = build_run_spec(settings)
        config = spec["configuration"]
        self.assertEqual(spec["run_name"], "maestro-worker")
        self.assertEqual(config["type"], "service")
        self.assertEqual(config["image"], "ghcr.io/quentinriviere/maestro-worker:latest")
        self.assertEqual(config["max_duration"], "4h")
        self.assertEqual(config["max_price"], 0.8)
        self.assertTrue(config["auth"])
        self.assertNotIn("volumes", config)
        self.assertNotIn("registry_auth", config)
        self.assertEqual(config["fleets"], ["studio-comfy-fleet"])
        self.assertTrue(config["probes"][0]["url"].startswith("/"))
        # The route opens on the first passing probe, so probe often until then.
        self.assertEqual((config["probes"][0]["interval"], config["probes"][0]["ready_after"]), (2, 1))
        self.assertTrue(config["probes"][0]["until_ready"])
        self.assertIn("--branch feat/remote", config["commands"][0])
        self.assertTrue(config["commands"][-1].endswith("remote_worker_boot.sh"))
        self.assertNotIn("R2_ACCOUNT_ID", config["env"])
        self.assertEqual(dstack_secrets(settings), {})

    def test_r2_and_registry_credentials_are_secret_references(self):
        settings = RemoteWorkerSettings(
            r2_account_id="acct", r2_access_key_id="key-id", r2_secret_access_key="key-secret",
            r2_bucket="videogen", registry_username="quentin", registry_token="ghp_x",
        )
        spec = build_run_spec(settings)
        rendered = json.dumps(spec)
        for value in ("acct", "key-id", "key-secret", "ghp_x"):
            self.assertNotIn(value, rendered)
        env = spec["configuration"]["env"]
        self.assertEqual(env["R2_SECRET_ACCESS_KEY"], "${{ secrets.MAESTRO_R2_SECRET_ACCESS_KEY }}")
        self.assertEqual(env["MAESTRO_R2_BUCKET"], "videogen")
        self.assertEqual(env["MAESTRO_R2_PREFIX"], "maestro")
        self.assertEqual(spec["configuration"]["registry_auth"]["password"],
                         "${{ secrets.MAESTRO_REGISTRY_TOKEN }}")
        self.assertEqual(set(dstack_secrets(settings).values()), {"acct", "key-id", "key-secret", "ghp_x"})


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

    def test_localhost_alias_for_containers(self):
        os.environ["MAESTRO_LOCALHOST_ALIAS"] = "host.docker.internal"
        self.addCleanup(os.environ.pop, "MAESTRO_LOCALHOST_ALIAS", None)
        client = DstackClient("http://127.0.0.1:3000", "main", "t", session=self.fake)
        self.assertEqual(client.server_url, "http://host.docker.internal:3000")
        remote = DstackClient("https://sky.dstack.ai", "main", "t", session=self.fake)
        self.assertEqual(remote.server_url, "https://sky.dstack.ai")

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
            proxy_check=lambda url, token: self.proxy_up,
        )
        self.proxy_up = True
        self.manager._preload = lambda url, token, model: self.preloads.append((url, model))
        self.worker_downloads = {}
        self.manager._fetch_downloads = lambda url, token: self.worker_downloads
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

    def test_start_applies_the_dedicated_fleet_before_the_run(self):
        self.manager.update_settings({"max_price": 1.5, "gpu": "80GB.."})
        self.manager.start()
        paths = [c[0] for c in self.fake.calls]
        self.assertLess(
            next(i for i, p in enumerate(paths) if p.endswith("/fleets/apply")),
            next(i for i, p in enumerate(paths) if p.endswith("/runs/apply")),
        )
        fleet = self.fake.fleets["maestro"]
        self.assertEqual(fleet["max_price"], 1.5)
        self.assertEqual(fleet["resources"]["gpu"], "80GB..")
        self.assertEqual(fleet["nodes"], {"min": 0, "max": 1})
        self.assertEqual(self.fake.run["run_spec"]["configuration"]["fleets"], ["maestro"])

    def test_unmanaged_fleet_is_left_alone(self):
        self.manager.update_settings({"fleet": "studio-comfy-fleet", "manage_fleet": False})
        self.manager.start()
        self.assertEqual(self.fake.fleets, {})
        self.assertEqual(self.fake.run["run_spec"]["configuration"]["fleets"], ["studio-comfy-fleet"])

    def test_ready_waits_for_the_proxy_route(self):
        self.manager.start()
        self.fake.to_running(ready=True)
        self.proxy_up = False
        status = self.manager.status()
        self.assertEqual(status["state"], "booting")
        self.assertIn("proxy", status["message"])
        self.proxy_up = True
        self.assertEqual(self.manager.status()["state"], "ready")
        self.proxy_up = False  # checked once per run, not on every poll
        self.assertEqual(self.manager.status()["state"], "ready")

    def test_review_pause_holds_the_worker_for_half_an_hour(self):
        self.manager.start()
        self.fake.to_running(ready=True)
        self.manager.status()
        paused_for = {"seconds": None}
        self.manager.review_pause = lambda: paused_for["seconds"]
        self.clock.now += 31 * 60
        paused_for["seconds"] = 20 * 60
        self.assertFalse(self.manager.idle_tick())
        self.assertTrue(self.manager.status()["review_hold"])
        paused_for["seconds"] = 31 * 60
        self.assertTrue(self.manager.idle_tick())

    def test_start_preloads_the_requested_then_upcoming_models(self):
        self.manager.update_settings({"r2_account_id": "acc", "r2_access_key_id": "kid",
                                      "r2_secret_access_key": "secret", "r2_bucket": "bucket"})
        self.manager.upcoming_models = lambda: ["flux2_klein_9b", "minimax_h3", "cloud_veo"]
        self.manager.start("minimax_h3")
        env = self.fake.run["run_spec"]["configuration"]["env"]
        self.assertEqual(env["MAESTRO_PRELOAD_MODELS"], "minimax_h3,flux2_klein_9b")
        self.assertEqual(self.manager.status()["preload_models"], ["minimax_h3", "flux2_klein_9b"])

    def test_boot_steps_come_from_the_pod_logs(self):
        self.manager.start()
        self.fake.run["submitted_at"] = "2026-10-13T11:00:00+00:00"
        self.fake.to_running(ready=False)
        self.fake.log_lines = [
            ("2026-10-13T11:00:40+00:00", '[Boot] {"phase": "clone", "state": "started"}'),
            ("2026-10-13T11:00:42+00:00", '[Boot] {"phase": "clone", "state": "done"}\n'
                                          '[Boot] {"phase": "prefetch", "state": "started", "model_type": "m"}'),
            ("2026-10-13T11:01:00+00:00", "[Maestro] Importing WanGP engine..."),
        ]
        steps = {s["label"]: s for s in self.manager.status()["boot"]["steps"]}
        self.assertEqual(steps["GPU host found and image pulled"]["seconds"], 40)
        self.assertEqual(steps["Code cloned"]["seconds"], 2)
        self.assertEqual(steps["Weights pulled from R2: m"]["state"], "started")
        self.assertEqual(steps["Engine loaded"]["state"], "started")

        self.fake.log_lines.append(("2026-10-13T11:01:30+00:00",
                                    '[Boot] {"phase": "prefetch", "state": "done", "model_type": "m", "gigabytes": 20}'))
        self.clock.now += 5
        steps = {s["label"]: s for s in self.manager.status()["boot"]["steps"]}
        self.assertEqual(steps["Weights pulled from R2: m"]["seconds"], 48)
        self.assertEqual(steps["Weights pulled from R2: m"]["details"], {"gigabytes": 20})

    def test_ready_status_lists_the_worker_downloads(self):
        self.manager.start()
        self.fake.to_running(ready=True)
        self.worker_downloads = {
            "minimax_h3": {"status": "downloading", "phase": "pulling_cache",
                           "progress": {"done_bytes": 12_300_000_000, "total_bytes": 41_000_000_000,
                                        "bytes_per_second": 640_000_000}},
            "flux2_klein_9b": {"status": "completed", "phase": "done"},
        }
        downloads = self.manager.status()["worker_downloads"]
        self.assertEqual([d["model_type"] for d in downloads], ["minimax_h3"])
        from services.remote_worker.job_forwarder import format_progress
        self.assertEqual(format_progress(downloads[0]["progress"]), " · 12.3 / 41.0 GB · 640 MB/s")

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

    def test_stop_when_queue_empty_skips_the_ttl(self):
        self.manager.update_settings({"stop_when_queue_empty": True})
        pending = [True]
        self.manager.pending_work = lambda: pending[0]
        self.manager.start()
        self.fake.to_running(ready=True)
        self.manager.status()
        self.clock.now += 5 * 60
        self.assertFalse(self.manager.idle_tick())  # queue still busy: TTL applies

        pending[0] = False
        self.assertTrue(self.manager.idle_tick())
        self.fake.finish()
        self.assertEqual(self.manager.status()["message"], "Stopped: generation queue is empty")

    def test_queue_empty_stop_waits_a_short_grace(self):
        self.manager.update_settings({"stop_when_queue_empty": True})
        self.manager.start()
        self.fake.to_running(ready=True)
        self.manager.status()
        self.manager.job_started()
        self.manager.job_finished()
        self.clock.now += 10
        self.assertFalse(self.manager.idle_tick())
        self.clock.now += 30
        self.assertTrue(self.manager.idle_tick())

    def test_failing_queue_check_keeps_the_worker(self):
        self.manager.update_settings({"stop_when_queue_empty": True})
        self.manager.pending_work = lambda: 1 / 0
        self.manager.start()
        self.fake.to_running(ready=True)
        self.manager.status()
        self.clock.now += 5 * 60
        self.assertFalse(self.manager.idle_tick())

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

    def test_start_pushes_secrets_before_apply(self):
        self.manager.update_settings({
            "r2_account_id": "acct", "r2_access_key_id": "key-id",
            "r2_secret_access_key": "key-secret", "r2_bucket": "videogen",
        })
        self.manager.start()
        paths = [c[0] for c in self.fake.calls]
        last_secret = max(i for i, p in enumerate(paths) if p.endswith("/secrets/create_or_update"))
        apply_index = paths.index("/api/project/main/runs/apply")
        self.assertLess(last_secret, apply_index)
        self.assertEqual(self.fake.secrets["MAESTRO_R2_SECRET_ACCESS_KEY"], "key-secret")

    def test_masked_secrets_from_ui_keep_stored_values(self):
        self.manager.update_settings({"r2_secret_access_key": "super-secret-value-123"})
        public = self.manager.settings.public()
        self.assertNotIn("super-secret", json.dumps(public))
        self.manager.update_settings({"r2_secret_access_key": public["r2_secret_access_key"]})
        self.assertEqual(self.manager.settings.r2_secret_access_key, "super-secret-value-123")

    def test_ensure_ready_respects_disabled_auto_start(self):
        self.manager.update_settings({"auto_start": False})
        with self.assertRaises(RemoteWorkerError):
            self.manager.ensure_ready(None)
        self.assertFalse(any(c[0].endswith("/runs/apply") for c in self.fake.calls))


if __name__ == "__main__":
    unittest.main()


def test_worker_spec_requires_a_ram_and_cpu_floor():
    from services.remote_worker.settings import RemoteWorkerSettings
    from services.remote_worker.worker_spec import build_fleet_spec, build_run_spec

    settings = RemoteWorkerSettings.from_mapping({"gpu": "80GB..", "disk": "200GB.."})
    run = build_run_spec(settings)["configuration"]["resources"]
    fleet = build_fleet_spec(settings)["configuration"]["resources"]
    assert run == fleet == {"gpu": "80GB..", "disk": "200GB..", "memory": "64GB..", "cpu": "8.."}
    assert "memory" not in build_run_spec(RemoteWorkerSettings.from_mapping({"memory": ""}))["configuration"]["resources"]


def test_worker_wait_changes_the_job_message():
    from services.remote_worker.job_forwarder import _worker_status_message

    starting = {"state": "starting"}
    assert _worker_status_message(starting, waited=5) == "Starting remote GPU worker…"
    assert _worker_status_message(starting, waited=314) == "Starting remote GPU worker… · 5 min 10 s"
    assert _worker_status_message(starting, waited=314) != _worker_status_message(starting, waited=325)
