"""Slow image pull watchdog: replace slow hosts, exclude data centers that keep being slow."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.remote_worker import RemoteWorkerManager, RemoteWorkerSettingsStore  # noqa: E402

REGIONS = ["EUR-IS-1", "EU-RO-1", "US-TX-3"]


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakeClient:
    """In-memory dstack: one run per name, scripted host regions."""

    def __init__(self, regions=None):
        self.runs = {}
        self.applied = []
        self.stopped = []
        self.plans = []
        self.hosts = []  # region of each next host
        self.regions = list(regions or REGIONS)
        self._submissions = 0

    def get_run(self, run_name):
        return self.runs.get(run_name)

    def apply(self, run_spec):
        self._submissions += 1
        self.applied.append(run_spec)
        name = run_spec["run_name"]
        region = self.hosts.pop(0) if self.hosts else "EU-RO-1"
        self.runs[name] = {
            "id": f"run-{self._submissions}", "status": "provisioning", "submitted_at": None,
            "latest_job_submission": {"id": f"sub-{self._submissions}", "status": "provisioning",
                                      "job_provisioning_data": {"region": region, "hostname": "10.0.0.1",
                                                                "price": 0.5}},
        }
        return self.runs[name]

    def stop_run(self, run_name, abort=False):
        self.stopped.append(run_name)
        self.runs[run_name]["status"] = "terminating"

    def finish(self, run_name):
        self.runs[run_name]["status"] = "terminated"

    def run_started(self, run_name):
        self.runs[run_name]["status"] = "running"

    def get_plan(self, run_spec, max_offers=5):
        self.plans.append(run_spec)
        return {"job_plans": [{"offers": [{"region": r, "backend": "runpod"} for r in self.regions]}]}

    def set_secret(self, name, value):
        pass

    def apply_fleet(self, spec):
        pass

    def poll_logs(self, *args, **kwargs):
        return []

    def service_base_url(self, run):
        return None


class SlowHostTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.clock = Clock()
        self.client = FakeClient()
        self.manager = RemoteWorkerManager(
            RemoteWorkerSettingsStore(self.tmp.name),
            client_factory=lambda s: self.client,
            clock=self.clock,
            sleep=lambda seconds: None,
            proxy_check=lambda url, token: False,
            environ={},
        )
        self.manager.update_settings({"token": "tok", "default_model_type": "qwen"})
        self.run = self.manager.run_name

    def _pull(self, seconds):
        """First poll sees the run provisioning, the second one ``seconds`` later."""
        self.manager.status()
        self.clock.now += seconds
        return self.manager.status()

    def _slow_host(self):
        """Let the current host pull too long, then let dstack end the stopped run."""
        status = self._pull(241)
        self.client.finish(self.run)
        return status, self.manager.status()

    def test_run_stuck_provisioning_is_stopped_and_submitted_again(self):
        self.manager.start("qwen")
        self.client.hosts = ["EU-RO-1"]
        status = self._pull(200)
        self.assertEqual(status["state"], "starting")
        self.assertEqual(self.client.stopped, [])

        self.clock.now += 41
        status = self.manager.status()
        self.assertEqual(self.client.stopped, [self.run])  # its own run only
        self.assertEqual(status["state"], "starting")  # queued jobs keep waiting
        self.assertEqual(status["message"], "Slow image pull on EU-RO-1, trying another host")
        self.assertEqual(status["boot"]["message"], status["message"])
        self.assertEqual(status["slow_host_retries"], 1)
        self.assertEqual(len(self.client.applied), 1)  # not before the old run ended

        self.client.finish(self.run)
        status = self.manager.status()
        self.assertEqual(len(self.client.applied), 2)
        self.assertEqual(status["state"], "starting")
        self.assertEqual(status["run_status"], "provisioning")
        first, second = (spec["configuration"] for spec in self.client.applied)
        self.assertEqual(second["env"].get("MAESTRO_PRELOAD_MODELS"), first["env"].get("MAESTRO_PRELOAD_MODELS"))
        self.assertEqual(self.client.applied[1]["run_name"], self.run)

    def test_one_slow_host_does_not_exclude_its_data_center(self):
        self.client.hosts = ["EUR-IS-1", "EU-RO-1"]
        self.manager.start("qwen")
        _, status = self._slow_host()
        self.assertEqual(status["excluded_regions"], [])
        self.assertNotIn("regions", self.client.applied[1]["configuration"])

    def test_two_slow_hosts_in_a_row_in_one_data_center_exclude_it(self):
        self.client.hosts = ["EUR-IS-1", "EUR-IS-1", "EU-RO-1"]
        self.manager.start("qwen")
        self._slow_host()
        _, status = self._slow_host()
        self.assertEqual([e["region"] for e in status["excluded_regions"]], ["EUR-IS-1"])
        self.assertEqual(status["excluded_regions"][0]["expires_in_seconds"], 1200)
        self.assertEqual(status["slow_host_retries"], 2)
        # The new run may use every planned data center but the excluded one.
        self.assertEqual(self.client.applied[2]["configuration"]["regions"], ["EU-RO-1", "US-TX-3"])
        self.assertTrue(all("fleets" not in plan["configuration"] for plan in self.client.plans))

    def test_slow_hosts_in_different_data_centers_exclude_nothing(self):
        self.client.hosts = ["EUR-IS-1", "EU-RO-1", "EUR-IS-1", "US-TX-3"]
        self.manager.start("qwen")
        for _ in range(3):
            _, status = self._slow_host()
        self.assertEqual(status["excluded_regions"], [])

    def test_exclusion_expires(self):
        self.manager.update_settings({"slow_region_exclusion_seconds": 600})
        self.client.hosts = ["EUR-IS-1", "EUR-IS-1", "EU-RO-1"]
        self.manager.start("qwen")
        self._slow_host()
        _, status = self._slow_host()
        self.assertEqual(status["excluded_regions"][0]["expires_in_seconds"], 600)
        self.clock.now += 601
        self.assertEqual(self.manager.status()["excluded_regions"], [])
        self.manager.stop()
        self.client.finish(self.run)
        self.manager.start("qwen")
        self.assertNotIn("regions", self.client.applied[-1]["configuration"])

    def test_cap_reached_keeps_the_current_host(self):
        self.manager.update_settings({"slow_pull_max_retries": 2})
        self.client.hosts = ["EUR-IS-1", "EU-RO-1", "US-TX-3"]
        self.manager.start("qwen")
        self._slow_host()
        self._slow_host()
        status = self._pull(1000)
        self.assertEqual(len(self.client.stopped), 2)
        self.assertEqual(len(self.client.applied), 3)
        self.assertEqual(status["state"], "starting")
        self.assertEqual(status["slow_host_retries"], 2)
        self.assertNotIn("Slow image pull", status["message"])

    def test_threshold_zero_disables_the_watchdog(self):
        self.manager.update_settings({"slow_pull_seconds": 0})
        self.manager.start("qwen")
        self._pull(3600)
        self.assertEqual(self.client.stopped, [])
        self.assertEqual(len(self.client.applied), 1)

    def test_never_excludes_every_data_center(self):
        self.client.regions = ["EUR-IS-1"]
        self.client.hosts = ["EUR-IS-1", "EUR-IS-1", "EUR-IS-1"]
        self.manager.start("qwen")
        self._slow_host()
        _, status = self._slow_host()
        self.assertNotIn("regions", self.client.applied[2]["configuration"])
        self.assertEqual(status["excluded_regions"], [])

    def test_a_fast_host_breaks_the_streak(self):
        self.client.hosts = ["EUR-IS-1", "EU-RO-1", "EUR-IS-1"]
        self.manager.start("qwen")
        self._slow_host()
        self.manager.status()  # second host seen provisioning
        self.clock.now += 30
        self.client.run_started(self.run)
        self.manager.status()  # left provisioning in time
        self.manager.stop()
        self.client.finish(self.run)
        self.manager.start("qwen")
        _, status = self._slow_host()
        self.assertEqual(status["excluded_regions"], [])

    def test_user_stop_cancels_a_pending_replacement(self):
        self.manager.start("qwen")
        self._pull(241)
        self.manager.stop()
        self.client.finish(self.run)
        self.assertEqual(self.manager.status()["state"], "stopped")
        self.assertEqual(len(self.client.applied), 1)


if __name__ == "__main__":
    unittest.main()
