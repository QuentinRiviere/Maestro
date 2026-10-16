"""Remote worker on an owned DGX Spark (dstack SSH fleet, linux/arm64, sm_121)."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
for path in (_APP_DIR, _HERE):
    if path not in sys.path:
        sys.path.insert(0, path)

from services import generation_provenance as gp  # noqa: E402
from services.remote_worker import (  # noqa: E402
    DstackClient,
    RemoteWorkerManager,
    RemoteWorkerSettings,
    RemoteWorkerSettingsStore,
    cache_warmer,
    kernel_cache,
)
from services.remote_worker.job_forwarder import _billing  # noqa: E402
from services.remote_worker.worker_spec import build_run_spec  # noqa: E402
from test_remote_worker import Clock, FakeDstack  # noqa: E402

# What docs/Remote-worker-DGX-Spark.md tells the user to set.
SPARK_SETTINGS = {
    "image": "ghcr.io/quentinriviere/maestro-worker:spark",
    "fleet": "spark",
    "manage_fleet": False,
    "backends": ["remote"],
    "cache_warm_backends": ["runpod"],
    "gpu": "GB10",
    "memory": "64GB..",
    "cpu": "8..",
    "disk": "100GB..",
    "slow_pull_seconds": 0,
    "pinned_memory": False,
    "host_data_path": "/var/lib/maestro-worker",
}


class SparkRunSpecTests(unittest.TestCase):
    def test_run_targets_the_existing_ssh_fleet_with_a_gb10_filter(self):
        settings = RemoteWorkerSettings.from_mapping(SPARK_SETTINGS)
        configuration = build_run_spec(settings)["configuration"]
        self.assertEqual(configuration["image"], "ghcr.io/quentinriviere/maestro-worker:spark")
        self.assertEqual(configuration["fleets"], ["spark"])
        self.assertEqual(configuration["backends"], ["remote"])
        # A name filter, not a VRAM floor: dstack reads 0 MiB of VRAM on the
        # GB10's unified memory, and a superchip name makes dstack match arm64.
        self.assertEqual(configuration["resources"]["gpu"], "GB10")
        self.assertEqual(configuration["volumes"], [
            {"instance_path": "/var/lib/maestro-worker", "path": "/workspace/data", "optional": True}])
        self.assertEqual(configuration["env"]["MAESTRO_PINNED_MEMORY"], "0")

    def test_cloud_runs_get_no_instance_volume(self):
        configuration = build_run_spec(RemoteWorkerSettings())["configuration"]
        self.assertNotIn("volumes", configuration)
        for bad in ("relative/path", "/", "/a:/b"):
            settings = RemoteWorkerSettings.from_mapping({"host_data_path": bad})
            self.assertNotIn("volumes", build_run_spec(settings)["configuration"], bad)

    def test_settings_accept_the_new_fields_from_text(self):
        settings = RemoteWorkerSettings()
        changed = settings.update({"cache_warm_backends": "runpod, aws", "host_data_path": " /data/m "})
        self.assertEqual(set(changed), {"cache_warm_backends", "host_data_path"})
        self.assertEqual(settings.cache_warm_backends, ["runpod", "aws"])
        self.assertEqual(settings.host_data_path, "/data/m")


class SparkManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.fake = FakeDstack()
        self.clock = Clock()
        self.manager = RemoteWorkerManager(
            RemoteWorkerSettingsStore(self.tmp.name),
            client_factory=lambda s: DstackClient(s.server_url, s.project, s.token, session=self.fake),
            clock=self.clock,
            sleep=self.clock.sleep,
            proxy_check=lambda url, token: True,
        )
        self.manager._preload = lambda url, token, model: None
        self.manager._fetch_downloads = lambda url, token: {}
        self.manager.update_settings({"token": self.fake.token, **SPARK_SETTINGS})

    def test_the_ssh_fleet_is_never_applied_and_an_owned_host_costs_nothing(self):
        self.manager.start()
        self.assertEqual(self.fake.fleets, {})
        paths = [call[0] for call in self.fake.calls]
        self.assertFalse(any(path.endswith("/fleets/apply") for path in paths))
        self.assertTrue(any(path.endswith("/runs/apply") for path in paths))
        self.fake.run["status"] = "running"
        self.fake.run["latest_job_submission"].update({
            "status": "running",
            "probes": [{"success_streak": 2}],
            "job_provisioning_data": {
                "backend": "remote", "region": "remote", "price": 0.0,
                "instance_type": {"name": "instance", "resources": {
                    "spot": False, "gpus": [{"name": "GB10", "memory_mib": 0, "vendor": "nvidia"}]}},
            },
        })
        self.fake.run["service"] = {"url": "/proxy/services/main/maestro-worker/"}
        status = self.manager.status()
        self.assertEqual(status["state"], "ready")
        self.assertEqual(status["price_per_hour"], 0.0)
        self.assertEqual(status["gpu_name"], "GB10")
        self.assertEqual(status["gpu_backend"], "remote")
        self.assertNotIn("gpu_memory_gb", status)  # unified memory: dstack reports 0


class SparkProvenanceTests(unittest.TestCase):
    def test_a_zero_price_is_kept_and_costs_zero(self):
        class Manager:
            run_name = "maestro-worker"

            @staticmethod
            def status():
                return {"price_per_hour": 0.0, "gpu_name": "GB10", "gpu_backend": "remote"}

        billing = _billing(Manager(), {"price_per_hour": 0.0}, waited=1.0, worked=36.0)
        self.assertEqual(billing["price_per_hour"], 0.0)
        record = gp.merge_remote({"hardware": {"gpu": "NVIDIA GB10"}}, billing)
        self.assertEqual(record["hardware"]["price_per_hour"], 0.0)
        self.assertEqual(record["cost"]["estimated_usd"], 0.0)

    def test_a_missing_price_still_means_no_cost(self):
        record = gp.merge_remote({}, {"work_seconds": 10.0, "price_per_hour": None})
        self.assertNotIn("cost", record)


class SparkCacheWarmerTests(unittest.TestCase):
    def test_the_warmer_never_targets_the_ssh_fleet(self):
        settings = RemoteWorkerSettings.from_mapping({"backends": ["remote"]})
        self.assertEqual(cache_warmer.trusted_backends(settings, "/nonexistent.yml"), [])

    def test_the_warmer_uses_its_own_backends(self):
        settings = RemoteWorkerSettings.from_mapping({"backends": ["remote"], "cache_warm_backends": ["aws"]})
        self.assertEqual(cache_warmer.trusted_backends(settings, "/nonexistent.yml"), ["aws"])
        spec = cache_warmer.build_warm_run_spec(settings, ["flux2_klein_9b"], ["aws"])
        self.assertEqual(spec["configuration"]["image"], "ghcr.io/quentinriviere/maestro-worker:cpu")

    def test_the_warmer_image_stays_the_amd64_cpu_tag(self):
        self.assertEqual(cache_warmer.warm_image("ghcr.io/quentinriviere/maestro-worker:spark"),
                         "ghcr.io/quentinriviere/maestro-worker:cpu")


class SparkKernelCacheTests(unittest.TestCase):
    def test_amd64_keys_are_unchanged_and_arm64_keys_are_separate(self):
        amd64 = kernel_cache.cache_key("sm_89", machine="x86_64")
        self.assertFalse(amd64.endswith("x86_64"))
        self.assertTrue(amd64.startswith("sm_89-torch"))
        spark = kernel_cache.cache_key("sm_121", machine="aarch64")
        self.assertTrue(spark.startswith("sm_121-torch"))
        self.assertTrue(spark.endswith("-aarch64"))
        self.assertNotEqual(kernel_cache.cache_key("sm_90", machine="aarch64"),
                            kernel_cache.cache_key("sm_90", machine="x86_64"))
        self.assertEqual(kernel_cache.cache_key("", machine="aarch64"), "")



if __name__ == "__main__":
    unittest.main()
