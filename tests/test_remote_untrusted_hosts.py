"""Pods on untrusted hosts: prefix-scoped keys, scoped outputs, no cache writes."""
from __future__ import annotations

import os
import sys

import pytest

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.remote_worker import cache_warmer, ephemeral_r2  # noqa: E402
from services.remote_worker.r2_store import R2Store  # noqa: E402
from services.remote_worker.settings import RemoteWorkerSettings  # noqa: E402
from services.remote_worker.worker_spec import build_run_spec, dstack_secrets  # noqa: E402

from test_remote_ephemeral_r2 import FakeCloudflare, _Response  # noqa: E402


def _settings(**overrides):
    values = dict(server_url="http://dstack", project="main", token="t", r2_account_id="acc",
                  r2_access_key_id="long-kid", r2_secret_access_key="long-secret", r2_bucket="videogen",
                  r2_prefix="maestro", cf_api_token="cf-admin", untrusted_hosts=True)
    values.update(overrides)
    return RemoteWorkerSettings.from_mapping(values)


class TempCloudflare(FakeCloudflare):
    def __init__(self):
        super().__init__()
        self.temp = []

    def request(self, method, url, headers=None, json=None, timeout=None):
        if url.endswith("r2/temp-access-credentials"):
            self.temp.append(json)
            n = len(self.temp)
            return _Response({"success": True, "result": {
                "accessKeyId": f"tmp{n}", "secretAccessKey": f"s{n}", "sessionToken": f"st{n}"}})
        return super().request(method, url, headers=headers, json=json, timeout=timeout)


def test_untrusted_pod_gets_read_only_inputs_and_its_own_output_folder(tmp_path):
    cloudflare = TempCloudflare()
    keys = ephemeral_r2.EphemeralR2Keys(tmp_path / "k.json", session=cloudflare, clock=lambda: 1_800_000_000.0)
    creds = keys.issue_untrusted(_settings(), "maestro-worker", 3600, "maestro-worker-abc")
    read, write = cloudflare.temp
    assert read["permission"] == "object-read-only" and read["parentAccessKeyId"] == "tok1"
    assert read["prefixes"] == ["maestro/ckpts/", "maestro/hf/", "maestro/loras/", "maestro/manifests/",
                                "maestro/assets/"]
    assert write == {**write, "permission": "object-read-write", "prefixes": ["maestro/outputs/maestro-worker-abc/"]}
    assert creds["session_token"] == "st1" and creds["write_session_token"] == "st2"

    secrets = dstack_secrets(_settings(), creds)
    assert "long-secret" not in secrets.values()
    env = build_run_spec(_settings(), "maestro-worker-abc")["configuration"]["env"]
    assert env["MAESTRO_R2_CACHE_READONLY"] == "1" and env["MAESTRO_R2_REQUIRE_HASHES"] == "1"
    assert env["MAESTRO_R2_OUTPUT_SCOPE"] == "maestro-worker-abc"
    assert env["R2_WRITE_SESSION_TOKEN"] == "${{ secrets.MAESTRO_R2_WRITE_SESSION_TOKEN }}"
    assert "MAESTRO_R2_CACHE_READONLY" not in build_run_spec(_settings(untrusted_hosts=False))["configuration"]["env"]


class _Client:
    def __init__(self):
        self.uploads = []

    def upload_file(self, path, bucket, key):
        self.uploads.append(key)


def test_store_writes_scoped_outputs_with_the_write_set():
    read, write = _Client(), _Client()
    store = R2Store(account_id="a", access_key_id="r", secret_access_key="s", bucket="b",
                    client=read, write_client=write, output_scope="maestro-worker-abc")
    key = store.output_key("job1", "clip.mp4")
    assert key == "maestro/outputs/maestro-worker-abc/job1/clip.mp4" and store.is_output_key(key)
    store.upload(__file__, key, skip_if_exists=False)
    assert write.uploads == [key] and read.uploads == []


def test_cache_warming_and_unscoped_keys_are_refused_on_untrusted_hosts(tmp_path):
    warmer = cache_warmer.CacheWarmer(lambda: _settings(), lambda: None)
    with pytest.raises(cache_warmer.CacheWarmerError, match="untrusted"):
        warmer.start(["flux2_klein_9b"])

    from services.remote_worker.manager import RemoteWorkerError, RemoteWorkerManager
    from services.remote_worker.settings import RemoteWorkerSettingsStore

    manager = RemoteWorkerManager(RemoteWorkerSettingsStore(tmp_path), environ={})
    with pytest.raises(RemoteWorkerError, match="Cloudflare API token"):
        manager.pod_r2_keys(_settings(cf_api_token=""), "maestro-worker")
