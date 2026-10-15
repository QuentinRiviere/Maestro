"""R2 cache warming on a CPU dstack task."""
from __future__ import annotations

import datetime
import io
import json
import os
import sys

import pytest

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.remote_worker import cache_warmer  # noqa: E402
from services.remote_worker.settings import RemoteWorkerSettings  # noqa: E402


def _settings(**overrides):
    values = dict(
        server_url="http://dstack", project="main", token="t",
        r2_account_id="acc", r2_access_key_id="kid", r2_secret_access_key="secret", r2_bucket="bucket",
        registry_username="me", registry_token="ghp",
    )
    values.update(overrides)
    return RemoteWorkerSettings.from_mapping(values)


def test_warm_spec_is_a_cpu_task_that_never_inlines_secrets():
    spec = cache_warmer.build_warm_run_spec(_settings(), ["flux2_klein_9b", "minimax_h3"])
    config = spec["configuration"]
    assert config["type"] == "task"
    assert config["resources"]["gpu"] == 0
    assert config["fleets"] == [cache_warmer.FLEET_NAME]
    assert config["max_price"] <= 0.5
    assert "--cache-warm flux2_klein_9b,minimax_h3" in config["commands"][1]
    rendered = json.dumps(spec)
    assert "secret" not in rendered.replace("secrets.", "") and "ghp" not in rendered


def test_warm_spec_runs_the_cpu_tag_on_the_given_backends():
    spec = cache_warmer.build_warm_run_spec(_settings(), ["flux2_klein_9b"], ["runpod"])
    assert spec["configuration"]["image"] == "ghcr.io/quentinriviere/maestro-worker:cpu"
    assert spec["configuration"]["backends"] == ["runpod"]
    assert cache_warmer.warm_image("registry:5000/team/worker") == "registry:5000/team/worker:cpu"
    assert cache_warmer.warm_image("ghcr.io/a/w:deps-1@sha256:abc") == "ghcr.io/a/w:cpu"


def test_only_backends_without_community_hosts_are_trusted(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(
        "projects:\n- name: main\n  backends:\n"
        "  - type: runpod\n    community_cloud: false\n"
        "  - type: vastai\n    community_cloud: true\n"
    )
    settings = _settings(backends=["runpod", "vastai", "aws"])
    assert cache_warmer.trusted_backends(settings, str(config)) == ["runpod", "aws"]
    assert cache_warmer.trusted_backends(settings, str(tmp_path / "missing.yml")) == ["aws"]


def test_model_types_are_validated_and_deduplicated():
    assert cache_warmer.valid_model_types(["a", "a", " b "]) == ["a", "b"]
    with pytest.raises(cache_warmer.CacheWarmerError):
        cache_warmer.valid_model_types(["x; rm -rf /"])
    with pytest.raises(cache_warmer.CacheWarmerError):
        cache_warmer.valid_model_types([])


def test_progress_is_read_from_marked_log_lines():
    lines = [
        "noise",
        '[CacheWarm] {"model_type": "flux2_klein_9b", "state": "downloading"}',
        '12:00 [CacheWarm] {"model_type": "flux2_klein_9b", "state": "cached", "files": 7}',
        '[CacheWarm] not json',
    ]
    assert cache_warmer.parse_progress(lines) == [
        {"model_type": "flux2_klein_9b", "state": "cached", "files": 7}]


class _FakeS3:
    def __init__(self, objects):
        self.objects = objects

    def get_paginator(self, _name):
        objects = self.objects

        class _Pages:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [
                    {"Key": key, "LastModified": datetime.datetime(2026, 10, 12, tzinfo=datetime.timezone.utc)}
                    for key in objects if key.startswith(Prefix)]}
        return _Pages()

    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}


class _FakeStore:
    bucket = "bucket"

    def __init__(self, objects):
        self._client = _FakeS3(objects)

    def _key(self, *parts):
        return "/".join(("maestro", *parts))


def test_cached_models_come_from_manifests():
    store = _FakeStore({
        "maestro/manifests/minimax_h3.json": json.dumps(
            {"files": [{"root": "ckpts", "path": "a", "size": 10}, {"root": "hf", "path": "b", "size": 5}]}).encode(),
        "maestro/manifests/broken.json": b"{",
        "maestro/ckpts/a": b"x",
    })
    assert cache_warmer.list_cached_models(store) == [{
        "model_type": "minimax_h3", "files": 2, "bytes": 15, "hashed": False,
        "updated_at": datetime.datetime(2026, 10, 12, tzinfo=datetime.timezone.utc).timestamp(),
    }]


class _FakeClient:
    def __init__(self, run=None):
        self.run = run
        self.applied, self.fleets, self.secrets = [], [], {}

    def get_run(self, _name):
        return self.run

    def set_secret(self, name, value):
        self.secrets[name] = value

    def apply_fleet(self, spec):
        self.fleets.append(spec)

    def apply(self, spec):
        self.applied.append(spec)
        self.run = {"status": "submitted", "latest_job_submission": {}}

    def poll_logs(self, *_args, **_kwargs):
        return []


def test_start_refuses_without_r2_and_while_a_job_runs(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("projects:\n- name: main\n  backends:\n  - type: runpod\n    community_cloud: false\n")
    client = _FakeClient()
    warmer = cache_warmer.CacheWarmer(lambda: _settings(r2_bucket=""), lambda: client, config_path=str(config))
    with pytest.raises(cache_warmer.CacheWarmerError):
        warmer.start(["flux2_klein_9b"])

    warmer = cache_warmer.CacheWarmer(lambda: _settings(), lambda: client, config_path=str(config))
    status = warmer.start(["flux2_klein_9b"])
    assert status["state"] == "starting" and status["requested"] == ["flux2_klein_9b"]
    assert client.fleets[0]["configuration"]["resources"]["gpu"] == 0
    assert client.fleets[0]["configuration"]["backends"] == ["runpod"]
    with pytest.raises(cache_warmer.CacheWarmerError):
        warmer.start(["minimax_h3"])


def test_warm_pod_reads_its_own_r2_secret_names():
    from services.remote_worker.worker_spec import (
        SECRET_R2_ACCESS_KEY_ID, SECRET_R2_SECRET_ACCESS_KEY, build_run_spec, dstack_secrets,
    )
    settings = _settings()
    env = cache_warmer.build_warm_run_spec(settings, ["minimax_h3"])["configuration"]["env"]
    assert env["R2_ACCESS_KEY_ID"] == "${{ secrets.%s }}" % (SECRET_R2_ACCESS_KEY_ID + cache_warmer.SECRET_SUFFIX)
    assert env["R2_SECRET_ACCESS_KEY"] == "${{ secrets.%s }}" % (
        SECRET_R2_SECRET_ACCESS_KEY + cache_warmer.SECRET_SUFFIX)
    # A GPU worker starting at the same time writes the unsuffixed names only.
    worker = dstack_secrets(settings, {"access_key_id": "worker-id", "secret_access_key": "worker-secret"})
    warm = dstack_secrets(settings, {"access_key_id": "warm-id", "secret_access_key": "warm-secret"},
                          suffix=cache_warmer.SECRET_SUFFIX)
    assert not (set(worker) & set(warm)) - {"MAESTRO_R2_ACCOUNT_ID", "MAESTRO_REGISTRY_TOKEN"}
    assert warm[SECRET_R2_ACCESS_KEY_ID + cache_warmer.SECRET_SUFFIX] == "warm-id"
    assert "secrets.%s }}" % SECRET_R2_ACCESS_KEY_ID in json.dumps(build_run_spec(settings))
