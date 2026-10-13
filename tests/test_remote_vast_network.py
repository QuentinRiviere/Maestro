"""Slow Vast.ai hosts are excluded before the worker run is submitted."""
from __future__ import annotations

import os
import sys

import pytest

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.remote_worker import vast_network  # noqa: E402
from services.remote_worker.settings import RemoteWorkerSettings  # noqa: E402
from services.remote_worker.worker_spec import build_run_spec  # noqa: E402

OFFERS = [
    {"backend": "vastai", "instance": {"name": "111"}},
    {"backend": "runpod", "instance": {"name": "NVIDIA A100-SXM4-80GB"}},
    {"backend": "vastai", "instance": {"name": "222"}},
    {"backend": "runpod", "instance": {"name": "NVIDIA A100-SXM4-80GB"}},
]


def _settings(**overrides):
    values = dict(backends=["runpod", "vastai"], min_download_mbps=800, vast_api_key="vk")
    values.update(overrides)
    return RemoteWorkerSettings.from_mapping(values)


def test_slow_vast_hosts_are_dropped_and_other_backends_kept():
    allowed, summary = vast_network.allowed_instance_types(OFFERS, 800, lambda ids: {111: 150.0, 222: 2400.0})
    assert allowed == ["NVIDIA A100-SXM4-80GB", "222"]
    assert summary == {"vast_offers": 2, "vast_fast": 1, "min_mbps": 800, "other_offers": 1}


def test_run_spec_is_pinned_from_a_fleetless_plan(monkeypatch):
    seen = {}
    monkeypatch.setattr(vast_network, "download_speeds", lambda ids, key, session=None: {111: 900.0, 222: 10.0})
    spec = build_run_spec(_settings(fleet="maestro"))

    def plan(probe):
        backend = probe["configuration"]["backends"]
        seen.setdefault("backends", []).append(backend)
        assert "fleets" not in probe["configuration"]
        return {"job_plans": [{"offers": [o for o in OFFERS if [o["backend"]] == backend]}]}

    vast_network.pin_fast_offers(spec, plan, _settings())
    assert seen["backends"] == [["runpod"], ["vastai"]]
    assert spec["configuration"]["instance_types"] == ["NVIDIA A100-SXM4-80GB", "111"]
    assert spec["configuration"]["fleets"] == ["maestro"]


def test_filter_is_skipped_or_refuses_cleanly(monkeypatch):
    spec = build_run_spec(_settings(backends=["runpod"]))
    assert vast_network.pin_fast_offers(spec, lambda s: {}, _settings(backends=["runpod"])) == {"skipped": True}
    assert vast_network.pin_fast_offers(build_run_spec(_settings()), lambda s: {},
                                        _settings(min_download_mbps=0)) == {"skipped": True}
    monkeypatch.setattr(vast_network, "download_speeds", lambda ids, key, session=None: {})
    with pytest.raises(vast_network.VastFilterError, match="No offer left"):
        vast_network.pin_fast_offers(build_run_spec(_settings()),
                                     lambda s: {"job_plans": [{"offers": [OFFERS[0]]}]}, _settings())


def test_vast_key_is_read_from_the_dstack_config(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("projects:\n- name: main\n  backends:\n  - type: vastai\n    creds:\n      type: api_key\n"
                      "      api_key: from-dstack\n")
    assert vast_network.vast_api_key(_settings(vast_api_key=""), str(config)) == "from-dstack"
    assert vast_network.vast_api_key(_settings(), str(config)) == "vk"
