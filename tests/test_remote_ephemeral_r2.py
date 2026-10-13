"""Per-pod R2 keys: issued at start, scoped to one bucket, revoked at stop."""
from __future__ import annotations

import hashlib
import os
import sys

import pytest

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.remote_worker import ephemeral_r2  # noqa: E402
from services.remote_worker.settings import RemoteWorkerSettings  # noqa: E402
from services.remote_worker.worker_spec import SECRET_R2_SECRET_ACCESS_KEY, dstack_secrets  # noqa: E402


class _Response:
    def __init__(self, payload, status=200):
        self._payload, self.status_code, self.ok, self.text = payload, status, status < 400, str(payload)

    def json(self):
        return self._payload


class FakeCloudflare:
    def __init__(self):
        self.tokens, self.calls, self.fail_delete, self.created = {}, [], False, 0

    def request(self, method, url, headers=None, json=None, timeout=None):
        self.calls.append((method, url.split("/client/v4/")[1], json))
        assert headers["Authorization"] == "Bearer cf-admin"
        if url.endswith("tokens/permission_groups"):
            return _Response({"success": True, "result": [
                {"id": "pg-read", "name": "Workers R2 Storage Bucket Item Read"},
                {"id": "pg-write", "name": ephemeral_r2.PERMISSION_GROUP}]})
        if method == "POST":
            self.created += 1
            token_id = f"tok{self.created}"
            self.tokens[token_id] = json
            return _Response({"success": True, "result": {"id": token_id, "value": f"value-{token_id}"}})
        if method == "DELETE":
            if self.fail_delete:
                return _Response({"success": False, "errors": [{"message": "upstream error"}]}, 500)
            token_id = url.rsplit("/", 1)[1]
            if self.tokens.pop(token_id, None) is None:
                return _Response({"success": False, "errors": [{"message": "Token not found"}]}, 404)
            return _Response({"success": True, "result": {"id": token_id}})
        raise AssertionError(url)


def _settings(**overrides):
    values = dict(r2_account_id="acc", r2_access_key_id="long-kid", r2_secret_access_key="long-secret",
                  r2_bucket="videogen", cf_api_token="cf-admin")
    values.update(overrides)
    return RemoteWorkerSettings.from_mapping(values)


@pytest.fixture
def keys(tmp_path):
    clock = [1_800_000_000.0]
    cloudflare = FakeCloudflare()
    issuer = ephemeral_r2.EphemeralR2Keys(tmp_path / "keys.json", session=cloudflare, clock=lambda: clock[0])
    return issuer, cloudflare, clock


def test_issued_key_is_bucket_scoped_expiring_and_never_the_long_lived_one(keys):
    issuer, cloudflare, _ = keys
    creds = issuer.issue(_settings(), "maestro-worker", ephemeral_r2.duration_seconds("4h"))
    body = cloudflare.tokens["tok1"]
    policy = body["policies"][0]
    assert policy["resources"] == {"com.cloudflare.edge.r2.bucket.acc_default_videogen": "*"}
    assert policy["permission_groups"] == [{"id": "pg-write"}]
    assert body["expires_on"] == "2027-01-15T12:15:00Z"  # start + 4 h + 15 min margin
    assert creds == {"access_key_id": "tok1", "secret_access_key": hashlib.sha256(b"value-tok1").hexdigest()}
    secrets = dstack_secrets(_settings(), creds)
    assert secrets[SECRET_R2_SECRET_ACCESS_KEY] == creds["secret_access_key"]
    assert "long-secret" not in secrets.values()
    assert "value-tok1" not in (issuer._path.read_text())  # only ids are stored


def test_stop_revokes_and_reissue_replaces_the_previous_key(keys):
    issuer, cloudflare, _ = keys
    issuer.issue(_settings(), "maestro-worker", 3600)
    issuer.issue(_settings(), "maestro-worker", 3600)
    assert list(cloudflare.tokens) == ["tok2"]
    assert issuer.revoke(_settings(), "maestro-worker") == 1
    assert cloudflare.tokens == {} and issuer.tracked() == []


def test_reconcile_revokes_finished_runs_after_the_submit_grace(keys):
    issuer, cloudflare, clock = keys
    issuer.issue(_settings(), "maestro-cache-warmer", 7200)
    assert issuer.reconcile(_settings(), lambda run: False) == 0  # just issued
    clock[0] += ephemeral_r2.SUBMIT_GRACE_SECONDS + 1
    assert issuer.reconcile(_settings(), lambda run: True) == 0
    assert issuer.reconcile(_settings(), lambda run: False) == 1
    assert cloudflare.tokens == {}


def test_failed_revoke_stays_tracked_and_missing_tokens_are_forgotten(keys):
    issuer, cloudflare, _ = keys
    issuer.issue(_settings(), "maestro-worker", 3600)
    cloudflare.fail_delete = True
    assert issuer.revoke(_settings(), "maestro-worker") == 0
    assert len(issuer.tracked()) == 1
    cloudflare.fail_delete = False
    cloudflare.tokens.clear()  # expired on its own
    assert issuer.revoke(_settings(), "maestro-worker") == 0
    assert issuer.tracked() == []


def test_disabled_without_a_cloudflare_token():
    assert not ephemeral_r2.EphemeralR2Keys.enabled(_settings(cf_api_token=""))
    assert dstack_secrets(_settings(cf_api_token=""))[SECRET_R2_SECRET_ACCESS_KEY] == "long-secret"
    assert ephemeral_r2.duration_seconds("2h30m") == 9000
    assert ephemeral_r2.duration_seconds("") == ephemeral_r2.DEFAULT_TTL_SECONDS
