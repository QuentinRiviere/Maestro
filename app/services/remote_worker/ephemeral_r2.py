"""Per-pod R2 credentials: issued at start, revoked when the pod is gone.

Remote pods run on hardware Maestro does not control, so they never receive
the account's long-lived R2 keys when a Cloudflare API token is configured.
Instead, each run gets its own account-owned API token limited to object
read/write on the one bucket, expiring on its own shortly after the run's
``max_duration``. The token is deleted as soon as the run stops (user stop,
idle stop, finished task, failed run), and on every watchdog pass for any run
that is no longer active, so a key only lives as long as its pod.

R2 derives S3 credentials from an API token: the access key ID is the token
id and the secret access key is the SHA-256 hex digest of the token value.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("maestro.remote_worker.ephemeral_r2")

API = "https://api.cloudflare.com/client/v4"
PERMISSION_GROUP = "Workers R2 Storage Bucket Item Write"
TOKEN_PREFIX = "maestro-pod-"
# Extra life after max_duration, so a run never outlives its key.
EXPIRY_MARGIN_SECONDS = 15 * 60
DEFAULT_TTL_SECONDS = 4 * 3600
# A key is issued a moment before dstack registers its run; don't mistake
# that window (or a slow submit) for a finished run.
SUBMIT_GRACE_SECONDS = 300
MAX_TEMP_TTL_SECONDS = 7 * 86400
# What an untrusted pod may read: the model cache and content-addressed inputs.
UNTRUSTED_READ_PREFIXES = ("ckpts", "hf", "loras", "manifests", "assets")


class EphemeralKeyError(RuntimeError):
    pass


def duration_seconds(value: str | None, default: int = DEFAULT_TTL_SECONDS) -> int:
    """Seconds in a dstack duration such as ``4h``, ``90m`` or ``2h30m``."""
    total = 0
    for amount, unit in re.findall(r"(\d+)\s*([dhms])", str(value or "").lower()):
        total += int(amount) * {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]
    return total or default


def s3_credentials(token_id: str, token_value: str) -> dict[str, str]:
    return {
        "access_key_id": token_id,
        "secret_access_key": hashlib.sha256(token_value.encode()).hexdigest(),
    }


class EphemeralR2Keys:
    def __init__(
        self,
        state_path: str | Path,
        *,
        session: requests.Session | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._path = Path(state_path)
        self._session = session or requests.Session()
        self._clock = clock
        self._lock = threading.Lock()
        self._group_id: str | None = None

    @staticmethod
    def enabled(settings: Any) -> bool:
        return bool(getattr(settings, "cf_api_token", "") and getattr(settings, "r2_configured", False))

    # ── Cloudflare API ─────────────────────────────────────────────────────

    def _call(self, settings: Any, method: str, path: str, body: dict | None = None) -> Any:
        try:
            response = self._session.request(
                method, f"{API}/accounts/{settings.r2_account_id}/{path}",
                headers={"Authorization": f"Bearer {settings.cf_api_token}"},
                json=body, timeout=30,
            )
        except requests.RequestException as exc:
            raise EphemeralKeyError(f"Cloudflare API unreachable: {exc}") from exc
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not response.ok or not payload.get("success", False):
            errors = "; ".join(str(e.get("message")) for e in payload.get("errors") or []) or response.text[:200]
            raise EphemeralKeyError(f"Cloudflare API {method} {path.split('/')[0]} failed: {errors}")
        return payload.get("result")

    def _permission_group(self, settings: Any) -> str:
        if self._group_id is None:
            groups = self._call(settings, "GET", "tokens/permission_groups") or []
            match = next((g for g in groups if g.get("name") == PERMISSION_GROUP), None)
            if not match:
                raise EphemeralKeyError(f"Permission group {PERMISSION_GROUP!r} not available to this token")
            self._group_id = str(match["id"])
        return self._group_id

    # ── bookkeeping (token ids only, never secrets) ────────────────────────

    def _load(self) -> list[dict]:
        try:
            records = json.loads(self._path.read_text())
            return [r for r in records if isinstance(r, dict) and r.get("token_id")]
        except (OSError, ValueError):
            return []

    def _save(self, records: list[dict]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(records, indent=1))
        tmp.replace(self._path)

    def tracked(self) -> list[dict]:
        with self._lock:
            return self._load()

    # ── lifecycle ──────────────────────────────────────────────────────────

    def issue(self, settings: Any, run_name: str, ttl_seconds: int) -> dict[str, str]:
        """A fresh bucket-scoped key for ``run_name``; older ones are revoked first."""
        self.revoke(settings, run_name)
        expires = dt.datetime.fromtimestamp(
            self._clock() + ttl_seconds + EXPIRY_MARGIN_SECONDS, tz=dt.timezone.utc,
        ).replace(microsecond=0)
        resource = f"com.cloudflare.edge.r2.bucket.{settings.r2_account_id}_default_{settings.r2_bucket}"
        name = f"{TOKEN_PREFIX}{run_name}-{int(self._clock())}"
        result = self._call(settings, "POST", "tokens", {
            "name": name,
            "policies": [{
                "effect": "allow",
                "resources": {resource: "*"},
                "permission_groups": [{"id": self._permission_group(settings)}],
            }],
            "expires_on": expires.isoformat().replace("+00:00", "Z"),
        }) or {}
        token_id, value = str(result.get("id") or ""), str(result.get("value") or "")
        if not token_id or not value:
            raise EphemeralKeyError("Cloudflare returned no token id or value")
        with self._lock:
            records = self._load()
            records.append({"token_id": token_id, "run_name": run_name, "name": name,
                            "created_at": self._clock(), "expires_on": expires.isoformat()})
            self._save(records)
        log.info("issued R2 key %s for %s (expires %s)", token_id, run_name, expires.isoformat())
        return s3_credentials(token_id, value)

    def _temporary(self, settings: Any, parent: dict[str, str], permission: str,
                   prefixes: list[str], ttl_seconds: int) -> dict[str, str]:
        result = self._call(settings, "POST", "r2/temp-access-credentials", {
            "bucket": settings.r2_bucket,
            "parentAccessKeyId": parent["access_key_id"],
            "permission": permission,
            "ttlSeconds": int(min(MAX_TEMP_TTL_SECONDS, ttl_seconds + EXPIRY_MARGIN_SECONDS)),
            "prefixes": prefixes,
        }) or {}
        if not all(result.get(key) for key in ("accessKeyId", "secretAccessKey", "sessionToken")):
            raise EphemeralKeyError("Cloudflare returned incomplete temporary credentials")
        return {"access_key_id": result["accessKeyId"], "secret_access_key": result["secretAccessKey"],
                "session_token": result["sessionToken"]}

    def issue_untrusted(self, settings: Any, run_name: str, ttl_seconds: int, output_scope: str) -> dict[str, str]:
        """Prefix-scoped keys for a pod on an untrusted host.

        Read-only on the model cache and the input assets, read/write only on
        ``outputs/<output_scope>/``: the pod can neither read other projects'
        renders nor alter the cache other pods load. The per-pod parent key
        stays on this machine and is revoked with the run, which also voids
        both derived sets.
        """
        parent = self.issue(settings, run_name, ttl_seconds)
        base = f"{settings.r2_prefix.strip('/')}/" if settings.r2_prefix.strip("/") else ""
        read = self._temporary(settings, parent, "object-read-only",
                               [f"{base}{part}/" for part in UNTRUSTED_READ_PREFIXES], ttl_seconds)
        write = self._temporary(settings, parent, "object-read-write",
                                [f"{base}outputs/{output_scope}/"], ttl_seconds)
        return {**read, **{f"write_{key}": value for key, value in write.items()}}

    def revoke(self, settings: Any, run_name: str | None = None) -> int:
        """Delete the keys of ``run_name`` (all tracked keys when None)."""
        with self._lock:
            records = self._load()
            targets = [r for r in records if run_name is None or r.get("run_name") == run_name]
            kept = [r for r in records if r not in targets]
            revoked = 0
            for record in targets:
                try:
                    self._call(settings, "DELETE", f"tokens/{record['token_id']}")
                    revoked += 1
                except EphemeralKeyError as exc:
                    if "not found" in str(exc).lower() or "does not exist" in str(exc).lower():
                        continue  # already gone or expired
                    # Keep it tracked so the next pass retries; it expires anyway.
                    log.warning("could not revoke R2 key %s: %s", record["token_id"], exc)
                    kept.append(record)
            self._save(kept)
        if revoked:
            log.info("revoked %d R2 key(s)%s", revoked, f" of {run_name}" if run_name else "")
        return revoked

    def reconcile(self, settings: Any, is_active: Callable[[str], bool]) -> int:
        """Revoke keys whose run is no longer active."""
        if not self.enabled(settings):
            return 0
        revoked = 0
        records = self.tracked()
        for run_name in sorted({r.get("run_name") for r in records}):
            newest = max(float(r.get("created_at") or 0) for r in records if r.get("run_name") == run_name)
            if self._clock() - newest < SUBMIT_GRACE_SECONDS:
                continue  # issued just before its run was submitted
            try:
                active = is_active(run_name)
            except Exception as exc:  # unknown state: keep the key, it expires anyway
                log.warning("R2 key reconcile skipped for %s: %s", run_name, exc)
                continue
            if not active:
                revoked += self.revoke(settings, run_name)
        return revoked
