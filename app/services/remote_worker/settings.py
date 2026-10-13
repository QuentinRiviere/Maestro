"""Persisted remote worker settings (``app/settings/remote_worker.json``).

The dstack token is stored in full in the gitignored settings directory and
only ever returned masked by the API, like the other provider keys.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


DEFAULT_REPO_URL = "https://github.com/QuentinRiviere/Maestro.git"
DEFAULT_IMAGE = "ghcr.io/quentinriviere/maestro-worker:latest"

# Never returned in clear by the API; a masked value sent back is ignored.
SECRET_FIELDS = ("token", "registry_token", "r2_access_key_id", "r2_secret_access_key", "cf_api_token")


@dataclass
class RemoteWorkerSettings:
    # dstack access
    server_url: str = "http://127.0.0.1:3000"
    project: str = "main"
    token: str = ""
    # what to run
    run_name: str = "maestro-worker"
    repo_url: str = DEFAULT_REPO_URL
    repo_ref: str = "main"
    image: str = DEFAULT_IMAGE
    # Only needed while the GHCR package is private.
    registry_username: str = ""
    registry_token: str = ""
    default_model_type: str = ""
    # where and how much
    backends: list[str] = field(default_factory=lambda: ["runpod"])
    # dstack 0.20 provisions inside fleets only, so a run without its own
    # fleet inherits the price/GPU caps of whatever fleet the project has.
    # Maestro therefore keeps a dedicated fleet mirroring the limits below.
    fleet: str = "maestro"
    manage_fleet: bool = True
    gpu: str = "48GB.."
    disk: str = "200GB.."
    # System RAM and vCPUs: marketplace GPUs can ship with 8-16 GB of RAM,
    # too little to load the video and image checkpoints.
    memory: str = "64GB.."
    cpu: str = "8.."
    spot_policy: str = "on-demand"
    max_price: float = 0.8
    max_duration: str = "4h"
    # Cloudflare R2 checkpoint cache (optional)
    r2_account_id: str = ""
    r2_access_key_id: str = ""
    r2_secret_access_key: str = ""
    r2_bucket: str = ""
    r2_prefix: str = "maestro"
    # Cloudflare API token allowed to create and delete account API tokens.
    # When set, pods get their own bucket-scoped R2 key, revoked when they
    # stop, instead of the long-lived keys above (which stay on this machine).
    cf_api_token: str = ""
    # Pods may run on hosts Maestro cannot trust (marketplace or community
    # GPUs): they then get read-only access to the model cache and inputs,
    # write access to their own output folder only, and verify cache hashes.
    untrusted_hosts: bool = False
    # behaviour
    idle_ttl_minutes: int = 30
    # Stop as soon as nothing is queued or running instead of waiting the TTL.
    stop_when_queue_empty: bool = False
    route_generations: bool = False
    auto_start: bool = True

    @property
    def configured(self) -> bool:
        return bool(self.server_url and self.project and self.token)

    @property
    def r2_configured(self) -> bool:
        return bool(
            self.r2_account_id and self.r2_access_key_id
            and self.r2_secret_access_key and self.r2_bucket
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "RemoteWorkerSettings":
        settings = cls()
        settings.update(data or {})
        return settings

    def update(self, data: Mapping[str, Any]) -> list[str]:
        """Apply known keys with type coercion; return the keys changed."""
        changed: list[str] = []
        for item in fields(self):
            if item.name not in data:
                continue
            value = _coerce(item.name, data[item.name], getattr(self, item.name))
            if value != getattr(self, item.name):
                setattr(self, item.name, value)
                changed.append(item.name)
        return changed

    def public(self, env_locked: frozenset[str] = frozenset()) -> dict[str, Any]:
        payload = asdict(self)
        payload["env_locked"] = sorted(env_locked)
        for name in SECRET_FIELDS:
            payload[name] = _mask(getattr(self, name))
            payload[f"{name}_set"] = bool(getattr(self, name))
        payload["configured"] = self.configured
        payload["r2_configured"] = self.r2_configured
        return payload


ENV_PREFIX = "MAESTRO_REMOTE_"
_FALSE_STRINGS = {"", "0", "false", "no", "off"}


def _coerce(name: str, value: Any, current: Any) -> Any:
    if isinstance(current, bool):
        if isinstance(value, str):
            return value.strip().lower() not in _FALSE_STRINGS
        return bool(value)
    if isinstance(current, int):
        number = int(value)
        if name == "idle_ttl_minutes" and number < 1:
            raise ValueError("idle_ttl_minutes must be at least 1")
        return number
    if isinstance(current, float):
        number = float(value)
        if number <= 0:
            raise ValueError(f"{name} must be positive")
        return number
    if isinstance(current, list):
        if isinstance(value, str):
            value = [part.strip() for part in value.split(",")]
        return [str(part) for part in value if str(part).strip()]
    return str(value or "").strip()


def is_masked(value: object) -> bool:
    return isinstance(value, str) and ("…" in value or value == "••••")


def _mask(secret: str) -> str:
    if not secret:
        return ""
    return f"{secret[:4]}…{secret[-4:]}" if len(secret) > 12 else "••••"


def env_overrides(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Settings fixed by ``MAESTRO_REMOTE_<FIELD>`` variables (e.g. in Docker).

    They take precedence over stored values, are never written to the
    settings file, and are read-only in the UI.
    """
    env = os.environ if environ is None else environ
    names = {item.name for item in fields(RemoteWorkerSettings)}
    return {
        name: env[f"{ENV_PREFIX}{name.upper()}"]
        for name in names
        if f"{ENV_PREFIX}{name.upper()}" in env
    }


class RemoteWorkerSettingsStore:
    def __init__(self, settings_dir: str | os.PathLike[str]) -> None:
        self.path = Path(settings_dir) / "remote_worker.json"

    def load(self) -> RemoteWorkerSettings:
        try:
            with open(self.path, encoding="utf-8") as handle:
                return RemoteWorkerSettings.from_mapping(json.load(handle))
        except FileNotFoundError:
            return RemoteWorkerSettings()
        except (OSError, ValueError):
            return RemoteWorkerSettings()

    def save(self, settings: RemoteWorkerSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".remote_worker.", suffix=".tmp", dir=str(self.path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(asdict(settings), handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def read_local_dstack_config(path: str | os.PathLike[str] | None = None) -> dict[str, str] | None:
    """Default project from ``~/.dstack/config.yml`` (server URL, name, token)."""
    config_path = Path(path) if path else Path.home() / ".dstack" / "config.yml"
    try:
        import yaml  # PyYAML is installed as an omegaconf dependency.

        with open(config_path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
    except Exception:
        return None
    projects = data.get("projects") or []
    if not isinstance(projects, list) or not projects:
        return None
    chosen = next((p for p in projects if isinstance(p, dict) and p.get("default")), projects[0])
    if not isinstance(chosen, dict) or not chosen.get("token"):
        return None
    return {
        "server_url": str(chosen.get("url") or ""),
        "project": str(chosen.get("name") or ""),
        "token": str(chosen.get("token") or ""),
    }
