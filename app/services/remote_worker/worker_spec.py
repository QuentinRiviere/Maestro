"""dstack run spec for a headless Maestro worker service."""
from __future__ import annotations

import shlex
from typing import Any

from services.remote_worker.settings import RemoteWorkerSettings


WORKER_PORT = 7860
SOURCE_DIR = "/workspace/maestro"
# Cheap endpoint that answers once launch.py has finished importing wgp.
HEALTH_PATH = "/api/v1/system-stats"

# dstack project secrets pushed by the manager before each start. The run spec
# only references them, so credentials never appear in run specs or logs.
SECRET_R2_ACCOUNT_ID = "MAESTRO_R2_ACCOUNT_ID"
SECRET_R2_ACCESS_KEY_ID = "MAESTRO_R2_ACCESS_KEY_ID"
SECRET_R2_SECRET_ACCESS_KEY = "MAESTRO_R2_SECRET_ACCESS_KEY"
SECRET_REGISTRY_TOKEN = "MAESTRO_REGISTRY_TOKEN"
# Untrusted hosts: session token of the read-only set, and the write-only set.
SECRET_R2_SESSION_TOKEN = "MAESTRO_R2_SESSION_TOKEN"
SECRET_R2_WRITE_ACCESS_KEY_ID = "MAESTRO_R2_WRITE_ACCESS_KEY_ID"
SECRET_R2_WRITE_SECRET_ACCESS_KEY = "MAESTRO_R2_WRITE_SECRET_ACCESS_KEY"
SECRET_R2_WRITE_SESSION_TOKEN = "MAESTRO_R2_WRITE_SESSION_TOKEN"
_SCOPED_SECRETS = {
    "session_token": (SECRET_R2_SESSION_TOKEN, "R2_SESSION_TOKEN"),
    "write_access_key_id": (SECRET_R2_WRITE_ACCESS_KEY_ID, "R2_WRITE_ACCESS_KEY_ID"),
    "write_secret_access_key": (SECRET_R2_WRITE_SECRET_ACCESS_KEY, "R2_WRITE_SECRET_ACCESS_KEY"),
    "write_session_token": (SECRET_R2_WRITE_SESSION_TOKEN, "R2_WRITE_SESSION_TOKEN"),
}


def _secret_ref(name: str) -> str:
    return "${{ secrets.%s }}" % name


def dstack_secrets(
    settings: RemoteWorkerSettings, r2_keys: dict[str, str] | None = None,
) -> dict[str, str]:
    """Secrets the run spec references, with their values.

    ``r2_keys`` (a per-pod key from ephemeral_r2) replaces the long-lived R2
    keys, which then never leave this machine.
    """
    secrets: dict[str, str] = {}
    if settings.r2_configured:
        secrets[SECRET_R2_ACCOUNT_ID] = settings.r2_account_id
        secrets[SECRET_R2_ACCESS_KEY_ID] = (r2_keys or {}).get("access_key_id") or settings.r2_access_key_id
        secrets[SECRET_R2_SECRET_ACCESS_KEY] = (
            (r2_keys or {}).get("secret_access_key") or settings.r2_secret_access_key
        )
        for field, (secret, _env) in _SCOPED_SECRETS.items():
            if (r2_keys or {}).get(field):
                secrets[secret] = r2_keys[field]
    if settings.registry_username and settings.registry_token:
        secrets[SECRET_REGISTRY_TOKEN] = settings.registry_token
    return secrets


def worker_resources(settings: RemoteWorkerSettings) -> dict[str, Any]:
    """GPU, disk, and the RAM/CPU floor the worker needs to load checkpoints."""
    resources: dict[str, Any] = {"gpu": settings.gpu, "disk": settings.disk}
    if settings.memory:
        resources["memory"] = settings.memory
    if settings.cpu:
        resources["cpu"] = settings.cpu
    return resources


def pod_r2_env(settings: RemoteWorkerSettings, output_scope: str = "") -> dict[str, str]:
    """R2 variables of a pod: secret references plus untrusted-host switches."""
    env = {
        "R2_ACCOUNT_ID": _secret_ref(SECRET_R2_ACCOUNT_ID),
        "R2_ACCESS_KEY_ID": _secret_ref(SECRET_R2_ACCESS_KEY_ID),
        "R2_SECRET_ACCESS_KEY": _secret_ref(SECRET_R2_SECRET_ACCESS_KEY),
        "MAESTRO_R2_BUCKET": settings.r2_bucket,
        "MAESTRO_R2_PREFIX": settings.r2_prefix,
    }
    if settings.untrusted_hosts:
        env.update({variable: _secret_ref(secret) for secret, variable in _SCOPED_SECRETS.values()})
        env.update({
            "MAESTRO_R2_CACHE_READONLY": "1",
            "MAESTRO_R2_REQUIRE_HASHES": "1",
            "MAESTRO_R2_OUTPUT_SCOPE": output_scope,
        })
    return env


def _boot_event(phase: str, state: str) -> str:
    """Shell that prints a startup step the controller shows (see boot_events.py)."""
    return f"echo '[Boot] {{\"phase\": \"{phase}\", \"state\": \"{state}\"}}'"


def build_run_spec(settings: RemoteWorkerSettings, output_scope: str = "",
                   preload_models: list[str] | None = None) -> dict[str, Any]:
    """Service spec: clone Maestro at ``repo_ref`` and run the boot script.

    The image (docker/remote-worker) carries the Python environment, so a boot
    is: pull image, clone code, start Maestro. Safety nets live in the spec so
    a crashed or closed Maestro never leaves a pod billing forever:
    ``max_duration`` and ``max_price`` are always set. The idle stop is
    enforced by Maestro because dstack's ``idle_duration`` is not honoured on
    RunPod.
    """
    clone = (
        f"{_boot_event('clone', 'started')} && rm -rf {SOURCE_DIR} && "
        f"git clone --quiet --depth 1 --single-branch --branch {shlex.quote(settings.repo_ref)} "
        f"{shlex.quote(settings.repo_url)} {SOURCE_DIR} && {_boot_event('clone', 'done')}"
    )
    env: dict[str, str] = {"SERVER_PORT": str(WORKER_PORT)}
    if not settings.pinned_memory:
        env["MAESTRO_PINNED_MEMORY"] = "0"
    if settings.r2_configured:
        env.update(pod_r2_env(settings, output_scope))
        if preload_models:
            # Pulled from R2 by the boot script while Maestro starts.
            env["MAESTRO_PRELOAD_MODELS"] = ",".join(preload_models)
    configuration: dict[str, Any] = {
        "type": "service",
        "name": settings.run_name,
        "image": settings.image,
        "commands": [clone, f"bash {SOURCE_DIR}/app/scripts/remote_worker_boot.sh"],
        "port": WORKER_PORT,
        "env": env,
        "resources": worker_resources(settings),
        "backends": list(settings.backends),
        "spot_policy": settings.spot_policy,
        "max_price": settings.max_price,
        "max_duration": settings.max_duration,
        "gateway": False,
        "auth": True,
        "replicas": 1,
        # dstack only opens the proxy route once the probe passes: probe
        # often so the route follows Maestro's startup within seconds, and
        # stop probing once ready.
        "probes": [{
            "type": "http",
            "url": HEALTH_PATH,
            "interval": 2,
            "timeout": 5,
            "ready_after": 1,
            "until_ready": True,
        }],
    }
    if settings.registry_username and settings.registry_token:
        configuration["registry_auth"] = {
            "username": settings.registry_username,
            "password": _secret_ref(SECRET_REGISTRY_TOKEN),
        }
    if settings.fleet:
        configuration["fleets"] = [settings.fleet]
    return {"run_name": settings.run_name, "configuration": configuration}


# Seconds an instance stays in the fleet after the worker run ends. Maestro
# owns the idle TTL, so the instance should go away right after the run.
FLEET_IDLE_SECONDS = 60


def build_fleet_spec(settings: RemoteWorkerSettings) -> dict[str, Any]:
    """dstack fleet that bounds the worker: one node, Maestro's GPU and price."""
    configuration: dict[str, Any] = {
        "type": "fleet",
        "name": settings.fleet,
        "nodes": {"min": 0, "max": 1},
        "resources": worker_resources(settings),
        "backends": list(settings.backends),
        "spot_policy": settings.spot_policy,
        "max_price": settings.max_price,
        "idle_duration": FLEET_IDLE_SECONDS,
    }
    return {"configuration": configuration, "profile": {"name": "default"}, "autocreated": False}
