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


def _secret_ref(name: str) -> str:
    return "${{ secrets.%s }}" % name


def dstack_secrets(settings: RemoteWorkerSettings) -> dict[str, str]:
    """Secrets the run spec references, with their values."""
    secrets: dict[str, str] = {}
    if settings.r2_configured:
        secrets[SECRET_R2_ACCOUNT_ID] = settings.r2_account_id
        secrets[SECRET_R2_ACCESS_KEY_ID] = settings.r2_access_key_id
        secrets[SECRET_R2_SECRET_ACCESS_KEY] = settings.r2_secret_access_key
    if settings.registry_username and settings.registry_token:
        secrets[SECRET_REGISTRY_TOKEN] = settings.registry_token
    return secrets


def build_run_spec(settings: RemoteWorkerSettings) -> dict[str, Any]:
    """Service spec: clone Maestro at ``repo_ref`` and run the boot script.

    The image (docker/remote-worker) carries the Python environment, so a boot
    is: pull image, clone code, start Maestro. Safety nets live in the spec so
    a crashed or closed Maestro never leaves a pod billing forever:
    ``max_duration`` and ``max_price`` are always set. The idle stop is
    enforced by Maestro because dstack's ``idle_duration`` is not honoured on
    RunPod.
    """
    clone = (
        f"rm -rf {SOURCE_DIR} && "
        f"git clone --depth 1 --branch {shlex.quote(settings.repo_ref)} "
        f"{shlex.quote(settings.repo_url)} {SOURCE_DIR}"
    )
    env: dict[str, str] = {"SERVER_PORT": str(WORKER_PORT)}
    if settings.r2_configured:
        env.update({
            "R2_ACCOUNT_ID": _secret_ref(SECRET_R2_ACCOUNT_ID),
            "R2_ACCESS_KEY_ID": _secret_ref(SECRET_R2_ACCESS_KEY_ID),
            "R2_SECRET_ACCESS_KEY": _secret_ref(SECRET_R2_SECRET_ACCESS_KEY),
            "MAESTRO_R2_BUCKET": settings.r2_bucket,
            "MAESTRO_R2_PREFIX": settings.r2_prefix,
        })
    configuration: dict[str, Any] = {
        "type": "service",
        "name": settings.run_name,
        "image": settings.image,
        "commands": [clone, f"bash {SOURCE_DIR}/app/scripts/remote_worker_boot.sh"],
        "port": WORKER_PORT,
        "env": env,
        "resources": {"gpu": settings.gpu, "disk": settings.disk},
        "backends": list(settings.backends),
        "spot_policy": settings.spot_policy,
        "max_price": settings.max_price,
        "max_duration": settings.max_duration,
        "gateway": False,
        "auth": True,
        "replicas": 1,
        "probes": [{
            "type": "http",
            "url": HEALTH_PATH,
            "interval": 15,
            "ready_after": 2,
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
        "resources": {"gpu": settings.gpu, "disk": settings.disk},
        "backends": list(settings.backends),
        "spot_policy": settings.spot_policy,
        "max_price": settings.max_price,
        "idle_duration": FLEET_IDLE_SECONDS,
    }
    return {"configuration": configuration, "profile": {"name": "default"}, "autocreated": False}
