"""dstack run spec for a headless Maestro worker service."""
from __future__ import annotations

import shlex
from typing import Any

from services.remote_worker.settings import RemoteWorkerSettings


WORKER_PORT = 7860
SOURCE_DIR = "/workspace/maestro"
# Cheap endpoint that answers once launch.py has finished importing wgp.
HEALTH_PATH = "/api/v1/system-stats"


def build_run_spec(settings: RemoteWorkerSettings) -> dict[str, Any]:
    """Service spec: clone Maestro at ``repo_ref`` and run the boot script.

    Safety nets live in the spec itself so a crashed or closed Maestro can
    never leave a pod billing forever: ``max_duration`` and ``max_price`` are
    always set. The 30 min idle stop is enforced by Maestro, because dstack's
    ``idle_duration`` is not honoured on RunPod.
    """
    clone = (
        f"rm -rf {SOURCE_DIR} && "
        f"git clone --depth 1 --branch {shlex.quote(settings.repo_ref)} "
        f"{shlex.quote(settings.repo_url)} {SOURCE_DIR}"
    )
    configuration: dict[str, Any] = {
        "type": "service",
        "name": settings.run_name,
        "image": settings.image,
        "commands": [
            "export DEBIAN_FRONTEND=noninteractive",
            "command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git >/dev/null)",
            clone,
            f"bash {SOURCE_DIR}/app/scripts/remote_worker_boot.sh",
        ],
        "port": WORKER_PORT,
        "env": {
            "SERVER_PORT": str(WORKER_PORT),
            "MAESTRO_VOLUME": settings.volume_path,
            "MAESTRO_TORCH_INDEX": settings.torch_index,
        },
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
    if settings.volume_name:
        configuration["volumes"] = [{"name": settings.volume_name, "path": settings.volume_path}]
    if settings.fleet:
        configuration["fleets"] = [settings.fleet]
    return {"run_name": settings.run_name, "configuration": configuration}
