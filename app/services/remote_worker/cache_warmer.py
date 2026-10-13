"""Fill the R2 checkpoint cache from a cheap CPU pod instead of the GPU worker.

Downloading a 20-45 GB checkpoint and copying it to R2 needs bandwidth and
disk, not a GPU. The warmer runs the remote worker image as a dstack *task* on
a CPU instance: it clones Maestro, resolves each requested model with Maestro's
own download logic (so the manifest paths are exactly the ones the GPU worker
hydrates), publishes the files to R2 and exits, which releases the pod.

The warmer reuses the GPU worker's dstack connection, repository and R2
credentials. It runs the worker image's ``cpu`` tag (docker/cache-warmer), a
few GB instead of the ~20 GB CUDA image a CPU pod would spend a quarter of an
hour pulling. Since it writes the cache every later pod loads, it only runs on
backends whose dstack config excludes community hosts, with a bucket-wide key
even when GPU workers use untrusted hosts.
"""
from __future__ import annotations

import json
import logging
import re
import shlex
import threading
from collections.abc import Callable, Mapping
from typing import Any

from services.remote_worker.dstack_client import DstackClient, DstackError
from services.remote_worker.settings import RemoteWorkerSettings
from services.remote_worker.vast_network import DSTACK_SERVER_CONFIG, dstack_backend_config
from services.remote_worker.worker_spec import (
    SECRET_R2_ACCESS_KEY_ID,
    SECRET_R2_ACCOUNT_ID,
    SECRET_R2_SECRET_ACCESS_KEY,
    SECRET_REGISTRY_TOKEN,
    SOURCE_DIR,
    dstack_secrets,
)

log = logging.getLogger("maestro.remote_worker.cache_warmer")

RUN_NAME = "maestro-cache-warmer"
FLEET_NAME = "maestro-cache"
# Enough disk for the largest checkpoints Maestro ships (H3, LTX: ~50 GB each).
# 8 GB of RAM gets the download killed on a 20 GB single-file checkpoint (Qwen-Image-Edit 2511).
CPU_RESOURCES = {"cpu": "4..", "memory": "32GB..", "disk": "120GB..", "gpu": 0}
CPU_IMAGE_TAG = "cpu"
# Backends with a ``community_cloud`` switch; they rent third-party hosts
# unless it is explicitly off in the dstack server config.
COMMUNITY_BACKENDS = {"runpod", "vastai"}
MAX_PRICE_PER_HOUR = 0.5
MAX_DURATION = "2h"
FLEET_IDLE_SECONDS = 60
# Progress lines the worker prints (see launch.py ``--cache-warm``).
LOG_MARKER = "[CacheWarm]"

_MODEL_TYPE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")
_FINISHED = {"terminated", "failed", "done"}


class CacheWarmerError(RuntimeError):
    pass


def _secret_ref(name: str) -> str:
    return "${{ secrets.%s }}" % name


def valid_model_types(model_types: list[str]) -> list[str]:
    cleaned: list[str] = []
    for model_type in model_types:
        model_type = str(model_type or "").strip()
        if not _MODEL_TYPE.match(model_type):
            raise CacheWarmerError(f"Invalid model type: {model_type!r}")
        if model_type not in cleaned:
            cleaned.append(model_type)
    if not cleaned:
        raise CacheWarmerError("Choose at least one model to cache")
    return cleaned


def warm_image(image: str) -> str:
    """The ``cpu`` tag of the worker image's repository."""
    repository = image.split("@", 1)[0]
    if ":" in repository.rsplit("/", 1)[-1]:
        repository = repository.rsplit(":", 1)[0]
    return f"{repository}:{CPU_IMAGE_TAG}"


def trusted_backends(settings: RemoteWorkerSettings, config_path: str = DSTACK_SERVER_CONFIG) -> list[str]:
    """The worker's backends that only rent datacenter hosts."""
    trusted = []
    for backend in settings.backends:
        if backend in COMMUNITY_BACKENDS:
            config = dstack_backend_config(backend, config_path)
            if not config or config.get("community_cloud") is not False:
                continue
        trusted.append(backend)
    return trusted


def build_warm_run_spec(settings: RemoteWorkerSettings, model_types: list[str],
                        backends: list[str] | None = None) -> dict[str, Any]:
    """Task spec: clone Maestro, cache the models in R2, exit."""
    clone = (
        f"rm -rf {SOURCE_DIR} && "
        f"git clone --depth 1 --branch {shlex.quote(settings.repo_ref)} "
        f"{shlex.quote(settings.repo_url)} {SOURCE_DIR}"
    )
    warm = (
        f"bash {SOURCE_DIR}/app/scripts/remote_worker_boot.sh "
        f"--cache-warm {shlex.quote(','.join(model_types))}"
    )
    configuration: dict[str, Any] = {
        "type": "task",
        "name": RUN_NAME,
        "image": warm_image(settings.image),
        "commands": [clone, warm],
        "env": {
            "R2_ACCOUNT_ID": _secret_ref(SECRET_R2_ACCOUNT_ID),
            "R2_ACCESS_KEY_ID": _secret_ref(SECRET_R2_ACCESS_KEY_ID),
            "R2_SECRET_ACCESS_KEY": _secret_ref(SECRET_R2_SECRET_ACCESS_KEY),
            "MAESTRO_R2_BUCKET": settings.r2_bucket,
            "MAESTRO_R2_PREFIX": settings.r2_prefix,
        },
        "resources": dict(CPU_RESOURCES),
        "backends": list(backends if backends is not None else settings.backends),
        "spot_policy": "on-demand",
        "max_price": MAX_PRICE_PER_HOUR,
        "max_duration": MAX_DURATION,
        "fleets": [FLEET_NAME],
    }
    if settings.registry_username and settings.registry_token:
        configuration["registry_auth"] = {
            "username": settings.registry_username,
            "password": _secret_ref(SECRET_REGISTRY_TOKEN),
        }
    return {"run_name": RUN_NAME, "configuration": configuration}


def build_warm_fleet_spec(settings: RemoteWorkerSettings, backends: list[str] | None = None) -> dict[str, Any]:
    """CPU-only fleet so the warmer can never land on (and pay for) a GPU."""
    configuration: dict[str, Any] = {
        "type": "fleet",
        "name": FLEET_NAME,
        "nodes": {"min": 0, "max": 1},
        "resources": dict(CPU_RESOURCES),
        "backends": list(backends if backends is not None else settings.backends),
        "spot_policy": "on-demand",
        "max_price": MAX_PRICE_PER_HOUR,
        "idle_duration": FLEET_IDLE_SECONDS,
    }
    return {"configuration": configuration, "profile": {"name": "default"}, "autocreated": False}


def parse_progress(lines: list[str]) -> list[dict[str, Any]]:
    """Per-model state from the worker's ``[CacheWarm] {json}`` lines."""
    models: dict[str, dict[str, Any]] = {}
    for line in lines:
        index = line.find(LOG_MARKER)
        if index < 0:
            continue
        try:
            event = json.loads(line[index + len(LOG_MARKER):].strip())
        except ValueError:
            continue
        model_type = event.get("model_type")
        if isinstance(model_type, str) and model_type:
            models.setdefault(model_type, {"model_type": model_type}).update(event)
    return list(models.values())


class CacheWarmer:
    """Start, follow and stop the single cache-warming task."""

    def __init__(
        self,
        settings: Callable[[], RemoteWorkerSettings],
        client: Callable[[], DstackClient],
        pod_r2_keys: Callable[..., dict[str, str] | None] = lambda *_args, **_kwargs: None,
        config_path: str = DSTACK_SERVER_CONFIG,
    ) -> None:
        self._config_path = config_path
        self._settings = settings
        self._client = client
        # Per-pod key issuer (RemoteWorkerManager.pod_r2_keys); revocation is
        # the manager watchdog's reconcile once this task's run has ended.
        self._pod_r2_keys = pod_r2_keys
        self._lock = threading.Lock()
        self._model_types: list[str] = []
        self._log_lines: list[str] = []
        self._log_cursor: str | None = None
        self._submission: str | None = None

    def start(self, model_types: list[str]) -> dict[str, Any]:
        settings = self._settings()
        if not settings.r2_configured:
            raise CacheWarmerError("Configure the R2 bucket in Remote GPU settings first")
        model_types = valid_model_types(model_types)
        # Filling the cache means writing what every later pod loads: only
        # datacenter hosts, whatever the GPU worker is allowed to use.
        backends = trusted_backends(settings, self._config_path)
        if not backends:
            raise CacheWarmerError("Cache warming writes the shared model cache and needs a backend without "
                                   "community hosts: set community_cloud: false for it in the dstack server config")
        client = self._client()
        with self._lock:
            run = client.get_run(RUN_NAME)
            if run and str(run.get("status") or "") not in _FINISHED and not run.get("deleted"):
                raise CacheWarmerError("A cache job is already running")
            keys = self._pod_r2_keys(settings, RUN_NAME, MAX_DURATION, trusted=True)
            for name, value in dstack_secrets(settings, keys).items():
                client.set_secret(name, value)
            client.apply_fleet(build_warm_fleet_spec(settings, backends))
            client.apply(build_warm_run_spec(settings, model_types, backends))
            self._model_types = model_types
            self._log_lines = []
            self._log_cursor = None
            self._submission = None
            log.info("cache warmer submitted for %s on %s", ", ".join(model_types), ", ".join(backends))
        return self.status()

    def stop(self) -> dict[str, Any]:
        client = self._client()
        run = client.get_run(RUN_NAME)
        if run and str(run.get("status") or "") not in _FINISHED:
            client.stop_run(RUN_NAME, abort=False)
        return self.status()

    def status(self) -> dict[str, Any]:
        try:
            client = self._client()
            run = client.get_run(RUN_NAME)
        except (DstackError, RuntimeError) as exc:
            return {"state": "unknown", "message": str(exc), "models": []}
        if not run or run.get("deleted"):
            return {"state": "idle", "models": []}
        run_status = str(run.get("status") or "")
        submission = run.get("latest_job_submission") or {}
        provisioning = submission.get("job_provisioning_data") or {}
        self._follow_logs(client, str(submission.get("id") or ""))
        state = (
            "done" if run_status == "done"
            else "failed" if run_status == "failed"
            else "stopped" if run_status == "terminated"
            else "running" if run_status == "running"
            else "starting"
        )
        message = run.get("status_message") or submission.get("status_message") or ""
        if run_status == "failed":
            message = submission.get("termination_reason_message") or submission.get("termination_reason") or message
        return {
            "state": state,
            "message": str(message or ""),
            "requested": list(self._model_types),
            "models": parse_progress(self._log_lines),
            "price_per_hour": provisioning.get("price"),
            "cost": run.get("cost"),
            "submitted_at": run.get("submitted_at"),
        }

    def _follow_logs(self, client: DstackClient, submission_id: str) -> None:
        if not submission_id:
            return
        with self._lock:
            if submission_id != self._submission:
                self._submission = submission_id
                self._log_lines = []
                self._log_cursor = None
            try:
                lines = client.poll_logs(RUN_NAME, submission_id, start_time=self._log_cursor, limit=500)
            except DstackError:
                return
            for line in lines:
                if LOG_MARKER in line.get("message", ""):
                    self._log_lines.extend(line["message"].splitlines())
            if lines:
                self._log_cursor = lines[-1].get("timestamp") or self._log_cursor


def list_cached_models(store: Any) -> list[dict[str, Any]]:
    """Models whose files are in the R2 cache, from their manifests."""
    client, bucket = store._client, store.bucket
    prefix = store._key("manifests") + "/"
    cached: list[dict[str, Any]] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            key = obj["Key"]
            if not key.endswith(".json"):
                continue
            try:
                payload = json.loads(client.get_object(Bucket=bucket, Key=key)["Body"].read())
            except (ValueError, KeyError):
                continue
            files = (payload.get("files") or []) if isinstance(payload, Mapping) else []
            cached.append({
                "model_type": key[len(prefix):-len(".json")],
                "files": len(files),
                "bytes": sum(int(f.get("size") or 0) for f in files if isinstance(f, Mapping)),
                # Untrusted pods only use a cache whose files all carry a hash.
                "hashed": bool(files) and all(isinstance(f, Mapping) and f.get("sha256") for f in files),
                "updated_at": obj["LastModified"].timestamp(),
            })
    return sorted(cached, key=lambda item: item["model_type"])
