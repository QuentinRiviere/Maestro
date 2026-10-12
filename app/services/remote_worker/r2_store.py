"""Cloudflare R2 (S3 API) store shared by the controller and the GPU worker.

Layout under ``<bucket>/<prefix>/``:

* ``assets/<sha256>/<name>``   reference inputs (images, audio, videos, LoRAs),
  content-addressed so a file is uploaded once whatever the number of pods.
* ``outputs/<job_id>/<name>``  generated media and their sidecars, kept in the
  bucket as a remote copy of the gallery.

The controller uploads inputs and downloads outputs; the worker downloads
inputs and uploads outputs. Neither goes through the dstack service proxy,
which is not meant for large transfers.
"""
from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class R2StoreError(RuntimeError):
    pass


def safe_name(name: str) -> str:
    cleaned = _UNSAFE.sub("_", os.path.basename(name or "")).strip("._")
    return cleaned or "file"


class R2Store:
    def __init__(
        self,
        *,
        account_id: str,
        access_key_id: str,
        secret_access_key: str,
        bucket: str,
        prefix: str = "maestro",
        client: Any = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is None:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "s3",
                endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                region_name="auto",
                config=Config(retries={"max_attempts": 5, "mode": "standard"}),
            )
        self._client = client

    # ── construction from the two sides' configuration ─────────────────────

    @classmethod
    def from_settings(cls, settings: Any) -> "R2Store | None":
        """Controller side: built from RemoteWorkerSettings (UI or env)."""
        if not getattr(settings, "r2_configured", False):
            return None
        return cls(
            account_id=settings.r2_account_id,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            bucket=settings.r2_bucket,
            prefix=settings.r2_prefix or "maestro",
        )

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "R2Store | None":
        """Worker side: built from the variables the run spec injects."""
        env = os.environ if environ is None else environ
        required = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "MAESTRO_R2_BUCKET")
        if not all(env.get(name) for name in required):
            return None
        return cls(
            account_id=env["R2_ACCOUNT_ID"],
            access_key_id=env["R2_ACCESS_KEY_ID"],
            secret_access_key=env["R2_SECRET_ACCESS_KEY"],
            bucket=env["MAESTRO_R2_BUCKET"],
            prefix=env.get("MAESTRO_R2_PREFIX", "maestro"),
        )

    # ── keys ───────────────────────────────────────────────────────────────

    def _key(self, *parts: str) -> str:
        return "/".join(p.strip("/") for p in (self.prefix, *parts) if p)

    def asset_key(self, sha256: str, name: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{64}", sha256 or ""):
            raise R2StoreError("asset keys need a sha256 hex digest")
        return self._key("assets", sha256, safe_name(name))

    def output_key(self, job_id: str, name: str) -> str:
        return self._key("outputs", safe_name(job_id), safe_name(name))

    def is_asset_key(self, key: str) -> bool:
        return key.startswith(self._key("assets") + "/") and ".." not in key.split("/")

    def is_output_key(self, key: str) -> bool:
        return key.startswith(self._key("outputs") + "/") and ".." not in key.split("/")

    # ── transfers ──────────────────────────────────────────────────────────

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self.bucket, Key=key)
            return True
        except Exception as exc:
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise R2StoreError(f"R2 head {key} failed: {exc}") from exc

    def upload(self, local_path: str, key: str, *, skip_if_exists: bool = True) -> str:
        if skip_if_exists and self.exists(key):
            return key
        try:
            self._client.upload_file(local_path, self.bucket, key)
        except Exception as exc:
            raise R2StoreError(f"R2 upload of {os.path.basename(local_path)} failed: {exc}") from exc
        return key

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            raise R2StoreError(f"R2 delete of {key} failed: {exc}") from exc

    def presigned_url(self, key: str, *, expires_seconds: int = 6 * 3600) -> str:
        """Time-limited GET URL; lets a third party read one object only."""
        try:
            return self._client.generate_presigned_url(
                "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=int(expires_seconds),
            )
        except Exception as exc:
            raise R2StoreError(f"R2 presign of {key} failed: {exc}") from exc

    def download(self, key: str, local_path: str) -> str:
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
        part = f"{local_path}.part"
        try:
            self._client.download_file(self.bucket, key, part)
            if os.path.getsize(part) == 0:
                raise R2StoreError(f"R2 object {key} is empty")
            os.replace(part, local_path)
        except R2StoreError:
            raise
        except Exception as exc:
            raise R2StoreError(f"R2 download of {key} failed: {exc}") from exc
        finally:
            if os.path.exists(part):
                os.remove(part)
        return local_path
