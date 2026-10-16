"""Compiled GPU kernels kept in R2 between worker pods.

Triton (Quanto's int8 kernels, SageAttention's Triton path) and Inductor
(``torch.compile``) compile kernels for the card on first use and keep them
in a local cache that dies with the pod. The boot script points both caches
under ``$MAESTRO_DATA/kernel-cache`` and pulls the archive matching the card
and the installed versions; the worker republishes the archive once it has
grown and stopped changing.

A kernel compiled for one GPU architecture, torch or Triton version is of no
use to another, so each combination has its own archive. Hosts that only
read the R2 cache (untrusted hosts) never publish: the archive holds code
other pods execute.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
import tarfile
import tempfile
import threading
import time

from services.remote_worker.checkpoint_cache import R2CheckpointCache

# Seconds between two looks at the local cache, and an upload is only made
# once the cache has stopped growing for one full interval.
PUBLISH_INTERVAL_SECONDS = 120
# A pull or a publish of a few hundred MB never holds anything up for long.
TRANSFER_TIMEOUT_SECONDS = 300


def cache_root() -> str:
    return os.path.join(os.environ.get("MAESTRO_DATA", "/workspace/data"), "kernel-cache")


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "none"


def _compute_capability() -> str:
    """``sm_89`` for the first GPU, read from nvidia-smi (no torch import)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""
    first = (out.strip().splitlines() or [""])[0].strip()
    return "sm_" + first.replace(".", "") if first else ""


def _host_arch_suffix(machine: str | None = None) -> str:
    """``-aarch64`` on non-x86 hosts, "" on x86_64 (keeps the existing keys).

    Triton caches host-native launcher stubs next to the GPU cubins, so an
    archive built on an arm64 host (DGX Spark) must never reach an amd64 pod.
    """
    machine = (platform.machine() if machine is None else machine).lower()
    return "" if machine in ("x86_64", "amd64", "") else f"-{machine}"


def cache_key(capability: str | None = None, machine: str | None = None) -> str:
    """The archive name for this card and environment ("" without a GPU)."""
    capability = _compute_capability() if capability is None else capability
    if not capability:
        return ""
    python = f"py{sys.version_info.major}{sys.version_info.minor}"
    return (f"{capability}-torch{_version('torch')}-triton{_version('triton')}-{python}"
            f"{_host_arch_suffix(machine)}")


def _file_count(root: str) -> int:
    return sum(len(names) for _directory, _dirs, names in os.walk(root)) if os.path.isdir(root) else 0


class KernelCache:
    def __init__(self, cache: R2CheckpointCache, key: str, root: str | None = None) -> None:
        self._cache = cache
        self._key = key
        self._root = root or cache_root()
        self._published = 0

    @property
    def readonly(self) -> bool:
        return self._cache.readonly

    @property
    def key(self) -> str:
        return self._key

    @property
    def url(self) -> str:
        return self._cache.object_url("kernels", f"{self._key}.tar")

    def pull(self) -> int:
        """Unpack the archive of this card; return how many files it brought."""
        os.makedirs(self._root, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "kernels.tar")
            if not self._cache.copy(self.url, local, timeout=TRANSFER_TIMEOUT_SECONDS):
                self._published = _file_count(self._root)
                return 0
            before = _file_count(self._root)
            with tarfile.open(local) as archive:
                members = [m for m in archive.getmembers()
                           if (m.isfile() or m.isdir()) and not os.path.isabs(m.name) and ".." not in m.name.split("/")
                           and not os.path.exists(os.path.join(self._root, m.name))]
                try:
                    archive.extractall(self._root, members=members, filter="data")
                except TypeError:  # Python without tarfile extraction filters
                    archive.extractall(self._root, members=members)
        self._published = _file_count(self._root)
        return self._published - before

    def publish_if_grown(self) -> bool:
        """Upload the local cache when it holds more files than the archive."""
        if self._cache.readonly:
            return False
        count = _file_count(self._root)
        if count <= self._published:
            return False
        with tempfile.TemporaryDirectory() as tmp:
            local = os.path.join(tmp, "kernels.tar")
            with tarfile.open(local, "w") as archive:
                for name in sorted(os.listdir(self._root)):
                    archive.add(os.path.join(self._root, name), arcname=name)
            if not self._cache.copy(local, self.url, timeout=TRANSFER_TIMEOUT_SECONDS):
                return False
        self._published = count
        return True

    def watch(self, interval: float = PUBLISH_INTERVAL_SECONDS) -> threading.Thread:
        """Publish in the background each time the cache grows, then settles."""
        def loop() -> None:
            previous = _file_count(self._root)
            while True:
                time.sleep(interval)
                count = _file_count(self._root)
                if count != previous:
                    print(f"[Worker] kernel cache: {count} files ({self._key})", flush=True)
                if count == previous and count > self._published:
                    try:
                        if self.publish_if_grown():
                            print(f"[Worker] kernel cache published ({count} files, {self._key})", flush=True)
                    except Exception as exc:  # the cache only saves time
                        print(f"[Worker] kernel cache publish failed: {exc}", flush=True)
                previous = count
        thread = threading.Thread(target=loop, daemon=True, name="kernel-cache-publish")
        thread.start()
        return thread


def from_environment(roots: dict[str, str]) -> KernelCache | None:
    cache = R2CheckpointCache.from_environment(roots)
    key = cache_key() if cache is not None else ""
    return KernelCache(cache, key) if cache is not None and key else None


def main() -> int:
    """Boot: pull this card's kernels, then publish them as they grow.

    The boot script runs ``python -m services.remote_worker.kernel_cache`` in
    the background next to Maestro; the process lives as long as the pod.
    """
    from services.remote_worker.prefetch import boot_event, worker_roots

    kernels = from_environment(worker_roots())
    if kernels is None:
        boot_event("kernels", "skipped", reason="no R2 cache or no GPU")
        return 0
    started = time.time()
    try:
        pulled = kernels.pull()
    except Exception as exc:
        boot_event("kernels", "failed", error=str(exc)[:300])
    else:
        boot_event("kernels", "done", key=kernels.key, files=pulled, seconds=round(time.time() - started, 1))
    if kernels.readonly:
        return 0
    kernels.watch().join()
    return 0


if __name__ == "__main__":
    sys.exit(main())
