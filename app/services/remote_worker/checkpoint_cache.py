"""Write-through checkpoint cache in Cloudflare R2, used on the remote worker.

Maestro's own download logic stays the source of truth for *which* files a
model needs. The cache only remembers, per ``model_type``, the files that
logic ended up putting on disk, so the next pod can pull them from R2 instead
of Hugging Face / CivitAI:

1. ``hydrate``: read ``<prefix>/manifests/<model_type>.json`` and pull the
   listed files into the local roots (``ckpts``, ``hf`` cache, ...).
2. Maestro downloads whatever is still missing (nothing, on a warm cache).
3. ``publish``: upload files that appeared during step 2 and rewrite the
   manifest as the union of the previous one and the new files.

Symlinks (the Hugging Face cache links snapshots to blobs) are recorded in the
manifest and recreated, never uploaded as file copies. ``s5cmd run`` can
report refusals on stderr while exiting 0, so every pulled file is checked
for presence and size afterwards.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


log = logging.getLogger("maestro.remote_worker.cache")

_SKIP_SUFFIXES = (".part", ".lock", ".incomplete", ".tmp")
# huggingface_hub's local_dir bookkeeping: rewritten by every later download
# into the same folder (so its cached size goes stale), and never needed since
# Maestro only checks that the weights themselves are on disk.
_SKIP_DIR = ".cache"


def _cached_path(path: str) -> bool:
    """Whether a root-relative path belongs in the cache."""
    return not path.endswith(_SKIP_SUFFIXES) and _SKIP_DIR not in path.split("/")
Runner = Callable[..., subprocess.CompletedProcess]
# A publish that drops below this rate is abandoned: the cache only saves
# time, and a stalled upload must never hold a generation hostage.
_MIN_UPLOAD_BYTES_PER_S = 15_000_000
_MIN_UPLOAD_TIMEOUT_S = 300


class CheckpointCacheError(RuntimeError):
    pass


@dataclass(frozen=True)
class FileEntry:
    root: str
    path: str
    size: int
    # SHA-256 written by a trusted publisher; pods on untrusted hosts can
    # only read the cache, so a tampered file cannot match it.
    sha256: str = ""


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class LinkEntry:
    root: str
    path: str
    target: str


Snapshot = dict[tuple[str, str], int]


class R2CheckpointCache:
    def __init__(
        self,
        *,
        bucket: str,
        prefix: str,
        endpoint_url: str,
        roots: Mapping[str, str],
        s5cmd: str = "s5cmd",
        runner: Runner = subprocess.run,
        readonly: bool = False,
        require_hashes: bool = False,
    ) -> None:
        # Untrusted hosts: never publish, and never use a file whose hash the
        # manifest does not vouch for.
        self.readonly = readonly
        self.require_hashes = require_hashes
        self._base = f"s3://{bucket}/{prefix.strip('/')}" if prefix.strip("/") else f"s3://{bucket}"
        self._endpoint_url = endpoint_url
        self._roots = {name: os.path.abspath(path) for name, path in roots.items()}
        self._s5cmd = s5cmd
        self._runner = runner

    @classmethod
    def from_environment(cls, roots: Mapping[str, str]) -> "R2CheckpointCache | None":
        """Build from the env the worker run spec provides; None when disabled."""
        endpoint = os.environ.get("R2_ENDPOINT_URL")
        bucket = os.environ.get("MAESTRO_R2_BUCKET")
        if not endpoint or not bucket:
            return None
        if not os.environ.get("R2_ACCESS_KEY_ID") or not os.environ.get("R2_SECRET_ACCESS_KEY"):
            return None
        return cls(
            bucket=bucket,
            prefix=os.environ.get("MAESTRO_R2_PREFIX", "maestro"),
            endpoint_url=endpoint,
            roots=roots,
            readonly=os.environ.get("MAESTRO_R2_CACHE_READONLY") == "1",
            require_hashes=os.environ.get("MAESTRO_R2_REQUIRE_HASHES") == "1",
        )

    # ── s5cmd plumbing ─────────────────────────────────────────────────────

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        # s5cmd reads the standard AWS variable names.
        env["AWS_ACCESS_KEY_ID"] = os.environ.get("R2_ACCESS_KEY_ID", "")
        env["AWS_SECRET_ACCESS_KEY"] = os.environ.get("R2_SECRET_ACCESS_KEY", "")
        # Temporary prefix-scoped credentials come with a session token.
        if os.environ.get("R2_SESSION_TOKEN"):
            env["AWS_SESSION_TOKEN"] = os.environ["R2_SESSION_TOKEN"]
        env.setdefault("AWS_REGION", "auto")
        return env

    def _s5(self, *args: str, check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
        try:
            result = self._runner(
                [self._s5cmd, "--endpoint-url", self._endpoint_url, *args],
                capture_output=True,
                text=True,
                env=self._env(),
                **({"timeout": timeout} if timeout else {}),
            )
        except subprocess.TimeoutExpired as error:
            raise CheckpointCacheError(f"s5cmd {args[0]} timed out after {int(error.timeout)}s") from error
        if check and result.returncode != 0:
            raise CheckpointCacheError(
                f"s5cmd {args[0]} failed ({result.returncode}): {(result.stderr or '').strip()[:500]}"
            )
        return result

    def _run_batch(self, lines: list[str], timeout: float | None = None) -> None:
        if not lines:
            return
        with tempfile.NamedTemporaryFile("w", suffix=".s5cmd", delete=False) as handle:
            handle.write("\n".join(lines) + "\n")
            batch = handle.name
        try:
            self._s5("run", batch, timeout=timeout)
        finally:
            os.unlink(batch)

    def _remote(self, root: str, path: str) -> str:
        return f"{self._base}/{root}/{path}"

    def _manifest_url(self, model_type: str) -> str:
        return f"{self._base}/manifests/{model_type}.json"

    # ── manifest ───────────────────────────────────────────────────────────

    def read_manifest(self, model_type: str) -> tuple[list[FileEntry], list[LinkEntry]]:
        result = self._s5("cat", self._manifest_url(model_type), check=False)
        if result.returncode != 0:
            return [], []
        try:
            payload = json.loads(result.stdout or "{}")
        except ValueError:
            log.warning("ignoring unreadable manifest for %s", model_type)
            return [], []
        files = [
            FileEntry(str(item["root"]), str(item["path"]), int(item["size"]), str(item.get("sha256") or ""))
            for item in payload.get("files") or []
            if item.get("root") in self._roots and _safe_relative(item.get("path")) and _cached_path(item["path"])
        ]
        links = [
            LinkEntry(str(item["root"]), str(item["path"]), str(item["target"]))
            for item in payload.get("links") or []
            if item.get("root") in self._roots and _safe_relative(item.get("path"))
        ]
        return files, links

    def _write_manifest(self, model_type: str, files: list[FileEntry], links: list[LinkEntry]) -> None:
        payload = {
            "model_type": model_type,
            "files": [vars(entry) for entry in sorted(files, key=lambda e: (e.root, e.path))],
            "links": [vars(entry) for entry in sorted(links, key=lambda e: (e.root, e.path))],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            json.dump(payload, handle, indent=1)
            local = handle.name
        try:
            self._s5("cp", local, self._manifest_url(model_type))
        finally:
            os.unlink(local)

    # ── local state ────────────────────────────────────────────────────────

    def snapshot(self) -> tuple[Snapshot, dict[tuple[str, str], str]]:
        files: Snapshot = {}
        links: dict[tuple[str, str], str] = {}
        for root_name, root in self._roots.items():
            if not os.path.isdir(root):
                continue
            for directory, dirs, names in os.walk(root):
                dirs[:] = [d for d in dirs if d != _SKIP_DIR]
                for name in names:
                    full = os.path.join(directory, name)
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    if not _cached_path(rel):
                        continue
                    if os.path.islink(full):
                        links[(root_name, rel)] = os.readlink(full)
                    elif os.path.isfile(full):
                        files[(root_name, rel)] = os.path.getsize(full)
        return files, links

    def _local(self, root: str, path: str) -> str:
        return os.path.join(self._roots[root], *path.split("/"))

    # ── public operations ─────────────────────────────────────────────────

    def hydrate(self, model_type: str) -> int:
        """Pull cached files for ``model_type``; return how many were fetched."""
        files, links = self.read_manifest(model_type)
        if self.require_hashes and any(not entry.sha256 for entry in files):
            raise CheckpointCacheError(
                f"the {model_type} cache has files without a trusted hash; downloading from the source instead"
            )
        missing = [
            entry for entry in files
            if not (os.path.isfile(self._local(entry.root, entry.path))
                    and os.path.getsize(self._local(entry.root, entry.path)) == entry.size)
        ]
        lines = []
        for entry in missing:
            destination = self._local(entry.root, entry.path)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            lines.append(f'cp "{self._remote(entry.root, entry.path)}" "{destination}"')
        self._run_batch(lines)
        problems = [
            entry.path for entry in missing
            if not (os.path.isfile(self._local(entry.root, entry.path))
                    and os.path.getsize(self._local(entry.root, entry.path)) == entry.size)
        ]
        if problems:
            raise CheckpointCacheError(
                f"R2 pull incomplete for {model_type}: {', '.join(problems[:5])}"
            )
        tampered = self._verify([entry for entry in missing if entry.sha256])
        if tampered:
            raise CheckpointCacheError(
                f"R2 cache files of {model_type} do not match their hashes (removed): {', '.join(tampered[:5])}"
            )
        for link in links:
            local = self._local(link.root, link.path)
            if os.path.lexists(local):
                continue
            os.makedirs(os.path.dirname(local), exist_ok=True)
            os.symlink(link.target, local)
        log.info("hydrated %s: %d file(s) pulled from R2", model_type, len(missing))
        return len(missing)

    def _verify(self, entries: list[FileEntry]) -> list[str]:
        """Paths whose local content does not match the manifest hash (deleted)."""
        def check(entry: FileEntry) -> str | None:
            local = self._local(entry.root, entry.path)
            if file_sha256(local) == entry.sha256:
                return None
            os.remove(local)
            return entry.path
        with ThreadPoolExecutor(max_workers=4) as pool:
            return [path for path in pool.map(check, entries) if path]

    def _with_hashes(self, entries: list[FileEntry]) -> list[FileEntry]:
        def hashed(entry: FileEntry) -> FileEntry:
            return FileEntry(entry.root, entry.path, entry.size, file_sha256(self._local(entry.root, entry.path)))
        with ThreadPoolExecutor(max_workers=4) as pool:
            return list(pool.map(hashed, entries))

    def has_hashes(self, model_type: str) -> bool:
        files, _ = self.read_manifest(model_type)
        return bool(files) and all(entry.sha256 for entry in files)

    def rehash(self, model_type: str) -> int:
        """Add hashes to an older manifest from files pulled on this trusted host."""
        if self.readonly:
            raise CheckpointCacheError("cannot rehash from a read-only host")
        files, links = self.read_manifest(model_type)
        unhashed = [entry for entry in files if not entry.sha256]
        if not unhashed:
            return 0
        for entry in unhashed:
            local = self._local(entry.root, entry.path)
            if not (os.path.isfile(local) and os.path.getsize(local) == entry.size):
                raise CheckpointCacheError(f"{entry.path} is not on disk; hydrate {model_type} first")
        rehashed = {(e.root, e.path): e for e in self._with_hashes(unhashed)}
        self._write_manifest(model_type, [rehashed.get((e.root, e.path), e) for e in files], links)
        log.info("rehashed %s: %d file(s)", model_type, len(rehashed))
        return len(rehashed)

    def publish(
        self,
        model_type: str,
        before: tuple[Snapshot, dict[tuple[str, str], str]],
        after: tuple[Snapshot, dict[tuple[str, str], str]] | None = None,
    ) -> int:
        """Upload files that appeared between ``before`` and ``after`` (default:
        now); update the manifest."""
        if self.readonly:
            log.info("cache is read-only on this host; %s not published", model_type)
            return 0
        before_files, before_links = before
        after_files, after_links = after if after is not None else self.snapshot()
        old_files, old_links = self.read_manifest(model_type)

        new_files = self._with_hashes([
            FileEntry(root, path, size)
            for (root, path), size in after_files.items()
            if before_files.get((root, path)) != size
        ])
        new_links = [
            LinkEntry(root, path, target)
            for (root, path), target in after_links.items()
            if (root, path) not in before_links
        ]
        upload_bytes = sum(entry.size for entry in new_files)
        self._run_batch([
            f'cp "{self._local(entry.root, entry.path)}" "{self._remote(entry.root, entry.path)}"'
            for entry in new_files
        ], timeout=max(_MIN_UPLOAD_TIMEOUT_S, upload_bytes / _MIN_UPLOAD_BYTES_PER_S))
        if new_files or new_links or not old_files:
            merged_files = {(e.root, e.path): e for e in old_files}
            merged_files.update({(e.root, e.path): e for e in new_files})
            merged_links = {(e.root, e.path): e for e in old_links}
            merged_links.update({(e.root, e.path): e for e in new_links})
            self._write_manifest(model_type, list(merged_files.values()), list(merged_links.values()))
        log.info("published %s: %d new file(s) to R2", model_type, len(new_files))
        return len(new_files)


def _safe_relative(path: Any) -> bool:
    if not isinstance(path, str) or not path:
        return False
    parts = path.split("/")
    return not path.startswith("/") and ".." not in parts
