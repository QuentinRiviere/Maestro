"""Write-through checkpoint cache in Cloudflare R2, used on the remote worker.

Maestro's own download logic stays the source of truth for *which* files a
model needs. The cache only remembers, per ``model_type``, the files that
logic ended up putting on disk, so the next pod can pull them from R2 instead
of Hugging Face / CivitAI:

1. ``hydrate``: read ``<prefix>/manifests/<model_type>.json`` and pull the
   listed files into the local roots (``ckpts``, ``hf`` cache, ...).
2. Maestro downloads whatever is still missing (nothing, on a warm cache).
3. ``publish``: upload files that appeared during step 2 and rewrite the
   manifest as the union of the previous one and the new files. Files that
   step 2 found already on disk (a text encoder another model fetched first)
   are added too when ``used`` names them: ``record_checks`` collects them.

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
import threading
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any


log = logging.getLogger("maestro.remote_worker.cache")

_SKIP_SUFFIXES = (".part", ".lock", ".incomplete", ".tmp")
# huggingface_hub's local_dir bookkeeping: rewritten by every later download
# into the same folder (so its cached size goes stale), and never needed since
# Maestro only checks that the weights themselves are on disk. ``xet`` holds
# hf_xet's download logs, which keep growing after the snapshot (a pull of the
# listed size then reads as incomplete).
_SKIP_DIRS = {".cache", "xet"}


def _cached_path(path: str) -> bool:
    """Whether a root-relative path belongs in the cache."""
    return not path.endswith(_SKIP_SUFFIXES) and not _SKIP_DIRS.intersection(path.split("/"))


Runner = Callable[..., subprocess.CompletedProcess]
# Checkpoints are few and large (up to ~20 GB a file): s5cmd's defaults
# (5 parts of 50 MB in flight per file) leave most of a fast link idle.
_CP_FLAGS = "--concurrency 16 --part-size 64"
_HASH_WORKERS = max(4, min(8, os.cpu_count() or 4))
PROGRESS_INTERVAL_SECONDS = 2.0
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
        # A test runner cannot stream; files are then verified after the batch.
        self._stream = runner is subprocess.run

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
                f"s5cmd {args[0]} failed ({result.returncode}): {(result.stderr or '').strip()[-1500:]}"
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

    def _stream_batch(self, lines: list[str], on_line: Callable[[str], None]) -> None:
        """``s5cmd run`` reporting each finished command as it completes."""
        with tempfile.NamedTemporaryFile("w", suffix=".s5cmd", delete=False) as handle:
            handle.write("\n".join(lines) + "\n")
            batch = handle.name
        try:
            process = subprocess.Popen(
                [self._s5cmd, "--endpoint-url", self._endpoint_url, "run", batch],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=self._env(),
            )
            assert process.stdout is not None
            for line in process.stdout:
                on_line(line.rstrip("\n"))
            stderr = process.stderr.read() if process.stderr else ""
            if process.wait() != 0:
                raise CheckpointCacheError(f"s5cmd run failed ({process.returncode}): {stderr.strip()[-1500:]}")
        finally:
            os.unlink(batch)

    def _pull(self, entries: list["FileEntry"], on_file: Callable[["FileEntry"], None]) -> None:
        """Download ``entries``, calling ``on_file`` for each one as soon as it lands."""
        pending = {self._local(e.root, e.path): e for e in entries}
        lines = []
        for destination, entry in pending.items():
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            lines.append(f'cp {_CP_FLAGS} "{self._remote(entry.root, entry.path)}" "{destination}"')
        if not lines:
            return
        if not self._stream:
            self._run_batch(lines)
            for entry in entries:
                on_file(entry)
            return

        def finished(line: str) -> None:
            for destination in list(pending):
                if line.endswith(" " + destination):
                    on_file(pending.pop(destination))
                    return
        self._stream_batch(lines, finished)
        for entry in pending.values():  # reported nothing: checked by the caller
            on_file(entry)

    def object_url(self, *parts: str) -> str:
        """URL of an object under the cache prefix (outside the checkpoint roots)."""
        return "/".join([self._base, *parts])

    def copy(self, source: str, destination: str, *, timeout: float | None = None) -> bool:
        """``s5cmd cp`` between a local path and an object URL; False when it fails."""
        return self._s5("cp", source, destination, check=False, timeout=timeout).returncode == 0

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
                dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
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

    def _tree_bytes(self) -> int:
        """Bytes under the cache roots, partial downloads included."""
        total = 0
        for root in self._roots.values():
            for directory, _dirs, names in os.walk(root):
                for name in names:
                    try:
                        total += os.lstat(os.path.join(directory, name)).st_size
                    except OSError:
                        pass
        return total

    def _report_progress(self, total_bytes: int, files_total: int, landed: list,
                         on_progress: Callable[[dict], None], stop: threading.Event) -> None:
        """Call ``on_progress`` every few seconds while a pull runs."""
        baseline, started = self._tree_bytes(), time.time()
        while True:
            finished = stop.wait(PROGRESS_INTERVAL_SECONDS)
            done = total_bytes if finished else min(total_bytes, max(0, self._tree_bytes() - baseline))
            elapsed = max(0.001, time.time() - started)
            try:
                on_progress({"done_bytes": done, "total_bytes": total_bytes, "files_done": len(landed),
                             "files_total": files_total, "bytes_per_second": int(done / elapsed)})
            except Exception:  # progress is cosmetic; never break the pull
                pass
            if finished:
                return

    def hydrate(self, model_type: str, on_progress: Callable[[dict], None] | None = None) -> int:
        """Pull cached files for ``model_type``; return how many were fetched.

        ``on_progress`` receives ``{done_bytes, total_bytes, files_done,
        files_total, bytes_per_second}`` every few seconds during the pull.
        """
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
        # Each file is hashed as soon as it lands, while the rest download.
        landed_files: list[FileEntry] = []
        stop = threading.Event()
        reporter = None
        if on_progress is not None and missing:
            reporter = threading.Thread(
                target=self._report_progress, daemon=True, name=f"r2-progress-{model_type}",
                args=(sum(e.size for e in missing), len(missing), landed_files, on_progress, stop),
            )
            reporter.start()
        try:
            with ThreadPoolExecutor(max_workers=_HASH_WORKERS) as pool:
                checks = []

                def landed(entry: FileEntry) -> None:
                    landed_files.append(entry)
                    if entry.sha256:
                        checks.append(pool.submit(self._matches, entry))
                self._pull(missing, landed)
                tampered = [path for path in (check.result() for check in checks) if path]
        finally:
            stop.set()
            if reporter is not None:
                reporter.join(timeout=5)
        problems = [
            entry.path for entry in missing
            if entry.path not in tampered
            and not (os.path.isfile(self._local(entry.root, entry.path))
                     and os.path.getsize(self._local(entry.root, entry.path)) == entry.size)
        ]
        if problems:
            raise CheckpointCacheError(
                f"R2 pull incomplete for {model_type}: {', '.join(problems[:5])}"
            )
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

    def _matches(self, entry: FileEntry) -> str | None:
        """Hash-check a pulled file: None if it matches, else delete it and return its path.

        A missing or short file is left to the size check (incomplete pull).
        """
        local = self._local(entry.root, entry.path)
        if not (os.path.isfile(local) and os.path.getsize(local) == entry.size):
            return None
        if file_sha256(local) == entry.sha256:
            return None
        os.remove(local)
        return entry.path

    def _with_hashes(self, entries: list[FileEntry]) -> list[FileEntry]:
        def hashed(entry: FileEntry) -> FileEntry:
            return FileEntry(entry.root, entry.path, entry.size, file_sha256(self._local(entry.root, entry.path)))
        with ThreadPoolExecutor(max_workers=_HASH_WORKERS) as pool:
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

    @contextmanager
    def record_checks(self) -> Iterator[set[str]]:
        """Collect the paths under the roots that existence checks ask about.

        Maestro's download logic checks every file a model needs and skips
        those already on disk; what it checked is the model's footprint.
        """
        checked: set[str] = set()
        roots = tuple(root + os.sep for root in self._roots.values())
        originals = (os.path.isfile, os.path.exists, os.path.isdir)

        def watch(check: Callable[[Any], bool]) -> Callable[[Any], bool]:
            def wrapped(path: Any) -> bool:
                try:
                    full = os.path.abspath(os.fspath(path))
                except TypeError:
                    return check(path)
                if full.startswith(roots):
                    checked.add(full)
                return check(path)
            return wrapped

        os.path.isfile, os.path.exists, os.path.isdir = (watch(check) for check in originals)
        try:
            yield checked
        finally:
            os.path.isfile, os.path.exists, os.path.isdir = originals

    def used_files(self, paths: Iterable[str]) -> set[tuple[str, str]]:
        """Root-relative files behind checked paths (a checked folder brings its files)."""
        found: set[tuple[str, str]] = set()
        for path in paths:
            for root_name, root in self._roots.items():
                if not path.startswith(root + os.sep):
                    continue
                if os.path.isfile(path) and not os.path.islink(path):
                    candidates = [path]
                elif os.path.isdir(path):
                    candidates = [os.path.join(directory, name) for directory, dirs, names in os.walk(path)
                                  if not _SKIP_DIRS.intersection(os.path.relpath(directory, root).split(os.sep))
                                  for name in names]
                else:
                    candidates = []
                for full in candidates:
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    if _cached_path(rel) and os.path.isfile(full) and not os.path.islink(full):
                        found.add((root_name, rel))
        return found

    def _published_entries(self) -> dict[tuple[str, str], FileEntry]:
        """Every file some manifest already lists (already in R2, with its hash)."""
        result = self._s5("ls", self.object_url("manifests", "*"), check=False)
        entries: dict[tuple[str, str], FileEntry] = {}
        for line in (result.stdout or "").splitlines():
            name = line.split()[-1] if line.split() else ""
            if name.endswith(".json"):
                for entry in self.read_manifest(name[:-5])[0]:
                    entries.setdefault((entry.root, entry.path), entry)
        return entries

    def publish(
        self,
        model_type: str,
        before: tuple[Snapshot, dict[tuple[str, str], str]],
        after: tuple[Snapshot, dict[tuple[str, str], str]] | None = None,
        used: Iterable[tuple[str, str]] = (),
    ) -> int:
        """Upload files that appeared between ``before`` and ``after`` (default:
        now); update the manifest. ``used`` files already on disk join the
        manifest too, uploaded only when no manifest lists them yet."""
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
        listed = {(e.root, e.path) for e in old_files} | {(e.root, e.path) for e in new_files}
        reused = [key for key in used if key in after_files and key not in listed]
        shared: list[FileEntry] = []
        if reused:
            published = self._published_entries()
            shared = [published[key] for key in reused
                      if key in published and published[key].size == after_files[key] and published[key].sha256]
            new_files += self._with_hashes([FileEntry(*key, after_files[key]) for key in reused
                                            if key not in {(e.root, e.path) for e in shared}])
            # Listed by another manifest: already in R2, keep its trusted hash.
            old_files = old_files + shared
        new_links = [
            LinkEntry(root, path, target)
            for (root, path), target in after_links.items()
            if (root, path) not in before_links
        ]
        upload_bytes = sum(entry.size for entry in new_files)
        self._run_batch([
            f'cp {_CP_FLAGS} "{self._local(entry.root, entry.path)}" "{self._remote(entry.root, entry.path)}"'
            for entry in new_files
        ], timeout=max(_MIN_UPLOAD_TIMEOUT_S, upload_bytes / _MIN_UPLOAD_BYTES_PER_S))
        if new_files or new_links or shared or not old_files:
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
