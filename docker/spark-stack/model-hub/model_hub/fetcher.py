"""Verified model downloads from the Hugging Face Hub, shared by the CLI and the web app.

For a repo and a revision it resolves the revision to a commit, lists the
repo's files through the Hub API, refuses pickle-based and code formats,
downloads into <models>/.staging with resume (HTTP Range), checks every file
against the hash the Hub publishes (SHA-256 for LFS files, the git blob SHA-1
otherwise), then moves it atomically into <models>/<owner>/<name>/<file>.
<models>/manifest.json records every verified file; <models>/models.sha256 is
the same list in `sha256sum -c` format.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import threading
import time
import urllib.error
from dataclasses import dataclass, field
from typing import Callable

from . import hf
from .paths import SHA_RE, check_filename, check_repo_id, inside, model_file, repo_dir, resolve_url

ALLOWED_SUFFIXES = (".gguf", ".safetensors", ".json", ".txt", ".md", ".model", ".jinja", ".tiktoken")
REFUSED_SUFFIXES = (".bin", ".pt", ".pth", ".ckpt", ".pkl", ".pickle", ".npy", ".npz", ".py", ".sh", ".so",
                    ".exe", ".dll", ".dylib", ".joblib", ".h5", ".keras", ".onnx_data", ".zip", ".tar", ".gz")
CHUNK = 8 << 20
DISK_MARGIN = 1 << 30  # keep 1 GiB free after a download

_MANIFEST_LOCK = threading.RLock()


class FetchError(RuntimeError):
    pass


class Cancelled(FetchError):
    pass


@dataclass
class RemoteFile:
    name: str
    size: int
    expected: str
    algo: str  # "sha256" (LFS) or "git-sha1"

    def as_dict(self) -> dict:
        return {"name": self.name, "size": self.size, "expected": self.expected, "algo": self.algo}


@dataclass
class Snapshot:
    repo: str
    commit: str
    files: list[RemoteFile]
    info: dict = field(default_factory=dict, repr=False)

    def by_name(self) -> dict[str, RemoteFile]:
        return {f.name: f for f in self.files}


def is_refused(name: str) -> bool:
    return name.lower().endswith(REFUSED_SUFFIXES)


def is_allowed(name: str) -> bool:
    low = name.lower()
    return low.endswith(ALLOWED_SUFFIXES) and not is_refused(name)


def remote_files(info: dict) -> list[RemoteFile]:
    files = []
    for sibling in info.get("siblings") or []:
        name = sibling.get("rfilename")
        if not isinstance(name, str):
            continue
        try:
            check_filename(name)
        except ValueError:
            continue  # never offered, so never downloaded
        lfs = sibling.get("lfs") or {}
        size = int(lfs.get("size") or sibling.get("size") or 0)
        if lfs.get("sha256"):
            files.append(RemoteFile(name, size, str(lfs["sha256"]).lower(), "sha256"))
        elif sibling.get("blobId"):
            files.append(RemoteFile(name, size, str(sibling["blobId"]).lower(), "git-sha1"))
    return files


def resolve(repo: str, revision: str = "main") -> Snapshot:
    """Resolve revision to a commit and list the repo's files with their hashes."""
    check_repo_id(repo)
    info = hf.repo_info(repo, revision)
    commit = info["sha"]
    if not SHA_RE.match(commit):
        raise FetchError(f"Hub returned an invalid commit for {repo}: {commit!r}")
    if SHA_RE.match(revision) and commit != revision:
        raise FetchError(f"Hub resolved {revision} to {commit}")
    return Snapshot(repo, commit, remote_files(info), info)


def select(snapshot: Snapshot, include: list[str] | None = None,
           exclude: list[str] | None = None) -> tuple[list[RemoteFile], list[str]]:
    """Files matching include/exclude globs, and the refused (unsafe) matches."""
    include = list(include or [])
    exclude = list(exclude or [])
    chosen, refused = [], []
    for f in snapshot.files:
        if include and not any(fnmatch.fnmatchcase(f.name, p) for p in include):
            continue
        if any(fnmatch.fnmatchcase(f.name, p) for p in exclude):
            continue
        if is_refused(f.name):
            refused.append(f.name)
        elif is_allowed(f.name):
            chosen.append(f)
    return chosen, refused


def glob_escape(name: str) -> str:
    """A glob that only matches this exact file name."""
    return "".join(f"[{c}]" if c in "*?[" else c for c in name)


def _hash(path: str, algo: str, cancel: threading.Event | None = None) -> str:
    if algo == "sha256":
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1(f"blob {os.path.getsize(path)}\0".encode())
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            if cancel is not None and cancel.is_set():
                raise Cancelled("cancelled")
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: str) -> str:
    return _hash(path, "sha256")


def git_blob_sha1(path: str) -> str:
    return _hash(path, "git-sha1")


ProgressFn = Callable[[dict], None]


class Store:
    """The models directory: staging, manifest, verified files."""

    def __init__(self, models_dir: str):
        self.models_dir = os.path.realpath(models_dir)
        self.staging = os.path.join(self.models_dir, ".staging")
        self.manifest_path = os.path.join(self.models_dir, "manifest.json")

    # -- manifest ---------------------------------------------------------
    def load_manifest(self) -> dict:
        with _MANIFEST_LOCK:
            if not os.path.exists(self.manifest_path):
                return {}
            with open(self.manifest_path, encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}

    def _save_manifest(self, manifest: dict) -> None:
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
        os.replace(tmp, self.manifest_path)
        tmp = os.path.join(self.models_dir, "models.sha256.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            for rel, item in sorted(manifest.items()):
                if item.get("sha256"):
                    handle.write(f"{item['sha256']}  {rel}\n")
        os.replace(tmp, os.path.join(self.models_dir, "models.sha256"))

    def update_manifest(self, changes: dict[str, dict | None]) -> dict:
        with _MANIFEST_LOCK:
            manifest = self.load_manifest()
            for rel, item in changes.items():
                if item is None:
                    manifest.pop(rel, None)
                else:
                    manifest[rel] = item
            self._save_manifest(manifest)
            return manifest

    # -- paths ------------------------------------------------------------
    def target(self, repo: str, name: str) -> str:
        return model_file(self.models_dir, repo, name)

    def part_path(self, repo: str, name: str) -> str:
        key = hashlib.sha1(f"{repo}/{name}".encode()).hexdigest()
        return inside(self.models_dir, ".staging", key + ".part")

    def staged_bytes(self, repo: str, name: str) -> int:
        part = self.part_path(repo, name)
        return os.path.getsize(part) if os.path.exists(part) else 0

    def free_bytes(self) -> int:
        return shutil.disk_usage(self.models_dir).free

    # -- checks -----------------------------------------------------------
    def is_verified(self, repo: str, f: RemoteFile) -> bool:
        """In place, same size, and recorded with this exact expected hash."""
        target = self.target(repo, f.name)
        known = self.load_manifest().get(f"{repo}/{f.name}")
        return bool(os.path.exists(target) and known and known.get("expected") == f.expected
                    and os.path.getsize(target) == f.size)

    def missing_bytes(self, repo: str, files: list[RemoteFile]) -> int:
        return sum(max(f.size - self.staged_bytes(repo, f.name), 0)
                   for f in files if not self.is_verified(repo, f))

    def check_disk(self, repo: str, files: list[RemoteFile]) -> None:
        need = self.missing_bytes(repo, files)
        free = self.free_bytes()
        if need and free < need + DISK_MARGIN:
            raise FetchError(f"not enough disk space: {need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free")

    def verify(self, repo: str, name: str, cancel: threading.Event | None = None) -> bool:
        """Re-hash an installed file against its manifest entry."""
        rel = f"{repo}/{name}"
        known = self.load_manifest().get(rel)
        target = self.target(repo, name)
        if not known or not os.path.exists(target):
            return False
        if os.path.getsize(target) != int(known.get("size", -1)):
            return False
        return _hash(target, known["algo"], cancel) == known["expected"]

    # -- download ---------------------------------------------------------
    def _download(self, url: str, part: str, size: int, progress: ProgressFn | None,
                  cancel: threading.Event | None) -> None:
        done = os.path.getsize(part) if os.path.exists(part) else 0
        if done > size:
            os.remove(part)
            done = 0
        if done == size:
            return
        headers = {"Range": f"bytes={done}-"} if done else {}
        try:
            resp = hf.OPENER.open(hf.request(url, headers), timeout=120)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and done:
                os.remove(part)
                return self._download(url, part, size, progress, cancel)
            raise FetchError(f"download failed: HTTP {exc.code}") from exc
        with resp:
            if done and resp.status != 206:
                done = 0  # server ignored the range: start over
            started, start_bytes, last = time.monotonic(), done, 0.0
            with open(part, "ab" if done else "wb") as out:
                while True:
                    if cancel is not None and cancel.is_set():
                        raise Cancelled("cancelled")
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    out.write(block)
                    done += len(block)
                    if done > size:
                        raise FetchError("server sent more bytes than the Hub listed")
                    now = time.monotonic()
                    if progress and now - last > 0.5:
                        rate = (done - start_bytes) / max(now - started, 1e-6)
                        progress({"phase": "download", "done": done, "size": size, "rate": rate})
                        last = now
        if os.path.getsize(part) != size:
            raise FetchError(f"size mismatch: {os.path.getsize(part)} != {size}")

    def fetch_file(self, repo: str, commit: str, f: RemoteFile, *, force_verify: bool = False,
                   progress: ProgressFn | None = None, cancel: threading.Event | None = None) -> str:
        """Download and verify one file. Returns 'skipped', 'in-place' or 'downloaded'."""
        check_repo_id(repo)
        check_filename(f.name)
        if not is_allowed(f.name):
            raise FetchError(f"refused file type: {f.name}")
        if not SHA_RE.match(commit):
            raise FetchError(f"not a commit sha: {commit}")
        rel = f"{repo}/{f.name}"
        target = self.target(repo, f.name)
        if not force_verify and self.is_verified(repo, f):
            return "skipped"
        if os.path.exists(target):
            if progress:
                progress({"phase": "verify", "done": 0, "size": f.size, "rate": 0})
            actual = _hash(target, f.algo, cancel)
            if actual == f.expected:
                sha = actual if f.algo == "sha256" else sha256_file(target)
                self.update_manifest({rel: self._entry(commit, f, sha)})
                return "in-place"
        os.makedirs(self.staging, exist_ok=True)
        part = self.part_path(repo, f.name)
        self.check_disk(repo, [f])
        url = resolve_url(repo, commit, f.name)
        try:
            self._download(url, part, f.size, progress, cancel)
        except (urllib.error.URLError, OSError, hf.HubError) as exc:
            raise FetchError(f"download failed: {exc}") from exc
        if progress:
            progress({"phase": "verify", "done": f.size, "size": f.size, "rate": 0})
        actual = _hash(part, f.algo, cancel)
        if actual != f.expected:
            os.remove(part)
            raise FetchError(f"{f.algo} mismatch for {f.name} (expected {f.expected}, got {actual}); file discarded")
        sha = actual if f.algo == "sha256" else sha256_file(part)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        os.chmod(part, 0o644)
        os.replace(part, target)
        self.update_manifest({rel: self._entry(commit, f, sha)})
        return "downloaded"

    @staticmethod
    def _entry(commit: str, f: RemoteFile, sha256: str) -> dict:
        return {"commit": commit, "expected": f.expected, "algo": f.algo, "sha256": sha256, "size": f.size}

    # -- installed --------------------------------------------------------
    def installed(self) -> dict[str, list[dict]]:
        """Manifest entries grouped by repo, with presence on disk."""
        repos: dict[str, list[dict]] = {}
        for rel, item in self.load_manifest().items():
            owner, name, *rest = rel.split("/")
            repo, fname = f"{owner}/{name}", "/".join(rest)
            try:
                path = self.target(repo, fname)
            except ValueError:
                continue
            repos.setdefault(repo, []).append({**item, "name": fname, "present": os.path.exists(path)})
        return repos

    def delete_repo(self, repo: str) -> int:
        """Delete a repo's files (and its manifest entries). Returns bytes freed."""
        directory = repo_dir(self.models_dir, repo)
        freed = 0
        if os.path.isdir(directory):
            for root, _dirs, files in os.walk(directory):
                freed += sum(os.path.getsize(os.path.join(root, n)) for n in files)
            shutil.rmtree(directory)
            owner_dir = os.path.dirname(directory)
            if owner_dir != self.models_dir and os.path.isdir(owner_dir) and not os.listdir(owner_dir):
                os.rmdir(owner_dir)
        prefix = repo + "/"
        self.update_manifest({rel: None for rel in self.load_manifest() if rel.startswith(prefix)})
        return freed

    def drop_partials(self, repo: str, names: list[str]) -> None:
        for name in names:
            part = self.part_path(repo, name)
            if os.path.exists(part):
                os.remove(part)
