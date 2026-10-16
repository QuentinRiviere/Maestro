"""Validation of everything that becomes a path or a URL.

Repository ids, revisions and file names come from the Hub API, models.yml or
the browser. Each one is checked here before it is joined to the models
directory or put in a URL, so a hostile value can neither escape the models
directory nor make the fetcher talk to another host.
"""
from __future__ import annotations

import os
import re
import urllib.parse

HUB = "https://huggingface.co"

# Hosts the fetcher may talk to: the Hub and its download CDNs
# (cdn-lfs*.hf.co, cas-bridge.xethub.hf.co, ...).
ALLOWED_HOST_SUFFIXES = ("huggingface.co", "hf.co")

# Hub repo ids: <owner>/<name>, letters, digits, '-', '_', '.', no leading dot.
_REPO_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_REVISION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]{0,127}$")
# One path segment of a file inside a repo.
_FILE_PART = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9 ._+()\[\]-]{0,199}$")


class UnsafePathError(ValueError):
    """A repo id, revision or file name that must not be used."""


def check_repo_id(repo: str) -> str:
    if not isinstance(repo, str):
        raise UnsafePathError("repo id must be a string")
    parts = repo.split("/")
    if len(parts) != 2 or not all(_REPO_PART.match(p) for p in parts):
        raise UnsafePathError(f"invalid repo id: {repo!r}")
    if any(p in (".", "..") or ".." in p for p in parts):
        raise UnsafePathError(f"invalid repo id: {repo!r}")
    return repo


def check_revision(revision: str) -> str:
    if not isinstance(revision, str) or not _REVISION.match(revision) or ".." in revision:
        raise UnsafePathError(f"invalid revision: {revision!r}")
    return revision


def check_filename(name: str) -> str:
    """A repo-relative file name: forward slashes, no '..', not absolute."""
    if not isinstance(name, str) or not name or len(name) > 400:
        raise UnsafePathError(f"invalid file name: {name!r}")
    if name.startswith("/") or "\\" in name or "\0" in name:
        raise UnsafePathError(f"invalid file name: {name!r}")
    parts = name.split("/")
    if len(parts) > 4 or not all(_FILE_PART.match(p) for p in parts):
        raise UnsafePathError(f"invalid file name: {name!r}")
    if any(p in (".", "..") or p.startswith(".") for p in parts):
        raise UnsafePathError(f"invalid file name: {name!r}")
    return name


def inside(root: str, *parts: str) -> str:
    """Join parts under root and refuse any result outside root."""
    base = os.path.realpath(root)
    target = os.path.realpath(os.path.join(base, *parts))
    if target != base and not target.startswith(base + os.sep):
        raise UnsafePathError(f"path escapes {root}: {'/'.join(parts)}")
    return target


def repo_dir(models_dir: str, repo: str) -> str:
    check_repo_id(repo)
    return inside(models_dir, *repo.split("/"))


def model_file(models_dir: str, repo: str, name: str) -> str:
    check_repo_id(repo)
    check_filename(name)
    return inside(models_dir, *repo.split("/"), *name.split("/"))


def allowed_host(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not host:
        return False
    return any(host == s or host.endswith("." + s) for s in ALLOWED_HOST_SUFFIXES)


def resolve_url(repo: str, commit: str, name: str) -> str:
    check_repo_id(repo)
    check_filename(name)
    if not SHA_RE.match(commit):
        raise UnsafePathError(f"not a commit sha: {commit!r}")
    return f"{HUB}/{repo}/resolve/{commit}/{urllib.parse.quote(name)}"
