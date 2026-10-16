"""models.yml: the single list of fetched models and how llama-swap serves them.

Each entry:

    - repo: owner/name
      revision: <40-hex commit>          # or a branch; the CLI prints the commit to pin
      include: ["*-Q4_K.gguf", ...]      # fnmatch globs (exact names when added by the web app)
      exclude: [...]                     # optional
      serve:                             # optional: map into llama-swap.yaml
        id: qwen36-35b
        model: file-Q4_K.gguf            # first part for split models
        mmproj: mmproj-model-f16.gguf    # optional
        ctx_size: 65536
        parallel: 1
        mtp: true
        mtp_draft_n_max: 6
        extra_args: []
        ttl: 0
        group: resident
        preload: true
        aliases: [small-llm]
        enabled: true

The fetch CLI reads repo / revision / include / exclude and ignores serve.
"""
from __future__ import annotations

import os
import re
import threading

import yaml

from .paths import SHA_RE, UnsafePathError, check_filename, check_repo_id, check_revision

HEADER = """\
# Models fetched into $MODELS_DIR/<owner>/<repo>/<file>, verified against the
# Hub hashes. Written by the model hub (and readable by `fetch`); hand edits
# are fine while the hub is not saving. Entries with a `serve:` section are
# rendered into llama-swap.yaml (see docs/Spark-model-hub.md).
"""

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
# Extra llama-server arguments: one token each, no shell or macro syntax
# (`$` would let a value expand ${env.*} secrets into a command line).
ARG_RE = re.compile(r"^[A-Za-z0-9_.,:=/+@-]{1,200}$")
SERVE_FILE_RE = re.compile(r"^[A-Za-z0-9_./+-]+$")

LOCK = threading.RLock()


class ConfigError(ValueError):
    pass


def load(path: str) -> list[dict]:
    with LOCK:
        if not os.path.exists(path):
            return []
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    if data is None:
        return []
    if not isinstance(data, list):
        raise ConfigError(f"{path}: expected a list of entries")
    for entry in data:
        validate_entry(entry)
    return data


def save(path: str, entries: list[dict]) -> None:
    for entry in entries:
        validate_entry(entry)
    text = HEADER + "\n" + yaml.safe_dump(entries, sort_keys=False, allow_unicode=True, width=200)
    with LOCK:
        atomic_write(path, text)


def atomic_write(path: str, text: str, backup: bool = False) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if backup and os.path.exists(path):
        with open(path, encoding="utf-8") as src:
            old = src.read()
        _write_replace(path + ".bak", old, directory)
    _write_replace(path, text, directory)


def _write_replace(path: str, text: str, directory: str) -> None:
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


EXAMPLES = (("models.yml", "models.example.yml"), ("llama-swap.base.yaml", "llama-swap.base.example.yaml"))


def seed_config(config_dir: str) -> list[str]:
    """Copy config/*.example.* to the user-owned names when those are missing."""
    created = []
    for name, example in EXAMPLES:
        target, source = os.path.join(config_dir, name), os.path.join(config_dir, example)
        if not os.path.exists(target) and os.path.exists(source):
            with open(source, encoding="utf-8") as handle:
                text = handle.read()
            try:
                with LOCK:
                    atomic_write(target, text)
            except OSError:
                continue  # read-only mount (the fetch service): callers fall back to the example
            created.append(name)
    return created


def find(entries: list[dict], repo: str) -> dict | None:
    return next((e for e in entries if e.get("repo") == repo), None)


def validate_entry(entry) -> None:
    if not isinstance(entry, dict):
        raise ConfigError("each entry must be a mapping")
    try:
        check_repo_id(entry.get("repo"))
        check_revision(str(entry.get("revision", "main")))
    except UnsafePathError as exc:
        raise ConfigError(str(exc)) from exc
    for key in ("include", "exclude"):
        value = entry.get(key)
        if value is not None and (not isinstance(value, list) or not all(isinstance(v, str) for v in value)):
            raise ConfigError(f"{entry['repo']}: {key} must be a list of strings")
    if entry.get("serve") is not None:
        entry["serve"] = normalize_serve(entry["serve"])


def _int(serve: dict, key: str, default: int, lo: int, hi: int) -> int:
    value = serve.get(key, default)
    if value is None:
        value = default
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        raise ConfigError(f"serve.{key} must be an integer in [{lo}, {hi}]")
    return value


def _bool(serve: dict, key: str, default: bool) -> bool:
    value = serve.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"serve.{key} must be true or false")
    return value


def _gguf(name, key: str) -> str:
    try:
        check_filename(name)
    except UnsafePathError as exc:
        raise ConfigError(f"serve.{key}: {exc}") from exc
    if not name.lower().endswith(".gguf"):
        raise ConfigError(f"serve.{key} must be a .gguf file")
    if not SERVE_FILE_RE.match(name):
        raise ConfigError(f"serve.{key}: file names used in a llama-swap cmd may only contain [A-Za-z0-9_./+-]")
    return name


def normalize_serve(serve) -> dict:
    """Validated serve section with defaults filled in."""
    if not isinstance(serve, dict):
        raise ConfigError("serve must be a mapping")
    known = {"id", "name", "model", "mmproj", "ctx_size", "parallel", "mtp", "mtp_draft_n_max", "extra_args",
             "ttl", "group", "preload", "aliases", "enabled", "unlisted"}
    unknown = set(serve) - known
    if unknown:
        raise ConfigError(f"unknown serve keys: {', '.join(sorted(unknown))}")
    model_id = serve.get("id")
    if not isinstance(model_id, str) or not ID_RE.match(model_id):
        raise ConfigError("serve.id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    name = serve.get("name") or ""
    if not isinstance(name, str) or len(name) > 120 or any(c in name for c in "\n\r$"):
        raise ConfigError("serve.name must be a single line without '$' (max 120 characters)")
    out = {"id": model_id}
    if name:
        out["name"] = name
    out["model"] = _gguf(serve.get("model"), "model")
    if serve.get("mmproj"):
        out["mmproj"] = _gguf(serve["mmproj"], "mmproj")
    out["ctx_size"] = _int(serve, "ctx_size", 32768, 0, 1 << 21)
    out["parallel"] = _int(serve, "parallel", 1, 1, 64)
    out["mtp"] = _bool(serve, "mtp", False)
    if out["mtp"]:
        out["mtp_draft_n_max"] = _int(serve, "mtp_draft_n_max", 6, 1, 32)
    extra = serve.get("extra_args") or []
    if not isinstance(extra, list) or len(extra) > 40 or not all(isinstance(a, str) and ARG_RE.match(a) for a in extra):
        raise ConfigError("serve.extra_args: up to 40 tokens of [A-Za-z0-9_.,:=/+@-] (no spaces, quotes or '$')")
    out["extra_args"] = list(extra)
    out["ttl"] = _int(serve, "ttl", 600, -1, 7 * 86400)
    group = serve.get("group") or "on-demand"
    if not isinstance(group, str) or not ID_RE.match(group):
        raise ConfigError("serve.group must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
    out["group"] = group
    out["preload"] = _bool(serve, "preload", False)
    aliases = serve.get("aliases") or []
    if not isinstance(aliases, list) or len(aliases) > 10 or not all(isinstance(a, str) and ID_RE.match(a) for a in aliases):
        raise ConfigError("serve.aliases: up to 10 ids")
    out["aliases"] = list(aliases)
    out["enabled"] = _bool(serve, "enabled", True)
    out["unlisted"] = _bool(serve, "unlisted", False)
    return out


def is_pinned(entry: dict) -> bool:
    return bool(SHA_RE.match(str(entry.get("revision", ""))))
