"""catalog.yml: the hand-curated model list."""
from __future__ import annotations

import yaml

from . import registry
from .paths import SHA_RE, check_repo_id

CATEGORIES = ("general", "abliterated", "vision", "code", "small")


class CatalogError(ValueError):
    pass


def load(path: str, default_group: str = "on-demand") -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or []
    if not isinstance(data, list):
        raise CatalogError(f"{path}: expected a list")
    out, seen = [], set()
    for item in data:
        if not isinstance(item, dict):
            raise CatalogError("catalog entries must be mappings")
        repo = check_repo_id(item.get("repo"))
        if repo in seen:
            raise CatalogError(f"duplicate catalog repo: {repo}")
        seen.add(repo)
        if not SHA_RE.match(str(item.get("revision", ""))):
            raise CatalogError(f"{repo}: revision must be a pinned 40-hex commit")
        if item.get("category") not in CATEGORIES:
            raise CatalogError(f"{repo}: category must be one of {', '.join(CATEGORIES)}")
        include = item.get("include")
        if not isinstance(include, list) or not include or not all(isinstance(p, str) for p in include):
            raise CatalogError(f"{repo}: include must be a non-empty list of globs")
        serve = dict(item.get("serve") or {})
        if serve:
            serve.setdefault("group", default_group)
            serve = registry.normalize_serve(serve)
        out.append({
            "name": str(item.get("name") or repo),
            "repo": repo,
            "revision": item["revision"],
            "category": item["category"],
            "include": include,
            "size_gb": float(item.get("size_gb") or 0),
            "notes": str(item.get("notes") or ""),
            "serve": serve or None,
        })
    return out
