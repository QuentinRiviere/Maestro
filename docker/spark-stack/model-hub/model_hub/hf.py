"""Minimal Hugging Face Hub client (stdlib only).

Only https://huggingface.co and its *.hf.co / *.huggingface.co download hosts
are reachable: a redirect anywhere else is refused. HF_TOKEN, when set, is only
sent to huggingface.co itself and stripped when a redirect leaves it.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request

from .paths import HUB, allowed_host, check_repo_id, check_revision

USER_AGENT = "spark-model-hub/1"


class HubError(RuntimeError):
    """The Hub refused a request or answered something unusable."""


class _SafeRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not allowed_host(newurl):
            raise HubError(f"refused redirect to a non Hugging Face host: {urllib.parse.urlparse(newurl).hostname}")
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlparse(newurl).hostname != "huggingface.co":
            new.remove_header("Authorization")
        return new


OPENER = urllib.request.build_opener(_SafeRedirect)


def _token() -> str:
    return os.environ.get("HF_TOKEN", "").strip()


def request(url: str, headers: dict | None = None) -> urllib.request.Request:
    if not allowed_host(url):
        raise HubError(f"refused non Hugging Face URL: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    token = _token()
    if token and urllib.parse.urlparse(url).hostname == "huggingface.co":
        req.add_header("Authorization", f"Bearer {token}")
    return req


def get_json(url: str, timeout: float = 60):
    try:
        with OPENER.open(request(url), timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        raise HubError(f"Hub answered {exc.code} for {url.split('?')[0]}") from exc
    except urllib.error.URLError as exc:
        raise HubError(f"Hub unreachable: {exc.reason}") from exc


def get_text(url: str, timeout: float = 30, limit: int = 512 * 1024) -> str:
    try:
        with OPENER.open(request(url), timeout=timeout) as resp:
            return resp.read(limit).decode("utf-8", "replace")
    except (urllib.error.URLError, HubError):
        return ""


def repo_info(repo: str, revision: str = "main") -> dict:
    """Model info at a revision, with file sizes and LFS hashes."""
    check_repo_id(repo)
    check_revision(revision)
    rev = urllib.parse.quote(revision, safe="")
    info = get_json(f"{HUB}/api/models/{repo}/revision/{rev}?blobs=true")
    if not isinstance(info, dict) or not isinstance(info.get("sha"), str):
        raise HubError(f"unexpected answer for {repo}@{revision}")
    return info


def readme(repo: str, commit: str) -> str:
    check_repo_id(repo)
    check_revision(commit)
    return get_text(f"{HUB}/{repo}/raw/{urllib.parse.quote(commit, safe='')}/README.md")


SORTS = ("downloads", "trendingScore", "likes", "lastModified")


def search(query: str = "", sort: str = "downloads", limit: int = 30, author: str | None = None) -> list[dict]:
    if sort not in SORTS:
        raise HubError(f"unknown sort: {sort}")
    params = {"filter": "gguf", "sort": sort, "direction": "-1", "limit": str(max(1, min(int(limit), 100))),
              "full": "false"}
    if query:
        params["search"] = query[:100]
    if author:
        params["author"] = author
    data = get_json(f"{HUB}/api/models?{urllib.parse.urlencode(params)}")
    if not isinstance(data, list):
        raise HubError("unexpected search answer")
    out = []
    for item in data:
        repo = item.get("id") or item.get("modelId")
        if not isinstance(repo, str):
            continue
        out.append({
            "repo": repo,
            "author": repo.split("/")[0],
            "downloads": item.get("downloads", 0),
            "likes": item.get("likes", 0),
            "trending": item.get("trendingScore"),
            "last_modified": item.get("lastModified") or item.get("createdAt"),
            "tags": [t for t in item.get("tags", []) if isinstance(t, str)][:30],
            "pipeline": item.get("pipeline_tag"),
            "gated": bool(item.get("gated")),
        })
    return out
