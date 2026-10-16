"""Spark model hub: browse, download (verified) and map GGUF models into llama-swap.

Run: uvicorn --factory model_hub.app:main_app --host 0.0.0.0 --port 8090 (see the Dockerfile).
The app only writes to MODEL_HUB_MODELS_DIR and MODEL_HUB_CONFIG_DIR. It never
talks to Docker: llama-swap picks the rendered config up by watching it.
"""
from __future__ import annotations

import hmac
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import yaml
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import auth, catalog, fetcher, ggufs, hf, registry, render
from .paths import SHA_RE, UnsafePathError, check_filename, check_repo_id, check_revision

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
DEFAULT_TRUSTED = ("ggml-org,unsloth,bartowski,Qwen,huihui-ai,google,openai,mistralai,microsoft,nvidia,"
                   "lmstudio-community,meta-llama")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name) or default)
    except ValueError:
        return default


@dataclass
class Settings:
    token: str
    models_dir: str = "/models"
    config_dir: str = "/config"
    catalog_path: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "catalog.yml")
    trusted_authors: list[str] = field(default_factory=lambda: DEFAULT_TRUSTED.split(","))
    memory_gb: float = 128.0
    reserved_gb: float = 10.0
    default_group: str = "on-demand"
    llama_swap_url: str = ""
    llama_swap_api_key: str = ""
    cookie_secure: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ
        return cls(
            token=env.get("MODEL_HUB_TOKEN", ""),
            models_dir=env.get("MODEL_HUB_MODELS_DIR", "/models"),
            config_dir=env.get("MODEL_HUB_CONFIG_DIR", "/config"),
            catalog_path=env.get("MODEL_HUB_CATALOG", cls.catalog_path),
            trusted_authors=[a.strip() for a in (env.get("MODEL_HUB_TRUSTED_AUTHORS") or DEFAULT_TRUSTED).split(",")
                             if a.strip()],
            memory_gb=_env_float("MODEL_HUB_MEMORY_GB", 128.0),
            reserved_gb=_env_float("MODEL_HUB_RESERVED_GB", 10.0),
            default_group=env.get("MODEL_HUB_DEFAULT_GROUP") or "on-demand",
            llama_swap_url=env.get("LLAMA_SWAP_URL", "").rstrip("/"),
            llama_swap_api_key=env.get("LLAMA_SWAP_API_KEY", ""),
            cookie_secure=env.get("MODEL_HUB_COOKIE_SECURE", "0").lower() in ("1", "true", "yes"),
        )

    @property
    def models_yml(self) -> str:
        return os.path.join(self.config_dir, "models.yml")

    @property
    def base_yaml(self) -> str:
        return os.path.join(self.config_dir, "llama-swap.base.yaml")

    @property
    def rendered_yaml(self) -> str:
        return os.path.join(self.config_dir, "llama-swap.yaml")


class LoginBody(BaseModel):
    token: str = Field(max_length=512)


class DownloadBody(BaseModel):
    repo: str
    revision: str
    files: list[str] = Field(min_length=1, max_length=64)
    serve: dict | None = None


class ServeBody(BaseModel):
    repo: str
    serve: dict | None = None


class VerifyBody(BaseModel):
    repo: str


class Hub:
    """State shared by the routes."""

    def __init__(self, settings: Settings):
        self.s = settings
        self.sessions = auth.Sessions(settings.token)
        self.limiter = auth.LoginLimiter()
        os.makedirs(settings.models_dir, exist_ok=True)
        self.store = fetcher.Store(settings.models_dir)
        self.queue = fetcher_queue(self)
        self._snap_cache: dict[tuple[str, str], tuple[float, fetcher.Snapshot, str]] = {}
        self._verify_lock = threading.Lock()
        self.verify_state: dict[str, dict] = {}

    # -- files / config ---------------------------------------------------
    def available(self, repo: str, name: str) -> bool:
        try:
            return os.path.exists(self.store.target(repo, name))
        except UnsafePathError:
            return False

    def entries(self) -> list[dict]:
        return registry.load(self.s.models_yml)

    def base(self) -> dict:
        if not os.path.exists(self.s.base_yaml):
            raise HTTPException(500, f"missing {self.s.base_yaml}")
        return render.load_base(self.s.base_yaml)

    def render_entries(self, entries: list[dict], write: bool) -> dict:
        config, warnings = render.render(self.base(), entries, self.available)
        text = render.dump(config)
        old = render.read_text(self.s.rendered_yaml)
        changed = old != text
        if write and changed:
            render.write(self.s.rendered_yaml, text)
        return {"changed": changed, "warnings": warnings, "diff": render.diff(old, text), "text": text,
                "mapped": sorted(config.get("models") or {})}

    def snapshot(self, repo: str, revision: str) -> tuple[fetcher.Snapshot, str]:
        key = (repo, revision)
        hit = self._snap_cache.get(key)
        if hit and time.time() - hit[0] < 300:
            return hit[1], hit[2]
        snap = fetcher.resolve(repo, revision)
        readme = hf.readme(repo, snap.commit)
        self._snap_cache[key] = (time.time(), snap, readme)
        if len(self._snap_cache) > 200:
            self._snap_cache.pop(next(iter(self._snap_cache)))
        return snap, readme

    # -- memory -------------------------------------------------------------
    def budget(self, exclude_repo: str | None = None) -> ggufs.Budget:
        """Total minus reserved minus models that stay loaded (resident)."""
        budget = ggufs.Budget(self.s.memory_gb, self.s.reserved_gb)
        try:
            groups = render.base_groups(self.base())
            entries = self.entries()
        except Exception:  # noqa: BLE001 - budget is advisory
            return budget
        manifest = self.store.load_manifest()
        for entry in entries:
            serve = entry.get("serve")
            if not serve or entry["repo"] == exclude_repo or not serve.get("enabled", True):
                continue
            group = groups.get(serve.get("group")) or {}
            if not (group.get("persistent") or serve.get("preload")):
                continue
            size = lambda n: int((manifest.get(f"{entry['repo']}/{n}") or {}).get("size", 0))  # noqa: E731
            est = ggufs.estimate_memory(size(serve["model"]), serve.get("ctx_size", 0),
                                        size(serve["mmproj"]) if serve.get("mmproj") else 0)
            budget.resident_gb += est["total"] / ggufs.GB
            budget.resident.append(serve["id"])
        return budget


def fetcher_queue(hub: "Hub"):
    from .jobs import Queue

    def on_done(job) -> str:
        return hub_record_download(hub, job.repo, job.commit, [f.name for f in job.files], job.serve)

    return Queue(hub.store, on_done)


def hub_record_download(hub: Hub, repo: str, commit: str, names: list[str], serve: dict | None) -> str:
    """Add the downloaded files to models.yml, then re-render llama-swap.yaml."""
    with registry.LOCK:
        entries = hub.entries()
        entry = registry.find(entries, repo)
        globs = [fetcher.glob_escape(n) for n in names]
        if entry is None:
            entry = {"repo": repo, "revision": commit, "include": globs}
            entries.append(entry)
        elif entry.get("revision") != commit:
            entry["revision"], entry["include"] = commit, globs
            entry.pop("exclude", None)
        else:
            include = list(entry.get("include") or [])
            entry["include"] = include + [g for g in globs if g not in include]
        if serve:
            entry["serve"] = registry.normalize_serve(serve)
        registry.save(hub.s.models_yml, entries)
        result = hub.render_entries(entries, write=True)
    msg = f"verified {len(names)} file(s)"
    if serve:
        msg += f"; mapped as {entry['serve']['id']}" if entry["serve"]["id"] in result["mapped"] else "; not mapped"
    return msg + (" (llama-swap.yaml updated)" if result["changed"] else "")


def _bad(exc: Exception, code: int = 400) -> HTTPException:
    return HTTPException(code, str(exc))


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    hub = Hub(settings)  # raises TokenMissing without a token
    app = FastAPI(title="Spark model hub", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.hub = hub

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    def require_auth(request: Request) -> None:
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            if hub.sessions.token_ok(header[7:].strip()):
                return
            raise HTTPException(401, "invalid token")
        session = hub.sessions.get(request.cookies.get(auth.COOKIE))
        if session is None:
            raise HTTPException(401, "login required")
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            sent = request.headers.get("x-csrf-token", "")
            if not hmac.compare_digest(sent.encode(), session.csrf.encode()):
                raise HTTPException(403, "missing or invalid CSRF token")

    public = APIRouter(prefix="/api")
    api = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @public.get("/session")
    def session_info(request: Request):
        session = hub.sessions.get(request.cookies.get(auth.COOKIE))
        if session is None:
            return JSONResponse({"authenticated": False}, status_code=401)
        return {"authenticated": True, "csrf": session.csrf}

    @public.post("/login")
    def login(body: LoginBody, request: Request, response: Response):
        client = request.client.host if request.client else "?"
        if hub.limiter.blocked(client):
            raise HTTPException(429, "too many failed logins; wait a few minutes")
        if not hub.sessions.token_ok(body.token):
            hub.limiter.failed(client)
            raise HTTPException(401, "invalid token")
        hub.limiter.reset(client)
        sid, session = hub.sessions.create()
        response.set_cookie(auth.COOKIE, sid, max_age=auth.SESSION_TTL, httponly=True, samesite="strict",
                            secure=settings.cookie_secure, path="/")
        return {"authenticated": True, "csrf": session.csrf}

    @api.post("/logout")
    def logout(request: Request, response: Response):
        hub.sessions.drop(request.cookies.get(auth.COOKIE))
        response.delete_cookie(auth.COOKIE, path="/")
        return {"ok": True}

    # -- overview -----------------------------------------------------------
    @api.get("/status")
    def status():
        usage = shutil.disk_usage(settings.models_dir)
        try:
            groups = sorted(render.base_groups(hub.base()))
            base_error = ""
        except HTTPException as exc:
            groups, base_error = [], str(exc.detail)
        except Exception as exc:  # noqa: BLE001
            groups, base_error = [], str(exc)
        return {"disk": {"free": usage.free, "total": usage.total}, "budget": hub.budget().as_dict(),
                "groups": groups, "default_group": settings.default_group, "base_error": base_error,
                "trusted_authors": settings.trusted_authors, "hf_token": bool(os.environ.get("HF_TOKEN")),
                "llama_swap": bool(settings.llama_swap_url)}

    @api.get("/catalog")
    def get_catalog():
        try:
            items = catalog.load(settings.catalog_path, settings.default_group)
        except (OSError, ValueError) as exc:
            raise _bad(exc, 500)
        installed = hub.store.installed()
        entries = {e["repo"]: e for e in hub.entries()}
        budget = hub.budget()
        for item in items:
            files = installed.get(item["repo"], [])
            item["installed"] = bool(files) and all(f["present"] for f in files)
            item["in_models_yml"] = item["repo"] in entries
            serve = item.get("serve") or {}
            est = ggufs.estimate_memory(int(item["size_gb"] * ggufs.GB), serve.get("ctx_size", 32768))
            item["memory_gb"] = round(est["total"] / ggufs.GB, 1)
            item["fit"] = ggufs.fit_badge(est["total"], budget)
        return {"items": items, "budget": budget.as_dict()}

    # -- explore ------------------------------------------------------------
    @api.get("/search")
    def search(q: str = "", sort: str = "downloads", limit: int = 30, all: bool = False):
        limit = max(1, min(limit, 100))
        trusted = {a.lower() for a in settings.trusted_authors}
        try:
            if all:
                results = hf.search(q, sort, limit)
            else:
                with ThreadPoolExecutor(max_workers=6) as pool:
                    batches = list(pool.map(lambda a: hf.search(q, sort, limit, author=a), settings.trusted_authors))
                key = {"downloads": "downloads", "likes": "likes", "trendingScore": "trending",
                       "lastModified": "last_modified"}.get(sort, "downloads")
                results = sorted((r for b in batches for r in b), key=lambda r: r.get(key) or 0, reverse=True)[:limit]
        except hf.HubError as exc:
            raise _bad(exc, 502)
        for r in results:
            r["trusted"] = r["author"].lower() in trusted
        return {"results": results, "trusted_only": not all}

    @api.get("/repo/{owner}/{name}")
    def repo_detail(owner: str, name: str, revision: str = "main", ctx: int = 32768):
        repo = f"{owner}/{name}"
        try:
            check_repo_id(repo)
            check_revision(revision)
            snap, readme = hub.snapshot(repo, revision)
        except (UnsafePathError, hf.HubError, fetcher.FetchError) as exc:
            raise _bad(exc, 400 if isinstance(exc, UnsafePathError) else 502)
        ctx = max(0, min(ctx, 1 << 21))
        groups = ggufs.group_files([{"name": f.name, "size": f.size} for f in snap.files])
        mmprojs = [g for g in groups if g.kind == "mmproj"]
        default_mmproj = next((g for g in mmprojs if "f16" in g.key.lower() and "bf16" not in g.key.lower()),
                              mmprojs[0] if mmprojs else None)
        budget = hub.budget(exclude_repo=repo)
        tags = [t for t in snap.info.get("tags", []) if isinstance(t, str)]
        out = []
        for g in groups:
            item = g.as_dict()
            if g.kind == "model":
                est = ggufs.estimate_memory(g.size, ctx, default_mmproj.size if default_mmproj else 0)
                item["memory"] = {k: round(v / ggufs.GB, 2) if k != "kv_per_token" else v for k, v in est.items()}
                item["fit"] = ggufs.fit_badge(est["total"], budget)
            item["installed"] = all(hub.available(repo, f) for f in g.files)
            out.append(item)
        refused = sorted(f.name for f in snap.files if fetcher.is_refused(f.name))
        author = owner.lower()
        return {
            "repo": repo, "commit": snap.commit, "revision": revision, "pinned": bool(SHA_RE.match(revision)),
            "last_modified": snap.info.get("lastModified"), "gated": bool(snap.info.get("gated")),
            "license": (snap.info.get("cardData") or {}).get("license"), "tags": tags[:40],
            "downloads": snap.info.get("downloads"), "likes": snap.info.get("likes"),
            "trusted": author in {a.lower() for a in settings.trusted_authors},
            "ggufs": out, "default_mmproj": default_mmproj.first if default_mmproj else None,
            "mtp": ggufs.detect_mtp(repo, tags, readme), "refused": refused, "ctx": ctx,
            "budget": budget.as_dict(), "in_models_yml": registry.find(hub.entries(), repo) is not None,
        }

    # -- downloads ------------------------------------------------------------
    @api.post("/downloads")
    def start_download(body: DownloadBody):
        try:
            repo = check_repo_id(body.repo)
            if not SHA_RE.match(body.revision):
                raise UnsafePathError("revision must be the 40-hex commit shown by the repo view")
            for n in body.files:
                check_filename(n)
            snap, _ = hub.snapshot(repo, body.revision)
        except (UnsafePathError, hf.HubError, fetcher.FetchError) as exc:
            raise _bad(exc, 400 if isinstance(exc, UnsafePathError) else 502)
        by_name = snap.by_name()
        wanted: list[fetcher.RemoteFile] = []
        for n in body.files:
            f = by_name.get(n)
            if f is None:
                raise HTTPException(400, f"not in {repo}@{snap.commit[:12]}: {n}")
            if not fetcher.is_allowed(n) or not n.lower().endswith(".gguf"):
                raise HTTPException(400, f"refused file type: {n}")
            if f not in wanted:
                wanted.append(f)
        # Complete split models: asking for one part downloads every part.
        for g in ggufs.group_files([{"name": f.name, "size": f.size} for f in snap.files]):
            if any(f.name in g.files for f in wanted):
                if not g.complete:
                    raise HTTPException(400, f"split model {g.key} is incomplete on the Hub")
                wanted += [by_name[n] for n in g.files if by_name[n] not in wanted]
        serve = None
        if body.serve:
            try:
                serve = registry.normalize_serve(body.serve)
                names = {f.name for f in wanted}
                for key in ("model", "mmproj"):
                    if serve.get(key) and serve[key] not in names and not hub.available(repo, serve[key]):
                        raise registry.ConfigError(f"serve.{key} {serve[key]} is not part of this download")
                _preview_with(hub, repo, serve, assume_available=names)
            except (registry.ConfigError, render.RenderError) as exc:
                raise _bad(exc)
        try:
            hub.store.check_disk(repo, wanted)
            job = hub.queue.submit(repo, snap.commit, wanted, serve)
        except (fetcher.FetchError, ValueError) as exc:
            raise _bad(exc, 409)
        return job.as_dict()

    @api.get("/downloads")
    def list_downloads():
        return {"jobs": hub.queue.list()}

    @api.post("/downloads/{job_id}/cancel")
    def cancel_download(job_id: str):
        try:
            return hub.queue.cancel(job_id).as_dict()
        except KeyError:
            raise HTTPException(404, "no such job")

    @api.post("/downloads/{job_id}/resume")
    def resume_download(job_id: str):
        try:
            return hub.queue.resume(job_id).as_dict()
        except KeyError:
            raise HTTPException(404, "no such job")
        except (ValueError, fetcher.FetchError) as exc:
            raise _bad(exc, 409)

    # -- installed --------------------------------------------------------------
    @api.get("/installed")
    def installed():
        entries = {e["repo"]: e for e in hub.entries()}
        rendered = render.read_text(settings.rendered_yaml)
        try:
            mapped = set(((yaml.safe_load(rendered) or {}).get("models") or {}))
        except Exception:  # noqa: BLE001
            mapped = set()
        repos = hub.store.installed()
        out = []
        for repo in sorted(set(repos) | set(entries)):
            files = repos.get(repo, [])
            entry = entries.get(repo) or {}
            serve = entry.get("serve")
            out.append({
                "repo": repo, "files": files, "bytes": sum(f["size"] for f in files if f["present"]),
                "commit": next((f["commit"] for f in files), entry.get("revision")),
                "in_models_yml": bool(entry), "serve": serve,
                "mapped": bool(serve and serve["id"] in mapped),
                "verify": hub.verify_state.get(repo),
                "downloading": repo in hub.queue.active_repos(),
            })
        usage = shutil.disk_usage(settings.models_dir)
        return {"repos": out, "disk": {"free": usage.free, "total": usage.total}}

    @api.post("/installed/verify")
    def verify(body: VerifyBody):
        try:
            repo = check_repo_id(body.repo)
        except UnsafePathError as exc:
            raise _bad(exc)
        files = hub.store.installed().get(repo)
        if not files:
            raise HTTPException(404, "not installed")
        if not hub._verify_lock.acquire(blocking=False):
            raise HTTPException(409, "a verification is already running")

        def run():
            try:
                hub.verify_state[repo] = {"state": "running", "started": time.time()}
                results = {f["name"]: hub.store.verify(repo, f["name"]) for f in files}
                hub.verify_state[repo] = {"state": "ok" if all(results.values()) else "mismatch",
                                          "files": results, "finished": time.time()}
            except Exception as exc:  # noqa: BLE001
                hub.verify_state[repo] = {"state": "error", "error": str(exc)}
            finally:
                hub._verify_lock.release()

        threading.Thread(target=run, daemon=True).start()
        return {"state": "running"}

    @api.delete("/installed/{owner}/{name}")
    def delete(owner: str, name: str, confirm: str = ""):
        repo = f"{owner}/{name}"
        try:
            check_repo_id(repo)
        except UnsafePathError as exc:
            raise _bad(exc)
        if confirm != repo:
            raise HTTPException(400, "confirmation must repeat the repo id")
        if repo in hub.queue.active_repos():
            raise HTTPException(409, "cancel its download first")
        with registry.LOCK:
            entries = [e for e in hub.entries() if e["repo"] != repo]
            try:
                result = hub.render_entries(entries, write=False)
            except (render.RenderError, registry.ConfigError) as exc:
                raise _bad(exc, 409)
            registry.save(settings.models_yml, entries)
            render.write(settings.rendered_yaml, result["text"])
        freed = hub.store.delete_repo(repo)
        return {"deleted": repo, "freed": freed, "config_changed": result["changed"]}

    # -- serve settings / llama-swap config -----------------------------------
    @api.get("/config")
    def get_config():
        try:
            preview = hub.render_entries(hub.entries(), write=False)
            error = ""
        except (render.RenderError, registry.ConfigError) as exc:
            preview, error = None, str(exc)
        return {"rendered": render.read_text(settings.rendered_yaml), "pending_diff": preview and preview["diff"],
                "warnings": preview and preview["warnings"], "error": error,
                "models_yml": render.read_text(settings.models_yml)}

    @api.post("/serve/preview")
    def serve_preview(body: ServeBody):
        try:
            serve = registry.normalize_serve(body.serve) if body.serve else None
            # Preview as if the files were installed (it often runs before the download).
            assumed = {serve["model"], serve.get("mmproj")} - {None} if serve else set()
            return _preview_with(hub, check_repo_id(body.repo), serve, assume_available=assumed)
        except (UnsafePathError, registry.ConfigError, render.RenderError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

    @api.post("/serve")
    def serve_save(body: ServeBody):
        try:
            repo = check_repo_id(body.repo)
            serve = registry.normalize_serve(body.serve) if body.serve else None
            with registry.LOCK:
                entries = hub.entries()
                entry = registry.find(entries, repo)
                if entry is None:
                    raise registry.ConfigError(f"{repo} is not in models.yml; download it first")
                if serve:
                    entry["serve"] = serve
                else:
                    entry.pop("serve", None)
                result = hub.render_entries(entries, write=False)
                registry.save(settings.models_yml, entries)
                if result["changed"]:
                    render.write(settings.rendered_yaml, result["text"])
        except (UnsafePathError, registry.ConfigError, render.RenderError) as exc:
            raise _bad(exc)
        return {"changed": result["changed"], "warnings": result["warnings"], "diff": result["diff"]}

    @api.post("/render")
    def rerender():
        try:
            result = hub.render_entries(hub.entries(), write=True)
        except (render.RenderError, registry.ConfigError) as exc:
            raise _bad(exc, 409)
        return {"changed": result["changed"], "warnings": result["warnings"], "diff": result["diff"]}

    @api.get("/llama-swap/running")
    def running():
        if not settings.llama_swap_url:
            return {"reachable": False, "error": "LLAMA_SWAP_URL not set"}
        req = urllib.request.Request(settings.llama_swap_url + "/running")
        if settings.llama_swap_api_key:
            req.add_header("Authorization", f"Bearer {settings.llama_swap_api_key}")
        try:
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read(1 << 20))
        except (urllib.error.URLError, OSError, ValueError) as exc:
            return {"reachable": False, "error": str(getattr(exc, "reason", exc))}
        running_models = [{"model": m.get("model"), "state": m.get("state"), "name": m.get("name"), "ttl": m.get("ttl")}
                          for m in (data.get("running") or []) if isinstance(m, dict)]
        return {"reachable": True, "running": running_models}

    app.include_router(public)
    app.include_router(api)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(STATIC_DIR, "index.html"))

    # Render once at startup so llama-swap (which waits for this service to be
    # healthy) always finds a config.
    for name in registry.seed_config(settings.config_dir):
        print(f"model hub: created {name} from its example", flush=True)
    try:
        if os.path.exists(settings.base_yaml):
            hub.render_entries(hub.entries(), write=True)
        else:
            print(f"model hub: {settings.base_yaml} is missing, nothing rendered", flush=True)
    except (render.RenderError, registry.ConfigError) as exc:
        print(f"model hub: initial render failed: {exc}", flush=True)
    return app


def _preview_with(hub: Hub, repo: str, serve: dict | None, assume_available: set[str] | None = None) -> dict:
    """Render as if `serve` were saved for repo (nothing is written)."""
    entries = hub.entries()
    entry = registry.find(entries, repo)
    if entry is None:
        entry = {"repo": repo, "revision": "main", "include": []}
        entries.append(entry)
    if serve:
        entry["serve"] = serve
    else:
        entry.pop("serve", None)
    assumed = assume_available or set()
    original = hub.available
    available = lambda r, n: (r == repo and n in assumed) or original(r, n)  # noqa: E731
    config, warnings = render.render(hub.base(), entries, available)
    text = render.dump(config)
    old = render.read_text(hub.s.rendered_yaml)
    return {"diff": render.diff(old, text), "warnings": warnings, "changed": old != text}


def main_app() -> FastAPI:
    """uvicorn --factory entry point: refuses to start without MODEL_HUB_TOKEN."""
    try:
        return create_app()
    except auth.TokenMissing as exc:
        raise SystemExit(f"model hub: {exc}")
