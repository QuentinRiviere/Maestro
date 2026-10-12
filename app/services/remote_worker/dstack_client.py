"""Minimal dstack server REST client (validated against dstack 0.20.29).

Every route is ``POST`` with a JSON body and ``Authorization: Bearer``.  Run
routes live under ``/api/project/{project}/runs/*``.  Nothing here shells out
to the dstack CLI, so Maestro only needs the server URL, project and token.
"""
from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import requests


class DstackError(RuntimeError):
    def __init__(self, message: str, *, status: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status = status
        self.code = code


class DstackClient:
    def __init__(
        self,
        server_url: str,
        project: str,
        token: str,
        *,
        timeout: float = 30.0,
        session: requests.Session | None = None,
    ) -> None:
        self.server_url = server_url.rstrip("/")
        self.project = project
        self._token = token
        self._timeout = timeout
        self._session = session or requests.Session()

    # ── transport ──────────────────────────────────────────────────────────

    def _post(self, path: str, body: Mapping[str, Any] | None = None) -> Any:
        try:
            response = self._session.post(
                f"{self.server_url}{path}",
                json=dict(body or {}),
                headers={"Authorization": f"Bearer {self._token}"},
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            raise DstackError(f"dstack server unreachable at {self.server_url}: {exc}") from exc
        if response.status_code in (401, 403):
            raise DstackError("dstack rejected the token", status=response.status_code)
        if not response.ok:
            message, code = _error_detail(response)
            raise DstackError(
                f"dstack {path} failed: HTTP {response.status_code} {message}",
                status=response.status_code,
                code=code,
            )
        if not response.content:
            return None
        return response.json()

    def _project_path(self, suffix: str) -> str:
        return f"/api/project/{quote(self.project, safe='')}/{suffix}"

    # ── server / project ───────────────────────────────────────────────────

    def get_info(self) -> Mapping[str, Any]:
        return self._post("/api/server/get_info")

    def list_fleets(self) -> list[Mapping[str, Any]]:
        return self._post(self._project_path("fleets/list")) or []

    def list_volumes(self) -> list[Mapping[str, Any]]:
        return self._post(self._project_path("volumes/list")) or []

    def create_volume(self, *, name: str, region: str, size: str, backend: str = "runpod") -> Mapping[str, Any]:
        return self._post(self._project_path("volumes/create"), {
            "configuration": {
                "type": "volume",
                "backend": backend,
                "name": name,
                "region": region,
                "size": size,
                "auto_cleanup_duration": "off",
            },
        })

    # ── runs ───────────────────────────────────────────────────────────────

    def get_plan(self, run_spec: Mapping[str, Any], *, max_offers: int = 5) -> Mapping[str, Any]:
        return self._post(self._project_path("runs/get_plan"), {
            "run_spec": run_spec,
            "max_offers": max_offers,
        })

    def apply(self, run_spec: Mapping[str, Any]) -> Mapping[str, Any]:
        """Submit a new run. A finished run with the same name is replaced."""
        return self._post(self._project_path("runs/apply"), {
            "plan": {"run_spec": run_spec},
            "force": False,
        })

    def get_run(self, run_name: str) -> Mapping[str, Any] | None:
        try:
            return self._post(self._project_path("runs/get"), {"run_name": run_name})
        except DstackError as exc:
            if exc.code == "resource_not_exists":
                return None
            raise

    def stop_run(self, run_name: str, *, abort: bool = False) -> None:
        self._post(self._project_path("runs/stop"), {"runs_names": [run_name], "abort": abort})

    def poll_logs(
        self,
        run_name: str,
        job_submission_id: str,
        *,
        start_time: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, str]]:
        """Return decoded log lines after ``start_time`` (exclusive)."""
        payload = self._post(self._project_path("logs/poll"), {
            "run_name": run_name,
            "job_submission_id": job_submission_id,
            "start_time": start_time,
            "end_time": None,
            "descending": False,
            "next_token": None,
            "limit": limit,
            "diagnose": False,
        }) or {}
        lines: list[dict[str, str]] = []
        for item in payload.get("logs") or []:
            raw = item.get("message") or ""
            try:
                message = base64.b64decode(raw).decode("utf-8", errors="replace")
            except (ValueError, TypeError):
                message = str(raw)
            lines.append({
                "timestamp": str(item.get("timestamp") or ""),
                "source": str(item.get("log_source") or "stdout"),
                "message": message,
            })
        return lines

    def service_base_url(self, run: Mapping[str, Any]) -> str | None:
        """Absolute proxy URL of a service run (``run.service.url`` is relative)."""
        service = run.get("service") or {}
        url = service.get("url") if isinstance(service, Mapping) else None
        if not isinstance(url, str) or not url:
            return None
        if url.startswith("http://") or url.startswith("https://"):
            return url.rstrip("/") + "/"
        return f"{self.server_url}/{url.strip('/')}/"


def _error_detail(response: requests.Response) -> tuple[str, str | None]:
    try:
        payload = response.json()
    except ValueError:
        return response.text[:300], None
    detail = payload.get("detail") if isinstance(payload, Mapping) else None
    if isinstance(detail, list) and detail:
        first = detail[0] if isinstance(detail[0], Mapping) else {}
        return str(first.get("msg") or detail)[:300], first.get("code")
    return str(payload)[:300], None
