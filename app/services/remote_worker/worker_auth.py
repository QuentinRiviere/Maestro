"""Bearer token in front of a worker that is reachable without dstack.

On dstack, the service proxy authenticates every request before it reaches
the worker. A static worker (a DGX Spark on the LAN or tailnet) has no such
proxy: with ``MAESTRO_WORKER_TOKEN`` set, every request must carry
``Authorization: Bearer <token>``, except the liveness and readiness probes,
which return nothing sensitive. Without the variable nothing changes.
"""
from __future__ import annotations

import hmac
import json
import os
from collections.abc import Mapping

TOKEN_ENV = "MAESTRO_WORKER_TOKEN"
# Unauthenticated probes (llama-swap's checkEndpoint, container health checks).
LIVENESS_PATH = "/healthz"
READINESS_PATH = "/readyz"
EXEMPT_PATHS = frozenset({LIVENESS_PATH, READINESS_PATH})


def worker_token_from_env(environ: Mapping[str, str] | None = None) -> str:
    env = os.environ if environ is None else environ
    return (env.get(TOKEN_ENV) or "").strip()


def authorized(header: str | None, token: str) -> bool:
    """Whether an ``Authorization`` header carries ``token`` (constant-time compare)."""
    if not header or not token:
        return False
    scheme, _, credentials = header.strip().partition(" ")
    if scheme.lower() != "bearer":
        return False
    return hmac.compare_digest(credentials.strip().encode("utf-8"), token.encode("utf-8"))


class BearerTokenMiddleware:
    """ASGI middleware: 401 for any HTTP or WebSocket request without the token."""

    def __init__(self, app, token: str, exempt_paths: frozenset[str] = EXEMPT_PATHS) -> None:
        if not token:
            raise ValueError("BearerTokenMiddleware needs a token")
        self.app = app
        self._token = token
        self._exempt = exempt_paths

    async def __call__(self, scope, receive, send):
        if scope.get("type") not in ("http", "websocket") or scope.get("path") in self._exempt:
            await self.app(scope, receive, send)
            return
        header = None
        for name, value in scope.get("headers") or []:
            if name == b"authorization":
                header = value.decode("latin-1")
                break
        if authorized(header, self._token):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        body = json.dumps({"detail": "Unauthorized"}).encode()
        await send({
            "type": "http.response.start",
            "status": 401,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
                (b"www-authenticate", b"Bearer"),
            ],
        })
        await send({"type": "http.response.body", "body": body})
