"""Token login, session cookies, CSRF tokens and login rate limiting.

* MODEL_HUB_TOKEN is required (the app refuses to start without it).
* POST /api/login compares the submitted token in constant time and sets an
  HttpOnly, SameSite=Strict session cookie. Sessions live in memory: a restart
  logs everyone out.
* Every mutating request (POST/PUT/DELETE) made with the cookie must carry the
  session's CSRF token in X-CSRF-Token. Scripts may instead send
  `Authorization: Bearer <MODEL_HUB_TOKEN>` (not ambient, so no CSRF token).
* Failed logins are limited per client address and globally.
"""
from __future__ import annotations

import hmac
import secrets
import threading
import time
from dataclasses import dataclass

COOKIE = "model_hub_session"
SESSION_TTL = 12 * 3600
MIN_TOKEN_LENGTH = 16


class TokenMissing(RuntimeError):
    pass


def check_token(token: str | None) -> str:
    token = (token or "").strip()
    if len(token) < MIN_TOKEN_LENGTH:
        raise TokenMissing(f"MODEL_HUB_TOKEN must be set (at least {MIN_TOKEN_LENGTH} characters); "
                           "generate one with: openssl rand -hex 24")
    return token


@dataclass
class Session:
    csrf: str
    expires: float


class Sessions:
    def __init__(self, token: str, ttl: float = SESSION_TTL):
        self._token = check_token(token).encode()
        self._ttl = ttl
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    def token_ok(self, candidate: str | None) -> bool:
        return hmac.compare_digest((candidate or "").encode(), self._token)

    def create(self) -> tuple[str, Session]:
        sid = secrets.token_urlsafe(32)
        session = Session(secrets.token_urlsafe(32), time.time() + self._ttl)
        with self._lock:
            now = time.time()
            for key in [k for k, s in self._sessions.items() if s.expires < now]:
                del self._sessions[key]
            self._sessions[sid] = session
        return sid, session

    def get(self, sid: str | None) -> Session | None:
        if not sid:
            return None
        with self._lock:
            session = self._sessions.get(sid)
            if session is None:
                return None
            if session.expires < time.time():
                del self._sessions[sid]
                return None
            return session

    def drop(self, sid: str | None) -> None:
        with self._lock:
            self._sessions.pop(sid or "", None)


class LoginLimiter:
    """At most `per_client` failures per client and `global_limit` overall per window."""

    def __init__(self, per_client: int = 5, global_limit: int = 30, window: float = 300.0):
        self.per_client, self.global_limit, self.window = per_client, global_limit, window
        self._failures: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _recent(self, key: str, now: float) -> list[float]:
        recent = [t for t in self._failures.get(key, []) if now - t < self.window]
        self._failures[key] = recent
        return recent

    def blocked(self, client: str) -> bool:
        now = time.time()
        with self._lock:
            return (len(self._recent(client, now)) >= self.per_client
                    or len(self._recent("*", now)) >= self.global_limit)

    def failed(self, client: str) -> None:
        now = time.time()
        with self._lock:
            self._recent(client, now).append(now)
            self._recent("*", now).append(now)

    def reset(self, client: str) -> None:
        with self._lock:
            self._failures.pop(client, None)
