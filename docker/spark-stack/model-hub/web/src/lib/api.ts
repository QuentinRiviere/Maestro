// Thin fetch wrapper: same-origin cookie session, CSRF header on writes,
// JSON in and out. Mirrors the `api()` helper of the previous vanilla UI.

export class ApiError extends Error {
  status: number;
  constructor(message: string, status = 0) {
    super(message);
    this.status = status;
  }
}

export const LOGIN_REQUIRED = "login required";

type Listener = () => void;

const state = { csrf: "", onUnauthorized: null as Listener | null };

export function setCsrf(token: string) {
  state.csrf = token;
}

export function onUnauthorized(listener: Listener | null) {
  state.onUnauthorized = listener;
}

export interface ApiOptions {
  method?: "GET" | "POST" | "DELETE";
  body?: unknown;
  /** Do not treat 401 as a logout. */
  allow401?: boolean;
  /** Return the JSON body even on a non-2xx status. */
  allowError?: boolean;
}

export async function api<T = unknown>(path: string, opts: ApiOptions = {}): Promise<T> {
  const method = opts.method || "GET";
  const headers: Record<string, string> = {};
  if (method !== "GET") headers["X-CSRF-Token"] = state.csrf;
  const init: RequestInit = { method, credentials: "same-origin", headers };
  if (opts.body !== undefined) {
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(opts.body);
  }
  const res = await fetch(path, init);
  let data: unknown = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }
  if (res.status === 401 && !opts.allow401) {
    state.onUnauthorized?.();
    throw new ApiError(LOGIN_REQUIRED, 401);
  }
  if (!res.ok && !opts.allowError) {
    const d = data as { detail?: unknown; error?: unknown } | null;
    const detail = d && (d.detail ?? d.error);
    throw new ApiError(typeof detail === "string" ? detail : `${res.status} ${res.statusText}`, res.status);
  }
  return data as T;
}

// The hub's sessions live in its memory: a hub restart (or a new tab after
// one) would ask for the token again. The token stays in sessionStorage,
// per tab and gone when the tab closes, and the app logs in again with it.
const TOKEN_KEY = "model_hub_token";

export function rememberToken(token: string) {
  try {
    sessionStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* storage blocked: the session cookie still works until the tab closes */
  }
}

export function storedToken(): string {
  try {
    return sessionStorage.getItem(TOKEN_KEY) || "";
  } catch {
    return "";
  }
}

export function forgetToken() {
  try {
    sessionStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

/** POST /api/login with a token; true when the hub accepted it. */
export async function login(token: string): Promise<{ ok: boolean; detail?: string }> {
  const r = await api<{ authenticated?: boolean; detail?: string } | null>("/api/login", {
    method: "POST",
    body: { token },
    allow401: true,
    allowError: true,
  });
  if (!r || !r.authenticated) return { ok: false, detail: (r && r.detail) || "Login failed" };
  rememberToken(token);
  return { ok: true };
}

export const isLoginRequired = (err: unknown) => err instanceof ApiError && err.message === LOGIN_REQUIRED;

export const errorMessage = (err: unknown) =>
  err instanceof Error ? err.message : typeof err === "string" ? err : String(err);
