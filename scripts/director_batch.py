#!/usr/bin/env python3
"""Run a Director payload several times in a row and review each run: the nightly batch.

    python3 scripts/director_batch.py --payload klove.json --runs 4 [--host http://localhost:7860]
                                      [--seed-step 1] [--until 07:00] [--review-takes 1]

Each run goes through the controller's HTTP API: start, wait (any review pause
is passed as is), review (storyboard + video judges, the local ones when the
controller is set up that way), then the next run. The insights registry
records every run on its own. The batch stops at ``--until`` (local HH:MM) or
after ``--runs`` runs, whichever comes first, and never interrupts a run.
Exit code 1 when a run failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
import urllib.error
import urllib.request

POLL_SECONDS = 20
RUN_TIMEOUT_SECONDS = 4 * 3600
REVIEW_TIMEOUT_SECONDS = 1800
TERMINAL = {"completed", "failed", "stopped", "cancelled"}


def _call(host: str, path: str, body: dict | None = None, timeout: float = 60) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(f"{host.rstrip('/')}{path}", data=data,
                                     headers={"content-type": "application/json"} if data else {},
                                     method="POST" if data is not None else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _state(host: str, pid: str) -> dict:
    try:
        return _call(host, f"/api/v1/director/pipeline/{pid}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return {}


def wait_for_run(host: str, pid: str, log) -> str:
    """Poll one run to its end, passing every review pause; returns its final status."""
    deadline = time.monotonic() + RUN_TIMEOUT_SECONDS
    last = ""
    while time.monotonic() < deadline:
        state = _state(host, pid)
        status = str(state.get("status") or "")
        line = f"{status} | {state.get('phase')} | {(state.get('progress') or {}).get('message') or ''}"
        if line != last:
            log(f"  {line}")
            last = line
        if status == "paused":
            _call(host, f"/api/v1/director/pipeline/{pid}/continue", {})
            log(f"  passed the {state.get('pause_reason') or 'review'} pause")
            time.sleep(5)
            continue
        if status in TERMINAL:
            return status
        time.sleep(POLL_SECONDS)
    return "timeout"


def review_run(host: str, pid: str, takes: int, log) -> dict:
    """Run the video review and wait for it; returns the review record ({} on failure)."""
    _call(host, f"/api/v1/director/pipeline/{pid}/review", {"takes": takes} if takes > 1 else {})
    deadline = time.monotonic() + REVIEW_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        review = _state(host, pid).get("video_review") or {}
        if review.get("status") in ("completed", "failed"):
            return review
        time.sleep(10)
    log("  review timed out")
    return {}


def guard_skill(host: str, skill: str, min_runs: int, margin: float) -> dict:
    """Ask the controller whether the skill's current version lost to an earlier one, and apply the rollback."""
    skill_id = skill.removeprefix("custom:")
    try:
        return _call(host, f"/api/v1/director/skills/{skill_id}/rollback",
                     {"apply": True, "min_runs": min_runs, "margin": margin})
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        return {"reason": f"guard failed: {error}"}


def summarise(review: dict) -> str:
    episode = review.get("episode") or {}
    shots = ", ".join(str(s.get("score")) for s in review.get("shots") or [])
    return (f"clips [{shots}] episode {episode.get('score')} {episode.get('verdict') or ''} "
            f"(${review.get('cost_usd') or 0}, {review.get('backend') or 'fal'})")


def past(until: str | None) -> bool:
    if not until:
        return False
    hour, minute = (int(part) for part in until.split(":", 1))
    now = dt.datetime.now()
    limit = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    # A limit earlier in the day than the start means tomorrow morning.
    return now >= limit if limit > _START else now >= limit + dt.timedelta(days=1)


_START = dt.datetime.now()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payload", required=True, help="JSON body of POST /api/v1/director/pipeline/start")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--host", default="http://localhost:7860")
    parser.add_argument("--until", help="stop starting runs at this local time, HH:MM")
    parser.add_argument("--review-takes", type=int, default=1)
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument("--guard-skill", metavar="SKILL_ID",
                        help="after each reviewed run, roll the custom skill back to an earlier version "
                             "when the registry shows it scored better (POST .../skills/{id}/rollback)")
    parser.add_argument("--guard-min-runs", type=int, default=3)
    parser.add_argument("--guard-margin", type=float, default=0.5)
    args = parser.parse_args(argv)

    with open(args.payload, encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["storyboard_review"] = False
    payload.setdefault("auto_mode", True)

    def log(message: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)

    failed = 0
    results = []
    for index in range(args.runs):
        if past(args.until):
            log(f"stopping before run {index + 1}: past {args.until}")
            break
        started = _call(args.host, "/api/v1/director/pipeline/start", payload)
        pid = started.get("pipeline_id")
        if not pid:
            log(f"run {index + 1}: start failed: {started}")
            failed += 1
            continue
        log(f"run {index + 1}/{args.runs}: {pid}")
        status = wait_for_run(args.host, pid, log)
        log(f"run {pid}: {status}")
        if status != "completed":
            failed += 1
            results.append({"pid": pid, "status": status})
            continue
        review = {} if args.no_review else review_run(args.host, pid, args.review_takes, log)
        if review:
            log(f"run {pid}: {summarise(review)}")
        results.append({"pid": pid, "status": status,
                        "episode_score": (review.get("episode") or {}).get("score"),
                        "verdict": (review.get("episode") or {}).get("verdict")})
        if args.guard_skill and review:
            guard = guard_skill(args.host, args.guard_skill, args.guard_min_runs, args.guard_margin)
            log(f"skill guard: {guard.get('reason') or guard}")
            if guard.get("applied"):
                log(f"skill guard: rolled {args.guard_skill} back to version {guard.get('revert_to')} "
                    f"(now version {guard.get('new_version')})")
    log("batch done: " + json.dumps(results))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
