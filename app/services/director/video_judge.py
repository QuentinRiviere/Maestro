"""Read-only AI quality review of Director clips.

Each shot is sent on its own (the pipeline knows the exact shot boundaries,
so the judge never segments a video) to a video-capable LLM through fal's
``openrouter/router/video`` endpoint, together with the shot plan it was
rendered from. The verdict is validated and stored beside the project; this
stage never regenerates anything.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping

import httpx

FAL_ROUTER = "openrouter/router/video"
FAL_QUEUE_URL = "https://queue.fal.run"
DEFAULT_MODEL = "google/gemini-3.8-flash"
REVIEW_FILE_PREFIX = "_director_review_"
CRITERIA = ("plan_adherence", "character_consistency", "anatomy_faces_hands",
            "physics_motion", "audio_dialogue", "continuity")
SEVERITIES = ("minor", "major", "critical")
FIXES = ("retake_segment", "new_seed", "prompt_revision", "new_start_image", "none_model_limit")
ACTIONS = ("accept", "retake_segment", "regenerate", "regenerate_start_image")
POLL_SECONDS = 2.0
TIMEOUT_SECONDS = 300

SYSTEM_PROMPT = f"""You are the quality-control judge of an AI video pipeline. You receive ONE generated shot
and the plan it was rendered from. Judge the video against the plan: anything the plan asks for (a spill, a
crash, a reaction) is intended, not an error. Report only what you actually see or hear; time positions are
seconds from the start of THIS clip. Separate fixable problems from limits of the video model.

Return ONLY a JSON object, no prose, no markdown:
{{"score": 0-10,
 "scores": {{{", ".join(f'"{c}": 0-10' for c in CRITERIA)}}},
 "issues": [{{"at_sec": number, "severity": "{'|'.join(SEVERITIES)}",
             "fixable_by": "{'|'.join(FIXES)}", "description": "short"}}],
 "action": "{'|'.join(ACTIONS)}",
 "retake_range_sec": [start, end] or null,
 "revised_prompt_hint": "what to add or remove in the prompt, or null",
 "summary": "one sentence"}}
Use "accept" unless a major or critical issue is fixable. Score 10 = flawless and exactly as planned."""


class JudgeError(RuntimeError):
    pass


def review_path(out_dir: str, pid: str) -> str:
    return os.path.join(out_dir, f"{REVIEW_FILE_PREFIX}{pid}.json")


def load_review(out_dir: str, pid: str) -> dict | None:
    try:
        with open(review_path(out_dir, pid), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def save_review(out_dir: str, pid: str, review: dict) -> None:
    path = review_path(out_dir, pid)
    temporary = f"{path}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(review, handle, indent=1, ensure_ascii=False)
    os.replace(temporary, path)


# ── Shots and segments ────────────────────────────────────────────────


def shot_brief(clip: Mapping[str, Any]) -> dict:
    """The plan facts a judge needs, from a saved or live clip record."""
    planned = clip.get("planned_clip") or {}
    brief = {
        "label": planned.get("label"),
        "duration_sec": planned.get("duration_sec") or clip.get("_director_duration_sec"),
        "video_prompt": clip.get("video_prompt"),
        "subjects": clip.get("_director_subjects_on_screen"),
        "dialogue": clip.get("_director_dialogue_beats"),
        "audio_plan": clip.get("_director_audio_plan"),
        "environment": clip.get("_director_environment"),
        "opening": clip.get("_director_opening_blocking"),
        "closing": clip.get("_director_closing_blocking"),
    }
    return {key: value for key, value in brief.items() if value not in (None, "", [], {})}


def plan_segments(clips: list[dict], output_files: list[str], out_dir: str) -> list[dict]:
    """One segment per shot: its own clip file, or a time range of the joined video."""
    def resolve(name):
        if not name:
            return None
        path = name if os.path.isabs(name) else os.path.join(out_dir, name)
        return path if os.path.isfile(path) else None

    own = [resolve(clip.get("video_filename")) for clip in clips]
    if clips and all(own):
        return [{"index": i, "path": path, "start": None, "end": None} for i, path in enumerate(own)]
    joined = next((resolve(name) for name in reversed(output_files or []) if resolve(name)), None)
    if joined is None:
        return []
    starts = [float((c.get("planned_clip") or {}).get("start") or 0) for c in clips]
    origin = min(starts) if starts else 0.0
    segments, cursor = [], 0.0
    for i, clip in enumerate(clips):
        planned = clip.get("planned_clip") or {}
        duration = float(planned.get("duration_sec") or 0) or max(
            0.0, float(planned.get("end") or 0) - float(planned.get("start") or 0))
        start = float(planned.get("start", cursor) or 0) - origin if "start" in planned else cursor
        segments.append({"index": i, "path": joined, "start": round(start, 3), "end": round(start + duration, 3)})
        cursor = start + duration
    return segments


def cut_segment(source: str, start: float | None, end: float | None, dest: str) -> str:
    """Re-encode a shot (frame-accurate) at a judge-friendly size."""
    command = ["ffmpeg", "-v", "error", "-y"]
    if start is not None:
        command += ["-ss", f"{start:.3f}"]
    command += ["-i", source]
    if start is not None and end is not None:
        command += ["-t", f"{max(0.1, end - start):.3f}"]
    command += ["-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", dest]
    subprocess.run(command, check=True, capture_output=True, timeout=300)
    return dest


# ── Verdicts ──────────────────────────────────────────────────────────


def _clamp_score(value: Any) -> int | None:
    try:
        return max(0, min(10, int(round(float(value)))))
    except (TypeError, ValueError):
        return None


def parse_verdict(text: str, duration: float | None) -> dict:
    """Extract and normalise the judge's JSON; raise JudgeError when unusable."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise JudgeError("The judge returned no JSON")
    try:
        raw = json.loads(match.group(0))
    except ValueError as error:
        raise JudgeError(f"The judge returned invalid JSON: {error}") from error
    score = _clamp_score(raw.get("score"))
    if score is None:
        raise JudgeError("The judge returned no overall score")
    limit = float(duration) if duration else None

    def clip_time(value):
        try:
            seconds = max(0.0, float(value))
        except (TypeError, ValueError):
            return None
        return round(min(seconds, limit), 2) if limit else round(seconds, 2)

    issues = []
    for issue in raw.get("issues") or []:
        if not isinstance(issue, dict) or not str(issue.get("description") or "").strip():
            continue
        issues.append({
            "at_sec": clip_time(issue.get("at_sec")),
            "severity": issue.get("severity") if issue.get("severity") in SEVERITIES else "minor",
            "fixable_by": issue.get("fixable_by") if issue.get("fixable_by") in FIXES else "none_model_limit",
            "description": str(issue["description"]).strip()[:400],
        })
    action = raw.get("action") if raw.get("action") in ACTIONS else "accept"
    retake = raw.get("retake_range_sec")
    if isinstance(retake, list) and len(retake) == 2:
        a, b = clip_time(retake[0]), clip_time(retake[1])
        retake = [a, b] if a is not None and b is not None and b > a else None
    else:
        retake = None
    return {
        "score": score,
        "scores": {c: _clamp_score((raw.get("scores") or {}).get(c)) for c in CRITERIA},
        "issues": issues,
        "action": action,
        "retake_range_sec": retake,
        "revised_prompt_hint": (str(raw.get("revised_prompt_hint")).strip()[:400]
                                if raw.get("revised_prompt_hint") else None),
        "summary": str(raw.get("summary") or "").strip()[:400],
    }


def judge_shot(
    video_url: str,
    brief: dict,
    *,
    index: int,
    count: int,
    previous: dict | None,
    fal_key: str,
    model: str = DEFAULT_MODEL,
    http: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Ask the video LLM for a verdict on one shot (one retry on bad JSON)."""
    prompt = (f"Shot {index + 1} of {count}. Plan for this shot:\n"
              + json.dumps(brief, ensure_ascii=False, indent=1))
    if previous:
        prompt += ("\nThe previous shot ended with: "
                   + json.dumps(previous, ensure_ascii=False)
                   + "\nJudge continuity with it.")
    owns_http = http is None
    http = http or httpx.Client(timeout=120)
    headers = {"Authorization": f"Key {fal_key}"}
    try:
        last_error: Exception | None = None
        for _ in range(2):
            submitted = http.post(f"{FAL_QUEUE_URL}/{FAL_ROUTER}", headers=headers, json={
                "model": model, "system_prompt": SYSTEM_PROMPT, "prompt": prompt, "video_urls": [video_url],
                # Gemini 3.x Flash refuses requests without reasoning; give it room.
                "reasoning": True, "temperature": 0.2, "max_tokens": 12000,
            })
            if submitted.status_code >= 400:
                raise JudgeError(f"fal.ai error {submitted.status_code}: {submitted.text[:300]}")
            handle = submitted.json()
            deadline = time.monotonic() + TIMEOUT_SECONDS
            while http.get(handle["status_url"], headers=headers).json().get("status") != "COMPLETED":
                if time.monotonic() > deadline:
                    raise JudgeError("The video judge timed out")
                sleep(POLL_SECONDS)
            result = http.get(handle["response_url"], headers=headers)
            body = result.json()
            if result.status_code >= 400 or "output" not in body:
                raise JudgeError(str(body.get("detail") or body)[:300])
            try:
                verdict = parse_verdict(body.get("output") or "", brief.get("duration_sec"))
            except JudgeError as error:
                last_error = error
                continue
            verdict["model"] = model
            verdict["cost_usd"] = (body.get("usage") or {}).get("cost")
            return verdict
        raise last_error or JudgeError("The judge returned no verdict")
    finally:
        if owns_http:
            http.close()


def review_clips(
    clips: list[dict],
    output_files: list[str],
    out_dir: str,
    *,
    fal_key: str,
    publish: Callable[[str], str],
    model: str = DEFAULT_MODEL,
    on_progress: Callable[[dict], None] = lambda review: None,
    judge: Callable[..., dict] = judge_shot,
) -> dict:
    """Review every shot. Per-shot failures are recorded, never raised."""
    segments = plan_segments(clips, output_files, out_dir)
    review = {"status": "running", "model": model, "started_at": time.time(), "shots": []}
    if not segments:
        review.update(status="failed", error="No rendered video to review")
        return review
    with tempfile.TemporaryDirectory(prefix="director-review-") as tmp:
        for segment in segments:
            index = segment["index"]
            brief = shot_brief(clips[index])
            entry: dict[str, Any] = {"index": index, "start_sec": segment["start"], "end_sec": segment["end"]}
            try:
                clip_path = cut_segment(segment["path"], segment["start"], segment["end"],
                                        os.path.join(tmp, f"shot_{index + 1}.mp4"))
                previous = shot_brief(clips[index - 1]) if index else None
                entry.update(judge(
                    publish(clip_path), brief, index=index, count=len(segments),
                    previous={k: previous[k] for k in ("closing", "subjects") if k in previous} if previous else None,
                    fal_key=fal_key, model=model,
                ))
            except Exception as error:  # noqa: BLE001 - review is advisory
                entry["error"] = str(error)[:300]
            review["shots"].append(entry)
            on_progress(review)
    scores = [shot["score"] for shot in review["shots"] if isinstance(shot.get("score"), int)]
    costs = [shot.get("cost_usd") or 0 for shot in review["shots"]]
    review.update(
        status="completed" if scores else "failed",
        score=round(sum(scores) / len(scores), 1) if scores else None,
        cost_usd=round(sum(costs), 4),
        finished_at=time.time(),
    )
    if not scores:
        review["error"] = next((s["error"] for s in review["shots"] if s.get("error")), "No verdict")
    return review
