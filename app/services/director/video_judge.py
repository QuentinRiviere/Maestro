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
# Valid in every style: realism is never the target, craft in the intended
# medium is. Stylisation the contract allows is not a defect.
CRITERIA = ("plan_adherence", "temporal_stability", "identity_consistency", "anatomy_physics_in_style",
            "motion_quality", "staging", "audio_fit")
CATEGORIES = ("morphing", "flicker", "identity_drift", "anatomy", "physics", "object_permanence",
              "motion", "text", "audio", "staging", "plan")
SEVERITIES = ("minor", "major", "critical")
FIXES = ("retake_segment", "new_seed", "prompt_revision", "new_start_image", "none_model_limit")
ACTIONS = ("accept", "retake_segment", "regenerate", "regenerate_start_image")
POLL_SECONDS = 2.0
TIMEOUT_SECONDS = 300

SYSTEM_PROMPT = f"""You are the quality-control judge of an AI video pipeline. The goal is footage that looks
like a professional production IN ITS OWN MEDIUM (live action, 3D animation, anime, cartoon, stop-motion...),
never like a failed AI generation. You receive ONE generated shot, its plan and the project's style contract.
- Stylisation the contract allows (cartoon physics, exaggerated proportions, limited animation) is correct.
- Artifacts are defects in every style: morphing or melting shapes, flicker or boiling textures, identity or
  design drift, anatomy that breaks the style's own design, objects appearing or vanishing, physics that break
  the style's own rules, floaty or jittery motion foreign to the medium, garbled text, mismatched audio.
- Anything the plan asks for (a spill, a crash, a reaction) is intended.
Report only what you actually see or hear; times are seconds from the start of THIS clip. Separate fixable
problems from limits of the video model.

Return ONLY a JSON object, no prose, no markdown:
{{"score": 0-10,
 "scores": {{{", ".join(f'"{c}": 0-10' for c in CRITERIA)}}},
 "issues": [{{"at_sec": number, "severity": "{'|'.join(SEVERITIES)}", "category": "{'|'.join(CATEGORIES)}",
             "fixable_by": "{'|'.join(FIXES)}", "description": "short"}}],
 "action": "{'|'.join(ACTIONS)}",
 "retake_range_sec": [start, end] or null,
 "revised_prompt_hint": "what to add or remove in the prompt, or null",
 "summary": "one sentence"}}
Use "accept" unless a major or critical issue is fixable. Score 10 = indistinguishable from a professional
production in this medium and exactly as planned."""

COMPARE_PROMPT = """You compare two takes of the SAME planned shot. Pick the one that looks more like a
professional production in the project's medium (see the style contract) and follows the plan better.
Artifacts (morphing, flicker, identity drift, broken anatomy or physics for the style, vanishing objects,
unnatural motion for the medium) weigh most. Video 1 is take A, video 2 is take B.
Return ONLY JSON: {"winner": "A"|"B"|"tie", "confidence": 0-1, "reason": "one sentence"}"""


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
            "category": issue.get("category") if issue.get("category") in CATEGORIES else "plan",
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


def contract_text(contract: dict | None) -> str:
    if not contract or contract.get("medium") in (None, "", "unspecified"):
        return "Style contract: not specified; infer the intended medium from the plan."
    return "Style contract:\n" + json.dumps(contract, ensure_ascii=False, indent=1)


def _ask(http: httpx.Client, fal_key: str, model: str, system: str, prompt: str, video_urls: list[str],
         sleep: Callable[[float], None]) -> tuple[str, float | None]:
    """One fal openrouter/router/video call; returns (text, cost)."""
    headers = {"Authorization": f"Key {fal_key}"}
    submitted = http.post(f"{FAL_QUEUE_URL}/{FAL_ROUTER}", headers=headers, json={
        "model": model, "system_prompt": system, "prompt": prompt, "video_urls": video_urls,
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
    return body.get("output") or "", (body.get("usage") or {}).get("cost")


def judge_shot(
    video_url: str,
    brief: dict,
    *,
    index: int,
    count: int,
    previous: dict | None,
    fal_key: str,
    model: str = DEFAULT_MODEL,
    style_contract: dict | None = None,
    http: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Ask the video LLM for a verdict on one shot (one retry on bad JSON)."""
    prompt = (contract_text(style_contract) + f"\n\nShot {index + 1} of {count}. Plan for this shot:\n"
              + json.dumps(brief, ensure_ascii=False, indent=1))
    if previous:
        prompt += ("\nThe previous shot ended with: "
                   + json.dumps(previous, ensure_ascii=False)
                   + "\nJudge continuity with it.")
    owns_http = http is None
    http = http or httpx.Client(timeout=120)
    try:
        last_error: Exception | None = None
        for _ in range(2):
            text, cost = _ask(http, fal_key, model, SYSTEM_PROMPT, prompt, [video_url], sleep)
            try:
                verdict = parse_verdict(text, brief.get("duration_sec"))
            except JudgeError as error:
                last_error = error
                continue
            verdict["model"] = model
            verdict["cost_usd"] = cost
            return verdict
        raise last_error or JudgeError("The judge returned no verdict")
    finally:
        if owns_http:
            http.close()


def _parse_choice(text: str) -> tuple[str, float, str]:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise JudgeError("The comparison returned no JSON")
    raw = json.loads(match.group(0))
    winner = str(raw.get("winner") or "tie").strip().upper()
    winner = winner if winner in ("A", "B") else "tie"
    try:
        confidence = max(0.0, min(1.0, float(raw.get("confidence"))))
    except (TypeError, ValueError):
        confidence = 0.0
    return winner, confidence, str(raw.get("reason") or "").strip()[:300]


def compare_takes(
    url_a: str,
    url_b: str,
    brief: dict,
    *,
    fal_key: str,
    model: str = DEFAULT_MODEL,
    style_contract: dict | None = None,
    http: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Pairwise choice, asked in both orders to cancel position bias.

    Returns {"winner": "A"|"B"|"tie", "confidence", "reasons", "cost_usd"};
    disagreeing orders give a tie (callers keep the existing take).
    """
    prompt = contract_text(style_contract) + "\n\nPlan for this shot:\n" + json.dumps(brief, ensure_ascii=False, indent=1)
    owns_http = http is None
    http = http or httpx.Client(timeout=120)
    try:
        text_1, cost_1 = _ask(http, fal_key, model, COMPARE_PROMPT, prompt, [url_a, url_b], sleep)
        text_2, cost_2 = _ask(http, fal_key, model, COMPARE_PROMPT, prompt, [url_b, url_a], sleep)
        first, conf_1, reason_1 = _parse_choice(text_1)
        second, conf_2, reason_2 = _parse_choice(text_2)
        # The second call saw B first: translate its answer back.
        second = {"A": "B", "B": "A"}.get(second, "tie")
        winner = first if first == second else "tie"
        return {
            "winner": winner,
            "confidence": round((conf_1 + conf_2) / 2, 2) if winner != "tie" else 0.0,
            "reasons": [r for r in (reason_1, reason_2) if r],
            "cost_usd": round((cost_1 or 0) + (cost_2 or 0), 5),
        }
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
    style_contract: dict | None = None,
    on_progress: Callable[[dict], None] = lambda review: None,
    judge: Callable[..., dict] = judge_shot,
) -> dict:
    """Review every shot. Per-shot failures are recorded, never raised."""
    segments = plan_segments(clips, output_files, out_dir)
    review = {"status": "running", "model": model, "style_contract": style_contract,
              "started_at": time.time(), "shots": []}
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
                    fal_key=fal_key, model=model, style_contract=style_contract,
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


def select_takes(
    clips: list[dict],
    out_dir: str,
    *,
    extra_takes: int,
    rerun: Callable[[int], str],
    restore: Callable[[int, str], None],
    publish: Callable[[str], str],
    compare: Callable[..., dict],
    on_progress: Callable[[list[dict]], None] = lambda takes: None,
    min_confidence: float = 0.5,
) -> list[dict]:
    """Render extra takes per shot and keep the one the judge prefers.

    ``rerun`` renders a new take of a shot (the project then points at it) and
    returns its filename; ``restore`` points the project back at a kept take.
    The incumbent stays unless the challenger wins in both orders with enough
    confidence, so judge noise never swaps a good take for a coin flip.
    """
    results: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="director-takes-") as tmp:
        def preview(name: str, label: str) -> str:
            source = name if os.path.isabs(name) else os.path.join(out_dir, name)
            return publish(cut_segment(source, None, None, os.path.join(tmp, f"{label}.mp4")))

        for index, clip in enumerate(clips):
            current = clip.get("video_filename")
            record: dict[str, Any] = {"index": index, "takes": [current] if current else [], "kept": current,
                                      "comparisons": []}
            if not current:
                record["error"] = "This shot has no video of its own"
                results.append(record)
                continue
            best, latest = current, current
            brief = shot_brief(clip)
            for take in range(extra_takes):
                try:
                    latest = rerun(index)
                    record["takes"].append(latest)
                    choice = compare(preview(best, f"s{index}_best_{take}"), preview(latest, f"s{index}_new_{take}"),
                                     brief)
                except Exception as error:  # noqa: BLE001 - keep what we have
                    record["error"] = str(error)[:300]
                    break
                record["comparisons"].append({"incumbent": best, "challenger": latest, **choice})
                if choice.get("winner") == "B" and float(choice.get("confidence") or 0) >= min_confidence:
                    best = latest
            if best != latest:
                restore(index, best)
            record["kept"] = best
            results.append(record)
            on_progress(results)
    return results
