"""Read-only AI quality review of Director clips.

Each shot is sent on its own (the pipeline knows the exact shot boundaries,
so the judge never segments a video) to a video-capable LLM, together with
the shot plan it was rendered from. The call goes through fal's
``openrouter/router/video`` endpoint by default, or through a local
OpenAI-compatible vision server that sees the clip as sampled frames
(``judge_backend``: every entry point takes a ``backend``). A second, stricter pass then judges the stitched episode as
one scene (blocking, continuity across cuts, coverage, rhythm, AI tells) and
traces each defect to the skill, the planner, the pipeline or the model. The
verdicts are validated and stored beside the project; this stage never
regenerates anything.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import httpx

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .judge_backend import JudgeBackend

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

# Episode review: the stitched video judged as one scene, by a stronger model.
DEFAULT_EPISODE_MODEL = "google/gemini-3.1-pro-preview"
EPISODE_TIMEOUT_SECONDS = 900
EPISODE_CRITERIA = ("scene_geography", "cut_continuity", "motivated_camera", "coverage", "editing_rhythm",
                    "acting", "lighting_continuity", "dialogue_audio", "authenticity")
EPISODE_VERDICTS = ("broadcast_ready", "needs_polish", "generic_ai", "broken")
ROOT_CAUSES = ("skill", "planner", "pipeline", "model_limit")
SKILL_FIELDS = ("rules.goal", "rules.structure", "rules.pacing", "rules.camera", "rules.dialogue", "rules.sound",
                "rules.do", "rules.dont", "shot_duration", "max_shots", "style_hint",
                # Optional film grammar of the genre (custom skills).
                "scene_grammar.coverage", "scene_grammar.cutting_rhythm", "scene_grammar.camera_default",
                "scene_grammar.blocking", "scene_grammar.lighting_continuity", "scene_grammar.acting")
# Strictness enforced after parsing: footage that reads as AI never scores above this.
GENERIC_AI_SCORE_CAP = 5
BROKEN_SCORE_CAP = 3

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

EPISODE_PROMPT = f"""You are a senior editor and script supervisor reviewing a finished, AI-generated episode as
if it were a cut delivered for broadcast in its genre (infer the genre from the brief: K-drama, sitcom, thriller,
anime, commercial...). You receive the WHOLE stitched episode, its shot list with time ranges, the plan of every
shot and the skill (the genre rules the planner followed). Per-shot artifacts were reviewed separately; here you
judge SCENE GRAMMAR: would a professional crew in this genre have shot and cut it this way?

Judge, comparing against real productions of the genre:
- scene_geography: who is where, stable blocking across cuts, screen direction, 180-degree rule, matching eyelines.
- cut_continuity: cut on action; the end state of shot N is the start state of shot N+1 (positions, hands,
  props, gaze, posture); no jumps, resets or duplicated actions across a cut.
- motivated_camera: the default is a static or discreetly handheld frame; every move (push-in, orbit, crane,
  dolly) must be motivated by the story or a character; unmotivated drifting or orbiting is a defect.
- coverage: a readable pattern (establishing or master, shot-reverse-shot, inserts, reaction close-ups) instead
  of a string of unrelated beauty shots.
- editing_rhythm: shot lengths fit the beat; reactions get time; no dead air, no cut mid-thought.
- acting: restraint and micro-expressions, actions with intention; no filler motion (aimless walking, hair
  touching, gazing into the distance) or stock gestures.
- lighting_continuity: time of day, light direction, colour and weather consistent across the scene.
- dialogue_audio: lines sized to their shot, attributed to the right speaker, lip sync, no looping or repeated
  lines, ambience continuous across cuts.
- authenticity: absence of AI tells: aesthetic boilerplate (glossy golden-hour look, bokeh everywhere, postcard
  or symmetrical framing), floaty slow-motion filler, characters posing for the lens, generic stock staging.

Be strict. Calibration: 9-10 = indistinguishable from a broadcast scene of the genre; 8 = a professional could
air it with minor notes; 6-7 = clearly competent but a viewer notices something is off; 5 or less = generic
AI-looking content, even when every shot is clean; 3 or less = the scene does not hold together. Footage that
reads as AI content MUST get verdict "generic_ai" and score at most 5. Do not reward pretty images.

Report only what you actually see or hear; at_sec is seconds from the start of the WHOLE episode and "shot" is
the 1-based shot number from the shot list. For every defect name the root cause:
- skill: the genre rules themselves push the planner the wrong way (e.g. camera rule asks for constant motion);
- planner: the shot plan was wrong (blocking, coverage choice, lines, durations, missing transitions);
- pipeline: assembly or conditioning (start images that break continuity, audio stitching, cut points);
- model_limit: the video model could not render a correct plan.
Then give concrete fixes: skill_fixes name the skill field to change ({", ".join(SKILL_FIELDS)}) and the new
rule text; planner_fixes and pipeline_fixes are short imperative instructions.

Return ONLY a JSON object, no prose, no markdown:
{{"score": 0-10,
 "scores": {{{", ".join(f'"{c}": 0-10' for c in EPISODE_CRITERIA)}}},
 "verdict": "{'|'.join(EPISODE_VERDICTS)}",
 "summary": "two or three sentences, as an editor's note",
 "defects": [{{"at_sec": number, "shot": number, "severity": "{'|'.join(SEVERITIES)}",
              "criterion": "{'|'.join(EPISODE_CRITERIA)}", "root_cause": "{'|'.join(ROOT_CAUSES)}",
              "description": "short"}}],
 "skill_fixes": [{{"field": "one of the skill fields", "change": "the rule to add or replace"}}],
 "planner_fixes": ["short instruction"],
 "pipeline_fixes": ["short instruction"]}}"""


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


def _asker(backend: "JudgeBackend | None") -> Callable[..., tuple[str, float | None]]:
    """The call of ``backend``, fal's when none is given."""
    return backend.ask if backend is not None else _ask


def backend_name(backend: "JudgeBackend | None") -> str:
    return backend.provider if backend is not None else "fal"


def contract_text(contract: dict | None) -> str:
    if not contract or contract.get("medium") in (None, "", "unspecified"):
        return "Style contract: not specified; infer the intended medium from the plan."
    return "Style contract:\n" + json.dumps(contract, ensure_ascii=False, indent=1)


def _ask(http: httpx.Client, fal_key: str, model: str, system: str, prompt: str, video_urls: list[str],
         sleep: Callable[[float], None], *, max_tokens: int = 12000,
         timeout: float = TIMEOUT_SECONDS, router: str = FAL_ROUTER) -> tuple[str, float | None]:
    """One fal openrouter router call (``openrouter/router/video`` by default); returns (text, cost).

    A text-only call passes no video and another ``router`` (e.g. ``openrouter/router``).
    """
    headers = {"Authorization": f"Key {fal_key}"}
    payload: dict[str, Any] = {
        "model": model, "system_prompt": system, "prompt": prompt,
        # Gemini 3.x Flash refuses requests without reasoning; give it room.
        "reasoning": True, "temperature": 0.2, "max_tokens": max_tokens,
    }
    if video_urls or router == FAL_ROUTER:
        payload["video_urls"] = video_urls
    submitted = http.post(f"{FAL_QUEUE_URL}/{router}", headers=headers, json=payload)
    if submitted.status_code >= 400:
        raise JudgeError(f"fal.ai error {submitted.status_code}: {submitted.text[:300]}")
    handle = submitted.json()
    deadline = time.monotonic() + timeout
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
    backend: "JudgeBackend | None" = None,
) -> dict:
    """Ask the video LLM for a verdict on one shot (one retry on bad JSON).

    ``fal_key`` is the credential of ``backend`` (fal by default).
    """
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
            text, cost = _asker(backend)(http, fal_key, model, SYSTEM_PROMPT, prompt, [video_url], sleep)
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
    backend: "JudgeBackend | None" = None,
) -> dict:
    """Pairwise choice, asked in both orders to cancel position bias.

    Returns {"winner": "A"|"B"|"tie", "confidence", "reasons", "cost_usd"};
    disagreeing orders give a tie (callers keep the existing take).
    """
    prompt = contract_text(style_contract) + "\n\nPlan for this shot:\n" + json.dumps(brief, ensure_ascii=False, indent=1)
    owns_http = http is None
    http = http or httpx.Client(timeout=120)
    try:
        ask = _asker(backend)
        text_1, cost_1 = ask(http, fal_key, model, COMPARE_PROMPT, prompt, [url_a, url_b], sleep)
        text_2, cost_2 = ask(http, fal_key, model, COMPARE_PROMPT, prompt, [url_b, url_a], sleep)
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
    backend: "JudgeBackend | None" = None,
) -> dict:
    """Review every shot. Per-shot failures are recorded, never raised.

    ``publish(path)`` turns a clip file into what ``backend`` accepts (an
    URL for fal, the path itself for a local server).
    """
    segments = plan_segments(clips, output_files, out_dir)
    review = {"status": "running", "model": model, "backend": backend_name(backend),
              "style_contract": style_contract, "started_at": time.time(), "shots": []}
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
                    fal_key=fal_key, model=model, style_contract=style_contract, backend=backend,
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


# Takes worth spending on a shot: performance shots (lip-sync dialogue,
# emotional close-ups) fail often and are judged hardest, inserts rarely.
PERFORMANCE_TAKES = 3
MAX_TAKES_PER_SHOT = 4


def _is_close_framing(framing: str) -> bool:
    words = set(re.findall(r"[a-z]+", framing))
    return "close" in framing or bool(words & {"cu", "ecu", "mcu", "closeup"})


def shot_takes(clips: list[dict], default: int, *, maximum: int = MAX_TAKES_PER_SHOT) -> list[int]:
    """Takes to render for each shot, from its saved plan, with ``default`` as baseline.

    A shot the next one extends from its last frame keeps its single take (a
    new take would break the handoff), inserts get one, lip-sync dialogue and
    close-up reactions get at least ``PERFORMANCE_TAKES``, the rest ``default``.
    """
    default = max(1, int(default))
    counts: list[int] = []
    for index, clip in enumerate(clips):
        following = clips[index + 1] if index + 1 < len(clips) else {}
        scene_type = str(clip.get("_director_scene_type") or "").strip().lower()
        framing = str(clip.get("_director_camera_framing") or "").strip().lower()
        speaks = any(str(beat.get("spoken_text") or "").strip()
                     for beat in clip.get("_director_dialogue_beats") or [] if isinstance(beat, Mapping))
        if str(following.get("_director_continuity_strategy") or "") == "extend_previous" or scene_type == "insert":
            counts.append(1)
        elif speaks or (_is_close_framing(framing) and scene_type in ("reaction", "exchange", "")):
            counts.append(min(max(default, PERFORMANCE_TAKES), maximum))
        else:
            counts.append(default)
    return counts


def select_takes(
    clips: list[dict],
    out_dir: str,
    *,
    extra_takes: int | Sequence[int],
    rerun: Callable[[int], str],
    restore: Callable[[int, str], None],
    publish: Callable[[str], str],
    compare: Callable[..., dict],
    on_progress: Callable[[list[dict]], None] = lambda takes: None,
    min_confidence: float = 0.5,
) -> list[dict]:
    """Render extra takes per shot and keep the one the judge prefers.

    ``extra_takes`` is one count for every shot or one count per shot.
    ``rerun`` renders a new take of a shot (the project then points at it) and
    returns its filename; ``restore`` points the project back at a kept take.
    The incumbent stays unless the challenger wins in both orders with enough
    confidence, so judge noise never swaps a good take for a coin flip.
    """
    per_shot = [extra_takes] * len(clips) if isinstance(extra_takes, int) else list(extra_takes)
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
            for take in range(max(0, int(per_shot[index] if index < len(per_shot) else 0))):
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


# ── Episode review ────────────────────────────────────────────────────


def _resolve(out_dir: str, name: str | None) -> str | None:
    if not name:
        return None
    path = name if os.path.isabs(name) else os.path.join(out_dir, name)
    return path if os.path.isfile(path) else None


def find_stitched_video(output_files: list[str], out_dir: str) -> str | None:
    """The latest ``*_multiclip`` join of the project, if it exists on disk."""
    for name in reversed(output_files or []):
        if "_multiclip" in os.path.splitext(os.path.basename(str(name)))[0].lower():
            path = _resolve(out_dir, name)
            if path:
                return path
    return None


def concat_shots(paths: list[str], dest: str) -> str:
    """Join per-shot files in order, re-encoded at a judge-friendly size."""
    listing = f"{dest}.txt"
    with open(listing, "w", encoding="utf-8") as handle:
        for path in paths:
            handle.write("file '" + os.path.abspath(path).replace("'", "'\\''") + "'\n")
    command = ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
               "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
               "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", dest]
    subprocess.run(command, check=True, capture_output=True, timeout=600)
    return dest


def episode_source(clips: list[dict], output_files: list[str], out_dir: str,
                   edit_file: str | None = None) -> tuple[list[str], str] | None:
    """(paths, kind) of the episode to judge.

    ``kind`` is "edit" (the tight edit, what the viewer gets), "multiclip"
    (the stitched join), "concat" (per-shot files to join here) or "joined"
    (a single seamless render); None when nothing exists.
    """
    edit = _resolve(out_dir, edit_file) if edit_file else None
    if edit:
        return [edit], "edit"
    stitched = find_stitched_video(output_files, out_dir)
    if stitched:
        return [stitched], "multiclip"
    own = [_resolve(out_dir, clip.get("video_filename")) for clip in clips]
    if clips and all(own):
        return [path for path in own if path], "concat"
    if any(own):  # some shots are missing: no complete episode to judge
        return None
    joined = next((path for path in (_resolve(out_dir, n) for n in reversed(output_files or [])) if path), None)
    return ([joined], "joined") if joined else None


def _probe_duration(path: str) -> float | None:
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                 "default=nw=1:nk=1", path], check=True, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def episode_timeline(clips: list[dict], out_dir: str,
                     probe: Callable[[str], float | None] = _probe_duration,
                     keeps: Sequence[tuple[float, float] | None] | None = None) -> list[dict]:
    """Shot time ranges in the stitched episode: real file lengths, else planned ones.

    With ``keeps`` (the tight edit's windows, see ``edit_cut.plan_keeps``) a
    kept shot lasts its window.
    """
    timeline, cursor = [], 0.0
    for index, clip in enumerate(clips):
        planned = clip.get("planned_clip") or {}
        path = _resolve(out_dir, clip.get("video_filename"))
        keep = keeps[index] if keeps and index < len(keeps) else None
        duration = (keep[1] - keep[0]) if keep else (probe(path) if path else None) or float(
            planned.get("duration_sec") or clip.get("_director_duration_sec")
            or max(0.0, float(planned.get("end") or 0) - float(planned.get("start") or 0)))
        timeline.append({"shot": index + 1, "start_sec": round(cursor, 2), "end_sec": round(cursor + duration, 2)})
        cursor += duration
    return timeline


def episode_context(state: Mapping[str, Any]) -> dict:
    """What the episode judge needs to know about the project and its skill."""
    params = state.get("_params_snapshot") or {}
    skill_id = str(params.get("director_skill") or state.get("pipeline_type") or "").strip()
    skill = None
    if skill_id:
        try:
            from services.director.custom_skills import inspiration

            spec = inspiration(skill_id)
        except Exception:  # noqa: BLE001 - the skill is context, not a requirement
            spec = None
        if spec:
            skill = {key: spec.get(key) for key in ("name", "description", "rules", "scene_grammar", "shot_duration",
                                                     "max_shots", "style_hint") if spec.get(key) not in (None, "", [], {})}
    return {key: value for key, value in {
        "skill_id": skill_id or None,
        "skill": skill,
        "brief": str(state.get("scene_description") or "").strip()[:2000] or None,
        "seamless": state.get("seamless"),
    }.items() if value is not None}


def _text_list(value: Any, limit: int = 8) -> list[str]:
    if not isinstance(value, list):
        return []
    items = [str(item).strip()[:300] for item in value if isinstance(item, (str, int, float))]
    return [item for item in items if item][:limit]


def parse_episode_verdict(text: str, duration: float | None, timeline: list[dict]) -> dict:
    """Extract and normalise the episode judge's JSON; raise JudgeError when unusable."""
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise JudgeError("The episode judge returned no JSON")
    try:
        raw = json.loads(match.group(0))
    except ValueError as error:
        raise JudgeError(f"The episode judge returned invalid JSON: {error}") from error
    score = _clamp_score(raw.get("score"))
    if score is None:
        raise JudgeError("The episode judge returned no overall score")
    scores = {c: _clamp_score((raw.get("scores") or {}).get(c)) for c in EPISODE_CRITERIA}
    verdict = raw.get("verdict") if raw.get("verdict") in EPISODE_VERDICTS else None
    # Enforce the calibration the prompt asks for: AI-looking footage never passes.
    if verdict == "broken":
        score = min(score, BROKEN_SCORE_CAP)
    elif verdict == "generic_ai" or (scores["authenticity"] is not None and scores["authenticity"] <= 4):
        score = min(score, GENERIC_AI_SCORE_CAP)
    if verdict is None:
        verdict = ("broadcast_ready" if score >= 8 else "needs_polish" if score >= 6
                   else "generic_ai" if score > BROKEN_SCORE_CAP else "broken")
    limit = float(duration) if duration else None
    count = len(timeline)

    def at_time(value):
        try:
            seconds = max(0.0, float(value))
        except (TypeError, ValueError):
            return None
        return round(min(seconds, limit), 2) if limit else round(seconds, 2)

    def shot_of(value, seconds):
        try:
            number = int(value)
            if 1 <= number <= count:
                return number
        except (TypeError, ValueError):
            pass
        if seconds is not None:
            return next((s["shot"] for s in timeline if s["start_sec"] <= seconds < s["end_sec"]),
                        timeline[-1]["shot"] if timeline and seconds >= timeline[-1]["start_sec"] else None)
        return None

    defects = []
    for defect in raw.get("defects") or []:
        if not isinstance(defect, dict) or not str(defect.get("description") or "").strip():
            continue
        seconds = at_time(defect.get("at_sec"))
        defects.append({
            "at_sec": seconds,
            "shot": shot_of(defect.get("shot"), seconds),
            "severity": defect.get("severity") if defect.get("severity") in SEVERITIES else "minor",
            "criterion": defect.get("criterion") if defect.get("criterion") in EPISODE_CRITERIA else "authenticity",
            "root_cause": defect.get("root_cause") if defect.get("root_cause") in ROOT_CAUSES else "model_limit",
            "description": str(defect["description"]).strip()[:400],
        })
    skill_fixes = []
    for fix in raw.get("skill_fixes") or []:
        if isinstance(fix, dict) and fix.get("field") in SKILL_FIELDS and str(fix.get("change") or "").strip():
            skill_fixes.append({"field": fix["field"], "change": str(fix["change"]).strip()[:600]})
    return {
        "score": score,
        "scores": scores,
        "verdict": verdict,
        "summary": str(raw.get("summary") or "").strip()[:800],
        "defects": defects[:40],
        "skill_fixes": skill_fixes[:8],
        "planner_fixes": _text_list(raw.get("planner_fixes")),
        "pipeline_fixes": _text_list(raw.get("pipeline_fixes")),
    }


def judge_episode(
    video_url: str,
    context: dict,
    timeline: list[dict],
    shots: list[dict],
    *,
    fal_key: str,
    model: str = DEFAULT_EPISODE_MODEL,
    style_contract: dict | None = None,
    http: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    backend: "JudgeBackend | None" = None,
) -> dict:
    """Ask the video LLM for a scene-grammar verdict on the whole episode (one retry on bad JSON)."""
    shot_list = [{**span, **brief} for span, brief in zip(timeline, shots)]
    duration = timeline[-1]["end_sec"] if timeline else None
    prompt = (contract_text(style_contract)
              + "\n\nProject:\n" + json.dumps(context, ensure_ascii=False, indent=1)
              + f"\n\nShot list ({len(shot_list)} shots, {duration or 'unknown'} s in total):\n"
              + json.dumps(shot_list, ensure_ascii=False, indent=1))
    owns_http = http is None
    http = http or httpx.Client(timeout=120)
    try:
        last_error: Exception | None = None
        total_cost = 0.0
        for _ in range(2):
            text, cost = _asker(backend)(http, fal_key, model, EPISODE_PROMPT, prompt, [video_url], sleep,
                                         max_tokens=24000, timeout=EPISODE_TIMEOUT_SECONDS)
            total_cost += float(cost or 0)
            try:
                verdict = parse_episode_verdict(text, duration, timeline)
            except JudgeError as error:
                last_error = error
                continue
            verdict["model"] = model
            verdict["cost_usd"] = round(total_cost, 5)
            return verdict
        raise last_error or JudgeError("The episode judge returned no verdict")
    finally:
        if owns_http:
            http.close()


def review_episode(
    clips: list[dict],
    output_files: list[str],
    out_dir: str,
    *,
    context: dict,
    fal_key: str,
    publish: Callable[[str], str],
    model: str = DEFAULT_EPISODE_MODEL,
    style_contract: dict | None = None,
    judge: Callable[..., dict] = judge_episode,
    probe: Callable[[str], float | None] = _probe_duration,
    backend: "JudgeBackend | None" = None,
    edit_file: str | None = None,
) -> dict:
    """Judge the episode as one scene: the tight edit when there is one, else the join. Failures are recorded, never raised."""
    from services.director.edit_cut import plan_keeps

    review: dict[str, Any] = {"status": "running", "model": model, "backend": backend_name(backend),
                              "started_at": time.time()}
    source = episode_source(clips, output_files, out_dir, edit_file=edit_file)
    if source is None:
        review.update(status="failed", error="No rendered episode to review", finished_at=time.time())
        return review
    paths, kind = source
    review["source"] = kind
    timeline = episode_timeline(clips, out_dir, probe, keeps=plan_keeps(clips) if kind == "edit" else None)
    try:
        with tempfile.TemporaryDirectory(prefix="director-episode-") as tmp:
            dest = os.path.join(tmp, "episode.mp4")
            video = concat_shots(paths, dest) if kind == "concat" else cut_segment(paths[0], None, None, dest)
            review.update(judge(publish(video), context, timeline, [shot_brief(clip) for clip in clips],
                                fal_key=fal_key, model=model, style_contract=style_contract, backend=backend))
        review["status"] = "completed"
    except Exception as error:  # noqa: BLE001 - review is advisory
        review.update(status="failed", error=str(error)[:300])
    review["timeline"] = timeline
    review["finished_at"] = time.time()
    return review
