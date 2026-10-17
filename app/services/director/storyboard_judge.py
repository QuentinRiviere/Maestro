"""Storyboard judge: check every shot image before any video is rendered.

A shot image costs about 20 GPU seconds, the video rendered from it ten to
twenty times more, so a bad start image is cheapest to catch here:

- a free duplicate check: a perceptual hash (dHash + aHash, PIL only) of
  every shot image; a non-establishing shot nearly identical to another
  shot of its scene "copies shot N" (image editors often return the scene
  master almost unchanged);
- a vision check per shot with the backend the video judge uses (see
  ``judge_backend``: fal's ``openrouter/router/video`` with the still sent as
  a short clip, or a local OpenAI-compatible vision server with the still
  sent as an image): framing, expected people and their count, the right set.

A failing shot is regenerated with the judge's fix hint appended to its image
prompt (up to ``MAX_ATTEMPTS`` times, re-judged each time); a shot still
failing after that is flagged. Flags never stop an auto run: they are stored
beside the project (``_director_storyboard_<pid>.json``) and shown on the
pipeline status, and a project with ``storyboard_review`` pauses on the
existing image review where the human sees them.

Without a usable backend (no fal key, or no server URL for the local
backend) only the duplicate check runs, and a duplicate gets one regeneration
with a framing hint.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Mapping, Sequence

from . import judge_backend
from .judge_backend import JudgeBackend, load_json_object
from .video_judge import DEFAULT_MODEL, JudgeError

STAGE = "storyboard_judge"
REPORT_FILE_PREFIX = "_director_storyboard_"
MAX_ATTEMPTS = 2
DUPLICATE_ONLY_ATTEMPTS = 1
# Bits (of 64) two hashes may differ by and still be the same picture. Both
# hashes must agree: dHash follows edges, aHash the overall light layout.
DUPLICATE_MAX_DISTANCE = 6
VISION_WORKERS = 4
MAX_CAST_SHEETS = 2
OFF_VALUES = ("off", "false", "0", "no", "disabled")
ON_VALUES = ("on", "true", "1", "yes", "enabled")

_VISION_PROMPT_TEMPLATE = """You check one storyboard frame of an AI video pipeline BEFORE it is animated: the frame
becomes the first frame of an expensive video render, so reject only what would ruin that shot. You receive
the frame ({shown}, {noun} 1), optionally the character sheets of the cast expected in it
(the following {nouns}, in the order listed), and the shot plan.
Check:
- framing: the shot size and angle match the plan (an extreme wide shot is not a close-up, an insert of an
  object is not a group shot);
- people: every expected person is present, recognisable from their sheet when one is given, and the number of
  people matches the plan (no missing, extra or duplicated person);
- set: the location matches the plan;
- integrity: no collage, split screen, reference sheet or text overlay, no broken anatomy.
Stylisation that matches the rest of the plan is correct; minor differences (pose details, props in the
background, colour nuances) pass.
Return ONLY a JSON object, no prose, no markdown:
{"pass": true|false, "issues": ["short description", ...],
 "fix_hint": "one imperative sentence to append to the image prompt so the next render fixes the issues, or empty"}"""


_RANK_PROMPT_TEMPLATE = """You pick the best of several candidate storyboard frames for ONE shot of an AI video pipeline
BEFORE it is animated: the chosen frame is re-rendered at full resolution and becomes the first frame of an
expensive video render. You receive the candidates (each {shown}, {nouns} 1 to N, in the
order listed), optionally the character sheets of the cast expected in the shot (the following {nouns}), and the
shot plan. All candidates share the same prompt and references; they differ only by their random seed.
Check every candidate:
- framing: the shot size and angle match the plan (an extreme wide shot is not a close-up, an insert of an
  object is not a group shot, an over-the-shoulder shot shows the foreground person only as a cropped,
  out-of-focus shoulder and back of the head); an insert of hands or an object that shows any face, head or
  whole person fails, however good the frame looks;
- people: every expected person is present, recognisable from their sheet when one is given, and the number of
  people matches the plan (no missing, extra or duplicated person: a person drawn twice fails);
- set: the location matches the plan;
- integrity: no collage, split screen, reference sheet or text overlay, no broken anatomy.
Stylisation that matches the rest of the plan is correct; minor differences (pose details, props in the
background, colour nuances) pass. Pick as best the candidate that passes and reads the plan most clearly; when
none passes, the one closest to passing.
Return ONLY a JSON object, no prose, no markdown, candidates numbered from 1. "issues" lists problems only
(what is wrong against the plan or the sheets), never what is correct: an empty list for a clean candidate.
{"best": 1, "verdicts": [{"index": 1, "verdict": "pass"|"flagged", "issues": ["short description", ...]}, ...],
 "fix_hint": "one imperative sentence to append to the image prompt so the next renders fix the issues of the
 best candidate, or empty"}"""

_SHOWN = {"video": "shown as a short still video", "image": "shown as an image"}


def _fill(template: str, noun: str) -> str:
    # The templates hold JSON braces, so no str.format here.
    return (template.replace("{shown}", _SHOWN[noun]).replace("{nouns}", noun + "s")
            .replace("{noun}", noun))


# One prompt per media noun: "video" on fal (a still is a short clip there),
# "image" on a local OpenAI-compatible server.
_VISION_PROMPTS = {noun: _fill(_VISION_PROMPT_TEMPLATE, noun) for noun in _SHOWN}
_RANK_PROMPTS = {noun: _fill(_RANK_PROMPT_TEMPLATE, noun) for noun in _SHOWN}
VISION_PROMPT = _VISION_PROMPTS["video"]
RANK_PROMPT = _RANK_PROMPTS["video"]


def vision_prompt(noun: str = "video") -> str:
    """The per-shot check prompt, with media named ``noun``."""
    return _VISION_PROMPTS[noun]


def rank_prompt(noun: str = "video") -> str:
    """The candidate ranking prompt, with media named ``noun``."""
    return _RANK_PROMPTS[noun]


class StoryboardJudgeError(RuntimeError):
    pass


# ── Settings and report storage ───────────────────────────────────────


def settings(services: Mapping[str, Any] | None, params: Mapping[str, Any] | None,
             fal_key: str = "", backend: JudgeBackend | None = None) -> dict:
    """Whether the stage runs and how: {enabled, vision, fal_key, model, backend, provider}.

    The services flag ``director_storyboard_judge`` (default on) and the
    project param of the same name (``off``/``on``, the param wins) enable
    the stage; the vision check also needs a usable backend (a fal key, or
    the local server's URL), else only the free duplicate check runs.
    ``fal_key`` is the credential the fal backend uses (``fal_key`` of the
    result is the credential of whichever backend runs); ``backend``
    defaults to the one ``services`` configure.
    """
    services = services or {}
    params = params or {}
    backend = backend or judge_backend.settings(services)
    if backend.provider == "fal":
        backend = judge_backend.with_key(backend, fal_key)

    def flag(value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        text = str(value if value is not None else "").strip().lower()
        if text in OFF_VALUES:
            return False
        if text in ON_VALUES:
            return True
        return None

    project = flag(params.get("director_storyboard_judge"))
    service = flag(services.get("director_storyboard_judge"))
    enabled = project if project is not None else (service if service is not None else True)
    return {
        "enabled": enabled,
        "vision": bool(enabled and backend.available),
        "fal_key": backend.key if enabled else "",
        "model": backend.resolve_model(str(services.get("director_storyboard_judge_model") or DEFAULT_MODEL)),
        "backend": backend,
        "provider": backend.provider,
    }


def report_path(out_dir: str, pid: str) -> str:
    return os.path.join(out_dir, f"{REPORT_FILE_PREFIX}{pid}.json")


def load_report(out_dir: str, pid: str) -> dict | None:
    try:
        with open(report_path(out_dir, pid), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def save_report(out_dir: str, pid: str, report: dict) -> None:
    path = report_path(out_dir, pid)
    temporary = f"{path}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=1, ensure_ascii=False)
        os.replace(temporary, path)
    except OSError as error:
        print(f"[Storyboard judge] Could not save the report of {pid}: {error}")


# ── Duplicate check (free) ────────────────────────────────────────────


def image_hashes(path: str) -> tuple[int, int]:
    """(dHash, aHash) of an image, 64 bits each."""
    from PIL import Image

    with Image.open(path) as image:
        gray = image.convert("L")
        wide = list(gray.resize((9, 8), Image.LANCZOS).getdata())
        small = list(gray.resize((8, 8), Image.LANCZOS).getdata())
    dhash = 0
    for row in range(8):
        for col in range(8):
            dhash = (dhash << 1) | int(wide[row * 9 + col] > wide[row * 9 + col + 1])
    mean = sum(small) / len(small)
    ahash = 0
    for value in small:
        ahash = (ahash << 1) | int(value > mean)
    return dhash, ahash


def _distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def near_identical(a: tuple[int, int], b: tuple[int, int], max_distance: int = DUPLICATE_MAX_DISTANCE) -> bool:
    return _distance(a[0], b[0]) <= max_distance and _distance(a[1], b[1]) <= max_distance


def _establishing(shot: Mapping[str, Any]) -> bool:
    return str(shot.get("scene_type") or "").strip().lower() == "establishing"


def _checks_duplicates(shot: Mapping[str, Any]) -> bool:
    # A shot that continues the previous one from its last frame is meant to
    # look like it.
    return bool(shot.get("path")) and shot.get("strategy") != "extend_previous"


def duplicate_of(index: int, shots: Sequence[Mapping[str, Any]], hashes: Mapping[int, tuple[int, int]]) -> int | None:
    """The shot of the same scene that ``shots[index]`` copies, or None.

    An establishing shot never copies; between two other shots the later one
    is the copy, unless the earlier one is the establishing shot's twin.
    """
    shot = shots[index]
    if index not in hashes or _establishing(shot) or not _checks_duplicates(shot):
        return None
    for other_index, other in enumerate(shots):
        if other_index == index or other_index not in hashes or not _checks_duplicates(other):
            continue
        if (other.get("group") or "") != (shot.get("group") or ""):
            continue
        if not near_identical(hashes[index], hashes[other_index]):
            continue
        if _establishing(other) or other_index < index:
            return other_index
    return None


def _safe_hashes(shots: Sequence[Mapping[str, Any]]) -> dict[int, tuple[int, int]]:
    hashes = {}
    for index, shot in enumerate(shots):
        if not shot.get("path"):
            continue
        try:
            hashes[index] = image_hashes(shot["path"])
        except (OSError, ValueError) as error:
            print(f"[Storyboard judge] Shot {index + 1}: could not hash the image: {error}")
    return hashes


def duplicate_hint(shot: Mapping[str, Any], original: int) -> str:
    framing = str(shot.get("framing") or "").strip()
    target = f"a {framing} shot" if framing else "a clearly different shot size"
    return (f"Compose this as {target} with a new camera position and angle, visibly different from "
            f"shot {original + 1} of the same scene.")


# ── Vision check ──────────────────────────────────────────────────────


FAL_BACKEND = JudgeBackend(provider="fal")


def still_as_video(path: str, dest: str) -> str:
    """A two-frame mp4 of a still: fal's video router accepts only video."""
    command = ["ffmpeg", "-v", "error", "-y", "-loop", "1", "-t", "2", "-i", path,
               "-vf", "scale='min(768,iw)':-2,format=yuv420p", "-r", "1", "-c:v", "libx264",
               "-preset", "veryfast", "-crf", "24", dest]
    subprocess.run(command, check=True, capture_output=True, timeout=120)
    return dest


def data_uri(path: str) -> str:
    with open(path, "rb") as handle:
        return "data:video/mp4;base64," + base64.b64encode(handle.read()).decode()


def parse_check(text: str) -> dict:
    raw = load_json_object(text, "The storyboard judge")
    verdict = raw.get("pass")
    if isinstance(verdict, str):
        verdict = verdict.strip().lower() in ("true", "yes", "pass")
    if not isinstance(verdict, bool):
        raise JudgeError("The storyboard judge returned no pass/fail")
    issues = [str(item).strip()[:240] for item in raw.get("issues") or [] if str(item).strip()][:6]
    return {"pass": verdict, "issues": issues, "fix_hint": str(raw.get("fix_hint") or "").strip()[:400]}


def parse_rank(text: str, count: int) -> dict:
    """{best, verdicts, fix_hint} of a ranking of ``count`` candidates (numbered from 1).

    Every candidate gets a verdict ("flagged" when the model skipped it); a
    verdict for a number out of range is dropped. A ``best`` out of range or
    missing falls back to the first passing candidate, else candidate 1, and
    the result says so (``best_fallback``). No JSON at all raises JudgeError.
    """
    raw = load_json_object(text, "The storyboard judge")
    if not isinstance(raw, dict):
        raise JudgeError("The storyboard judge returned no ranking object")

    def number(value: Any) -> int | None:
        try:
            value = int(str(value).strip())
        except (TypeError, ValueError):
            return None
        return value if 1 <= value <= count else None

    verdicts: dict[int, dict] = {}
    for item in raw.get("verdicts") or []:
        if not isinstance(item, dict):
            continue
        index = number(item.get("index"))
        if index is None or index in verdicts:
            continue
        verdict = item.get("verdict", item.get("pass"))
        if isinstance(verdict, bool):
            passed = verdict
        else:
            passed = str(verdict or "").strip().lower() in ("pass", "passed", "true", "yes")
        issues = [str(issue).strip()[:240] for issue in item.get("issues") or [] if str(issue).strip()][:6]
        verdicts[index] = {"index": index, "verdict": "pass" if passed else "flagged", "issues": issues}
    ordered = [verdicts.get(index) or {"index": index, "verdict": "flagged", "issues": ["not ranked by the judge"]}
               for index in range(1, count + 1)]
    best = number(raw.get("best"))
    fallback = best is None
    if fallback:
        best = next((v["index"] for v in ordered if v["verdict"] == "pass"), 1)
    return {"best": best, "verdicts": ordered, "fix_hint": str(raw.get("fix_hint") or "").strip()[:400],
            "best_fallback": fallback}


def shot_plan_text(shot: Mapping[str, Any]) -> str:
    plan = {key: shot.get(key) for key in ("framing", "scene_type", "location", "environment", "subjects",
                                           "opening_blocking", "image_prompt")}
    return json.dumps({k: v for k, v in plan.items() if v not in (None, "", [], {})}, ensure_ascii=False, indent=1)


def check_shot(shot: Mapping[str, Any], *, fal_key: str, model: str = DEFAULT_MODEL,
               ask: Callable[..., tuple[str, float | None]] | None = None, http: Any = None,
               backend: JudgeBackend | None = None) -> dict:
    """One vision verdict {pass, issues, fix_hint, cost_usd}; one retry on bad JSON.

    ``fal_key`` is the credential of ``backend`` (fal by default); ``ask``
    overrides the backend's call (tests).
    """
    import httpx

    backend = backend or FAL_BACKEND
    ask = ask or backend.ask
    noun = backend.media_noun
    sheets = [s for s in shot.get("cast_sheets") or [] if s.get("path") and os.path.isfile(s["path"])]
    prompt = ""
    if sheets:
        prompt += "Character sheets: " + ", ".join(
            f"{noun} {n + 2} = {sheet.get('name') or 'cast member'}" for n, sheet in enumerate(sheets)) + ".\n"
    prompt += "Shot plan:\n" + shot_plan_text(shot)
    owns_http = http is None
    http = http or httpx.Client(timeout=120)
    cost = 0.0
    try:
        with tempfile.TemporaryDirectory(prefix="director-storyboard-") as tmp:
            urls = backend.media_for_images([shot["path"]] + [sheet["path"] for sheet in sheets], tmp)
            last_error: Exception | None = None
            for _ in range(2):
                text, call_cost = ask(http, fal_key, model, vision_prompt(noun), prompt, urls, time.sleep,
                                      max_tokens=4000, timeout=180)
                cost += float(call_cost or 0)
                try:
                    verdict = parse_check(text)
                except JudgeError as error:
                    last_error = error
                    continue
                verdict["cost_usd"] = round(cost, 5)
                return verdict
            raise last_error or JudgeError("The storyboard judge returned no verdict")
    finally:
        if owns_http:
            http.close()


def rank_candidates(shot: Mapping[str, Any], paths: Sequence[str], *, fal_key: str, model: str = DEFAULT_MODEL,
                    ask: Callable[..., tuple[str, float | None]] | None = None, http: Any = None,
                    backend: JudgeBackend | None = None) -> dict:
    """One vision call ranking a shot's candidate images: {best, verdicts, fix_hint, cost_usd}.

    The candidates are media 1 to N (videos on fal, images locally), the
    shot's character sheets follow, as in ``check_shot``; one retry on bad JSON.
    """
    import httpx

    backend = backend or FAL_BACKEND
    ask = ask or backend.ask
    noun = backend.media_noun
    sheets = [s for s in shot.get("cast_sheets") or [] if s.get("path") and os.path.isfile(s["path"])]
    count = len(paths)
    prompt = "Candidates: " + ", ".join(f"{noun} {n + 1} = candidate {n + 1}" for n in range(count)) + ".\n"
    if sheets:
        prompt += "Character sheets: " + ", ".join(
            f"{noun} {count + n + 1} = {sheet.get('name') or 'cast member'}" for n, sheet in enumerate(sheets)) + ".\n"
    prompt += "Shot plan:\n" + shot_plan_text(shot)
    owns_http = http is None
    http = http or httpx.Client(timeout=120)
    cost = 0.0
    try:
        with tempfile.TemporaryDirectory(prefix="director-storyboard-rank-") as tmp:
            urls = backend.media_for_images(list(paths) + [sheet["path"] for sheet in sheets], tmp)
            last_error: Exception | None = None
            for _ in range(2):
                text, call_cost = ask(http, fal_key, model, rank_prompt(noun), prompt, urls, time.sleep,
                                      max_tokens=4000, timeout=180)
                cost += float(call_cost or 0)
                try:
                    ranking = parse_rank(text, count)
                except JudgeError as error:
                    last_error = error
                    continue
                ranking["cost_usd"] = round(cost, 5)
                return ranking
            error = last_error or JudgeError("The storyboard judge returned no ranking")
            # The calls were paid for even without a ranking.
            error.cost_usd = round(cost, 5)  # type: ignore[attr-defined]
            raise error
    finally:
        if owns_http:
            http.close()

# ── The stage ─────────────────────────────────────────────────────────


def judge_storyboard(
    shots: list[dict],
    *,
    rerun: Callable[[int, str], str],
    check: Callable[[dict], dict] | None = None,
    record_cost: Callable[[int, dict], None] = lambda index, verdict: None,
    on_progress: Callable[[dict], None] = lambda report: None,
    is_cancelled: Callable[[], bool] = lambda: False,
    model: str | None = None,
    max_attempts: int = MAX_ATTEMPTS,
    workers: int = VISION_WORKERS,
    backend: str = "fal",
) -> dict:
    """Judge every shot image, regenerate failures, flag what stays bad.

    ``shots`` are dicts with ``path`` (absolute, "" when the shot has no
    image), ``image_prompt``, ``group``, ``scene_type``, ``framing``,
    ``strategy`` and the plan facts the vision check reads; their ``path`` is
    updated in place when a shot is regenerated. ``rerun(index, prompt)``
    renders a new image and returns its absolute path. ``check(shot)`` is
    the vision verdict (None: duplicate check only). Vision errors never fail
    a shot: the check is advisory.
    """
    vision = check is not None
    attempts_allowed = max_attempts if vision else DUPLICATE_ONLY_ATTEMPTS
    report: dict[str, Any] = {
        "status": "running", "mode": "vision" if vision else "duplicates_only", "model": model if vision else None,
        "backend": backend if vision else None,
        "started_at": time.time(), "cost_usd": 0.0,
        "shots": [{"index": i, "verdict": "pending" if shot.get("path") else "skipped", "attempts": 0,
                   "issues": [], "fix_hint": "", "duplicate_of": None,
                   "image": os.path.basename(shot.get("path") or ""), "cost_usd": 0.0}
                  for i, shot in enumerate(shots)],
    }
    on_progress(report)

    def add_cost(index: int, verdict: dict) -> None:
        cost = float(verdict.get("cost_usd") or 0)
        report["shots"][index]["cost_usd"] = round(report["shots"][index]["cost_usd"] + cost, 5)
        report["cost_usd"] = round(report["cost_usd"] + cost, 5)
        record_cost(index, verdict)

    def safe_check(index: int) -> dict | None:
        try:
            return check(shots[index])  # type: ignore[misc]
        except Exception as error:  # noqa: BLE001 - review is advisory
            return {"error": str(error)[:300]}

    def judge(index: int, hashes: Mapping[int, tuple[int, int]], verdict: dict | None) -> tuple[bool, list, str]:
        """(passed, issues, fix_hint) of one shot from its duplicate and vision checks."""
        entry = report["shots"][index]
        original = duplicate_of(index, shots, hashes)
        entry["duplicate_of"] = original
        issues: list[str] = []
        hints: list[str] = []
        if original is not None:
            issues.append(f"copies shot {original + 1}")
            hints.append(duplicate_hint(shots[index], original))
        if verdict is not None:
            if verdict.get("error"):
                entry["error"] = verdict["error"]
            else:
                entry.pop("error", None)
                add_cost(index, verdict)
                if not verdict.get("pass"):
                    issues.extend(verdict.get("issues") or ["the image does not match its plan"])
                    if verdict.get("fix_hint"):
                        hints.append(verdict["fix_hint"])
        return not issues, issues, " ".join(hints)

    pending = [i for i, shot in enumerate(shots) if shot.get("path")]
    hashes = _safe_hashes(shots)
    verdicts: dict[int, dict | None] = {i: None for i in pending}
    # A copy is regenerated anyway: judging it would be money spent twice.
    to_check = [i for i in pending if duplicate_of(i, shots, hashes) is None]
    if vision and to_check:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for index, verdict in zip(to_check, pool.map(safe_check, to_check)):
                verdicts[index] = verdict

    failing: list[tuple[int, list, str]] = []
    for index in pending:
        passed, issues, hint = judge(index, hashes, verdicts[index])
        entry = report["shots"][index]
        entry.update(issues=issues, fix_hint=hint, verdict="pass" if passed else "failed")
        if not passed:
            failing.append((index, issues, hint))
    on_progress(report)

    for index, issues, hint in failing:
        entry = report["shots"][index]
        base_prompt = str(shots[index].get("image_prompt") or "").strip()
        history: list[dict] = []
        passed = False
        while entry["attempts"] < attempts_allowed and not is_cancelled():
            entry["attempts"] += 1
            prompt = f"{base_prompt} {hint}".strip() if hint else base_prompt
            history.append({"image": entry["image"], "issues": issues, "fix_hint": hint})
            try:
                shots[index]["path"] = rerun(index, prompt)
            except Exception as error:  # noqa: BLE001 - keep the current image
                entry["error"] = f"Regeneration failed: {str(error)[:260]}"
                break
            entry["image"] = os.path.basename(shots[index]["path"])
            try:
                hashes[index] = image_hashes(shots[index]["path"])
            except (OSError, ValueError):
                hashes.pop(index, None)
            passed, issues, hint = judge(index, hashes, safe_check(index) if vision else None)
            entry.update(issues=issues, fix_hint=hint)
            on_progress(report)
            if passed:
                break
        entry["history"] = history
        entry["verdict"] = "regenerated" if passed else "flagged"
        on_progress(report)

    report["flagged"] = [e["index"] for e in report["shots"] if e["verdict"] == "flagged"]
    report["regenerated"] = [e["index"] for e in report["shots"] if e["verdict"] == "regenerated"]
    report.update(status="cancelled" if is_cancelled() else "completed", finished_at=time.time())
    on_progress(report)
    return report


def shots_from_plans(clip_plans: Sequence[Mapping[str, Any]], clip_images: Sequence[str], out_dir: str,
                     params: Mapping[str, Any] | None = None) -> list[dict]:
    """The judge's view of each shot: its image path and the plan facts to check."""
    params = params or {}
    sheets = [{"name": str(label), "path": path}
              for label, path in zip(params.get("character_ref_labels") or [], params.get("character_ref_paths") or [])
              if label and path]
    shots = []
    for index, plan in enumerate(clip_plans):
        name = clip_images[index] if index < len(clip_images) else ""
        path = (name if os.path.isabs(name) else os.path.join(out_dir, name)) if name else ""
        prompt = str(plan.get("image_prompt") or "")
        subjects = plan.get("_director_subjects_on_screen") or []
        mentioned = (prompt + " " + json.dumps(subjects, ensure_ascii=False, default=str)).lower()
        shots.append({
            "path": path if path and os.path.isfile(path) else "",
            "image_prompt": prompt,
            "group": str(plan.get("_director_continuity_group") or ""),
            "strategy": str(plan.get("_director_continuity_strategy") or ""),
            "scene_type": plan.get("_director_scene_type"),
            "framing": plan.get("_director_camera_framing"),
            "location": plan.get("_director_location"),
            "environment": plan.get("_director_environment"),
            "subjects": subjects,
            "opening_blocking": plan.get("_director_opening_blocking"),
            "cast_sheets": [s for s in sheets if s["name"].lower() in mentioned][:MAX_CAST_SHEETS],
        })
    return shots


def stage_config(params: Mapping[str, Any] | None) -> dict:
    """``settings`` of the running server: its services flags, judge backend and fal key."""
    from services import cloud_image
    from services import director_pipeline as pipeline

    server_config = getattr(pipeline._wgp, "server_config", None) if pipeline._wgp else None
    services = (server_config.get("services", {}) if isinstance(server_config, dict) else {}) or {}
    return settings(services, params, cloud_image.provider_api_key("fal", services))

def run_stage(pid: str, params: dict, clip_plans: list[dict], clip_images: list[str], out_dir: str) -> dict | None:
    """Pipeline hook between shot images and video: judge, regenerate, flag.

    Updates ``clip_images`` in place with regenerated images, publishes the
    report as the pipeline's ``storyboard_judge`` field and saves it beside
    the project. Never raises: a judge failure must not fail the render.
    """
    from services.director import cost_ledger
    from services import director_pipeline as pipeline

    try:
        config = stage_config(params)
        if not config["enabled"]:
            return None
        shots = shots_from_plans(clip_plans, clip_images, out_dir, params)
        if not any(shot["path"] for shot in shots):
            return None
        mode = (f"vision ({config['provider']}, {config['model']}) + duplicates" if config["vision"]
                else "duplicates only (no usable judge backend)")
        print(f"[Pipeline {pid}] Storyboard judge: {len(shots)} shot(s), {mode}")

        def is_cancelled() -> bool:
            return (pipeline._pipelines.get(pid) or {}).get("status") == "cancelled"

        def rerun(index: int, prompt: str) -> str:
            pipeline._update_pipeline(pid, progress={
                "current": index, "total": len(shots), "step": 0, "total_steps": 0,
                "message": f"Storyboard judge: regenerating shot {index + 1}"})
            result = pipeline._rerun_clip_image_impl(out_dir, pid, index, prompt_override=prompt)
            filename = result["filename"]
            clip_images[index] = filename
            # The live plan keeps its prompt; saving rewrites the clip from it.
            pipeline._update_pipeline(pid, clip_images=list(clip_images))
            pipeline._save_pipeline_state(pid)
            return filename if os.path.isabs(filename) else os.path.join(out_dir, filename)

        def record_cost(index: int, verdict: dict) -> None:
            cost_ledger.record(pid, "cloud", stage=STAGE, provider=config["provider"], model=config["model"],
                               shot=index, usd=verdict.get("cost_usd"))

        def publish(report: dict) -> None:
            pipeline._update_pipeline(pid, storyboard_judge=json.loads(json.dumps(report, default=str)))
            save_report(out_dir, pid, report)

        pipeline._update_pipeline(pid, progress={"current": 0, "total": len(shots), "step": 0, "total_steps": 0,
                                                 "message": "Storyboard judge: checking shot images"})
        check = None
        if config["vision"]:
            def check(shot: dict) -> dict:
                return check_shot(shot, fal_key=config["fal_key"], model=config["model"], backend=config["backend"])
        report = judge_storyboard(shots, rerun=rerun, check=check, record_cost=record_cost, on_progress=publish,
                                  is_cancelled=is_cancelled, model=config["model"], backend=config["provider"])
        print(f"[Pipeline {pid}] Storyboard judge: regenerated {report['regenerated']}, "
              f"flagged {report['flagged']}, cost ${report['cost_usd']}")
        return report
    except Exception as error:  # noqa: BLE001 - advisory stage
        print(f"[Pipeline {pid}] Storyboard judge skipped: {error}")
        return None
