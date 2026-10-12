"""Skill Creator: draft a declarative Director skill from text and references.

References (images, videos, audio) are analysed by a video-capable LLM through
fal's ``openrouter/router/video`` (which only accepts video, so images become
short stills and audio rides on a black frame). The analyses and the user's
description, optionally starting from an existing skill, are turned into a
skill definition that ``custom_skills.validate_spec`` then bounds.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from typing import Any, Callable

from .custom_skills import ICONS, RULE_FIELDS, WORKFLOWS, validate_spec
from .video_judge import DEFAULT_MODEL, JudgeError, _ask

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
MAX_REFERENCE_SECONDS = 60
MAX_REFERENCES = 8

ANALYSIS_PROMPTS = {
    "video": ("Analyse this reference video as a director who wants to reproduce its CRAFT, not its content. "
              "Describe: medium and visual style; structure and hook; average shot length and cut rhythm; "
              "framing and camera movement; transitions; lighting and palette; motion language; on-screen text; "
              "sound design, music and voice. Under 220 words, as reusable rules."),
    "image": ("Analyse this reference image (shown as a still video) as an art director. Describe the medium, "
              "rendering or photographic style, palette, lighting, composition, lens and texture. Under 150 words, "
              "as reusable style rules."),
    "audio": ("Only the soundtrack matters (the picture is black). Analyse it as a sound designer: genre, tempo, "
              "energy curve, structure, voice style and delivery, sound effects, mixing. Under 150 words, as "
              "reusable rules for the pacing and sound of a video."),
}

DRAFT_SYSTEM = f"""You design a DIRECTOR SKILL: reusable writing rules that turn any brief into a shot plan for
one kind of video. Use the user's description, the optional base skill, and the craft patterns extracted from
reference media. The skill must work for any brief of that kind, in any visual style the brief asks for; put
style only in style_hint unless the description makes a style part of the skill.
Return ONLY JSON:
{{"name": "short name", "description": "one sentence", "icon": "{'|'.join(ICONS)}",
 "workflow": "{'|'.join(WORKFLOWS)}" (story = written from the brief, audio = cut on uploaded audio),
 "default_duration": seconds, "shot_duration": [min_seconds, max_seconds], "max_shots": number,
 "style_hint": "default look if the brief gives none, or empty",
 "rules": {{{", ".join(f'"{f}": "..."' for f in RULE_FIELDS)}, "do": ["..."], "dont": ["..."]}},
 "options": [{{"key": "snake_case", "label": "Label", "values": ["..."], "default": "..."}}]}}
Rules are concrete and actionable for a shot planner (what each shot must contain, how scenes progress,
how long shots last, how the camera behaves). At most 4 options, only when users genuinely need to choose."""


class SkillCreatorError(RuntimeError):
    pass


def reference_kind(path: str) -> str | None:
    ext = os.path.splitext(path)[1].lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    return None


def reference_as_video(path: str, kind: str, dest: str) -> str:
    """Encode any reference as a small mp4 the video router accepts."""
    base = ["ffmpeg", "-v", "error", "-y"]
    if kind == "image":
        command = base + ["-loop", "1", "-t", "3", "-i", path,
                          "-vf", "scale='min(1280,iw)':-2,format=yuv420p", "-r", "8", "-c:v", "libx264", dest]
    elif kind == "audio":
        command = base + ["-f", "lavfi", "-i", "color=c=black:s=320x240:r=4", "-i", path,
                          "-t", str(MAX_REFERENCE_SECONDS), "-shortest", "-c:v", "libx264", "-c:a", "aac",
                          "-b:a", "128k", dest]
    else:
        command = base + ["-i", path, "-t", str(MAX_REFERENCE_SECONDS),
                          "-vf", "scale='min(960,iw)':-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
                          "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", dest]
    subprocess.run(command, check=True, capture_output=True, timeout=300)
    return dest


def analyze_references(
    paths: list[str],
    *,
    fal_key: str,
    publish: Callable[[str], str],
    model: str = DEFAULT_MODEL,
    ask: Callable[..., tuple[str, float | None]] | None = None,
) -> list[dict]:
    """One analysis per reference; failures are reported, never raised."""
    import httpx

    analyses = []
    with tempfile.TemporaryDirectory(prefix="skill-refs-") as tmp, httpx.Client(timeout=120) as http:
        for index, path in enumerate(paths[:MAX_REFERENCES]):
            kind = reference_kind(path)
            entry: dict[str, Any] = {"name": os.path.basename(path), "kind": kind or "unknown"}
            try:
                if kind is None:
                    raise SkillCreatorError("Unsupported reference type")
                if not os.path.isfile(path):
                    raise SkillCreatorError("Reference file not found")
                clip = reference_as_video(path, kind, os.path.join(tmp, f"ref_{index}.mp4"))
                text, cost = (ask or _ask)(http, fal_key, model, "You extract reusable craft rules from media.",
                                           ANALYSIS_PROMPTS[kind], [publish(clip)], time.sleep)
                entry.update(notes=text.strip()[:2000], cost_usd=cost)
            except (SkillCreatorError, JudgeError, subprocess.SubprocessError, OSError) as error:
                entry["error"] = str(error)[:300]
            analyses.append(entry)
    return analyses


def _json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        raise SkillCreatorError("The LLM returned no JSON skill")
    try:
        return json.loads(match.group(0))
    except ValueError as error:
        raise SkillCreatorError(f"The LLM returned invalid JSON: {error}") from error


def draft_skill(
    description: str,
    *,
    generate: Callable[..., str],
    base: dict | None = None,
    analyses: list[dict] | None = None,
    editing: bool = False,
) -> dict:
    """Ask the LLM for a definition, then validate it (one retry on bad JSON)."""
    if not str(description or "").strip() and not base:
        raise SkillCreatorError("Describe the skill, or pick one to start from")
    parts = [f"User description:\n{(description or '').strip() or '(none: improve the base skill)'}"]
    if base:
        role = "Skill to EDIT (keep its intent; apply the description as changes)" if editing else \
            "Base skill for inspiration (adapt freely)"
        parts.append(f"{role}:\n" + json.dumps({k: base.get(k) for k in (
            "name", "description", "workflow", "default_duration", "shot_duration", "max_shots", "style_hint",
            "rules", "options")}, ensure_ascii=False, indent=1))
    notes = [a for a in analyses or [] if a.get("notes")]
    if notes:
        parts.append("Craft patterns extracted from the reference media:\n"
                     + "\n".join(f"- [{a['kind']}] {a['name']}: {a['notes']}" for a in notes))
    prompt = "\n\n".join(parts)
    last_error: Exception | None = None
    for _ in range(2):
        text = generate(prompt=prompt, system_prompt=DRAFT_SYSTEM, max_new_tokens=2500, temperature=0.4,
                        enable_thinking=False)
        try:
            raw = _json_object(text)
            raw["references"] = [{"kind": a["kind"], "name": a["name"], "notes": a["notes"]} for a in notes]
            if base and not editing:
                raw["based_on"] = base.get("id")
            return validate_spec(raw, skill_id=base.get("id") if editing and base else None)
        except (SkillCreatorError, ValueError) as error:
            last_error = error
    raise SkillCreatorError(str(last_error or "No skill drafted"))
