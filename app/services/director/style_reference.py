"""Style references: learn staging, editing and rhythm from a reference video.

A user uploads a video whose filmmaking they want a series to follow (a
K-drama scene, a trailer, a short). It is analysed in two passes:

- a local, exact pass with ffprobe/ffmpeg: duration, frame rate, hard cuts
  (ffmpeg scene detection), shot lengths, average and median shot length,
  the shot-length distribution, cuts per minute and a coarse pacing curve
  (average shot length per 10 s window);
- a model pass through fal's ``openrouter/router/video`` (the provider the
  video judge uses): the video is downscaled and split into chunks of at most
  a minute, aligned on the detected cuts; each chunk goes with its exact cut
  list and comes back as one JSON record per shot (size, angle, movement,
  generic subject and action, why the cut happens, transition, light, sound).
  One text-only synthesis call then turns the shot records and the stats into
  a "staging grammar": rhythm targets, shot mix, recurring patterns, scene
  openings and closings, dialogue coverage, camera habits, light, sound, and
  5-10 rules a storyboard planner can follow. Only transferable principles
  are kept: never the reference's content, characters or story.

With the local judge backend (``judge_backend`` provider ``openai``) the model
pass costs nothing: the excerpts become silent stills, one or two per shot,
sent to the ``analyst`` alias of the local server (a Qwen vision model, the
family Qwen-Image encodes its prompts with, so the grammar comes out in the
image model's own dialect). The sound column is then a guess from the picture.

References live in a bible (``style_references``, files in the bible's
folder). An enabled, analysed reference reaches the Director planners as a
bounded "REFERENCE STAGING GRAMMAR" block (``planner_block``).
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import secrets
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

from fastapi import File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from . import bibles
from .bibles import BibleError
from .video_judge import DEFAULT_MODEL, FAL_ROUTER, JudgeError, _ask

# ── constants ──────────────────────────────────────────────────────────────

# ffmpeg ``scene`` score above which a frame counts as a hard cut (the value
# the H3 sequence splitter uses on Director renders).
SCENE_CUT_THRESHOLD = 0.27
# Cuts closer than this to the previous one are flashes, not shots.
MIN_SHOT_SEC = 0.25
PACING_WINDOW_SEC = 10.0
# Shot-length buckets of the distribution: (label, upper bound in seconds).
LENGTH_BUCKETS = (("<1s", 1.0), ("1-2s", 2.0), ("2-4s", 4.0), ("4-8s", 8.0), ("8s+", math.inf))

# Model pass: chunks of at most a minute, cut on shot boundaries, small but
# with sound (music and silence are part of the grammar).
MAX_CHUNK_SEC = 60.0
CHUNK_WIDTH = 480
CHUNK_CRF = 30
CHUNK_AUDIO_BITRATE = "48k"
# Only the first minutes go to the model (cost bound); stats cover the whole video.
MAX_ANALYSED_SEC = 15 * 60
CHUNK_ATTEMPTS = 2
SYNTHESIS_ATTEMPTS = 2
TEXT_ROUTER = "openrouter/router"
# Local backend: the alias of the vision model on the local server, and how an
# excerpt is shown to it (stills instead of a video: one per shot, two for a
# shot long enough to move, at most MAX_LOCAL_STILLS per call).
LOCAL_MODEL = "analyst"
MAX_LOCAL_STILLS = 24
LONG_SHOT_SEC = 3.0
STILL_LONG_SIDE = 768
# The local pass is free: a whole episode can go through it.
MAX_LOCAL_ANALYSED_SEC = 2 * 3600
# Consolidation of several references into one grammar (start_merge).
MERGE_ATTEMPTS = 2
MERGED_RULES = 14
MIN_MERGE_REFERENCES = 2
MAX_STORED_SHOTS = 1500
MAX_SYNTHESIS_SHOTS = 400

# Uploads: a whole episode fits; it is reduced to a 480p proxy before anything
# reads it (cut detection, stills, chunks), and the proxy replaces the upload.
MAX_UPLOAD_BYTES = 4 * 1024 * 1024 * 1024
PROXY_HEIGHT = 480
PROXY_CRF = 26
PROXY_AUDIO_BITRATE = "64k"
VIDEO_SUFFIXES = bibles.VIDEO_SUFFIXES
MEDIA_TYPES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".webm": "video/webm", ".mkv": "video/x-matroska"}

# Planner block bounds.
MAX_REFERENCE_CHARS = 1500
MAX_BLOCK_CHARS = 4000
CHOICE_AUTO = "auto"
CHOICE_OFF = "off"
# On-screen graphics (captions, subtitles, logos, lower thirds, text): left out
# of the staging, or read as part of the grammar.
GRAPHICS_MODES = bibles.STYLE_REFERENCE_GRAPHICS
GRAPHICS_IGNORE, GRAPHICS_INCLUDE = GRAPHICS_MODES
GRAPHICS_NOTES = {
    GRAPHICS_IGNORE: (
        "Burned-in subtitles, captions, broadcaster logos, watermarks, lower thirds and other on-screen graphics "
        "are NOT part of the staging: never describe them, never let them change the subject, shot_size or "
        "framing you report, and never mark a shot as credits because a subtitle, caption or channel logo sits on "
        "it; a shot is credits only when the picture itself is a title sequence, a credit roll, a logo animation, "
        "a recap or a preview."),
    GRAPHICS_INCLUDE: (
        "On-screen graphics are part of the grammar: for every shot set \"graphics\" to a short description of the "
        "captions, subtitles, lower thirds, logos, stickers, text animations or other graphic elements in frame "
        "(what they say or show, where they sit, how they move), or \"none\"."),
}
GRAPHICS_SYNTHESIS = {
    GRAPHICS_IGNORE: "",
    GRAPHICS_INCLUDE: ("\nThe shot breakdown has a graphics column: also cover on-screen graphics (captions, subtitles, "
                       "lower thirds, logos, stickers, text animations): when they appear, what they do, how they "
                       "sit and move, and add rules about them. Put that in \"graphics\" (short) in the JSON."),
}

SHOT_SIZES = ("extreme_wide", "wide", "medium", "medium_close_up", "close_up", "extreme_close_up", "insert",
              "ots", "pov", "two_shot")
SHOT_SIZE_LABELS = {
    "extreme_wide": "extreme wide", "wide": "wide", "medium": "medium", "medium_close_up": "medium close-up",
    "close_up": "close-up", "extreme_close_up": "extreme close-up", "insert": "insert", "ots": "over-the-shoulder",
    "pov": "POV", "two_shot": "two-shot",
}
ANGLES = ("eye_level", "high", "low", "overhead", "dutch", "ground")
MOVEMENTS = ("static", "handheld", "pan", "tilt", "push_in", "pull_out", "tracking", "dolly", "crane", "zoom",
             "orbit", "other")
CUT_REASONS = ("eyeline", "action", "dialogue_line", "reaction", "beat", "time_jump", "scene_change", "rhythm",
               "other")
TRANSITIONS = ("hard_cut", "match_cut", "j_cut", "l_cut", "dissolve", "fade", "whip", "smash_cut", "other")
AUDIO_KINDS = ("dialogue", "music", "silence", "ambience", "mixed")

SHOT_SYSTEM_PROMPT = f"""You are a film editor and cinematographer breaking down a reference video to learn its
staging, editing and rhythm. You receive ONE excerpt of a longer video and its exact shot list, measured by cut
detection (start and end in seconds from the start of THIS excerpt). Describe every listed shot, in order.
- Describe people generically by role or look ("a young woman", "the older man", "a waiter"), never by name; never
  transcribe dialogue, only say what a line does (a question, a refusal, a confession).
- cut_reason is why the editor cuts INTO this shot: eyeline (we see what someone looks at), action (cut on a
  movement), dialogue_line (a new line or speaker), reaction (to see a reaction), beat (a dramatic turn),
  time_jump, scene_change, rhythm (pace or music), other.
- If a listed cut is not a real cut (a flash, a fast pan, a light change), set "false_cut": true on the shot that
  starts there.
- The video may be a whole episode: set "credits": true on a shot that belongs to the opening titles or title
  sequence, the end credits, a broadcaster or studio logo, a title card, a recap or a preview of the next episode
  (text over the picture, cast names, a logo animation). Still describe it. Such shots are left out of the grammar.

Return ONLY a JSON object, no prose, no markdown:
{{"shots": [{{"shot": 1, "shot_size": "{'|'.join(SHOT_SIZES)}", "angle": "{'|'.join(ANGLES)}",
  "movement": "{'|'.join(MOVEMENTS)}", "subject": "who or what the frame shows, generic",
  "action": "what happens, short", "cut_reason": "{'|'.join(CUT_REASONS)}",
  "transition": "{'|'.join(TRANSITIONS)}", "lighting": "light and colour, short",
  "audio": "{'|'.join(AUDIO_KINDS)}", "false_cut": false, "credits": false, "graphics": "none or short"}}]}}"""

MERGE_SYSTEM_PROMPT = """You are a senior editor consolidating the staging grammars of SEVERAL reference videos
into ONE grammar for an AI storyboard planner that will direct NEW productions with their own story and cast.
You receive, for each reference, its label, the scenes it was chosen for (tags, or every scene), its weight (shot
count), its measured editing statistics (exact) and the grammar learned from it. Write the grammar the references
share: where they agree, state the principle once with the weighted numbers; where they differ, keep the habit of
the heavier references and say so ("most references..."); where a difference follows the scene tags, put it in
"scene_notes" keyed by tag. Never mention a reference's specific content, characters, names, places, plot or
dialogue; write "a character", "the listener", "the speaker", "an object".

Return ONLY a JSON object, no prose outside it:
{"rhythm": {"target_asl_sec": number, "pace_with_emotion": "how pace changes with emotion and tension",
            "notes": "short"},
 "shot_mix": {"close_up": percent, "medium": percent, "...": percent},
 "patterns": ["recurring pattern shared by the references, short"],
 "scene_openings": "short", "scene_closings": "short", "dialogue_coverage": "short",
 "camera_movement": "short", "lighting_color": "short", "music_silence": "short",
 "rules": ["8-14 imperative rules a planner can apply shot by shot, numbers where possible"],
 "scene_notes": {"<tag>": "what changes for scenes of this mood, short"},
 "summary_markdown": "a readable markdown summary for the user, 150-300 words, with short headings"}"""

SYNTHESIS_SYSTEM_PROMPT = """You are a senior editor writing the staging grammar of a reference video for an AI
storyboard planner that will direct NEW productions with their own story and cast. You receive measured editing
statistics (exact) and a shot-by-shot breakdown. Extract only TRANSFERABLE PRINCIPLES of staging, coverage, editing
and rhythm. Never mention the reference's specific content, characters, names, places, plot or dialogue; write
"a character", "the listener", "the speaker", "an object".

Cover: rhythm (target average shot length and how pace changes with emotion or tension), shot-type mix, recurring
patterns (shot/reverse-shot, a reaction after a line, body-part inserts such as hands or legs on entrances, push-ins
on realisations...), how scenes open and close, how dialogue is covered, camera movement habits, lighting and
colour, use of music and silence. Then write 5 to 10 concrete, imperative rules a planner can apply shot by shot
(numbers where possible: durations, proportions, orders of shots).

Return ONLY a JSON object, no prose outside it:
{"rhythm": {"target_asl_sec": number, "pace_with_emotion": "how pace changes with emotion and tension",
            "notes": "short"},
 "shot_mix": {"close_up": percent, "medium": percent, "...": percent},
 "patterns": ["recurring pattern, short"],
 "scene_openings": "short", "scene_closings": "short", "dialogue_coverage": "short",
 "camera_movement": "short", "lighting_color": "short", "music_silence": "short",
 "rules": ["5-10 imperative rules"],
 "summary_markdown": "a readable markdown summary for the user, 150-300 words, with short headings"}"""


class StyleReferenceError(RuntimeError):
    pass


# ── local exact pass ───────────────────────────────────────────────────────


def _rate(value: Any) -> float | None:
    try:
        if isinstance(value, str) and "/" in value:
            num, den = value.split("/", 1)
            return float(num) / float(den) if float(den) else None
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def probe(path: str, *, timeout: float = 60) -> dict:
    """Duration, frame rate, size and whether the file has sound (ffprobe)."""
    command = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", path]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=timeout)
        data = json.loads(result.stdout or "{}")
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise StyleReferenceError(f"ffprobe could not read the video: {str(error)[:200]}") from error
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise StyleReferenceError("The file has no video stream")
    duration = _rate((data.get("format") or {}).get("duration")) or _rate(video.get("duration"))
    if not duration or duration <= 0:
        raise StyleReferenceError("Could not read the video's duration")
    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate"))
    return {
        "duration_sec": round(duration, 3),
        "fps": round(fps, 3) if fps else None,
        "width": video.get("width"),
        "height": video.get("height"),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
    }


_PTS_TIME = re.compile(r"pts_time:([0-9.]+)")


def detect_cuts(path: str, *, threshold: float = SCENE_CUT_THRESHOLD, timeout: float = 1800) -> list[float]:
    """Times (seconds) of the hard cuts ffmpeg's scene detection finds (one decode pass, video only)."""
    command = ["ffmpeg", "-hide_banner", "-nostats", "-i", path, "-an", "-sn", "-dn",
               "-vf", f"select='gt(scene,{float(threshold)})',showinfo", "-f", "null", "-"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        raise StyleReferenceError(f"ffmpeg scene detection failed: {str(error)[:200]}") from error
    if result.returncode != 0:
        raise StyleReferenceError(f"ffmpeg scene detection failed: {result.stderr.strip()[-300:]}")
    times = {round(float(m.group(1)), 3) for line in result.stderr.splitlines() if "showinfo" in line
             for m in [_PTS_TIME.search(line)] if m}
    return sorted(t for t in times if t > 0)


def clean_cuts(cuts: Iterable[float], duration: float, *, min_shot: float = MIN_SHOT_SEC) -> list[float]:
    """Sorted cuts inside the video, dropping flashes shorter than ``min_shot``."""
    kept: list[float] = []
    for cut in sorted(float(c) for c in cuts):
        if cut - (kept[-1] if kept else 0.0) < min_shot or duration - cut < min_shot:
            continue
        kept.append(round(cut, 3))
    return kept


def shot_spans(cuts: Sequence[float], duration: float) -> list[dict]:
    edges = [0.0, *cuts, float(duration)]
    return [{"index": i, "start": round(a, 3), "end": round(b, 3), "duration": round(b - a, 3)}
            for i, (a, b) in enumerate(zip(edges, edges[1:]))]


def pacing_curve(shots: Sequence[Mapping[str, float]], duration: float,
                 window: float = PACING_WINDOW_SEC) -> list[dict]:
    """Average shot length per window: the window length over the shots it overlaps."""
    curve = []
    for k in range(max(1, math.ceil(duration / window))):
        start, end = k * window, min(duration, (k + 1) * window)
        if end <= start:
            break
        overlapping = [s for s in shots if s["start"] < end and s["end"] > start]
        cuts = sum(1 for s in shots if start < s["start"] < end)
        curve.append({"start": round(start, 2), "end": round(end, 2),
                      "asl_sec": round((end - start) / max(1, len(overlapping)), 2), "cuts": cuts})
    return curve


def length_stats(shots: Sequence[Mapping[str, float]], duration: float) -> dict:
    """Shot-length statistics of ``shots`` (pure): count, lengths, average, median, bounds, distribution, pacing."""
    lengths = [float(s["duration"]) for s in shots]
    covered = sum(lengths)
    distribution = []
    lower = 0.0
    for label_, upper in LENGTH_BUCKETS:
        count = sum(1 for d in lengths if lower <= d < upper)
        distribution.append({"bucket": label_, "count": count,
                             "percent": round(100 * count / len(lengths), 1) if lengths else 0.0})
        lower = upper
    return {
        "shot_count": len(shots),
        "shot_lengths": [round(d, 3) for d in lengths],
        "asl_sec": round(statistics.fmean(lengths), 2) if lengths else None,
        "median_sec": round(statistics.median(lengths), 2) if lengths else None,
        "min_sec": round(min(lengths), 2) if lengths else None,
        "max_sec": round(max(lengths), 2) if lengths else None,
        "cuts_per_minute": round(max(0, len(shots) - 1) / (covered / 60), 1) if covered > 0 else None,
        "distribution": distribution,
        "pacing": pacing_curve(shots, duration),
    }


def shot_stats(cuts: Sequence[float], duration: float) -> dict:
    """Editing statistics of a video from its cuts (pure)."""
    shots = shot_spans(cuts, duration)
    return {
        "duration_sec": round(float(duration), 3),
        "cut_count": len(cuts),
        "cuts": [round(c, 3) for c in cuts],
        **length_stats(shots, duration),
    }


def without_credits(shots: Sequence[Mapping[str, Any]], duration: float) -> tuple[list[dict], dict]:
    """(shots that are not credits, stats of those) once the model has flagged titles, credits and logos.

    The stats carry ``credits_shots`` and ``credits_sec`` for what was left out
    ({} when nothing was flagged: the measured stats then stand).
    """
    flagged = [s for s in shots if s.get("credits")]
    if not flagged:
        return [dict(s) for s in shots], {}
    kept = [dict(s) for s in shots if not s.get("credits")]
    return kept, {**length_stats(kept, duration), "credits_shots": len(flagged),
                  "credits_sec": round(sum(float(s["duration"]) for s in flagged), 1)}


def make_proxy(source: str, dest: str, *, height: int = PROXY_HEIGHT, has_audio: bool = True,
               timeout: float = 4 * 3600) -> str:
    """A 480p H.264 MP4 of ``source`` (never upscaled), mono AAC sound kept; raises StyleReferenceError."""
    command = ["ffmpeg", "-v", "error", "-y", "-i", source, "-map", "0:v:0",
               "-vf", f"scale=-2:'min({height},ih)',format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", str(PROXY_CRF)]
    if has_audio:
        command += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", PROXY_AUDIO_BITRATE, "-ac", "1"]
    else:
        command += ["-an"]
    command += ["-movflags", "+faststart", dest]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        raise StyleReferenceError(f"Could not reduce the video to {height}p: {str(error)[:200]}") from error
    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        raise StyleReferenceError(f"The {height}p proxy is empty")
    return dest


def ensure_proxy(bible_id: str, reference: Mapping[str, Any], *, height: int = PROXY_HEIGHT) -> dict:
    """Replace a reference's upload with its ``height``p proxy once; returns the (updated) reference.

    A video already at ``height`` or smaller is kept as is. The upload's size
    and dimensions stay on the reference as ``source``.
    """
    if reference.get("source"):
        return dict(reference)
    path = bibles.video_path(bible_id, reference.get("file") or "")
    if path is None:
        raise StyleReferenceError("The video file is missing")
    facts = probe(str(path))
    source = {"width": facts.get("width"), "height": facts.get("height"), "bytes": os.path.getsize(path),
              "duration_sec": facts.get("duration_sec")}
    small = isinstance(facts.get("height"), int) and facts["height"] <= height
    if small and str(path).lower().endswith(".mp4"):
        return _set(bible_id, reference["id"], source=source)
    dest = path.with_name(f"{path.stem}_{height}p.mp4")
    make_proxy(str(path), str(dest), height=height, has_audio=bool(facts.get("has_audio", True)))
    updated = _set(bible_id, reference["id"], file=dest.name, source=source)
    path.unlink(missing_ok=True)
    return updated


def local_analysis(path: str, *, threshold: float = SCENE_CUT_THRESHOLD) -> dict:
    """The exact pass: probe facts plus editing statistics."""
    facts = probe(path)
    cuts = clean_cuts(detect_cuts(path, threshold=threshold), facts["duration_sec"])
    return {**facts, **shot_stats(cuts, facts["duration_sec"]), "scene_threshold": threshold}


# ── chunks for the model pass ──────────────────────────────────────────────


def plan_chunks(shots: Sequence[Mapping[str, float]], *, max_sec: float = MAX_CHUNK_SEC,
                limit_sec: float | None = MAX_ANALYSED_SEC) -> list[dict]:
    """Consecutive shots grouped into chunks of at most ``max_sec``, cut on shot boundaries.

    A shot longer than ``max_sec`` gets a chunk of its own, cut at ``max_sec``.
    Shots starting after ``limit_sec`` are left out.
    """
    chunks: list[dict] = []
    current: list[Mapping[str, float]] = []

    def close() -> None:
        if current:
            start = current[0]["start"]
            end = min(current[-1]["end"], start + max_sec)
            chunks.append({"start": round(start, 3), "end": round(end, 3),
                           "shots": [int(s["index"]) for s in current]})

    for shot in shots:
        if limit_sec is not None and shot["start"] >= limit_sec:
            break
        if current and shot["end"] - current[0]["start"] > max_sec:
            close()
            current = []
        current.append(shot)
    close()
    return chunks


def encode_chunk(source: str, start: float, end: float, dest: str, *, has_audio: bool = True,
                 timeout: float = 600) -> str:
    """A small re-encode of one chunk: CHUNK_WIDTH wide, low bitrate, mono sound kept."""
    command = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", source, "-t", f"{max(0.1, end - start):.3f}",
               "-vf", f"scale='min({CHUNK_WIDTH},iw)':-2,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast",
               "-crf", str(CHUNK_CRF)]
    command += ["-c:a", "aac", "-b:a", CHUNK_AUDIO_BITRATE, "-ac", "1"] if has_audio else ["-an"]
    command += ["-movflags", "+faststart", dest]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError) as error:
        raise StyleReferenceError(f"Could not encode a chunk: {str(error)[:200]}") from error
    return dest


def data_uri(path: str) -> str:
    with open(path, "rb") as handle:
        return "data:video/mp4;base64," + base64.b64encode(handle.read()).decode()


def chunk_prompt(chunk: Mapping[str, Any], shots: Sequence[Mapping[str, Any]], number: int, count: int,
                 total: float) -> str:
    origin = float(chunk["start"])
    lines = []
    for n, index in enumerate(chunk["shots"], 1):
        shot = shots[index]
        end = min(float(shot["end"]), float(chunk["end"]))
        lines.append(f"{n}: {shot['start'] - origin:.2f}-{end - origin:.2f} s ({end - shot['start']:.2f} s)")
    return (f"Excerpt {number} of {count}: {origin:.1f}-{float(chunk['end']):.1f} s of a {total:.1f} s video.\n"
            f"Shot list of this excerpt ({len(lines)} shots):\n" + "\n".join(lines))


# ── stills for the local backend ──────────────────────────────────────────


def still_times(shot: Mapping[str, float], *, long_shot: float = LONG_SHOT_SEC) -> list[float]:
    """When a shot is sampled: its middle, or a quarter and three quarters in when it is long enough to move."""
    start, end = float(shot["start"]), float(shot["end"])
    length = max(0.0, end - start)
    if length >= long_shot:
        return [round(start + length * 0.25, 3), round(start + length * 0.75, 3)]
    return [round(start + length / 2, 3)]


def plan_local_chunks(shots: Sequence[Mapping[str, float]], *, max_stills: int = MAX_LOCAL_STILLS,
                      limit_sec: float | None = MAX_LOCAL_ANALYSED_SEC) -> list[dict]:
    """Consecutive shots grouped so that one call shows at most ``max_stills`` stills (same shape as ``plan_chunks``)."""
    chunks: list[dict] = []
    current: list[Mapping[str, float]] = []
    count = 0

    def close() -> None:
        if current:
            chunks.append({"start": round(float(current[0]["start"]), 3), "end": round(float(current[-1]["end"]), 3),
                           "shots": [int(s["index"]) for s in current]})

    for shot in shots:
        if limit_sec is not None and shot["start"] >= limit_sec:
            break
        needed = len(still_times(shot))
        if current and count + needed > max_stills:
            close()
            current, count = [], 0
        current.append(shot)
        count += needed
    close()
    return chunks


def shot_stills(source: str, chunk: Mapping[str, Any], shots: Sequence[Mapping[str, Any]], dest_dir: str,
                *, long_side: int = STILL_LONG_SIDE, timeout: float = 120) -> list[tuple[int, float, str]]:
    """(shot number in the chunk, time, jpeg path) of every still of ``chunk``; raises StyleReferenceError."""
    scale = f"scale=w='if(gt(iw,ih),min({long_side},iw),-2)':h='if(gt(iw,ih),-2,min({long_side},ih))'"
    stills: list[tuple[int, float, str]] = []
    for n, index in enumerate(chunk["shots"], 1):
        for seconds in still_times(shots[index]):
            dest = os.path.join(dest_dir, f"shot_{n:03d}_{len(stills):03d}.jpg")
            command = ["ffmpeg", "-v", "error", "-y", "-ss", f"{seconds:.3f}", "-i", source, "-frames:v", "1",
                       "-vf", scale, "-q:v", "4", dest]
            try:
                subprocess.run(command, check=True, capture_output=True, timeout=timeout)
            except (OSError, subprocess.SubprocessError) as error:
                raise StyleReferenceError(f"Could not extract a still: {str(error)[:200]}") from error
            if not os.path.exists(dest):
                raise StyleReferenceError(f"No still at {seconds:.1f} s")
            stills.append((n, seconds, dest))
    return stills


def stills_note(stills: Sequence[tuple[int, float, str]], origin: float) -> str:
    """How the stills map to the shot list: appended to the chunk prompt for the local backend."""
    labels = ", ".join(f"Image {i} = shot {n} at {seconds - origin:.1f} s" for i, (n, seconds, _) in enumerate(stills, 1))
    return ("This excerpt is shown as silent stills, one per shot (two for a longer shot, a quarter and three "
            f"quarters in): {labels}. You cannot hear the sound: set \"audio\" from what the picture suggests "
            "(dialogue when people talk to each other, ambience otherwise) and read movement from how the two "
            "stills of a shot differ.")


# ── parsing ────────────────────────────────────────────────────────────────


def extract_json(text: str) -> Any:
    """The JSON value in a model answer (fenced or surrounded by prose); raises StyleReferenceError."""
    body = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.I)
    try:
        return json.loads(body)
    except ValueError:
        pass
    for pattern in (r"\{.*\}", r"\[.*\]"):
        match = re.search(pattern, body, re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except ValueError:
                continue
    raise StyleReferenceError("The model returned no valid JSON")


def _enum(value: Any, allowed: Sequence[str], fallback: str) -> str:
    key = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    aliases = {"ecu": "extreme_close_up", "cu": "close_up", "mcu": "medium_close_up", "ms": "medium",
               "ws": "wide", "ews": "extreme_wide", "over_the_shoulder": "ots", "closeup": "close_up",
               "medium_closeup": "medium_close_up", "extreme_closeup": "extreme_close_up", "2_shot": "two_shot",
               "point_of_view": "pov", "dolly_in": "push_in", "dolly_out": "pull_out", "cut": "hard_cut"}
    key = aliases.get(key, key)
    return key if key in allowed else fallback


def _short(value: Any, limit: int = 200) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:limit]


def parse_chunk_shots(text: str, chunk: Mapping[str, Any]) -> dict[int, dict]:
    """Per-shot records of one chunk, keyed by the global shot index; raises when unusable."""
    raw = extract_json(text)
    items = raw.get("shots") if isinstance(raw, dict) else raw
    if not isinstance(items, list) or not items:
        raise StyleReferenceError("The model returned no shots")
    indices = list(chunk["shots"])
    records: dict[int, dict] = {}
    for position, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        try:
            number = int(item.get("shot")) - 1
        except (TypeError, ValueError):
            number = position
        if not 0 <= number < len(indices) or indices[number] in records:
            continue
        records[indices[number]] = {
            "shot_size": _enum(item.get("shot_size"), SHOT_SIZES, "medium"),
            "angle": _enum(item.get("angle"), ANGLES, "eye_level"),
            "movement": _enum(item.get("movement"), MOVEMENTS, "static"),
            "subject": _short(item.get("subject")),
            "action": _short(item.get("action")),
            "cut_reason": _enum(item.get("cut_reason"), CUT_REASONS, "other"),
            "transition": _enum(item.get("transition"), TRANSITIONS, "hard_cut"),
            "lighting": _short(item.get("lighting"), 160),
            "audio": _enum(item.get("audio"), AUDIO_KINDS, "mixed"),
            **({"false_cut": True} if item.get("false_cut") is True else {}),
            **({"credits": True} if item.get("credits") is True else {}),
            **({"graphics": _short(item.get("graphics"), 160)} if _short(item.get("graphics"), 160)
               and _short(item.get("graphics"), 160).lower() != "none" else {}),
        }
    if not records:
        raise StyleReferenceError("The model's shots match none of the listed shots")
    return records


def shot_mix(shots: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Percent of the described shots per shot size, largest first."""
    sizes = [s["shot_size"] for s in shots if s.get("shot_size") in SHOT_SIZES]
    if not sizes:
        return {}
    counts = {size: sizes.count(size) for size in SHOT_SIZES if size in sizes}
    return {size: round(100 * n / len(sizes), 1) for size, n in sorted(counts.items(), key=lambda kv: -kv[1])}


def motion_mix(shots: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Percent of the described shots per camera movement, largest first."""
    moves = [s["movement"] for s in shots if s.get("movement") in MOVEMENTS]
    if not moves:
        return {}
    counts = {move: moves.count(move) for move in MOVEMENTS if move in moves}
    return {move: round(100 * n / len(moves), 1) for move, n in sorted(counts.items(), key=lambda kv: -kv[1])}


def _median_p75(values: Sequence[float]) -> list[float] | None:
    values = sorted(float(v) for v in values if v and v > 0)
    if len(values) < 3:
        return None
    p75 = values[min(len(values) - 1, round(0.75 * (len(values) - 1)))]
    return [round(statistics.median(values), 2), round(p75, 2)]


def audio_lengths(shots: Sequence[Mapping[str, Any]]) -> dict[str, list[float]]:
    """[median, p75] length of shots under dialogue and of silent shots (silence or ambience only)."""
    groups = {"line_sec": [s.get("duration") for s in shots if s.get("audio") == "dialogue"],
              "silent_sec": [s.get("duration") for s in shots if s.get("audio") in ("silence", "ambience")]}
    result = {}
    for key, values in groups.items():
        pair = _median_p75([float(v) for v in values if isinstance(v, (int, float))])
        if pair:
            result[key] = pair
    return result


def reference_measures(reference: Mapping[str, Any], shots: Sequence[Mapping[str, Any]] | None = None) -> dict:
    """The measured numbers a reference brings to the cinema grammar calibration ({} without any).

    Exact cut statistics (average and median shot, cuts per minute), the
    shot-size mix, the static share of camera movement and the shot length
    under a line vs silent. Older analyses without the movement and sound
    splits in their stats get them from their per-shot breakdown (``shots``).
    The weight is the reference's shot count.
    """
    stats = reference.get("stats") or {}
    grammar = reference.get("grammar") or {}
    measures: dict[str, Any] = {}
    asl = _number(stats.get("asl_sec")) or _number((grammar.get("rhythm") or {}).get("target_asl_sec"))
    for key, value in (("asl_sec", asl), ("median_sec", _number(stats.get("median_sec"))),
                       ("cuts_per_minute", _number(stats.get("cuts_per_minute")))):
        if value and value > 0:
            measures[key] = value
    mix = stats.get("shot_mix") or grammar.get("shot_mix")
    if isinstance(mix, Mapping) and mix:
        measures["shot_mix"] = dict(mix)
    motion = stats.get("motion_mix")
    if not motion and shots:
        motion = motion_mix(shots)
    if motion:
        measures["static_share"] = round(float(motion.get("static", 0)) / 100, 3)
    lengths = stats.get("audio_lengths")
    if not lengths and shots:
        lengths = audio_lengths(shots)
    for key in ("line_sec", "silent_sec"):
        if (lengths or {}).get(key):
            measures[key] = list(lengths[key])
    if measures:
        measures["id"] = reference.get("id")
        measures["weight"] = int(stats.get("shot_count") or reference.get("described_shots") or 1)
    return measures


def _text_list(value: Any, limit: int, size: int = 300) -> list[str]:
    if isinstance(value, str):
        value = [line for line in re.split(r"\n+", value)]
    if not isinstance(value, list):
        return []
    items = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", _short(item, size)) for item in value
             if isinstance(item, (str, int, float))]
    return [item for item in items if item][:limit]


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_grammar(text: str, *, max_rules: int = 10, scene_notes: bool = False) -> tuple[dict, str]:
    """(grammar, summary markdown) from the synthesis answer; raises when it has no rules.

    ``scene_notes`` keeps the merged grammar's per-tag notes.
    """
    raw = extract_json(text)
    if not isinstance(raw, dict):
        raise StyleReferenceError("The synthesis returned no JSON object")
    rhythm = raw.get("rhythm") if isinstance(raw.get("rhythm"), dict) else {}
    target = _number(rhythm.get("target_asl_sec"))
    mix_raw = raw.get("shot_mix") if isinstance(raw.get("shot_mix"), dict) else {}
    mix = {}
    for key, value in mix_raw.items():
        size, percent = _enum(key, SHOT_SIZES, ""), _number(value)
        if size and percent is not None and percent > 0:
            mix[size] = round(min(100.0, percent), 1)
    rules = _text_list(raw.get("rules"), max_rules)
    if not rules:
        raise StyleReferenceError("The synthesis returned no rules")
    notes = raw.get("scene_notes") if scene_notes and isinstance(raw.get("scene_notes"), dict) else {}
    notes = {bibles._slug(str(k), 24): _short(v, 300) for k, v in list(notes.items())[:8] if _short(v, 300)}
    notes = {k: v for k, v in notes.items() if k}
    grammar = {
        "rhythm": {"target_asl_sec": round(target, 2) if target and target > 0 else None,
                   "pace_with_emotion": _short(rhythm.get("pace_with_emotion"), 400),
                   "notes": _short(rhythm.get("notes"), 400)},
        "shot_mix": mix,
        "patterns": _text_list(raw.get("patterns"), 8),
        **{key: _short(raw.get(key), 400) for key in ("scene_openings", "scene_closings", "dialogue_coverage",
                                                      "camera_movement", "lighting_color", "music_silence")},
        **({"graphics": _short(raw.get("graphics"), 400)} if _short(raw.get("graphics"), 400) else {}),
        "rules": rules,
        **({"scene_notes": notes} if notes else {}),
    }
    summary = str(raw.get("summary_markdown") or raw.get("summary") or "").strip()[:8000]
    return grammar, summary


# ── model pass ─────────────────────────────────────────────────────────────


def synthesis_prompt(stats: Mapping[str, Any], shots: Sequence[Mapping[str, Any]],
                     graphics: str = GRAPHICS_IGNORE) -> str:
    facts = {key: stats.get(key) for key in ("duration_sec", "shot_count", "asl_sec", "median_sec", "min_sec",
                                             "max_sec", "cuts_per_minute", "distribution", "pacing", "shot_mix")}
    described = [s for s in shots if s.get("shot_size")]
    if len(described) > MAX_SYNTHESIS_SHOTS:
        step = len(described) / MAX_SYNTHESIS_SHOTS
        described = [described[int(i * step)] for i in range(MAX_SYNTHESIS_SHOTS)]
    with_graphics = graphics == GRAPHICS_INCLUDE
    lines = [" | ".join([f"{s['index'] + 1}", f"{s['start']:.1f}s", f"{s['duration']:.1f}s", s["shot_size"],
                         s.get("angle", ""), s.get("movement", ""), s.get("cut_reason", ""),
                         s.get("transition", ""), s.get("audio", ""),
                         _short(f"{s.get('subject', '')}: {s.get('action', '')}", 140)]
                        + ([_short(s.get("graphics") or "none", 100)] if with_graphics else []))
             for s in described]
    return ("Measured editing statistics (exact):\n" + json.dumps(facts, ensure_ascii=False)
            + "\n\nShot breakdown (number | start | length | size | angle | movement | cut reason | transition | "
              "sound | subject: action" + (" | graphics" if with_graphics else "") + "):\n" + "\n".join(lines))


def _ask_text(http: Any, fal_key: str, model: str, system: str, prompt: str, ask: Callable[..., tuple[str, Any]],
              sleep: Callable[[float], None]) -> tuple[str, Any]:
    """A text-only call: fal's text router, else the video router without a video."""
    try:
        return ask(http, fal_key, model, system, prompt, [], sleep, max_tokens=12000, timeout=300,
                   router=TEXT_ROUTER)
    except JudgeError as error:
        if not str(error).startswith("fal.ai error 4"):
            raise
        return ask(http, fal_key, model, system, prompt, [], sleep, max_tokens=12000, timeout=300,
                   router=FAL_ROUTER)


def _resolve_backend(backend: Any, fal_key: str | None, model: str,
                     ask: Callable[..., tuple[str, Any]]) -> tuple[Callable[..., tuple[str, Any]], str, str, bool]:
    """(ask, key, model, local) for the model pass; raises StyleReferenceError when neither backend can be used."""
    if backend is not None and getattr(backend, "provider", "") == "openai":
        if not backend.available:
            raise StyleReferenceError("The local judge has no server URL (Settings → Services); "
                                      "the measured stats are saved")
        # The backend's own key authenticates (``key or self.key``), never fal's.
        return backend.ask, "", str(getattr(backend, "model", "") or LOCAL_MODEL), True
    if not fal_key:
        raise StyleReferenceError("The model pass needs a fal.ai API key or the local judge (Settings → Services); "
                                  "the measured stats are saved")
    return ask, str(fal_key), model, False


def analyse_reference(
    path: str,
    *,
    fal_key: str | None,
    model: str = DEFAULT_MODEL,
    ask: Callable[..., tuple[str, Any]] = _ask,
    http: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    on_progress: Callable[[str, dict], None] = lambda message, partial: None,
    local: Callable[[str], dict] = local_analysis,
    backend: Any = None,
    graphics: str = GRAPHICS_IGNORE,
) -> dict:
    """Both passes on one video: {stats, shots, grammar, summary, cost_usd, model, chunk_errors}.

    ``graphics`` (GRAPHICS_MODES) says whether captions, logos and other
    on-screen graphics are left out of the staging or read into the grammar.

    ``on_progress(message, partial)`` gets the local stats as soon as they
    exist, so they survive a failing model pass. Raises StyleReferenceError
    when the model pass gives nothing usable. A ``backend`` of the ``openai``
    provider (``judge_backend.JudgeBackend``) replaces fal: its ``ask`` is
    called with stills instead of video chunks, with its model (``LOCAL_MODEL``
    when it names none), for nothing.
    """
    stats = local(path)
    shots = [dict(s) for s in shot_spans(stats["cuts"], stats["duration_sec"])]
    on_progress(f"Measured {stats['shot_count']} shots", {"stats": stats})
    ask, fal_key, model, local_backend = _resolve_backend(backend, fal_key, model, ask)
    graphics = graphics if graphics in GRAPHICS_MODES else GRAPHICS_IGNORE
    shot_system = SHOT_SYSTEM_PROMPT + "\n" + GRAPHICS_NOTES[graphics]
    synthesis_system = SYNTHESIS_SYSTEM_PROMPT + GRAPHICS_SYNTHESIS[graphics]
    chunks = plan_local_chunks(shots) if local_backend else plan_chunks(shots)
    cost = 0.0
    errors: list[str] = []
    owns_http = http is None
    if owns_http:
        import httpx
        http = httpx.Client(timeout=120)
    try:
        with tempfile.TemporaryDirectory(prefix="director-styleref-") as tmp:
            for number, chunk in enumerate(chunks, 1):
                on_progress(f"Analysing excerpt {number} of {len(chunks)}", {})
                prompt = chunk_prompt(chunk, shots, number, len(chunks), stats["duration_sec"])
                try:
                    if local_backend:
                        still_dir = os.path.join(tmp, f"stills_{number}")
                        os.makedirs(still_dir, exist_ok=True)
                        stills = shot_stills(path, chunk, shots, still_dir)
                        media = [still for _, _, still in stills]
                        prompt += "\n" + stills_note(stills, float(chunk["start"]))
                    else:
                        clip = encode_chunk(path, chunk["start"], chunk["end"],
                                            os.path.join(tmp, f"chunk_{number}.mp4"),
                                            has_audio=bool(stats.get("has_audio", True)))
                        media = [data_uri(clip)]
                    last: Exception | None = None
                    for _ in range(CHUNK_ATTEMPTS):
                        text, call_cost = ask(http, fal_key, model, shot_system, prompt, media, sleep,
                                              max_tokens=12000)
                        cost += float(call_cost or 0)
                        try:
                            records = parse_chunk_shots(text, chunk)
                        except StyleReferenceError as error:
                            last = error
                            continue
                        for index, record in records.items():
                            shots[index].update(record)
                        last = None
                        break
                    if last:
                        raise last
                except (StyleReferenceError, JudgeError, OSError) as error:
                    errors.append(f"Excerpt {number}: {str(error)[:200]}")
        kept, trimmed = without_credits(shots, stats["duration_sec"])
        described = [s for s in kept if s.get("shot_size")]
        if not described:
            raise StyleReferenceError(errors[0] if errors else "The model described no shot")
        stats = {**stats, **trimmed, "shot_mix": shot_mix(described), "motion_mix": motion_mix(described),
                 "audio_lengths": audio_lengths(described), "described_shots": len(described)}
        if trimmed:
            on_progress(f"Left out {trimmed['credits_shots']} credits shots ({trimmed['credits_sec']:.0f} s)",
                        {"stats": stats})
        on_progress("Writing the staging grammar", {"stats": stats})
        prompt = synthesis_prompt(stats, kept, graphics)
        last_error: Exception | None = None
        for _ in range(SYNTHESIS_ATTEMPTS):
            text, call_cost = _ask_text(http, fal_key, model, synthesis_system, prompt, ask, sleep)
            cost += float(call_cost or 0)
            try:
                grammar, summary = parse_grammar(text)
            except StyleReferenceError as error:
                last_error = error
                continue
            break
        else:
            raise StyleReferenceError(f"Synthesis failed: {last_error}")
    finally:
        if owns_http:
            http.close()
    # Measured facts win over the model's estimates.
    if stats.get("shot_mix"):
        grammar["shot_mix"] = stats["shot_mix"]
    if not grammar["rhythm"].get("target_asl_sec"):
        grammar["rhythm"]["target_asl_sec"] = stats.get("asl_sec")
    return {
        "stats": stats,
        "shots": shots[:MAX_STORED_SHOTS],
        "grammar": grammar,
        "summary": summary,
        "cost_usd": round(cost, 5),
        "model": model,
        "chunk_errors": errors,
    }


# ── planner block ──────────────────────────────────────────────────────────


def _mix_text(mix: Mapping[str, float], limit: int = 6) -> str:
    return ", ".join(f"{SHOT_SIZE_LABELS.get(k, k)} {v:g}%" for k, v in list(mix.items())[:limit])


def reference_block(reference: Mapping[str, Any], limit: int = MAX_REFERENCE_CHARS, *, numbers: bool = True) -> str:
    """One reference's grammar as compact planner lines, at most ``limit`` characters.

    ``numbers=False`` leaves out the measured rhythm and shot mix when the
    CINEMA GRAMMAR block already carries them (calibrated from the references).
    """
    grammar = reference.get("grammar") or {}
    stats = reference.get("stats") or {}
    rhythm = grammar.get("rhythm") or {}
    tags = reference.get("scene_tags") or []
    head = f"Reference \"{_short(reference.get('label'), 60)}\"" + (
        f" (for scenes: {', '.join(tags)})" if tags else " (for every scene)") + ":"
    target = rhythm.get("target_asl_sec") or stats.get("asl_sec")
    rhythm_bits = []
    if target:
        rhythm_bits.append(f"target average shot {float(target):.1f} s")
    if stats.get("median_sec"):
        rhythm_bits.append(f"median {float(stats['median_sec']):.1f} s")
    if stats.get("cuts_per_minute"):
        rhythm_bits.append(f"{float(stats['cuts_per_minute']):g} cuts/min")
    distribution = [f"{d['bucket']} {d['percent']:g}%" for d in stats.get("distribution") or [] if d.get("percent")]
    if distribution:
        rhythm_bits.append("lengths " + ", ".join(distribution))
    lines = [head]
    if rhythm_bits and numbers:
        lines.append("- Rhythm: " + "; ".join(rhythm_bits) + ".")
    if rhythm.get("pace_with_emotion"):
        lines.append("- Pace: " + _short(rhythm["pace_with_emotion"], 220))
    rules = [f"- Rule: {_short(rule, 220)}" for rule in grammar.get("rules") or []]
    optional = []
    if grammar.get("shot_mix") and numbers:
        optional.append("- Shot mix: " + _mix_text(grammar["shot_mix"]) + ".")
    for key, title in (("dialogue_coverage", "Dialogue"), ("scene_openings", "Openings"),
                       ("scene_closings", "Closings"), ("camera_movement", "Camera"),
                       ("music_silence", "Sound"), ("lighting_color", "Light"), ("graphics", "Graphics")):
        if grammar.get(key):
            optional.append(f"- {title}: {_short(grammar[key], 180)}")
    optional += [f"- Pattern: {_short(p, 180)}" for p in grammar.get("patterns") or []]
    text = ""
    for line in lines + rules + optional:
        candidate = f"{text}\n{line}" if text else line
        if len(candidate) <= limit:  # a line that does not fit is skipped; a shorter one may still fit
            text = candidate
    return text


def _tag_matches(tag: str, brief: str) -> bool:
    """A tag appears in the brief, word for word or as a word sharing its first four letters (argument, argue)."""
    words = re.findall(r"[a-z]+", brief.lower())
    stem = tag.replace("_", " ").lower()
    return stem in brief.lower() or (len(stem) >= 4 and any(len(w) >= 4 and w[:4] == stem[:4] for w in words))


def usable(reference: Mapping[str, Any]) -> bool:
    return reference.get("status") == "done" and bool((reference.get("grammar") or {}).get("rules"))


def select_references(bible: Mapping[str, Any] | None, choice: Any = CHOICE_AUTO) -> list[dict]:
    """References a run uses: "auto" = every enabled, analysed one; "off" = none; else the one with that id."""
    choice = str(choice or CHOICE_AUTO).strip()
    references = [r for r in (bible or {}).get("style_references") or [] if usable(r)]
    if choice.lower() == CHOICE_OFF:
        return []
    if choice.lower() == CHOICE_AUTO:
        return [r for r in references if r.get("enabled", True)]
    return [r for r in references if r.get("id") == choice]


def merged_block(merged: Mapping[str, Any], references: Sequence[Mapping[str, Any]], limit: int, *,
                 brief: str = "", numbers: bool = True) -> str:
    """The consolidated grammar as planner lines: one block, then the notes of the tags the brief names first."""
    grammar = merged.get("grammar") or {}
    pseudo = {"label": f"all {len(references)} references combined", "grammar": grammar, "stats": None,
              "scene_tags": []}
    text = reference_block(pseudo, limit, numbers=numbers)
    notes = grammar.get("scene_notes") or {}
    ordered = sorted(notes.items(), key=lambda item: 0 if _tag_matches(item[0], brief) else 1)
    for tag, note in ordered:
        line = f"- For {tag.replace('_', ' ')} scenes: {_short(note, 300)}"
        if len(text) + 1 + len(line) <= limit:
            text += "\n" + line
    return text


def planner_block(references: Sequence[Mapping[str, Any]], *, brief: str = "",
                  total_limit: int = MAX_BLOCK_CHARS, per_reference: int = MAX_REFERENCE_CHARS,
                  numbers: bool = True, merged: Mapping[str, Any] | None = None) -> str:
    """The "REFERENCE STAGING GRAMMAR" planner block ("" without references), at most ``total_limit`` chars.

    With a ``merged`` grammar (``merged_grammar``) the block is that one
    grammar, so every reference speaks whatever their number. Otherwise the
    references are stacked: tagged ones whose tags appear in the brief first,
    then untagged ones, then the other tagged ones; references that no longer
    fit are named as left out.
    """
    if not references:
        return ""
    tagged = any(r.get("scene_tags") for r in references)
    rhythm_note = ("Shot durations stay inside the allowed range; follow the target rhythm within it." if numbers
                   else "Their measured rhythm and shot mix are in the CINEMA GRAMMAR block.")
    head = ("REFERENCE STAGING GRAMMAR (editing and staging principles learned from reference videos chosen for "
            "this series: apply them to this production's own story and cast, never copy a reference's content, "
            f"characters or story. {rhythm_note})")
    if merged and (merged.get("grammar") or {}).get("rules"):
        body = merged_block(merged, references, total_limit - len(head) - 1, brief=brief, numbers=numbers)
        return (head + "\n" + body)[:total_limit] if body else ""
    if tagged:
        head += ("\nEach reference names the scenes it guides: for each scene, follow the reference whose scenes "
                 "fit its mood; references for every scene always apply.")

    def rank(reference: Mapping[str, Any]) -> int:
        tags = reference.get("scene_tags") or []
        if not tags:
            return 1
        return 0 if any(_tag_matches(t, brief) for t in tags) else 2

    ordered = sorted(references, key=rank)
    budget = total_limit - len(head) - 1
    share = max(400, min(per_reference, budget // len(ordered)))
    parts, skipped = [], []
    for reference in ordered:
        block = reference_block(reference, share, numbers=numbers)
        if block and len(block) + 2 <= budget:
            parts.append(block)
            budget -= len(block) + 2
        else:
            skipped.append(_short(reference.get("label"), 40))
    if not parts:
        return ""
    text = head + "\n" + "\n\n".join(parts)
    if skipped:
        note = "\n(Left out for length: " + ", ".join(skipped) + ".)"
        if len(text) + len(note) <= total_limit:
            text += note
    return text[:total_limit]


# ── storage in the bible ───────────────────────────────────────────────────


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def find_reference(bible: Mapping[str, Any], ref_id: str) -> dict:
    reference = next((r for r in bible.get("style_references") or [] if r.get("id") == ref_id), None)
    if reference is None:
        raise BibleError("Unknown style reference")
    return reference


def list_references(bible_id: str) -> list[dict]:
    bible = bibles.get_bible(bible_id)
    if not bible:
        raise BibleError("Unknown bible")
    return bible.get("style_references") or []


def get_reference(bible_id: str, ref_id: str) -> dict:
    bible = bibles.get_bible(bible_id)
    if not bible:
        raise BibleError("Unknown bible")
    return find_reference(bible, ref_id)


def shots_path(bible_id: str, reference: Mapping[str, Any]):
    """The sidecar file of a reference's per-shot breakdown (kept out of bible.json, which is read often)."""
    return bibles._folder(bible_id) / f"{os.path.splitext(reference['file'])[0]}.shots.json"


def save_shots(bible_id: str, reference: Mapping[str, Any], shots: Sequence[Mapping[str, Any]]) -> None:
    path = shots_path(bible_id, reference)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(list(shots), ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def load_shots(bible_id: str, reference: Mapping[str, Any]) -> list[dict]:
    try:
        data = json.loads(shots_path(bible_id, reference).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [s for s in data if isinstance(s, dict)] if isinstance(data, list) else []


def parse_tags(value: Any) -> list[str]:
    items = value if isinstance(value, list) else re.split(r"[,;\n]+", str(value or ""))
    tags = [bibles._slug(str(item), 24) for item in items]
    return list(dict.fromkeys(t for t in tags if t))[:8]


def add_reference(bible_id: str, stored_file: str, *, source_filename: str = "", label: str = "",
                  scene_tags: Any = None, graphics: Any = GRAPHICS_IGNORE) -> dict:
    """Register a video already written into the bible's folder; returns the new reference."""
    name = os.path.basename(stored_file)

    def change(bible: dict) -> dict:
        references = bible.setdefault("style_references", [])
        if len(references) >= bibles.MAX_STYLE_REFERENCES:
            raise BibleError(f"A bible keeps at most {bibles.MAX_STYLE_REFERENCES} style references")
        reference = {
            "id": "ref_" + secrets.token_hex(4), "label": _short(label, 80) or _short(
                os.path.splitext(source_filename)[0], 80) or "Reference",
            "file": name, "source_filename": _short(source_filename, 160), "created_at": _now(),
            "updated_at": _now(), "status": "failed", "error": "Not analysed yet", "progress": "",
            "enabled": True, "scene_tags": parse_tags(scene_tags), "stats": None, "described_shots": 0,
            "graphics": parse_graphics(graphics), "grammar": None,
            "summary": "", "cost_usd": None, "model": "",
        }
        references.append(reference)
        return reference
    return bibles._mutate(bible_id, change)[1]


def parse_graphics(value: Any) -> str:
    """A GRAPHICS_MODES value from the API (a mode name, or a boolean meaning include)."""
    if isinstance(value, bool):
        return GRAPHICS_INCLUDE if value else GRAPHICS_IGNORE
    text = str(value or "").strip().lower()
    if text in GRAPHICS_MODES:
        return text
    return GRAPHICS_INCLUDE if text in ("true", "1", "yes", "on") else GRAPHICS_IGNORE


def update_reference(bible_id: str, ref_id: str, patch: Mapping[str, Any]) -> dict:
    """Edit what the user owns: label, enabled, scene_tags, graphics and the grammar rules (a list or one per line).

    A new ``graphics`` mode applies to the next analysis.
    """
    def change(bible: dict) -> dict:
        reference = find_reference(bible, ref_id)
        if "label" in patch:
            reference["label"] = _short(patch.get("label"), 80) or reference["label"]
        if "graphics" in patch:
            reference["graphics"] = parse_graphics(patch.get("graphics"))
        if "enabled" in patch:
            reference["enabled"] = bool(patch.get("enabled"))
        if "scene_tags" in patch:
            reference["scene_tags"] = parse_tags(patch.get("scene_tags"))
        if "rules" in patch:
            if not isinstance(reference.get("grammar"), dict):
                raise BibleError("Analyse the reference before editing its rules")
            reference["grammar"]["rules"] = _text_list(patch.get("rules"), 12, 400)
            reference["grammar"]["rules_edited"] = True
        reference["updated_at"] = _now()
        return reference
    return bibles._mutate(bible_id, change)[1]


def _set(bible_id: str, ref_id: str, **fields: Any) -> dict:
    def change(bible: dict) -> dict:
        reference = find_reference(bible, ref_id)
        reference.update(fields, updated_at=_now())
        return reference
    return bibles._mutate(bible_id, change)[1]


def delete_reference(bible_id: str, ref_id: str) -> list[dict]:
    def change(bible: dict) -> str:
        reference = find_reference(bible, ref_id)
        bible["style_references"] = [r for r in bible["style_references"] if r["id"] != ref_id]
        return reference.get("file") or ""
    with bibles._LOCK:
        bible, name = bibles._mutate(bible_id, change)
        if name and name not in {r.get("file") for r in bible.get("style_references") or []}:
            path = bibles.video_path(bible_id, name)
            if path is not None:
                path.unlink(missing_ok=True)
            shots_path(bible_id, {"file": name}).unlink(missing_ok=True)
    bibles._LIVE_ANALYSES.discard(ref_id)
    return bible.get("style_references") or []


# ── one grammar for several references ─────────────────────────────────────


def merge_sources(references: Sequence[Mapping[str, Any]]) -> list[dict]:
    """What a merged grammar was built from: enough to tell when it is stale."""
    return sorted(({"id": str(r.get("id") or ""), "updated_at": str(r.get("updated_at") or ""),
                    "scene_tags": list(r.get("scene_tags") or [])} for r in references), key=lambda s: s["id"])


def merge_candidates(bible: Mapping[str, Any] | None) -> list[dict]:
    """The references a merged grammar covers: every enabled, analysed one (the "auto" selection)."""
    return select_references(bible, CHOICE_AUTO)


def merged_grammar(bible: Mapping[str, Any] | None, references: Sequence[Mapping[str, Any]]) -> dict | None:
    """The bible's merged grammar when it is done and covers exactly ``references``, else None."""
    merged = (bible or {}).get("style_grammar") or {}
    if merged.get("status") != "done" or not (merged.get("grammar") or {}).get("rules"):
        return None
    if len(references) < MIN_MERGE_REFERENCES or merged.get("sources") != merge_sources(references):
        return None
    return merged


def merge_prompt(references: Sequence[Mapping[str, Any]]) -> str:
    parts = []
    for n, reference in enumerate(references, 1):
        stats = reference.get("stats") or {}
        facts = {key: stats.get(key) for key in ("asl_sec", "median_sec", "cuts_per_minute", "shot_mix",
                                                 "motion_mix", "distribution") if stats.get(key) is not None}
        tags = reference.get("scene_tags") or []
        parts.append(f"Reference {n}: \"{_short(reference.get('label'), 60)}\", for "
                     f"{', '.join(tags) + ' scenes' if tags else 'every scene'}, weight {reference_measures(reference).get('weight', 1)} shots.\n"
                     "Measured: " + json.dumps(facts, ensure_ascii=False) + "\n"
                     "Grammar: " + json.dumps(reference.get("grammar") or {}, ensure_ascii=False)[:6000])
    return f"{len(references)} references to consolidate.\n\n" + "\n\n".join(parts)


def merge_grammars(
    references: Sequence[Mapping[str, Any]],
    *,
    fal_key: str | None,
    model: str = DEFAULT_MODEL,
    ask: Callable[..., tuple[str, Any]] = _ask,
    http: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    backend: Any = None,
) -> dict:
    """One grammar for ``references``: {grammar, summary, cost_usd, model}; raises StyleReferenceError."""
    ask, fal_key, model, _ = _resolve_backend(backend, fal_key, model, ask)
    prompt = merge_prompt(references)
    cost = 0.0
    owns_http = http is None
    if owns_http:
        import httpx
        http = httpx.Client(timeout=120)
    try:
        last_error: Exception | None = None
        for _ in range(MERGE_ATTEMPTS):
            text, call_cost = _ask_text(http, fal_key, model, MERGE_SYSTEM_PROMPT, prompt, ask, sleep)
            cost += float(call_cost or 0)
            try:
                grammar, summary = parse_grammar(text, max_rules=MERGED_RULES, scene_notes=True)
            except StyleReferenceError as error:
                last_error = error
                continue
            break
        else:
            raise StyleReferenceError(f"Merge failed: {last_error}")
    finally:
        if owns_http:
            http.close()
    return {"grammar": grammar, "summary": summary, "cost_usd": round(cost, 4), "model": model}


def _set_grammar(bible_id: str, value: dict | None) -> dict | None:
    def change(bible: dict) -> dict | None:
        bible["style_grammar"] = value
        return value
    return bibles._mutate(bible_id, change)[1]


_MERGE_SLOT = threading.Semaphore(1)


def start_merge(bible_id: str, *, fal_key: str | None, model: str = DEFAULT_MODEL, backend: Any = None,
                merge: Callable[..., dict] | None = None, force: bool = False, wait: bool = False) -> dict | None:
    """Keep the bible's merged grammar in step with its references; returns the stored record.

    Fewer than MIN_MERGE_REFERENCES usable references: the merged grammar is
    dropped (the single reference speaks for itself). A merged grammar that
    covers the current references is kept unless ``force``. Otherwise a
    background thread rebuilds it and the record reads ``merging`` meanwhile.
    """
    bible = bibles.get_bible(bible_id)
    if not bible:
        raise BibleError("Unknown bible")
    references = merge_candidates(bible)
    if len(references) < MIN_MERGE_REFERENCES:
        if bible.get("style_grammar"):
            _set_grammar(bible_id, None)
        return None
    sources = merge_sources(references)
    current = bible.get("style_grammar") or {}
    if not force and current.get("sources") == sources and current.get("status") in ("done", "merging"):
        return current
    record = _set_grammar(bible_id, {"status": "merging", "sources": sources, "grammar": None, "summary": "",
                                     "model": "", "cost_usd": None, "updated_at": _now(), "error": ""})
    thread = threading.Thread(target=_run_merge, args=(bible_id, references, fal_key, model, backend,
                                                       merge or merge_grammars),
                              name=f"style-merge-{bible_id}", daemon=True)
    thread.start()
    if wait:
        thread.join()
    return record


def _run_merge(bible_id: str, references: list[dict], fal_key: str | None, model: str, backend: Any,
               merge: Callable[..., dict]) -> None:
    sources = merge_sources(references)
    try:
        with _MERGE_SLOT:
            try:
                result = merge(references, fal_key=fal_key, model=model, backend=backend)
            except Exception as error:  # noqa: BLE001 - recorded on the bible
                print(f"[Style reference] {bible_id}: merge failed: {error}")
                _set_grammar(bible_id, {"status": "failed", "sources": sources, "grammar": None, "summary": "",
                                        "model": "", "cost_usd": None, "updated_at": _now(),
                                        "error": _short(error, 400)})
                return
            _set_grammar(bible_id, {"status": "done", "sources": sources, "updated_at": _now(), "error": "",
                                    **result})
            print(f"[Style reference] {bible_id}: merged {len(references)} references, "
                  f"{len(result['grammar'].get('rules') or [])} rules, cost ${result.get('cost_usd')}")
    except BibleError:
        pass  # the bible was deleted meanwhile
    except Exception as error:  # noqa: BLE001
        print(f"[Style reference] {bible_id}: merge error: {error}")


# ── a link as a reference (video fetch service) ───────────────────────────

# Recent fetches per bible (in memory: a restart forgets them, the references
# they made stay).
MAX_FETCHES = 50
MAX_FETCH_BATCH = 20
FETCH_POLL_SEC = 3.0
FETCH_TIMEOUT_SEC = 3 * 3600
# Links of a batch wait here, then download two at a time on the service.
_FETCH_SLOT = threading.Semaphore(2)
_FETCHES: dict[str, list[dict]] = {}
_FETCH_LOCK = threading.Lock()


def list_fetches(bible_id: str) -> list[dict]:
    with _FETCH_LOCK:
        return [dict(f) for f in _FETCHES.get(bible_id, [])]


def _fetch_set(bible_id: str, fetch_id: str, **fields: Any) -> dict:
    with _FETCH_LOCK:
        for fetch in _FETCHES.get(bible_id, []):
            if fetch["id"] == fetch_id:
                fetch.update(fields, updated_at=_now())
                return dict(fetch)
    return {}


def _valid_video_url(url: str) -> bool:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.hostname)


def fetch_reference(bible_id: str, url: str, *, fetcher: tuple[str, str] | None, label: str = "",
                    scene_tags: Any = None, graphics: Any = GRAPHICS_IGNORE,
                    start: Callable[[str, str], Any] | None = None, http: Any = None,
                    sleep: Callable[[float], None] = time.sleep, wait: bool = False) -> dict:
    """Queue a link: the fetch service downloads it at 480p, the file joins the bible, the analysis starts.

    Returns the fetch record ({id, url, status, progress, error, reference_id,
    ...}); ``start(bible_id, ref_id)`` runs once the reference exists.
    """
    if not call_bible_exists(bible_id):
        raise BibleError("Unknown bible")
    url = str(url or "").strip()
    if len(url) > 2000 or not _valid_video_url(url):
        raise BibleError("Give the http(s) link of a video")
    if not fetcher or not fetcher[0]:
        raise BibleError("No video fetch service is configured (Settings → Services)")
    fetch = {"id": "fetch_" + secrets.token_hex(4), "url": url, "status": "queued", "progress": "Waiting",
             "error": "", "reference_id": "", "label": _short(label, 80), "created_at": _now(),
             "updated_at": _now()}
    with _FETCH_LOCK:
        fetches = _FETCHES.setdefault(bible_id, [])
        fetches.append(fetch)
        del fetches[:-MAX_FETCHES]
    thread = threading.Thread(target=_run_fetch, args=(bible_id, dict(fetch), fetcher, parse_tags(scene_tags),
                                                       parse_graphics(graphics), start, http, sleep),
                              name=f"style-fetch-{fetch['id']}", daemon=True)
    thread.start()
    if wait:
        thread.join()
    return dict(fetch)


def parse_urls(value: Any) -> list[str]:
    """Distinct http(s) links from a list or a text with one per line (commas and spaces split too)."""
    items = value if isinstance(value, list) else re.split(r"[\s,]+", str(value or ""))
    urls = [str(item).strip() for item in items if str(item).strip()]
    return list(dict.fromkeys(u for u in urls if len(u) <= 2000 and _valid_video_url(u)))[:MAX_FETCH_BATCH]


def fetch_references(bible_id: str, urls: Any, **kwargs: Any) -> list[dict]:
    """Queue every link of ``urls`` (``fetch_reference`` each); a label applies only to a single link."""
    links = parse_urls(urls)
    if not links:
        raise BibleError("Give the http(s) link of a video, one per line")
    if len(links) > 1:
        kwargs["label"] = ""
    return [fetch_reference(bible_id, url, **kwargs) for url in links]


def call_bible_exists(bible_id: str) -> bool:
    try:
        return bool(bibles.get_bible(bible_id))
    except BibleError:
        return False


def _run_fetch(bible_id: str, fetch: dict, fetcher: tuple[str, str], tags: list[str], graphics: str,
               start: Callable[[str, str], Any] | None, http: Any, sleep: Callable[[float], None]) -> None:
    base, token = fetcher[0].rstrip("/"), fetcher[1]
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    owns_http = http is None
    if owns_http:
        import httpx
        http = httpx.Client(timeout=60)
    target = None
    try:
        with _FETCH_SLOT:
            response = http.post(f"{base}/api/fetch", headers=headers,
                                 json={"url": fetch["url"], "max_height": PROXY_HEIGHT})
            if response.status_code >= 400:
                raise StyleReferenceError(f"The fetch service refused: HTTP {response.status_code}: {response.text[:200]}")
            job = (response.json() or {}).get("job") or {}
            job_id = str(job.get("id") or "")
            if not job_id:
                raise StyleReferenceError("The fetch service returned no job")
            _fetch_set(bible_id, fetch["id"], status="downloading", progress="Starting")
            deadline = time.monotonic() + FETCH_TIMEOUT_SEC
            while True:
                response = http.get(f"{base}/api/fetch/{job_id}", headers=headers)
                if response.status_code >= 400:
                    raise StyleReferenceError(f"The fetch service lost the job: HTTP {response.status_code}")
                job = (response.json() or {}).get("job") or {}
                if job.get("status") == "failed":
                    raise StyleReferenceError(f"Download failed: {str(job.get('error') or '')[:300]}")
                if job.get("status") == "done":
                    break
                _fetch_set(bible_id, fetch["id"], progress=_short(job.get("progress") or "Downloading", 100))
                if time.monotonic() > deadline:
                    raise StyleReferenceError("The download timed out")
                sleep(FETCH_POLL_SEC)
            _fetch_set(bible_id, fetch["id"], progress="Copying into the bible")
            folder = bibles._folder(bible_id)
            target = folder / f"styleref_{int(time.time() * 1000)}_{secrets.token_hex(3)}.mp4"
            written = 0
            with http.stream("GET", f"{base}/api/fetch/{job_id}/file", headers=headers, timeout=600) as stream:
                if stream.status_code >= 400:
                    raise StyleReferenceError(f"Could not read the file: HTTP {stream.status_code}")
                with open(target, "wb") as handle:
                    for chunk in stream.iter_bytes(1024 * 1024):
                        written += len(chunk)
                        if written > MAX_UPLOAD_BYTES:
                            raise StyleReferenceError("Video too large (max 4 GB)")
                        handle.write(chunk)
            if not written:
                raise StyleReferenceError("The downloaded file is empty")
            title = _short(job.get("title"), 80)
            reference = add_reference(bible_id, str(target), source_filename=f"{title or job_id}.mp4",
                                      label=fetch.get("label") or title, scene_tags=tags, graphics=graphics)
            try:
                http.delete(f"{base}/api/fetch/{job_id}", headers=headers)
            except Exception:  # noqa: BLE001 - the service cleans up on its own
                pass
            _fetch_set(bible_id, fetch["id"], status="done", progress="", reference_id=reference["id"])
            if start is not None:
                start(bible_id, reference["id"])
    except Exception as error:  # noqa: BLE001 - recorded on the fetch
        if target is not None:
            try:
                target.unlink(missing_ok=True)
            except OSError:
                pass
        print(f"[Style reference] {bible_id}: fetch {fetch['url'][:80]}: {error}")
        _fetch_set(bible_id, fetch["id"], status="failed", progress="", error=_short(error, 400))
    finally:
        if owns_http:
            http.close()


# ── background analysis ────────────────────────────────────────────────────

# One analysis at a time: each sends minutes of video to the model.
_ANALYSIS_SLOT = threading.Semaphore(1)


def start_analysis(bible_id: str, ref_id: str, *, fal_key: str | None, model: str = DEFAULT_MODEL,
                   analyse: Callable[..., dict] | None = None, wait: bool = False, backend: Any = None,
                   on_done: Callable[[str], Any] | None = None) -> dict:
    """Queue the analysis of one reference in a background thread; returns the queued reference.

    ``on_done(bible_id)`` runs after a successful analysis (the routes use it
    to refresh the merged grammar).
    """
    if ref_id in bibles._LIVE_ANALYSES:
        raise BibleError("This reference is already being analysed")
    bibles._LIVE_ANALYSES.add(ref_id)
    try:
        reference = _set(bible_id, ref_id, status="queued", error="", progress="Waiting for the analyser")
    except Exception:
        bibles._LIVE_ANALYSES.discard(ref_id)
        raise
    thread = threading.Thread(target=_run_analysis,
                              args=(bible_id, ref_id, fal_key, model, analyse or analyse_reference, backend,
                                    on_done),
                              name=f"style-reference-{ref_id}", daemon=True)
    thread.start()
    if wait:
        thread.join()
    return reference


def _run_analysis(bible_id: str, ref_id: str, fal_key: str | None, model: str,
                  analyse: Callable[..., dict], backend: Any = None,
                  on_done: Callable[[str], Any] | None = None) -> None:
    try:
        with _ANALYSIS_SLOT:
            if ref_id not in bibles._LIVE_ANALYSES:
                return  # deleted while queued
            reference = _set(bible_id, ref_id, status="analysing", progress=f"Reducing to {PROXY_HEIGHT}p")
            try:
                reference = ensure_proxy(bible_id, reference)
            except StyleReferenceError as error:
                print(f"[Style reference] {bible_id}/{ref_id}: {error}")
                _set(bible_id, ref_id, status="failed", error=_short(error, 400), progress="")
                return
            reference = _set(bible_id, ref_id, progress="Measuring cuts")
            path = bibles.video_path(bible_id, reference["file"])
            if path is None:
                raise StyleReferenceError("The video file is missing")

            def progress(message: str, partial: dict) -> None:
                _set(bible_id, ref_id, progress=_short(message, 200), **partial)

            try:
                result = analyse(str(path), fal_key=fal_key, model=model, on_progress=progress, backend=backend,
                                 graphics=reference.get("graphics") or GRAPHICS_IGNORE)
            except Exception as error:  # noqa: BLE001 - recorded on the reference
                print(f"[Style reference] {bible_id}/{ref_id}: {error}")
                _set(bible_id, ref_id, status="failed", error=_short(error, 400), progress="")
                return
            errors = result.pop("chunk_errors", []) or []
            shots = result.pop("shots", []) or []
            save_shots(bible_id, reference, shots)
            _set(bible_id, ref_id, status="done", progress="", described_shots=sum(
                1 for s in shots if s.get("shot_size") and not s.get("credits")),
                 error=("; ".join(errors))[:400] if errors else "", **result)
            print(f"[Style reference] {bible_id}/{ref_id}: done, {result['stats'].get('shot_count')} shots, "
                  f"ASL {result['stats'].get('asl_sec')} s, cost ${result.get('cost_usd')}")
            if on_done is not None:
                try:
                    on_done(bible_id)
                except Exception as error:  # noqa: BLE001
                    print(f"[Style reference] {bible_id}: after analysis: {error}")
    except BibleError:
        pass  # the bible or the reference was deleted meanwhile
    except Exception as error:  # noqa: BLE001
        try:
            _set(bible_id, ref_id, status="failed", error=_short(error, 400), progress="")
        except Exception:  # noqa: BLE001
            pass
    finally:
        bibles._LIVE_ANALYSES.discard(ref_id)


# ── HTTP routes ────────────────────────────────────────────────────────────


def _public(reference: Mapping[str, Any], *, bible_id: str = "", with_shots: bool = False) -> dict:
    """A reference for the API: lists leave out the per-cut arrays, one reference adds its shots."""
    data = dict(reference)
    if with_shots:
        data["shots"] = load_shots(bible_id, reference)
    else:
        stats = dict(data.get("stats") or {})
        for key in ("cuts", "shot_lengths"):
            stats.pop(key, None)
        data["stats"] = stats or None
    return data


def register_routes(api: Any, *, get_fal_key: Callable[[], str | None],
                    get_model: Callable[[], str] = lambda: DEFAULT_MODEL,
                    get_backend: Callable[[], Any] = lambda: None,
                    get_fetcher: Callable[[], tuple[str, str] | None] = lambda: None) -> None:
    """Mount the style reference endpoints under ``/api/v1/director/bibles/{bible_id}/style-references``.

    ``get_backend`` returns the local judge backend when the analysis should
    run there (see ``analyse_reference``), else None for fal. ``get_fetcher``
    returns the (url, token) of the video fetch service for links.
    """
    base = "/api/v1/director/bibles/{bible_id}/style-references"

    def call(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except BibleError as error:
            status = 404 if str(error).startswith("Unknown") or str(error) == "Invalid bible id" else 400
            raise HTTPException(status_code=status, detail=str(error)) from error

    def refresh_merge(bible_id: str, force: bool = False) -> dict | None:
        return start_merge(bible_id, fal_key=get_fal_key(), model=get_model(), backend=get_backend(), force=force)

    def analyse_new(bible_id: str, ref_id: str) -> None:
        start_analysis(bible_id, ref_id, fal_key=get_fal_key(), model=get_model(), backend=get_backend(),
                       on_done=refresh_merge)

    @api.post(base + "/from-url")
    async def style_reference_from_url(bible_id: str, request: Request):
        try:
            body = await request.json()
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from error
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        urls = body.get("urls") if body.get("urls") is not None else body.get("url")
        fetches = call(fetch_references, bible_id, urls, fetcher=get_fetcher(),
                       label=str(body.get("label") or ""), scene_tags=body.get("scene_tags"),
                       graphics=body.get("graphics"), start=analyse_new)
        return {"fetch": fetches[0], "fetches": fetches}

    @api.get(base + "/fetches")
    def style_reference_fetches(bible_id: str):
        if not call(bibles.get_bible, bible_id):
            raise HTTPException(status_code=404, detail="Bible not found")
        return {"fetches": list_fetches(bible_id)}

    @api.get(base + "/merged")
    def style_reference_merged(bible_id: str):
        bible = call(bibles.get_bible, bible_id)
        if not bible:
            raise HTTPException(status_code=404, detail="Bible not found")
        return {"merged": bible.get("style_grammar"), "references": len(merge_candidates(bible))}

    @api.post(base + "/merge")
    def style_reference_merge(bible_id: str):
        return {"merged": call(refresh_merge, bible_id, True)}

    @api.get(base)
    def style_references_list(bible_id: str):
        return {"references": [_public(r) for r in call(list_references, bible_id)]}

    @api.post(base)
    async def style_reference_upload(bible_id: str, request: Request, file: UploadFile = File(...),
                                     label: str = Form(""), scene_tags: str = Form(""), analyse: str = Form("true"),
                                     graphics: str = Form(GRAPHICS_IGNORE)):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_UPLOAD_BYTES + 1024 * 1024:
            raise HTTPException(status_code=413, detail="Video too large (max 4 GB)")
        suffix = os.path.splitext(file.filename or "")[1].lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(status_code=400, detail="Upload an MP4, MOV, WebM or MKV video")
        if not call(bibles.get_bible, bible_id):
            raise HTTPException(status_code=404, detail="Bible not found")
        folder = bibles._folder(bible_id)
        target = folder / f"styleref_{int(time.time() * 1000)}_{secrets.token_hex(3)}{suffix}"
        written = 0
        try:
            with open(target, "wb") as handle:
                while chunk := await file.read(1024 * 1024):
                    written += len(chunk)
                    if written > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail="Video too large (max 4 GB)")
                    handle.write(chunk)
            if not written:
                raise HTTPException(status_code=400, detail="The file is empty")
            reference = call(add_reference, bible_id, str(target), source_filename=file.filename or "",
                             label=label, scene_tags=scene_tags, graphics=graphics)
        except BaseException:
            target.unlink(missing_ok=True)
            raise
        if str(analyse).strip().lower() not in ("false", "0", "no", "off"):
            reference = call(start_analysis, bible_id, reference["id"], fal_key=get_fal_key(), model=get_model(), backend=get_backend(),
                             on_done=refresh_merge)
        return {"reference": _public(reference)}

    @api.get(base + "/{ref_id}")
    def style_reference_get(bible_id: str, ref_id: str):
        return {"reference": _public(call(get_reference, bible_id, ref_id), bible_id=bible_id, with_shots=True)}

    @api.patch(base + "/{ref_id}")
    async def style_reference_update(bible_id: str, ref_id: str, request: Request):
        try:
            body = await request.json()
        except ValueError as error:
            raise HTTPException(status_code=400, detail="Request body must be JSON") from error
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")
        reference = _public(call(update_reference, bible_id, ref_id, body))
        call(refresh_merge, bible_id)
        return {"reference": reference}

    @api.delete(base + "/{ref_id}")
    def style_reference_delete(bible_id: str, ref_id: str):
        references = call(delete_reference, bible_id, ref_id)
        call(refresh_merge, bible_id)
        return {"deleted": ref_id, "references": [_public(r) for r in references]}

    @api.post(base + "/{ref_id}/analyse")
    def style_reference_analyse(bible_id: str, ref_id: str):
        call(get_reference, bible_id, ref_id)
        reference = call(start_analysis, bible_id, ref_id, fal_key=get_fal_key(), model=get_model(), backend=get_backend(),
                             on_done=refresh_merge)
        return {"reference": _public(reference)}

    @api.get(base + "/{ref_id}/video")
    def style_reference_video(bible_id: str, ref_id: str):
        reference = call(get_reference, bible_id, ref_id)
        path = call(bibles.video_path, bible_id, reference.get("file") or "")
        if path is None:
            raise HTTPException(status_code=404, detail="Video not found")
        suffix = os.path.splitext(str(path))[1].lower()
        return FileResponse(str(path), media_type=MEDIA_TYPES.get(suffix, "video/mp4"))


def copy_into_bible(bible_id: str, source: str, *, label: str = "", scene_tags: Any = None) -> dict:
    """Add a local video file as a reference (scripts and tests); returns the reference, not analysed."""
    suffix = os.path.splitext(source)[1].lower()
    if suffix not in VIDEO_SUFFIXES:
        raise BibleError("Not a supported video file")
    target = bibles._folder(bible_id) / f"styleref_{int(time.time() * 1000)}_{secrets.token_hex(3)}{suffix}"
    shutil.copyfile(source, target)
    try:
        return add_reference(bible_id, str(target), source_filename=os.path.basename(source), label=label,
                             scene_tags=scene_tags)
    except Exception:
        target.unlink(missing_ok=True)
        raise
