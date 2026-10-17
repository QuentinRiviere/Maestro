"""Tight edit of a Director project: render long, cut short.

The video model renders at least ~5 s per shot, while drama editing cuts
reactions and inserts to 1.5-3 s. Planners therefore mark, per shot, the
part worth keeping (``keep_sec``); this module joins those parts into an
extra ``*_edit.mp4`` next to the full join, which is never replaced.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

MIN_KEEP_SEC = 1.0
EDIT_SUFFIX = "_edit.mp4"
# The most screen time a shot keeps, by scene type (drama cutting rhythm):
# planners write keep_sec but tend to keep nearly the whole render, and
# both video judges called 6 s reactions dead air.
KEEP_CAPS = {"insert": 3.0, "reaction": 3.0, "exchange": 4.0, "establishing": 5.0}
DEFAULT_KEEP_CAP = 5.0


def normalize_keep(value: Any, duration: float | None) -> tuple[float, float] | None:
    """A sane ``[start, end]`` inside the shot, or None to keep it whole."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        start, end = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    start = max(0.0, start)
    if duration:
        end = min(end, float(duration))
    if end - start < MIN_KEEP_SEC:
        return None
    if duration and start <= 0.05 and end >= float(duration) - 0.05:
        return None
    return round(start, 3), round(end, 3)


def _has_audio(path: str) -> bool:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=60,
    )
    return bool(result.stdout.strip())


def has_cuts(keeps: list[tuple[float, float] | None]) -> bool:
    return any(keep is not None for keep in keeps)


def build_edit(video_paths: list[str], keeps: list[tuple[float, float] | None], output_path: str) -> str:
    """Trim each shot to its keep range and join them (frame-accurate re-encode)."""
    if len(video_paths) != len(keeps) or not video_paths:
        raise ValueError("one keep range (or None) per shot is required")
    with tempfile.TemporaryDirectory(prefix="director-edit-") as tmp:
        parts = []
        for index, (source, keep) in enumerate(zip(video_paths, keeps)):
            part = os.path.join(tmp, f"part_{index:03d}.mp4")
            command = ["ffmpeg", "-v", "error", "-y"]
            if keep:
                command += ["-ss", f"{keep[0]:.3f}", "-i", source, "-t", f"{keep[1] - keep[0]:.3f}"]
            else:
                command += ["-i", source]
            if not _has_audio(source):
                # A silent shot still needs an audio track for the concat.
                command += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-map", "0:v", "-map", "1:a",
                            "-shortest"]
            # Same codec, rate and audio layout for every part so the concat
            # demuxer can join them without a second re-encode.
            command += ["-vf", "fps=24,format=yuv420p", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "160k", part]
            subprocess.run(command, check=True, capture_output=True, timeout=600)
            parts.append(part)
        listing = os.path.join(tmp, "parts.txt")
        with open(listing, "w", encoding="utf-8") as handle:
            handle.writelines(f"file '{part}'\n" for part in parts)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", listing,
             "-c", "copy", "-movflags", "+faststart", output_path],
            check=True, capture_output=True, timeout=600,
        )
    return output_path


def keep_cap(scene_type: Any) -> float:
    """The longest keep window of a scene type (``DEFAULT_KEEP_CAP`` for an unknown one)."""
    return KEEP_CAPS.get(str(scene_type or "").strip().lower(), DEFAULT_KEEP_CAP)


def cap_keep(keep: tuple[float, float] | None, scene_type: Any, duration: float | None) -> tuple[float, float] | None:
    """``keep`` shortened to its scene type's cap around its centre; a whole shot over the cap gets a centred window.

    None stays None when the shot already fits its cap (nothing to cut).
    """
    cap = keep_cap(scene_type)
    length = float(duration or 0)
    if keep is None:
        if length <= cap + 0.05:
            return None
        start = (length - cap) / 2
        return round(start, 3), round(start + cap, 3)
    start, end = keep
    if end - start <= cap + 0.05:
        return keep
    centre = (start + end) / 2
    start = max(0.0, centre - cap / 2)
    end = start + cap
    if length and end > length:
        end = length
        start = max(0.0, end - cap)
    return round(start, 3), round(end, 3)


def plan_keeps(clip_plans: list[dict]) -> list[tuple[float, float] | None]:
    """Each shot's keep window: the planner's, normalised and capped by its scene type."""
    return [
        cap_keep(normalize_keep(plan.get("_director_keep_sec"), plan.get("_director_duration_sec")),
                 plan.get("_director_scene_type"), plan.get("_director_duration_sec"))
        for plan in clip_plans
    ]

