"""The CINEMA GRAMMAR block a storyboard planner reads.

The knowledge lives in ``data/grammar.json`` (film grammar distilled from the
sources listed there) and ``data/dramaboard_rhythm.json`` (aggregate rhythm of
vertical micro-drama episodes). This module turns both into a short list of
instructions written for an AI planner whose shots are rendered one by one by
image-to-video models, under a hard character cap.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
GRAMMAR_PATH = DATA_DIR / "grammar.json"
RHYTHM_PATH = DATA_DIR / "dramaboard_rhythm.json"

HEADER = "CINEMA GRAMMAR (K-drama short episode; every shot is rendered alone by an image-to-video model):"
TARGET_CHARS = 2500
MAX_CHARS = 2800


@lru_cache(maxsize=1)
def load_grammar() -> dict:
    with open(GRAMMAR_PATH, encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_rhythm() -> dict:
    with open(RHYTHM_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _pct(value: float) -> str:
    return f"{round(value * 100)}%"


def _span(values: list[float]) -> str:
    low, high = values
    return f"{low:g}" if low == high else f"{low:g}-{high:g}"


def _clamp(value: float, low: float, high: float) -> float:
    return round(min(max(float(value), low), high), 1)


def rhythm_targets(target_duration: float | None = None, shot_range: tuple[float, float] | None = None) -> dict:
    """Shot-length and shot-count targets for an episode, from the DramaBoard aggregates.

    ``shot_range`` is the (min, max) length a shot may have with the chosen
    video model. Reference lengths are clamped into it, so a model that cannot
    render 1.3-second shots gets fewer, longer shots rather than an impossible plan.
    """
    rhythm = load_rhythm()
    episode = rhythm["episode"]
    durations = rhythm["shot_duration_sec"]
    low, high = (float(shot_range[0]), float(shot_range[1])) if shot_range else (0.0, 60.0)
    silent = durations["by_dialogue"]["silent"]
    spoken = durations["by_dialogue"]["with_dialogue"]
    targets = {
        "reference_episodes": rhythm["source"]["episodes"],
        "reference_episode_sec": round(episode["duration_sec"]["median"]),
        "reference_shots": round(episode["shots"]["median"]),
        "reference_asl_sec": episode["asl_sec"]["median"],
        "median_shot_sec": durations["overall"]["median"],
        # Silent shots (reactions, inserts) and shots under a line, median and p75.
        "silent_shot_sec": [_clamp(silent["median"], low, high), _clamp(silent["p75"], low, high)],
        "line_shot_sec": [_clamp(spoken["median"], low, high), _clamp(spoken["p75"], low, high)],
        "asl_sec": _clamp(episode["asl_sec"]["median"], low, high),
    }
    if target_duration:
        seconds = float(target_duration)
        # The p25-p75 band of per-episode ASL, clamped the same way.
        fast = max(_clamp(episode["asl_sec"]["p25"], low, high), 0.1)
        slow = max(_clamp(episode["asl_sec"]["p75"], low, high), 0.1)
        targets["target_duration_sec"] = round(seconds)
        targets["shots_min"] = max(1, round(seconds / slow))
        targets["shots_max"] = max(1, round(seconds / fast))
    return targets


def _rhythm_lines(target_duration: float | None, shot_range: tuple[float, float] | None) -> list[str]:
    rhythm = load_rhythm()
    scale = rhythm["distribution"]["shot_scale"]
    static = rhythm["distribution"]["camera_motion"].get("static", 0)
    coverage = rhythm["dialogue_coverage"]
    t = rhythm_targets(target_duration, shot_range)
    mix = ", ".join(f"{label} {_pct(scale.get(key, 0))}" for key, label in (
        ("medium", "medium"), ("close_up", "close-up"), ("medium_close_up", "MCU"),
        ("medium_wide", "medium wide"), ("wide", "wide")))
    count = ""
    if t.get("target_duration_sec"):
        band = (f"{t['shots_min']}" if t["shots_min"] == t["shots_max"]
                else f"{t['shots_min']}-{t['shots_max']}")
        count = f" A {t['target_duration_sec']}s episode carries about {band} shots when the count is yours."
    return [
        f"- RHYTHM (DramaBoard, {t['reference_episodes']:,} micro-drama episodes, median "
        f"{t['reference_episode_sec']}s / {t['reference_shots']} shots): {_pct(static)} static; mix {mix}. "
        f"Silent reactions and inserts ~{_span(t['silent_shot_sec'])}s, shots under a line "
        f"~{_span(t['line_shot_sec'])}s.{count}",
        f"- {_pct(coverage['shots_with_line'].get('listener_or_insert', 0))} of shots under a line show the "
        f"listener, not the speaker; {_pct(coverage['after_line_ends']['silent_shot_share'])} of line ends cut "
        "to a silent reaction, most often a close-up.",
    ]


def _rule_lines() -> list[str]:
    """Planner instructions from grammar.json, highest priority first; each is self-contained."""
    return [f"- {rule['text']}" for rule in load_grammar()["planner_rules"]]


def planner_block(target_duration: float | None = None, shot_range: tuple[float, float] | None = None,
                  max_chars: int = MAX_CHARS) -> str:
    """The CINEMA GRAMMAR block, never longer than ``max_chars``."""
    lines = [HEADER]
    rules = _rule_lines()
    candidates = rules[:2] + _rhythm_lines(target_duration, shot_range) + rules[2:]
    for line in candidates:
        if len("\n".join(lines + [line])) <= max_chars:
            lines.append(line)
    return "\n".join(lines) if len(lines) > 1 else ""
