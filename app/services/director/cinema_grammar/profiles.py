"""Layered profiles behind the CINEMA GRAMMAR block.

Three layers, each a data file under ``data/``:

- universal (``universal.json``): film grammar that holds for every format and
  genre (action -> reaction, one move per shot, continuity, acting, AI-video
  robustness). No genre numbers.
- format (``formats.json``): ``vertical_short`` (9:16) or ``horizontal``
  (16:9): framing habits and the base shot-size mix. Chosen from the Director
  aspect ratio unless ``director_format_profile`` names one.
- genre (``genres/<id>.json``, one file per genre): rhythm targets, shot-mix
  bias, camera, look, acting, motifs, do/don't. Chosen by
  ``director_genre_profile`` ("auto": the skill id, else ``neutral``).

Analysed style references calibrate the numbers: their measured rhythm, shot
mix and camera-motion share replace the genre and format defaults
(``calibrate``).
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping

DATA_DIR = Path(__file__).resolve().parent / "data"
GENRES_DIR = DATA_DIR / "genres"
UNIVERSAL_PATH = DATA_DIR / "universal.json"
FORMATS_PATH = DATA_DIR / "formats.json"
GRAMMAR_PATH = DATA_DIR / "grammar.json"
RHYTHM_PATH = DATA_DIR / "dramaboard_rhythm.json"

FORMAT_PARAM = "director_format_profile"
GENRE_PARAM = "director_genre_profile"
AUTO = "auto"
DEFAULT_GENRE = "neutral"

# The five sizes the format mixes are written over, in display order.
MIX_SIZES = ("medium", "close_up", "medium_close_up", "medium_wide", "wide")
SIZE_LABELS = {
    "extreme_wide": "extreme wide", "wide": "wide", "medium_wide": "medium wide", "medium": "medium",
    "medium_close_up": "MCU", "close_up": "close-up", "extreme_close_up": "extreme close-up", "insert": "insert",
    "ots": "over-the-shoulder", "pov": "POV", "two_shot": "two-shot",
}
# Calibration keys: scalars and [median, p75] pairs averaged across references.
_SCALARS = ("asl_sec", "median_sec", "cuts_per_minute", "static_share")
_PAIRS = ("line_sec", "silent_sec")


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def load_grammar() -> dict:
    return _read(GRAMMAR_PATH)


@lru_cache(maxsize=1)
def load_rhythm() -> dict:
    return _read(RHYTHM_PATH)


@lru_cache(maxsize=1)
def load_universal() -> dict:
    return _read(UNIVERSAL_PATH)


@lru_cache(maxsize=1)
def load_formats() -> dict:
    return _read(FORMATS_PATH)


@lru_cache(maxsize=1)
def load_genres() -> dict[str, dict]:
    """Every genre profile by id (one JSON file each; adding a genre is adding a file)."""
    genres = {}
    for path in sorted(GENRES_DIR.glob("*.json")):
        profile = _read(path)
        genres[profile["id"]] = profile
    return genres


def format_ids() -> list[str]:
    return list(load_formats()["profiles"])


def genre_ids() -> list[str]:
    return list(load_genres())


def format_profile(format_id: str) -> dict:
    return load_formats()["profiles"][format_id]


def genre_profile(genre_id: str) -> dict:
    return load_genres()[genre_id]


# ── selection ──────────────────────────────────────────────────────────────


def format_from_aspect(aspect_ratio: Any) -> str | None:
    """``vertical_short`` for portrait or square ("9:16", "3:4", "1:1"), ``horizontal`` for landscape."""
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*[:x/]\s*(\d+(?:\.\d+)?)\s*", str(aspect_ratio or ""))
    if not match or not float(match.group(2)):
        return None
    return "vertical_short" if float(match.group(1)) / float(match.group(2)) <= 1.0 else "horizontal"


def select_format(params: Mapping[str, Any]) -> tuple[str, str]:
    """(format id, source): the explicit ``director_format_profile``, else the aspect ratio, else the default."""
    choice = str(params.get(FORMAT_PARAM) or AUTO).strip().lower()
    if choice in load_formats()["profiles"]:
        return choice, "param"
    from_aspect = format_from_aspect(params.get("director_aspect_ratio"))
    if from_aspect:
        return from_aspect, "aspect_ratio"
    return load_formats()["default"], "default"


def genre_for_skill(skill_type: str) -> str | None:
    """The genre whose ``skill_pattern`` names a custom skill id (``custom:k_drama_mini`` -> kdrama)."""
    skill = str(skill_type or "")
    if not skill.startswith("custom:"):
        return None
    for genre_id, profile in load_genres().items():
        pattern = profile.get("skill_pattern")
        if pattern and re.search(pattern, skill, re.IGNORECASE):
            return genre_id
    return None


def explicit_genre(params: Mapping[str, Any]) -> str | None:
    choice = str(params.get(GENRE_PARAM) or AUTO).strip().lower()
    return choice if choice in load_genres() else None


def select_genre(params: Mapping[str, Any], skill_type: str = "") -> tuple[str, str]:
    """(genre id, source): the explicit ``director_genre_profile``, else the skill's genre, else neutral."""
    chosen = explicit_genre(params)
    if chosen:
        return chosen, "param"
    from_skill = genre_for_skill(skill_type)
    if from_skill:
        return from_skill, "skill"
    return DEFAULT_GENRE, "default"


# ── numbers ────────────────────────────────────────────────────────────────


def genre_rhythm(genre_id: str) -> dict:
    """A genre's rhythm defaults (seconds, static share) and the label of their source."""
    profile = genre_profile(genre_id)
    rhythm = dict(profile.get("rhythm") or {})
    if profile.get("rhythm_source") == "dramaboard":
        stats = load_rhythm()
        durations = stats["shot_duration_sec"]
        episode = stats["episode"]
        silent, spoken = durations["by_dialogue"]["silent"], durations["by_dialogue"]["with_dialogue"]
        rhythm.update({
            "median_sec": durations["overall"]["median"],
            "asl_sec": episode["asl_sec"]["median"],
            "asl_band_sec": [episode["asl_sec"]["p25"], episode["asl_sec"]["p75"]],
            "silent_sec": [silent["median"], silent["p75"]],
            "line_sec": [spoken["median"], spoken["p75"]],
            "static_share": stats["distribution"]["camera_motion"].get("static", 0),
            "reference_episodes": stats["source"]["episodes"],
        })
        rhythm["source_label"] = f"DramaBoard, {stats['source']['episodes']:,} micro-drama episodes"
    else:
        rhythm["source_label"] = f"{profile['label']} default"
    return rhythm


def _normalised(mix: Mapping[str, float]) -> dict[str, float]:
    total = sum(v for v in mix.values() if v > 0)
    if not total:
        return {}
    return {k: round(100 * v / total, 1) for k, v in mix.items() if v > 0}


def format_mix(format_id: str) -> tuple[dict[str, float], str]:
    """A format's base shot mix (percent over ``MIX_SIZES``) and its source label."""
    profile = format_profile(format_id)
    if profile.get("shot_mix_source") == "dramaboard":
        scale = load_rhythm()["distribution"]["shot_scale"]
        return _normalised({size: scale.get(size, 0) for size in MIX_SIZES}), "DramaBoard 9:16 mix"
    return _normalised(profile.get("shot_mix") or {}), profile.get("shot_mix_label") or f"{profile['label']} guideline"


def biased_mix(format_id: str, genre_id: str) -> tuple[dict[str, float], str]:
    """The format mix weighted by the genre's ``shot_mix_bias``, renormalised."""
    mix, label = format_mix(format_id)
    genre = genre_profile(genre_id)
    bias = genre.get("shot_mix_bias") or {}
    if not bias:
        return mix, label
    return _normalised({size: value * float(bias.get(size, 1.0)) for size, value in mix.items()}), \
        f"{label} biased for {genre['label']}"


def dialogue_coverage() -> dict:
    """Drama-general coverage numbers (DramaBoard): listener share under a line, silent reaction after a line."""
    coverage = load_rhythm()["dialogue_coverage"]
    return {
        "listener_share": coverage["shots_with_line"].get("listener_or_insert", 0),
        "silent_after_line_share": coverage["after_line_ends"]["silent_shot_share"],
    }


# ── calibration by style references ────────────────────────────────────────


def _positive(value: Any, *, allow_zero: bool = False) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 or (allow_zero and number == 0) else None


def calibrate(measures: Iterable[Mapping[str, Any]]) -> dict:
    """Weighted average of the references' measured numbers ({} when none has any).

    Each measure is ``style_reference.reference_measures()`` output: an id, a
    weight (its shot count) and any of asl_sec, median_sec, cuts_per_minute,
    static_share (0-1), line_sec and silent_sec ([median, p75]) and shot_mix
    (percent by size). A key averages only the references that measured it.
    """
    measures = [m for m in measures if m]
    result: dict[str, Any] = {}
    for key in _SCALARS:
        pairs = [(float(m.get("weight") or 1), _positive(m.get(key), allow_zero=key == "static_share"))
                 for m in measures]
        pairs = [(w, v) for w, v in pairs if v is not None]
        if pairs:
            result[key] = round(sum(w * v for w, v in pairs) / sum(w for w, _ in pairs), 3)
    for key in _PAIRS:
        pairs = [(float(m.get("weight") or 1), m.get(key)) for m in measures]
        pairs = [(w, v) for w, v in pairs if isinstance(v, (list, tuple)) and len(v) == 2 and all(
            _positive(x) for x in v)]
        if pairs:
            total = sum(w for w, _ in pairs)
            result[key] = [round(sum(w * float(v[i]) for w, v in pairs) / total, 2) for i in (0, 1)]
    mixes = [(float(m.get("weight") or 1), m.get("shot_mix")) for m in measures]
    mixes = [(w, mix) for w, mix in mixes if isinstance(mix, Mapping) and mix]
    if mixes:
        total = sum(w for w, _ in mixes)
        sizes = list(dict.fromkeys(size for _, mix in mixes for size in mix))
        averaged = {size: sum(w * float(mix.get(size) or 0) for w, mix in mixes) / total for size in sizes}
        result["shot_mix"] = dict(sorted(_normalised(averaged).items(), key=lambda kv: -kv[1]))
    if result:
        result["references"] = [str(m.get("id") or "") for m in measures]
    return result


def mix_text(mix: Mapping[str, float], limit: int = 6) -> str:
    ordered = sorted(mix.items(), key=lambda kv: -kv[1])[:limit]
    return ", ".join(f"{SIZE_LABELS.get(size, size.replace('_', ' '))} {round(value)}%" for size, value in ordered)
