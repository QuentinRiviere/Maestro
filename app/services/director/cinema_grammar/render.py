"""The CINEMA GRAMMAR block a storyboard planner reads.

The block stacks three layers (see ``profiles.py``): the universal film
grammar, the format (vertical short or horizontal) and the genre (K-drama,
micro-drama, animated mini-series, vlog, neutral). Analysed style references
calibrate the numbers. Every number names its source ("from your
references", a genre default, DramaBoard). The block is written for an AI
planner whose shots are rendered one by one by image-to-video models, under a
hard character cap.
"""
from __future__ import annotations

from typing import Any, Mapping

from . import profiles
from .profiles import DATA_DIR, GRAMMAR_PATH, RHYTHM_PATH, load_grammar, load_rhythm  # noqa: F401 - re-exported

TARGET_CHARS = 2500
MAX_CHARS = 2800
REFERENCES_LABEL = "from your references"
# A calibrated average shot length becomes a band of +-20-25% around it.
_BAND = (0.8, 1.25)


def _pct(value: float) -> str:
    return f"{round(value * 100)}%"


def _span(values: list[float]) -> str:
    low, high = values
    return f"{low:g}" if low == high else f"{low:g}-{high:g}"


def _clamp(value: float, low: float, high: float) -> float:
    return round(min(max(float(value), low), high), 1)


def rhythm_targets(target_duration: float | None = None, shot_range: tuple[float, float] | None = None, *,
                   genre_id: str = profiles.DEFAULT_GENRE, calibration: Mapping[str, Any] | None = None) -> dict:
    """Shot-length and shot-count targets for an episode, with the source of each number.

    The genre gives the defaults; ``calibration`` (``profiles.calibrate``
    output) replaces what the references measured. Without a measured
    silent/line split, the genre's holds are scaled by the references' pace.
    ``shot_range`` is the (min, max) length a shot may have with the chosen
    video model: every length is clamped into it, so a model that cannot
    render 1.3-second shots gets fewer, longer shots rather than an impossible plan.
    """
    rhythm = profiles.genre_rhythm(genre_id)
    calibration = calibration or {}
    low, high = (float(shot_range[0]), float(shot_range[1])) if shot_range else (0.0, 60.0)
    sources: dict[str, str] = {}

    def pick(key: str, default: Any) -> Any:
        if calibration.get(key) is not None:
            sources[key] = "references"
            return calibration[key]
        sources[key] = "genre"
        return default

    asl = float(pick("asl_sec", rhythm["asl_sec"]))
    pace = asl / float(rhythm["asl_sec"]) if sources["asl_sec"] == "references" else 1.0
    band = [asl * _BAND[0], asl * _BAND[1]] if sources["asl_sec"] == "references" else rhythm["asl_band_sec"]
    sources["asl_band_sec"] = sources["asl_sec"]
    median = calibration.get("median_sec")
    if median is not None:
        sources["median_sec"] = "references"
    else:
        median = float(rhythm["median_sec"]) * pace
        sources["median_sec"] = "scaled" if pace != 1.0 else "genre"
    targets: dict[str, Any] = {
        "genre": genre_id,
        "source_label": rhythm["source_label"],
        "median_sec": _clamp(median, low, high),
        "asl_sec": _clamp(asl, low, high),
        "asl_band_sec": [_clamp(band[0], low, high), _clamp(band[1], low, high)],
        "static_share": round(float(pick("static_share", rhythm["static_share"])), 2),
        "note": rhythm.get("note", ""),
    }
    for key in ("silent_sec", "line_sec"):
        if calibration.get(key):
            values, sources[key] = calibration[key], "references"
        else:
            values = [float(v) * pace for v in rhythm[key]]
            sources[key] = "scaled" if pace != 1.0 else "genre"
        targets[key] = [_clamp(values[0], low, high), _clamp(values[1], low, high)]
    if calibration.get("cuts_per_minute"):
        targets["cuts_per_minute"] = round(float(calibration["cuts_per_minute"]), 1)
        sources["cuts_per_minute"] = "references"
    if rhythm.get("reference_episodes"):
        targets["reference_episodes"] = rhythm["reference_episodes"]
    if target_duration:
        seconds = float(target_duration)
        fast, slow = max(targets["asl_band_sec"][0], 0.1), max(targets["asl_band_sec"][1], 0.1)
        targets["target_duration_sec"] = round(seconds)
        targets["shots_min"] = max(1, round(seconds / slow))
        targets["shots_max"] = max(1, round(seconds / fast))
    targets["sources"] = sources
    return targets


def _source_label(source: str, genre_label: str, rhythm_label: str) -> str:
    return {"references": REFERENCES_LABEL, "genre": rhythm_label,
            "scaled": f"{genre_label} default scaled to your references"}[source]


def _rhythm_line(targets: dict, genre_label: str) -> str:
    """One RHYTHM line; numbers are grouped under the source they come from."""
    sources = targets["sources"]
    groups: dict[str, list[str]] = {}

    def add(key: str, text: str) -> None:
        groups.setdefault(sources[key], []).append(text)

    add("median_sec", f"median shot ~{targets['median_sec']:g}s")
    pace = f"average {_span(targets['asl_band_sec'])}s"
    if targets.get("cuts_per_minute"):
        pace += f" ({targets['cuts_per_minute']:g} cuts/min)"
    add("asl_sec", pace)
    add("silent_sec", f"silent reactions and holds ~{_span(targets['silent_sec'])}s")
    add("line_sec", f"shots under a line ~{_span(targets['line_sec'])}s")
    add("static_share", f"{_pct(targets['static_share'])} static")
    order = ("references", "scaled", "genre")
    parts = [f"{_source_label(s, genre_label, targets['source_label'])}: {', '.join(groups[s])}"
             for s in order if s in groups]
    text = "- RHYTHM (" + "; ".join(parts) + ")."
    if targets.get("note"):
        text += f" {targets['note'][0].upper()}{targets['note'][1:]}."
    if targets.get("target_duration_sec"):
        band = (f"{targets['shots_min']}" if targets["shots_min"] == targets["shots_max"]
                else f"{targets['shots_min']}-{targets['shots_max']}")
        text += f" A {targets['target_duration_sec']}s episode carries about {band} shots when the count is yours."
    return text


def _format_line(format_id: str, genre_id: str, calibration: Mapping[str, Any]) -> str:
    rule = profiles.format_profile(format_id)["rule"]
    if calibration.get("shot_mix"):
        mix, label = calibration["shot_mix"], REFERENCES_LABEL
    else:
        mix, label = profiles.biased_mix(format_id, genre_id)
    return f"- {rule} Shot mix ({label}): {profiles.mix_text(mix)}."


def _coverage_line() -> str:
    numbers = profiles.dialogue_coverage()
    return ("- COVERAGE: over-the-shoulder shot/reverse at MCU, two-shot for intimacy, profiles for avoidance; "
            f"under a line show the listener ~{_pct(numbers['listener_share'])} of the time; "
            f"~{_pct(numbers['silent_after_line_share'])} of line ends cut to a silent reaction, usually a close-up "
            "(DramaBoard).")


def _genre_lines(genre_id: str) -> list[tuple[int, str]]:
    """(priority, line) for the genre layer."""
    profile = profiles.genre_profile(genre_id)
    name = profile.get("tag") or profile["label"].upper()
    lines = [(55, f"- {name} CAMERA: {profile['camera']}."), (65, f"- {name} ACTING: {profile['acting']}.")]
    if profile.get("motifs"):
        lines.append((70, f"- {name} MOTIFS: " + "; ".join(profile["motifs"]) + "."))
    rules = [f"DO: {'; '.join(profile['do'])}" if profile.get("do") else "",
             f"DON'T: {'; '.join(profile['dont'])}" if profile.get("dont") else ""]
    if any(rules):
        lines.append((85, f"- {name} " + ". ".join(r for r in rules if r) + "."))
    lines.append((84, f"- {name} LOOK: {profile['look']}."))
    return lines


def planner_block(target_duration: float | None = None, shot_range: tuple[float, float] | None = None,
                  max_chars: int = MAX_CHARS, *, format_id: str | None = None, genre_id: str = profiles.DEFAULT_GENRE,
                  calibration: Mapping[str, Any] | None = None) -> str:
    """The CINEMA GRAMMAR block, never longer than ``max_chars``.

    Lines are taken by priority (universal unit and camera, rhythm, format,
    coverage, genre camera, ...); a line that no longer fits is dropped whole
    and a shorter later one may still fit.
    """
    format_id = format_id or profiles.load_formats()["default"]
    calibration = calibration or {}
    genre = profiles.genre_profile(genre_id)
    targets = rhythm_targets(target_duration, shot_range, genre_id=genre_id, calibration=calibration)
    candidates: list[tuple[int, str]] = [(rule["priority"], f"- {rule['text']}")
                                         for rule in profiles.load_universal()["rules"]]
    candidates += [(30, _rhythm_line(targets, genre["label"])), (40, _format_line(format_id, genre_id, calibration))]
    if genre.get("coverage", True):
        candidates.append((50, _coverage_line()))
    candidates += _genre_lines(genre_id)
    header = (f"CINEMA GRAMMAR ({genre['label']}, {profiles.format_profile(format_id)['label']}; "
              "each shot is rendered alone by image-to-video):")
    lines = [header]
    for _, line in sorted(candidates, key=lambda item: item[0]):
        if len("\n".join(lines + [line])) <= max_chars:
            lines.append(line)
    return "\n".join(lines) if len(lines) > 1 else ""
