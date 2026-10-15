"""Film-grammar knowledge for the Director planner, in layers.

``data/universal.json`` holds the rules every production follows,
``data/formats.json`` the vertical and horizontal format profiles,
``data/genres/*.json`` one profile per genre (K-drama, micro-drama, animated
mini-series, vlog, neutral), ``data/grammar.json`` the distilled film-grammar
knowledge base and ``data/dramaboard_rhythm.json`` the aggregate rhythm of
micro-drama episodes. See README.md for sources and licenses.
"""
from __future__ import annotations

from .profiles import (
    AUTO, DEFAULT_GENRE, FORMAT_PARAM, GENRE_PARAM, calibrate, explicit_genre, format_ids, genre_for_skill, genre_ids,
    load_formats, load_genres, load_universal, select_format, select_genre,
)
from .render import MAX_CHARS, TARGET_CHARS, load_grammar, load_rhythm, planner_block, rhythm_targets
from .vocabulary import CAMERA_MOVES, MINIMAX_BRACKET_COMMANDS, camera_moves_in, canonical_camera_text

PARAM = "director_cinema_grammar"


def cinema_grammar_mode(params: dict, skill_type: str) -> str:
    """"on" or "off": the explicit ``director_cinema_grammar``; else on when a genre profile applies.

    A genre applies when ``director_genre_profile`` names one, or when a
    custom skill id names one (``custom:k_drama_mini`` -> kdrama).
    """
    value = str(params.get(PARAM) or "").strip().lower()
    if value in ("on", "true", "1"):
        return "on"
    if value in ("off", "false", "0"):
        return "off"
    return "on" if explicit_genre(params) or genre_for_skill(skill_type) else "off"


__all__ = [
    "AUTO", "CAMERA_MOVES", "DEFAULT_GENRE", "FORMAT_PARAM", "GENRE_PARAM", "MAX_CHARS", "MINIMAX_BRACKET_COMMANDS",
    "PARAM", "TARGET_CHARS", "calibrate", "camera_moves_in", "canonical_camera_text", "cinema_grammar_mode",
    "format_ids", "genre_for_skill", "genre_ids", "load_formats", "load_genres", "load_grammar", "load_rhythm",
    "load_universal", "planner_block", "rhythm_targets", "select_format", "select_genre",
]
