"""Film-grammar knowledge for the Director planner (K-drama short episodes).

``data/grammar.json`` holds the distilled grammar (shot sizes, angles, moves,
lenses, light, coverage, editing, K-drama motifs, acting) and the planner
rules; ``data/dramaboard_rhythm.json`` the aggregate rhythm of micro-drama
episodes. See README.md for sources and licenses.
"""
from __future__ import annotations

import re

from .render import MAX_CHARS, TARGET_CHARS, load_grammar, load_rhythm, planner_block, rhythm_targets
from .vocabulary import CAMERA_MOVES, MINIMAX_BRACKET_COMMANDS, camera_moves_in, canonical_camera_text

PARAM = "director_cinema_grammar"
# Skills whose id names a K-drama get the block unless the run turns it off.
_KDRAMA_SKILL = re.compile(r"k[_-]?drama", re.IGNORECASE)


def cinema_grammar_mode(params: dict, skill_type: str) -> str:
    """"on" or "off": the explicit ``director_cinema_grammar``, else on for K-drama skills."""
    value = str(params.get(PARAM) or "").strip().lower()
    if value in ("on", "true", "1"):
        return "on"
    if value in ("off", "false", "0"):
        return "off"
    return "on" if str(skill_type or "").startswith("custom:") and _KDRAMA_SKILL.search(skill_type) else "off"


__all__ = [
    "CAMERA_MOVES", "MAX_CHARS", "MINIMAX_BRACKET_COMMANDS", "PARAM", "TARGET_CHARS", "camera_moves_in",
    "canonical_camera_text", "cinema_grammar_mode", "load_grammar", "load_rhythm", "planner_block", "rhythm_targets",
]
