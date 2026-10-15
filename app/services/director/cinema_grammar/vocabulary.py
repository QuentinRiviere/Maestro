"""Closed camera-move vocabulary for compiled video prompts.

Planners write camera moves as free text ("slow dolly in", "camera drifts",
"follows her"). Image-to-video models follow a small set of named moves far
more reliably, so the H3 prompt compiler rewrites every recognised move to
its canonical phrase and turns vague motion into "static camera".

``CAMERA_MOVES`` is the single mapping table. Each entry also carries the
MiniMax bracket command for the move; ``MINIMAX_BRACKET_COMMANDS`` switches
that syntax on (off by default until it is validated on H3).
"""
from __future__ import annotations

import re

# Off until the bracket syntax is validated on H3; flip here only.
MINIMAX_BRACKET_COMMANDS = False

# (canonical phrase, MiniMax command or "" when it has none, synonym patterns). Order matters: the
# first table entry that matches a span wins, so specific phrases come first.
CAMERA_MOVES: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("static camera", "[Static shot]", (
        r"locked[- ]off(?: static)?(?: camera| shot)?", r"static (?:camera|shot|hold|frame)", r"fixed camera",
        r"still camera", r"no camera (?:movement|motion)", r"camera (?:holds|stays) (?:still|static)",
    )),
    ("push-in", "[Push in]", (
        r"dolly[- ]in", r"dolly(?:ing)? (?:forward|toward)", r"push(?:es|ing)?[- ]in",
        r"push(?:es|ing)? (?:forward|toward|closer)", r"creep(?:s|ing)? (?:in|forward|toward)",
        r"(?:slow |gentle )?zoom(?:s|ing)?[- ]in", r"moves? (?:in )?closer", r"camera approaches",
    )),
    ("pull-out", "[Pull out]", (
        r"dolly[- ]out", r"dolly(?:ing)? back", r"pull(?:s|ing)?[- ](?:out|back|away)", r"zoom(?:s|ing)?[- ]out",
        r"backs? away", r"camera retreats", r"widens? out",
    )),
    ("tracking shot", "[Tracking shot]", (
        r"tracking(?: shot)?", r"tracks? (?:with|beside|alongside|behind)", r"follow(?:s|ing)? (?:shot|him|her|them)",
        r"steadicam(?: follow)?", r"dolly (?:left|right|alongside)", r"trucks? (?:left|right)",
    )),
    ("pan left", "[Pan left]", (r"pan(?:s|ning)? (?:to the )?left", r"whip[- ]pan left")),
    ("pan right", "[Pan right]", (r"pan(?:s|ning)? (?:to the )?right", r"whip[- ]pan right")),
    ("tilt up", "[Tilt up]", (r"tilt(?:s|ing)? up(?:ward)?",)),
    ("tilt down", "[Tilt down]", (r"tilt(?:s|ing)? down(?:ward)?",)),
    ("crane up", "[Pedestal up]", (r"crane(?:s)? up", r"boom(?:s)? up", r"pedestal up", r"camera rises")),
    ("crane down", "[Pedestal down]", (r"crane(?:s)? down", r"boom(?:s)? down", r"pedestal down", r"camera descends")),
    ("slow arc", "", (r"(?:slow |partial )?(?:orbit|arc)(?:s|ing)?(?: around)?", r"circles? (?:around )?(?:him|her|them)")),
    ("handheld", "[Shake]", (r"hand[- ]?held(?: camera)?", r"shaky cam(?:era)?")),
)

# Motion words that promise movement without naming one the model knows.
# Unsure means static.
VAGUE_MOTION = (
    r"(?:digital |slow |gentle )?drift(?:s|ing)?", r"float(?:s|ing) (?:camera|around)", r"glid(?:es|ing)",
    r"cinematic camera(?: movement| motion)?", r"dynamic camera(?: movement| motion)?",
    r"camera (?:moves|wanders|sways)(?: around| freely)?", r"free[- ]?roaming camera",
)

# "the camera pushes in" becomes "push-in", not "the camera push-in".
_SUBJECT = r"(?:(?:the )?camera )?"
_MOVE_PATTERNS = [
    (canonical, command, re.compile(rf"(?<![\w-]){_SUBJECT}(?:{'|'.join(patterns)})(?![\w-])", re.IGNORECASE))
    for canonical, command, patterns in CAMERA_MOVES
]
# A vague motion takes the rest of its clause with it ("drifts slowly around her").
_VAGUE_PATTERN = re.compile(rf"(?<![\w-]){_SUBJECT}(?:{'|'.join(VAGUE_MOTION)})(?![\w-])[^,.;]*", re.IGNORECASE)
CANONICAL_MOVES = tuple(canonical for canonical, _, _ in CAMERA_MOVES)


def camera_moves_in(text: str) -> list[str]:
    """The canonical moves named in ``text``, in order of appearance."""
    found: list[tuple[int, str]] = []
    taken: list[tuple[int, int]] = []
    for canonical, _, pattern in _MOVE_PATTERNS:
        for match in pattern.finditer(text or ""):
            if any(start < match.end() and match.start() < end for start, end in taken):
                continue
            taken.append(match.span())
            found.append((match.start(), canonical))
    return [move for _, move in sorted(found)]


def canonical_camera_text(text: str, *, bracket_commands: bool | None = None) -> str:
    """``text`` with every camera move rewritten to the closed vocabulary.

    Framing, angle, lens and timing words are kept as written. A vague
    motion ("the camera drifts") becomes "static camera", or is dropped when
    the text already names a real move. With bracket commands on, the MiniMax
    command of each move follows its phrase.
    """
    text = str(text or "")
    if not text.strip():
        return text
    use_brackets = MINIMAX_BRACKET_COMMANDS if bracket_commands is None else bracket_commands
    spans: list[tuple[int, int, str]] = []
    for canonical, command, pattern in _MOVE_PATTERNS:
        for match in pattern.finditer(text):
            if any(start < match.end() and match.start() < end for start, end, _ in spans):
                continue
            spans.append((match.start(), match.end(),
                          f"{canonical} {command}" if use_brackets and command else canonical))
    named_move = bool(spans)
    for match in _VAGUE_PATTERN.finditer(text):
        if any(start < match.end() and match.start() < end for start, end, _ in spans):
            continue
        static = "" if named_move else "static camera [Static shot]" if use_brackets else "static camera"
        spans.append((match.start(), match.end(), static))
    for start, end, replacement in sorted(spans, reverse=True):
        text = text[:start] + replacement + text[end:]
    text = re.sub(r"\s*,\s*(?=,|$)", "", text)
    text = re.sub(r"^\s*,\s*", "", text)
    return re.sub(r"\s{2,}", " ", text).strip()
