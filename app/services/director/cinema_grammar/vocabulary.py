"""Closed camera-move vocabulary for compiled video prompts.

Planners write camera moves as free text ("slow dolly in", "the camera
drifts", "zooms in on her hand"). Image-to-video models follow a small set of
named moves more reliably, so the H3 prompt compiler rewrites synonyms to
their canonical names and turns vague motion into "static camera". The
rewrite keeps the sentence's grammar (a verb stays a verb) and leaves the
framing, subject and destination of the move as written.

``CAMERA_MOVES`` is the single table: canonical move, MiniMax bracket command,
and the phrases that name it. ``MINIMAX_BRACKET_COMMANDS`` switches the
bracket syntax on (off by default until it is validated on H3).
"""
from __future__ import annotations

import re

# Off until the bracket syntax is validated on H3; flip here only.
MINIMAX_BRACKET_COMMANDS = False

# canonical id -> (MiniMax command or "", detection patterns). Detection
# finds every move a text names, canonical or not.
CAMERA_MOVES: dict[str, tuple[str, tuple[str, ...]]] = {
    "static": ("[Static shot]", (
        r"static camera", r"locked[- ]off(?: camera| shot)?", r"fixed camera", r"still camera", r"static (?:shot|hold|frame)",
    )),
    "push_in": ("[Push in]", (
        r"push(?:es|ing)?[- ]in", r"dolly(?:ing|ies)?[- ]in", r"zoom(?:s|ing)?[- ]in", r"creep(?:s|ing)? (?:in|forward)",
    )),
    "pull_out": ("[Pull out]", (
        r"pull(?:s|ing)?[- ](?:out|back|away)", r"dolly(?:ing|ies)?[- ](?:out|back)", r"zoom(?:s|ing)?[- ]out",
    )),
    "tracking": ("[Tracking shot]", (
        r"tracking(?: shot)?", r"track(?:s)? (?:with|beside|alongside|behind)", r"steadicam(?: follow)?",
        r"dolly (?:alongside|left|right)", r"truck(?:s|ing)? (?:left|right)", r"follow(?:s|ing)",
    )),
    "pan_left": ("[Pan left]", (r"(?:whip[- ])?pan(?:s|ning)? (?:to the )?left",)),
    "pan_right": ("[Pan right]", (r"(?:whip[- ])?pan(?:s|ning)? (?:to the )?right",)),
    "tilt_up": ("[Tilt up]", (r"tilt(?:s|ing)? up(?:ward)?",)),
    "tilt_down": ("[Tilt down]", (r"tilt(?:s|ing)? down(?:ward)?",)),
    "crane_up": ("[Pedestal up]", (r"(?:crane|boom|pedestal)(?:s|ing)? up",)),
    "crane_down": ("[Pedestal down]", (r"(?:crane|boom|pedestal)(?:s|ing)? down",)),
    "arc": ("", (r"(?:orbit|arc)(?:s|ing)?(?: around)?",)),
    "handheld": ("[Shake]", (r"hand[- ]?held", r"shaky cam(?:era)?")),
}

# Synonym -> canonical wording, by grammatical form so a verb stays a verb.
# Order matters: the first pattern that matches a span wins.
REWRITES: tuple[tuple[str, str], ...] = (
    (r"(?:dolly|zoom)[- ]in", "push-in"),
    (r"(?:dollies|zooms) in|creeps (?:in|forward)", "pushes in"),
    (r"(?:dollying|zooming) in|creeping (?:in|forward)", "pushing in"),
    (r"(?:dolly|zoom)[- ](?:out|back)|pull[- ]back", "pull-out"),
    (r"(?:dollies|zooms) (?:out|back)", "pulls back"),
    (r"(?:dollying|zooming) (?:out|back)", "pulling back"),
    (r"steadicam(?: follow)?|dolly alongside", "tracking shot"),
    (r"(?:locked[- ]off|fixed|still) camera", "static camera"),
    (r"shaky cam(?:era)?", "handheld camera"),
)

# Motion that promises movement without naming one the model knows; it takes
# the rest of its clause with it ("drifts slowly around her"). Unsure means static.
VAGUE_MOTION = (
    r"(?:digital |slow |gentle )?drift(?:s|ing)?", r"float(?:s|ing)", r"glid(?:es|ing)",
    r"cinematic camera(?: movement| motion)?", r"dynamic camera(?: movement| motion)?",
    r"camera (?:moves|wanders|sways)(?: around| freely)", r"free[- ]?roaming camera",
)

_SUBJECT = r"(?:(?:the )?camera )?"


def _words(patterns, subject: str = "") -> re.Pattern:
    return re.compile(rf"(?<![\w-]){subject}(?:{'|'.join(patterns)})(?![\w-])", re.IGNORECASE)


_DETECT = [(move, command, _words(patterns)) for move, (command, patterns) in CAMERA_MOVES.items()]
_REWRITE = [(_words([pattern]), replacement) for pattern, replacement in REWRITES]
_VAGUE = re.compile(_words(VAGUE_MOTION, _SUBJECT).pattern + r"[^,.;:]*", re.IGNORECASE)
CANONICAL_MOVES = tuple(CAMERA_MOVES)


def _spans(text: str) -> list[tuple[int, int, str, str]]:
    """(start, end, move id, command) for each move named in ``text``; the first table entry wins."""
    found: list[tuple[int, int, str, str]] = []
    for move, command, pattern in _DETECT:
        for match in pattern.finditer(text):
            if not any(start < match.end() and match.start() < end for start, end, _, _ in found):
                found.append((match.start(), match.end(), move, command))
    return sorted(found)


def camera_moves_in(text: str) -> list[str]:
    """The canonical ids of the moves named in ``text``, in order of appearance."""
    return [move for _, _, move, _ in _spans(str(text or ""))]


def canonical_camera_text(text: str, *, bracket_commands: bool | None = None) -> str:
    """``text`` with camera-move synonyms in the closed vocabulary.

    A vague motion ("the camera drifts") becomes "static camera", or is
    dropped when the text already names a real move. Everything else is kept
    as written. With bracket commands on, each named move is followed by its
    MiniMax command.
    """
    text = str(text or "")
    if not text.strip():
        return text
    for pattern, replacement in _REWRITE:
        text = pattern.sub(replacement, text)
    if _VAGUE.search(text):
        text = _VAGUE.sub("" if _spans(text) else "static camera", text)
    use_brackets = MINIMAX_BRACKET_COMMANDS if bracket_commands is None else bracket_commands
    if use_brackets:
        for _, end, _, command in sorted(_spans(text), reverse=True):
            if command:
                text = f"{text[:end]} {command}{text[end:]}"
    text = re.sub(r"\s*,\s*(?=[,.;]|$)", "", text)
    text = re.sub(r"^\s*,\s*", "", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()
