"""MiniMax H3 sequences: consecutive shots of one scene in one generation.

Red Flag method: one H3 render holds 2-3 shots of the same scene joined by
timed hard cuts, so set, light and wardrobe stay the same across the cuts.
The render is then split back at the cut frames, so the edit, the review and
the Dashboard keep seeing one clip per shot.

Only base H3 (T2VA/I2VA/FL2VA) story runs without driving audio are grouped.
Ref2VA/Omni and shots chained with ``extend_previous`` keep rendering one shot
per generation. H3 does not cut exactly where the prompt asks (the drift grows
along the sequence and it sometimes adds cuts), so the split snaps each planned
cut to the nearest cut found by scene detection on the render. Everything here
is pure except :func:`detect_scene_cuts` and :func:`split_sequence_video`.
"""

from __future__ import annotations

import itertools
import re
from typing import Any, Mapping, MutableMapping, Sequence

from services.director.h3_dialogue import (
    H3DialogueContractError,
    _alignment_header,
    _extract_h3_fields,
    _normalized_space,
    _rewrite_outside_dialogue,
    normalize_h3_text,
    validate_h3_prompt_contract,
)
from services.director_video_strategy import BOUNDED_START_END, video_strategy

SEQUENCE_KEY = "_director_sequence"
# H3 drifts off the asked cut times more with every cut (up to ~1 s by the
# fourth shot), so a sequence holds at most three shots.
MAX_SEQUENCE_SHOTS = 3
# ffmpeg ``scene`` score above which a frame counts as a hard cut.
SCENE_CUT_THRESHOLD = 0.27
# A planned cut snaps to a detected cut at most this far away.
CUT_SNAP_TOLERANCE_SEC = 1.2
# No snapped cut may leave a shot shorter than this.
MIN_SEQUENCE_SHOT_SEC = 0.5
_SNAP_CANDIDATES = 4
_SEQUENCE_MODES = {"t2va", "i2va", "fl2va", "l2va"}
_OFF = {"off", "false", "0", "no"}
_ON = {"on", "true", "1", "yes"}


def h3_sequences_enabled(params: Mapping[str, Any], model_def: Mapping[str, Any] | None) -> bool:
    """Whether this run renders same-scene shots as H3 sequences.

    ``director_h3_sequences``: "off" disables it, "on" enables it for any
    eligible story run, and "auto" (the default) enables it for custom story
    skills only.
    """
    flag = params.get("director_h3_sequences", "auto")
    text = str(flag).strip().lower() if flag is not None else "auto"
    if flag is False or text in _OFF:
        return False
    video_model = str(params.get("video_model") or "").lower()
    model_def = model_def or {}
    if not video_model.startswith("minimax_h3") or "ref2va" in video_model:
        return False
    if video_strategy(model_def) != BOUNDED_START_END or any(
        model_def.get(key) for key in ("omni_reference", "audio_only", "minimax_h3_viggle")
    ):
        return False
    # Driving audio is sliced per shot; a sequence would need its own slice.
    if params.get("pipeline_type") != "short_film_story" or params.get("audio_path"):
        return False
    if flag is True or text in _ON:
        return True
    return str(params.get("director_skill") or "").startswith("custom:")


def h3_per_shot_story(params: Mapping[str, Any], model_def: Mapping[str, Any] | None) -> bool:
    """A run sequences could film that renders every shot on its own instead.

    Each shot is its own generation, anchored on its start image (and, for
    bounded start/end, the next same-scene shot's image as end). A shot
    planned shorter than the H3 minimum renders at the minimum and the edit
    trims it back (see :func:`per_shot_render_frames`).
    """
    return (not h3_sequences_enabled(params, model_def)
            and h3_sequences_enabled({**params, "director_h3_sequences": "on"}, model_def))


def per_shot_render_frames(
    planned_frames: Sequence[int], *, min_frames: int, max_frames: int, frame_step: int,
) -> list[int]:
    """Each shot's render length: the nearest native H3 length, at least the minimum.

    Shots are rounded one by one (no carried residual, unlike a media-driven
    schedule): a short shot raised to the minimum must not shorten the next.
    """
    step = max(1, int(frame_step or 1))
    minimum = int(min_frames)
    maximum = minimum + max(0, (int(max_frames) - minimum) // step) * step
    valid = range(minimum, maximum + 1, step)
    return [min(valid, key=lambda candidate: (abs(candidate - max(1, int(frames))), candidate))
            for frames in planned_frames]


def per_shot_keep_ranges(
    plans: Sequence[MutableMapping[str, Any]], planned_frames: Sequence[int], rendered_frames: Sequence[int],
    *, min_frames: int, fps: float,
) -> list[int]:
    """Trim shots planned shorter than the H3 minimum back to their planned length.

    Sets ``_director_keep_sec`` to ``[0, planned]`` on such a shot (unless it
    already has its own keep range) and returns the indexes it set.
    """
    trimmed = []
    for index, (plan, planned, rendered) in enumerate(zip(plans, planned_frames, rendered_frames)):
        if planned < int(min_frames) and planned < rendered and not plan.get("_director_keep_sec"):
            plan["_director_keep_sec"] = [0.0, round(planned / float(fps), 3)]
            trimmed.append(index)
    return trimmed


def sequence_total_frames(member_frames: Sequence[int], *, min_frames: int, frame_step: int) -> int:
    """The first native H3 length that holds every member's full length."""
    total = sum(int(frames) for frames in member_frames)
    step = max(1, int(frame_step or 1))
    if total <= min_frames:
        return int(min_frames)
    return int(min_frames) + -(-(total - int(min_frames)) // step) * step


def _group(plan: Mapping[str, Any]) -> str:
    return str(plan.get("_director_continuity_group") or "").strip()


def _chained(clip_plans: Sequence[Mapping[str, Any]], index: int) -> bool:
    """Same rule as the renderer: an extend_previous shot of the same scene."""
    if index <= 0 or index >= len(clip_plans):
        return False
    plan = clip_plans[index]
    return bool(
        str(plan.get("_director_continuity_strategy") or "").strip().lower() == "extend_previous"
        and _group(plan)
        and _group(plan) == _group(clip_plans[index - 1])
    )


def plan_h3_sequences(
    clip_plans: Sequence[Mapping[str, Any]],
    frames: Sequence[int],
    *,
    max_frames: int,
    min_frames: int,
    frame_step: int,
    max_shots: int = MAX_SEQUENCE_SHOTS,
    linked: Sequence[bool] | None = None,
) -> list[list[int]]:
    """Group consecutive shots of one continuity group into sequences.

    A sequence holds 2..``max_shots`` shots whose summed full lengths fit one
    native render. Shots on either side of an ``extend_previous`` link (or of
    any ``linked[i]`` last-frame handoff from shot i-1 to shot i) keep their
    chained single-shot rendering, and windowed shots stay alone.
    """
    count = min(len(clip_plans), len(frames))

    def chained(index: int) -> bool:
        if linked is not None and 0 < index < min(count, len(linked)) and linked[index]:
            return True
        return _chained(clip_plans, index)

    eligible = [
        bool(_group(clip_plans[i]))
        and not chained(i)
        and not chained(i + 1)
        and len(clip_plans[i].get("window_prompts") or []) <= 1
        and int(frames[i] or 0) > 0
        for i in range(count)
    ]
    groups: list[list[int]] = []
    index = 0
    while index < count:
        if not eligible[index]:
            index += 1
            continue
        members = [index]
        nxt = index + 1
        while (
            nxt < count
            and len(members) < max_shots
            and eligible[nxt]
            and _group(clip_plans[nxt]) == _group(clip_plans[index])
            and sequence_total_frames(
                [frames[i] for i in (*members, nxt)], min_frames=min_frames, frame_step=frame_step,
            ) <= max_frames
        ):
            members.append(nxt)
            nxt += 1
        if len(members) > 1:
            groups.append(members)
        index = nxt
    return groups


def annotate_h3_sequences(
    clip_plans: Sequence[MutableMapping[str, Any]],
    groups: Sequence[Sequence[int]],
    frames: Sequence[int],
    *,
    fps: float,
    min_frames: int,
    frame_step: int,
) -> None:
    """Store each member's sequence id and cut offsets on its clip plan.

    Members render their full planned length, so their render-long-cut-short
    ``keep_sec`` is parked in the sequence record (and restored if the shot
    later renders alone again).
    """
    for plan in clip_plans:
        stale = plan.pop(SEQUENCE_KEY, None)
        if isinstance(stale, Mapping) and stale.get("keep_sec") and not plan.get("_director_keep_sec"):
            plan["_director_keep_sec"] = stale["keep_sec"]
    fps = float(fps)
    for members in groups:
        member_frames = [int(frames[i]) for i in members]
        total = sequence_total_frames(member_frames, min_frames=min_frames, frame_step=frame_step)
        sequence_id = f"seq_{members[0] + 1:02d}"
        start = 0
        for position, index in enumerate(members):
            plan = clip_plans[index]
            keep = plan.pop("_director_keep_sec", None)
            plan[SEQUENCE_KEY] = {
                "id": sequence_id,
                "index": position,
                "size": len(members),
                "members": list(members),
                "start_frame": start,
                "frames": member_frames[position],
                "total_frames": total,
                "fps": fps,
                "offset_sec": round(start / fps, 3),
                "duration_sec": round(member_frames[position] / fps, 3),
                **({"keep_sec": keep} if keep else {}),
            }
            start += member_frames[position]


def sequence_units(clip_count: int, groups: Sequence[Sequence[int]]) -> list[list[int]]:
    """Render units in shot order: one per sequence, one per lone shot."""
    first = {members[0]: list(members) for members in groups}
    units: list[list[int]] = []
    index = 0
    while index < clip_count:
        unit = first.get(index, [index])
        units.append(unit)
        index = unit[-1] + 1
    return units


def saved_sequence_members(clips: Sequence[Mapping[str, Any]], clip_index: int) -> list[int]:
    """The members of a saved clip's sequence, or [] when it renders alone."""
    record = clips[clip_index].get(SEQUENCE_KEY) if 0 <= clip_index < len(clips) else None
    if not isinstance(record, Mapping):
        return []
    members = record.get("members")
    if not isinstance(members, list) or len(members) < 2 or clip_index not in members:
        return []
    if members != list(range(members[0], members[0] + len(members))) or members[-1] >= len(clips):
        return []
    for position, index in enumerate(members):
        other = clips[index].get(SEQUENCE_KEY)
        if not isinstance(other, Mapping) or other.get("id") != record.get("id") or other.get("index") != position:
            return []
    return list(members)


# Times a planner writes inside one shot ("0.0-2.0s: ...", "2.0s: ...") and
# the compiler's dialogue timing clause count from that shot's own start.
_TIME_RANGE_RE = re.compile(
    r"(?<![\d.:])(\d{1,3}(?:\.\d+)?)(\s*(?:-|–|to)\s*)(\d{1,3}(?:\.\d+)?)(\s*s\b)"
)
_TIME_POINT_RE = re.compile(r"(?<![\d.:\-–])(\d{1,3}(?:\.\d+)?)(s:)")
_DIALOGUE_TIMING_RE = re.compile(r"Dialogue timing:.*?through \d+(?:\.\d+)? seconds\.", re.DOTALL)


def _shift_number(value: str, offset: float) -> str:
    decimals = len(value.split(".")[1]) if "." in value else 1
    return f"{float(value) + offset:.{max(1, decimals)}f}"


def shift_local_times(body: str, offset: float) -> str:
    """Move a shot's local times onto the sequence timeline."""
    if offset <= 0:
        return body

    def rewrite(text: str) -> str:
        text = _DIALOGUE_TIMING_RE.sub(
            lambda match: re.sub(
                r"\d+\.\d+", lambda number: f"{float(number.group(0)) + offset:.2f}", match.group(0),
            ),
            text,
        )
        text = _TIME_RANGE_RE.sub(
            lambda m: f"{_shift_number(m.group(1), offset)}{m.group(2)}{_shift_number(m.group(3), offset)}{m.group(4)}",
            text,
        )
        return _TIME_POINT_RE.sub(lambda m: f"{_shift_number(m.group(1), offset)}{m.group(2)}", text)

    return _rewrite_outside_dialogue(body, rewrite)


def _speech_beats(plan: Mapping[str, Any]) -> list[Any]:
    audio_mode = _normalized_space((plan.get("_director_audio_plan") or {}).get("mode", "")).casefold()
    if audio_mode in {"audio_driven", "music_driven"}:
        return []
    return list(plan.get("_director_dialogue_beats") or [])


_CUT_FRAMING_MAX_WORDS = 6
_CUT_SUBJECT_MAX = 3


def _cut_framing(plan: Mapping[str, Any]) -> str:
    """The shot's framing in a few words ("a medium close-up")."""
    framing = _normalized_space(plan.get("_director_camera_framing", "")).strip(" .")
    framing = re.split(r"[,;:(]|\s+-\s+", framing, maxsplit=1)[0].strip()
    words = framing.split()[:_CUT_FRAMING_MAX_WORDS]
    if not words:
        return ""
    if words[0].casefold() in {"a", "an", "the"}:
        return " ".join(words)
    framing = " ".join(words).lower()
    article = "an" if framing[:1] in "aeiou" else "a"
    return f"{article} {framing}"


def _cut_subjects(plan: Mapping[str, Any]) -> str:
    names: list[str] = []
    for subject in plan.get("_director_subjects_on_screen") or []:
        if not isinstance(subject, Mapping):
            continue
        name = _normalized_space(subject.get("speaker_name") or "")
        if not name:
            text = _normalized_space(subject.get("visual_description") or "")
            text = re.split(r"[,;.(]|\s(?:in|with|wearing|who|holding)\s", text, maxsplit=1)[0]
            name = " ".join(text.split()[:5]).strip()
        if name and name not in names:
            names.append(name)
    names = names[:_CUT_SUBJECT_MAX]
    if len(names) > 1:
        return f"{', '.join(names[:-1])} and {names[-1]}"
    return names[0] if names else ""


def cut_target(plan: Mapping[str, Any]) -> str:
    """What the official H3 cut sentence cuts to: the shot's framing and subject.

    The H3 prompt guide writes every later shot as
    ``[Shot N] At MM:SS.mmm, the camera cuts to <framing/subject>...``.
    """
    framing, subjects = _cut_framing(plan), _cut_subjects(plan)
    if framing and subjects:
        return f"{framing} of {subjects}"
    if subjects:
        return subjects
    return framing or "the next shot"


def compile_h3_sequence_prompt(
    member_plans: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    offsets_sec: Sequence[float],
    total_duration_sec: float,
) -> str:
    """Join the members' compiled H3 prompts into one multi-shot prompt.

    Each member is first compiled alone by ``compile_h3_clip_plans`` (exact
    dialogue tags, speaker ids, identity anchors). This keeps those bodies,
    opens with ``[Shot 1]``, marks every later cut with the official H3 form
    ``[Shot N] At MM:SS.mmm, the camera cuts to <framing> of <subject>.`` and validates the result against the same Context-IR contract.
    """
    from services.h3_window_planner import _h3_shot_timestamp

    mode = str(mode or "t2va").strip().lower()
    if mode not in _SEQUENCE_MODES:
        raise H3DialogueContractError(f"H3 sequences do not support the {mode} prompt mode")
    if len(member_plans) < 2 or len(offsets_sec) != len(member_plans):
        raise H3DialogueContractError("An H3 sequence needs 2+ shots with one offset each")
    look = _normalized_space(member_plans[0].get("_director_look", ""))
    shots: list[str] = []
    soundscapes: list[str] = []
    music = "N/A"
    beats: list[Any] = []
    subjects: dict[str, Any] = {}
    anchors: list[str] = []
    for number, (plan, offset) in enumerate(zip(member_plans, offsets_sec), start=1):
        compiled = str(plan.get("_director_h3_compiled_prompt") or plan.get("video_prompt") or "")
        fields = _extract_h3_fields(compiled)
        body = _normalized_space(fields.get("integrated_multimodal_description", ""))
        if not body:
            raise H3DialogueContractError(f"Sequence shot {number} has no compiled H3 description")
        body = re.sub(r"^\[Shot\s+\d+\]\s*", "", body, flags=re.IGNORECASE)
        if look:
            body = _normalized_space(body.replace(f"Visual style: {look}", ""))
        body = shift_local_times(body, float(offset))
        lead = "[Shot 1]" if number == 1 else (
            f"[Shot {number}] At {_h3_shot_timestamp(offset)}, the camera cuts to {cut_target(plan)}."
        )
        shots.append(f"{lead} {body}")
        sound = _normalized_space(fields.get("overall_soundscape", "")).rstrip(". ")
        if sound and sound.upper() != "N/A" and sound not in soundscapes:
            soundscapes.append(sound)
        member_music = _normalized_space(fields.get("non_diegetic_music", "")).rstrip(". ")
        if music == "N/A" and member_music and member_music.upper() != "N/A":
            music = member_music
        beats.extend(_speech_beats(plan))
        for subject in plan.get("_director_subjects_on_screen") or []:
            key = _normalized_space(
                (subject or {}).get("character_id") or (subject or {}).get("speaker_name") or ""
            ) if isinstance(subject, Mapping) else ""
            subjects.setdefault(key or f"_{len(subjects)}", subject)
        anchors.extend(plan.get("_director_required_context_anchors") or [])
    description = " ".join(shots)
    if look:
        description = f"{description} Visual style: {look}"
    compiled = (
        f"integrated_multimodal_description: {description}\n\n"
        f"overall_soundscape: {'; '.join(soundscapes) or 'N/A'}.\n\n"
        f"non_diegetic_music: {music}"
    )
    header = _alignment_header(mode, total_duration_sec, len(shots))
    if header:
        compiled = f"{header}\n\n{compiled}"
    compiled = normalize_h3_text(compiled).strip()
    errors = validate_h3_prompt_contract(
        compiled, beats, mode=mode, subjects=list(subjects.values()),
        context_anchors=list(dict.fromkeys(anchors)),
    )
    found = [int(n) for n in re.findall(
        r"\[Shot\s+(\d+)\]", _extract_h3_fields(compiled).get("integrated_multimodal_description", ""),
    )]
    if found != list(range(1, len(shots) + 1)):
        errors.append(f"expected shots 1-{len(shots)} in order, found {found}")
    if errors:
        raise H3DialogueContractError("H3 sequence: " + "; ".join(errors))
    return compiled


def split_sequence_video(
    source: str,
    frame_ranges: Sequence[tuple[int, int]],
    fps: float,
    output_paths: Sequence[str],
) -> list[str]:
    """Cut one sequence render into its shots at exact frame boundaries."""
    import subprocess

    from services.director.edit_cut import _has_audio

    if len(frame_ranges) != len(output_paths) or not frame_ranges:
        raise ValueError("one output path per frame range is required")
    has_audio = _has_audio(source)
    for (start, end), output in zip(frame_ranges, output_paths):
        if end <= start:
            raise ValueError(f"empty frame range {start}-{end}")
        graph = f"[0:v]trim=start_frame={int(start)}:end_frame={int(end)},setpts=PTS-STARTPTS[v]"
        command = ["ffmpeg", "-v", "error", "-y", "-i", source]
        if has_audio:
            graph += (f";[0:a]atrim=start={start / fps:.6f}:end={end / fps:.6f},"
                      "asetpts=PTS-STARTPTS[a]")
        command += ["-filter_complex", graph, "-map", "[v]"]
        if has_audio:
            command += ["-map", "[a]", "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "160k"]
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", output]
        subprocess.run(command, check=True, capture_output=True, timeout=600)
    return list(output_paths)


_PTS_TIME_RE = re.compile(r"pts_time:\s*(-?\d+(?:\.\d+)?)")


def detect_scene_cuts(source: str, *, threshold: float = SCENE_CUT_THRESHOLD, timeout: float = 300) -> list[float]:
    """Times (seconds) of the hard cuts ffmpeg's scene detection finds in ``source``.

    One decode pass, video only. Raises when ffmpeg fails, so the caller can
    fall back to the planned cuts.
    """
    import subprocess

    command = [
        "ffmpeg", "-hide_banner", "-nostats", "-i", source, "-an", "-sn", "-dn",
        "-vf", f"select='gt(scene,{float(threshold)})',showinfo", "-f", "null", "-",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg scene detection failed: {result.stderr.strip()[-300:]}")
    times = {
        round(float(match.group(1)), 3)
        for line in result.stderr.splitlines() if "showinfo" in line
        for match in [_PTS_TIME_RE.search(line)] if match
    }
    return sorted(time for time in times if time > 0)


def snap_sequence_cuts(
    planned_cuts: Sequence[int],
    detected_sec: Sequence[float],
    *,
    fps: float,
    total_frames: int,
    tolerance_sec: float = CUT_SNAP_TOLERANCE_SEC,
    min_shot_sec: float = MIN_SEQUENCE_SHOT_SEC,
) -> tuple[list[int], list[float]]:
    """Snap each planned cut frame to the nearest detected cut.

    A planned cut matches a detected cut at most ``tolerance_sec`` away; each
    detected cut matches at most one planned cut. The result keeps the cuts
    strictly increasing with every shot at least ``min_shot_sec`` long, between
    0 and ``total_frames``. Among valid assignments the one matching the most
    cuts wins, then the one closest to the plan. Unmatched planned cuts keep
    their planned frame. Returns the cut frames and the detected cuts (seconds)
    that matched no planned cut.
    """
    fps = float(fps)
    planned = [int(cut) for cut in planned_cuts]
    detected = sorted({float(time) for time in detected_sec})
    if not planned or fps <= 0:
        return planned, detected
    min_frames = max(1, int(round(min_shot_sec * fps)))
    tolerance = float(tolerance_sec) * fps
    total = int(total_frames)

    def valid(cuts: Sequence[int]) -> bool:
        bounds = [0, *cuts, total]
        return all(b - a >= min_frames for a, b in zip(bounds, bounds[1:]))

    options: list[list[int | None]] = []
    for cut in planned:
        near = sorted(
            (index for index, time in enumerate(detected) if abs(time * fps - cut) <= tolerance),
            key=lambda index: abs(detected[index] * fps - cut),
        )
        options.append([None, *near[:_SNAP_CANDIDATES]])
    best: tuple[int, float] | None = None
    best_cuts: list[int] | None = None
    best_used: set[int] = set()
    for choice in itertools.product(*options):
        used = [index for index in choice if index is not None]
        if len(set(used)) != len(used):
            continue
        cuts = [
            cut if index is None else int(round(detected[index] * fps))
            for cut, index in zip(planned, choice)
        ]
        if not valid(cuts):
            continue
        score = (-len(used), sum(abs(a - b) for a, b in zip(cuts, planned)))
        if best is None or score < best:
            best, best_cuts, best_used = score, cuts, set(used)
    if best_cuts is None:
        return planned, detected
    return best_cuts, [time for index, time in enumerate(detected) if index not in best_used]


def sequence_frame_ranges(
    records: Sequence[Mapping[str, Any]],
    detected_sec: Sequence[float] | None,
    *,
    tolerance_sec: float = CUT_SNAP_TOLERANCE_SEC,
    min_shot_sec: float = MIN_SEQUENCE_SHOT_SEC,
) -> tuple[list[tuple[int, int]], list[float]]:
    """The frame range of each member in the render, snapped to real cuts.

    ``records`` are the members' sequence records in order. Without detected
    cuts (``None``: detection failed) the planned ranges are returned. The
    first shot starts at frame 0; the last keeps its planned length from its
    actual start, within the rendered length.
    """
    if not records:
        return [], []
    fps = float(records[0]["fps"])
    total = int(records[0].get("total_frames") or 0) or sum(int(r["frames"]) for r in records)
    planned_cuts = [int(record["start_frame"]) for record in records[1:]]
    if detected_sec is None:
        cuts, ignored = planned_cuts, []
    else:
        cuts, ignored = snap_sequence_cuts(
            planned_cuts, detected_sec, fps=fps, total_frames=total,
            tolerance_sec=tolerance_sec, min_shot_sec=min_shot_sec,
        )
    starts = [0, *cuts]
    last = records[-1]
    end = min(total, starts[-1] + int(last["frames"]))
    if detected_sec is None:
        end = int(last["start_frame"]) + int(last["frames"])
    ends = [*cuts, max(end, starts[-1] + 1)]
    return list(zip(starts, ends)), ignored
