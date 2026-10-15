"""MiniMax H3 sequences: consecutive shots of one scene in one generation.

Red Flag method: one H3 render holds 2-3 shots of the same scene joined by
timed hard cuts, so set, light and wardrobe stay the same across the cuts.
The render is then split back at the cut frames, so the edit, the review and
the Dashboard keep seeing one clip per shot.

Only base H3 (T2VA/I2VA/FL2VA) story runs without driving audio are grouped.
Ref2VA/Omni and shots chained with ``extend_previous`` keep rendering one shot
per generation. Everything here is pure except :func:`split_sequence_video`.
"""

from __future__ import annotations

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
    opens with ``[Shot 1]``, marks every later cut ``[Shot N] At MM:SS.mmm``
    and validates the result against the same Context-IR contract.
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
        lead = "[Shot 1]" if number == 1 else f"[Shot {number}] At {_h3_shot_timestamp(offset)}, hard cut."
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
