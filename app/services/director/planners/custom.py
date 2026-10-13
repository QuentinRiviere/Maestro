"""Generic planner for declarative (custom) Director skills.

The skill definition supplies the writing rules; this planner supplies the
shot schema, timing, references and the shared policies, so a definition can
never execute code or change the pipeline contract.
"""
from __future__ import annotations

import math
from typing import Any, Optional

from ..policies import build_camera_style_block, build_character_rules_block
from ..schema import (
    AssetRef, AudioPlan, CameraPlan, CharacterProfile, DialogueBeat, ProductionPlan, ReferenceAssets, ShotPlan,
    SubjectRef,
)
from .base import BasePlanner

_RULE_LABELS = {
    "goal": "Purpose", "structure": "Structure", "pacing": "Pacing", "camera": "Camera",
    "dialogue": "Dialogue", "sound": "Sound",
}


def model_shot_bounds(kwargs: dict) -> tuple[float, float] | None:
    """The video model's renderable shot length in seconds, when it has one.

    Shots shorter than the minimum are later merged two by two into one
    render, which keeps only the first shot's prompt, so planners must write
    shots the model can render on their own.
    """
    try:
        fps = float(kwargs.get("fps") or 0)
        minimum = int(kwargs.get("frames_minimum") or 0)
        maximum = int(kwargs.get("frames_maximum") or 0)
    except (TypeError, ValueError):
        return None
    if fps <= 0 or minimum <= 0 or maximum < minimum:
        return None
    return minimum / fps, maximum / fps


def fit_shot_range(low: float, high: float, bounds: tuple[float, float] | None) -> tuple[float, float]:
    """Clamp a skill's shot range into what the video model renders natively."""
    if not bounds:
        return float(low), float(high)
    model_low = math.ceil(bounds[0] * 10) / 10
    model_high = max(model_low, math.floor(bounds[1] * 10) / 10)
    low = min(max(float(low), model_low), model_high)
    return low, min(max(float(high), low), model_high)


def skill_rules_block(spec: dict, options: dict[str, str]) -> str:
    """The skill-specific instructions, in the definition's own words."""
    rules = spec.get("rules") or {}
    lines = [f"SKILL: {spec.get('name')} — {spec.get('description') or ''}".strip()]
    for field, label in _RULE_LABELS.items():
        if rules.get(field):
            lines.append(f"{label}: {rules[field]}")
    if rules.get("do"):
        lines.append("Always: " + "; ".join(rules["do"]))
    if rules.get("dont"):
        lines.append("Never: " + "; ".join(rules["dont"]))
    for option in spec.get("options") or []:
        value = options.get(option["key"], option.get("default"))
        if value:
            lines.append(f"{option['label']}: {value}")
    notes = [ref["notes"] for ref in spec.get("references") or [] if ref.get("notes")]
    if notes:
        lines.append("Patterns learned from the reference media (apply their craft, never copy their content):")
        lines += [f"- {note}" for note in notes[:6]]
    return "\n".join(lines)


class CustomSkillPlanner(BasePlanner):
    """Plans shots for a validated custom skill definition."""

    def __init__(self, spec: dict, llm_generate: Any = None, llm_generate_streaming: Any = None):
        super().__init__(llm_generate=llm_generate, llm_generate_streaming=llm_generate_streaming)
        self.spec = spec
        self.skill_type = "custom:" + str(spec.get("id") or "skill")

    def plan(
        self,
        concept: str = "",
        audio_path: Optional[str] = None,
        reference_image_path: Optional[str] = None,
        characters: Optional[list[dict]] = None,
        target_duration: Optional[int] = None,
        clips: Optional[list[dict]] = None,
        skill_options: Optional[dict] = None,
        **kwargs,
    ) -> ProductionPlan:
        spec = self.spec
        concept = concept or kwargs.get("scene_description") or kwargs.get("story_description") or ""
        if not concept.strip():
            raise ValueError(f"{spec.get('name')} needs a brief")
        options = {str(k): str(v) for k, v in (skill_options or {}).items()}
        has_reference = bool(reference_image_path)
        low, high = fit_shot_range(*(spec.get("shot_duration") or [3, 8])[:2], model_shot_bounds(kwargs))
        target_duration = int(target_duration or spec.get("default_duration") or 30)
        clip_durations = [
            max(0.5, float(c.get("duration_sec") or (float(c.get("end", 0)) - float(c.get("start", 0))) or low))
            for c in (clips or []) if isinstance(c, dict)
        ]
        if clip_durations:
            target_scenes = len(clip_durations)
            timing_rule = (f"The audio is already cut into {target_scenes} segments. Output exactly {target_scenes} "
                           "shot plans, in order, with these durations in seconds: "
                           + ", ".join(f"{d:.1f}" for d in clip_durations) + ".")
        else:
            average = (low + high) / 2
            target_scenes = max(1, min(int(spec.get("max_shots") or 12), round(target_duration / average)))
            timing_rule = (f"Output exactly {target_scenes} shot plans totaling about {target_duration}s, "
                           f"each between {low:g} and {high:g} seconds. Every shot is rendered on its own, so give "
                           "each one enough action to fill its full duration.")

        char_profiles = [
            CharacterProfile(id=f"char_{i}", display_name=c.get("name", ""),
                             physical_description=c.get("description", "person"),
                             wardrobe=c.get("wardrobe") or None)
            for i, c in enumerate(characters or [])
        ]
        system_prompt = f"""You are the director of a production written with a specific skill.

{skill_rules_block(spec, options)}

{"You are given a REFERENCE PHOTO; keep its subjects and look." if has_reference else ""}
{build_character_rules_block(has_reference, char_profiles or None)}

{build_camera_style_block()}

OUTPUT FORMAT — respond with ONLY a JSON array of shot objects:
[{{"scene_goal": "...", "scene_type": "short label", "duration_sec": {low},
  "subjects_on_screen": [{{"visual_description": "..."}}], "spatial_setup": "...", "environment": "...",
  "visual_style": "...", "lighting": "...", "mood": "...", "action_beats": ["one main action"],
  "dialogue_beats": [{{"spoken_text": "...", "delivery": "..."}}],
  "camera_plan": {{"framing": "...", "movement": "...", "movement_intensity": "subtle|moderate|dynamic"}},
  "audio_plan": {{"mode": "generated_audio"}}, "ending_beat": "..."}}]

{timing_rule} Go:"""
        if kwargs.get("polish_block"):
            system_prompt = f"{system_prompt}\n\n{kwargs['polish_block']}"
        user_prompt = f"Brief: {concept}\nTarget duration: {target_duration}s"

        shot_dicts = self._call_llm_json(
            user_prompt=user_prompt, system_prompt=system_prompt,
            max_tokens=max(1024, target_scenes * 320 + 512),
            image_paths=[reference_image_path] if has_reference else None,
        )
        if not shot_dicts:
            raise ValueError("The LLM returned no shots")
        if clip_durations:
            shot_dicts = (list(shot_dicts) + [shot_dicts[-1]] * len(clip_durations))[: len(clip_durations)]

        shots = []
        for i, raw in enumerate(shot_dicts):
            if clip_durations:
                duration = clip_durations[i]
            else:
                try:
                    duration = max(float(low), min(float(high), float(raw.get("duration_sec") or low)))
                except (TypeError, ValueError):
                    duration = float(low)
            cam = raw.get("camera_plan") or {}
            audio_raw = raw.get("audio_plan") or {}
            shots.append(ShotPlan(
                shot_id=self._make_shot_id(i, "cs"),
                index=i,
                duration_sec=duration,
                skill_type=self.skill_type,
                scene_goal=raw.get("scene_goal", f"Shot {i + 1}"),
                scene_type=raw.get("scene_type", "scene"),
                source_mode_preference="i2v" if has_reference else "t2v",
                image_strategy="reference_edit" if has_reference else "fresh_generation",
                continuity_strategy="independent",
                subjects_on_screen=[SubjectRef.from_dict(s) if isinstance(s, dict) else SubjectRef(visual_description=str(s))
                                    for s in raw.get("subjects_on_screen", [])],
                spatial_setup=raw.get("spatial_setup", ""),
                environment=raw.get("environment", ""),
                visual_style=raw.get("visual_style", ""),
                lighting=raw.get("lighting", ""),
                mood=raw.get("mood", ""),
                action_beats=raw.get("action_beats", []),
                dialogue_beats=[DialogueBeat.from_dict(d) for d in raw.get("dialogue_beats") or []] or None,
                camera_plan=CameraPlan(framing=cam.get("framing", "medium shot"), angle=cam.get("angle"),
                                       movement=cam.get("movement"),
                                       movement_intensity=cam.get("movement_intensity", "moderate")),
                audio_plan=AudioPlan(mode="audio_driven" if clip_durations else audio_raw.get("mode", "generated_audio"),
                                     ambience=audio_raw.get("ambience"),
                                     timing_anchor="audio" if clip_durations else "video"),
                ending_beat=raw.get("ending_beat", ""),
                metadata={"skill": spec.get("id"), "skill_version": spec.get("version"), **options},
            ))
        return ProductionPlan(
            skill_type=self.skill_type,
            title=None,
            global_style=f"{spec.get('name')} — {concept}"[:300],
            total_duration_sec=sum(s.duration_sec for s in shots),
            reference_assets=ReferenceAssets(
                start_image=AssetRef(id="ref_image", type="image", uri=reference_image_path) if has_reference else None,
                audio=AssetRef(id="audio", type="audio", uri=audio_path) if audio_path else None,
            ),
            characters=char_profiles or None,
            shots=shots,
            continuity_notes=[f"Custom skill {spec.get('name')} v{spec.get('version')}"],
        )
