"""Generic planner for declarative (custom) Director skills.

The skill definition supplies the writing rules; this planner supplies the
shot schema, timing, references and the shared policies, so a definition can
never execute code or change the pipeline contract.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

from ..policies import build_camera_style_block, build_character_rules_block
from ..schema import (
    AssetRef, AudioPlan, CameraPlan, CharacterProfile, DialogueBeat, ProductionPlan, ReferenceAssets, ShotPlan,
    SubjectRef,
)
from ..edit_cut import normalize_keep
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


_GRAMMAR_LABELS = {
    "coverage": "Coverage", "cutting_rhythm": "Cutting rhythm", "camera_default": "Camera default",
    "blocking": "Blocking", "lighting_continuity": "Lighting continuity", "acting": "Acting",
}

# Craft every narrative skill shares. The model renders each shot alone, so the
# scene has to be designed once and every shot written as a piece of it.
SCENE_CRAFT = """SCENE CRAFT (this is what separates a real scene from generated clips):
- Shots are coverage of continuous scenes, not separate postcards. Every shot belongs to one scene from the
  SCENE DESIGN and keeps that scene's place, time of day, light, props and character positions.
- Screen direction: each character keeps the screen side the scene geography gives them (180-degree rule);
  eyelines match across shot/reverse-shot.
- Cut on action: a shot opens exactly where the previous shot of the same scene ended (same pose, same grip,
  same distance). Write opening_blocking and closing_blocking as concrete positions and poses.
- Action is performance, second by second: a physical intention, the micro-expression it causes, the reaction.
  No filler motion (aimless walking, hair touching, looking around) to fill time.
- Camera is static (locked-off) by default. Move only when the story motivates it (a slow push-in on a
  realization, a follow when a character crosses the space) and say why in camera_plan.motivation.
- Framing follows the coverage: establishing or master to place people, over-the-shoulder shot/reverse-shot for
  exchanges, inserts for decisive details (a hand, an object), close-ups for reactions.
- Do not describe style, grade, lens flare or "aesthetic" in shots: the visual style is set once for the whole
  production. Shot fields describe what the camera sees and what people do.
- Dialogue: at most one line per shot, short enough to say inside keep_sec (about 2.5 words per second), then
  explicit silence (breath, stillness) until the end of the shot; never a line the model must repeat to fill time.
- keep_sec: the [start, end] seconds of the rendered shot the final edit keeps. Reactions and inserts keep
  1.5-3 s, exchanges 2-4 s, establishing shots 3-5 s; any dialogue and the decisive action fall inside it."""

SCENE_DESIGN_SYSTEM = """You are the director preparing the scenes of a short production before shooting.
From the brief, the skill rules and the cast, design the scenes. Return ONLY a JSON array, one object per scene:
[{"scene_id": "short_snake_case", "place": "the exact location", "time_of_day": "...",
  "light": "one lighting description kept for every shot of this scene",
  "geography": "the layout of the space: where each character stands or moves, which screen side each one keeps",
  "props": ["objects that matter and which hand or place holds them"],
  "beats": ["the dramatic beats of this scene in order, each one action and its reaction"]}]
Use as few scenes as the story needs (usually one or two for a short piece). Time and light only change
between scenes, never inside one."""


def scene_grammar_block(spec: dict) -> str:
    grammar = spec.get("scene_grammar") or {}
    if not isinstance(grammar, dict):
        return ""
    lines = [f"{label}: {grammar[key]}" for key, label in _GRAMMAR_LABELS.items() if grammar.get(key)]
    return ("GENRE SCENE GRAMMAR:\n" + "\n".join(lines)) if lines else ""


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
        narrative = spec.get("workflow", "story") == "story" and not clip_durations
        rules = skill_rules_block(spec, options)
        grammar = scene_grammar_block(spec)
        scenes = self._design_scenes(concept, rules, grammar, characters or []) if narrative else []
        scene_ids = [scene["scene_id"] for scene in scenes]
        scene_block = (
            "SCENE DESIGN (fixed; every shot uses one scene_id and keeps its place, light and geography):\n"
            + json.dumps(scenes, ensure_ascii=False, indent=1)
        ) if scenes else ""
        system_prompt = f"""You are the director of a production written with a specific skill.

{rules}

{grammar}

{"You are given a REFERENCE PHOTO; keep its subjects and look." if has_reference else ""}
{build_character_rules_block(has_reference, char_profiles or None)}

{SCENE_CRAFT if narrative else build_camera_style_block()}

{scene_block}

OUTPUT FORMAT — respond with ONLY a JSON array of shot objects:
[{{"scene_id": "{scene_ids[0] if scene_ids else 'scene_1'}", "scene_goal": "...", "scene_type": "establishing|exchange|insert|reaction|action",
  "duration_sec": {low}, "keep_sec": [0, {low}],
  "subjects_on_screen": [{{"visual_description": "...", "position_or_relation": "screen side, distance, facing"}}],
  "opening_blocking": "exact positions and poses at the first frame", "environment": "...",
  "lighting": "the scene's light", "mood": "...",
  "action_beats": ["second by second: action, micro-expression, reaction"],
  "dialogue_beats": [{{"spoken_text": "...", "delivery": "..."}}],
  "camera_plan": {{"framing": "...", "movement": "static unless motivated", "movement_intensity": "subtle|moderate|dynamic",
                  "motivation": "why the camera moves, or empty"}},
  "audio_plan": {{"mode": "generated_audio"}}, "closing_blocking": "exact positions and poses at the last frame"}}]

{timing_rule} Go:"""
        if kwargs.get("polish_block"):
            system_prompt = f"{system_prompt}\n\n{kwargs['polish_block']}"
        user_prompt = f"Brief: {concept}\nTarget duration: {target_duration}s"

        shot_dicts = self._call_llm_json(
            user_prompt=user_prompt, system_prompt=system_prompt,
            max_tokens=max(1536, target_scenes * 520 + 768),
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
            scene_id = str(raw.get("scene_id") or "").strip()
            if scene_ids and scene_id not in scene_ids:
                scene_id = shots[-1].metadata.get("continuity_group", scene_ids[0]) if shots else scene_ids[0]
            previous_group = shots[-1].metadata.get("continuity_group") if shots else None
            same_scene = bool(scene_id) and scene_id == previous_group
            closing = str(raw.get("closing_blocking") or raw.get("ending_beat") or "")
            shots.append(ShotPlan(
                shot_id=self._make_shot_id(i, "cs"),
                index=i,
                duration_sec=duration,
                skill_type=self.skill_type,
                scene_goal=raw.get("scene_goal", f"Shot {i + 1}"),
                scene_type=raw.get("scene_type", "scene"),
                source_mode_preference="i2v" if has_reference else "t2v",
                image_strategy="reference_edit" if has_reference else "fresh_generation",
                continuity_strategy="continuous" if same_scene else "independent",
                subjects_on_screen=[SubjectRef.from_dict(s) if isinstance(s, dict) else SubjectRef(visual_description=str(s))
                                    for s in raw.get("subjects_on_screen", [])],
                spatial_setup=raw.get("opening_blocking") or raw.get("spatial_setup", ""),
                environment=raw.get("environment", ""),
                # The style contract carries the look; per-shot style text only
                # repeats it and crowds out the action in the model prompt.
                visual_style="" if narrative else raw.get("visual_style", ""),
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
                ending_beat=closing,
                metadata={
                    "skill": spec.get("id"), "skill_version": spec.get("version"), **options,
                    **({"continuity_group": scene_id} if scene_id else {}),
                    **({"closing_blocking": closing} if closing else {}),
                    **({"keep_sec": keep} if (keep := normalize_keep(raw.get("keep_sec"), duration)) else {}),
                },
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
            continuity_notes=[f"Custom skill {spec.get('name')} v{spec.get('version')}"]
            + [f"Scene {scene['scene_id']}: {scene.get('place')}, {scene.get('time_of_day')}, {scene.get('light')}"
               for scene in scenes],
        )

    def _design_scenes(self, concept: str, rules: str, grammar: str, characters: list[dict]) -> list[dict]:
        """First pass: place, light, geography and beats of each scene."""
        cast = "\n".join(
            f"- {c.get('name')}: {c.get('description', '')}" for c in characters if isinstance(c, dict)
        )
        prompt = "\n\n".join(filter(None, [f"Brief: {concept}", rules, grammar, f"Cast:\n{cast}" if cast else ""]))
        try:
            raw = self._call_llm_json(user_prompt=prompt, system_prompt=SCENE_DESIGN_SYSTEM,
                                      max_tokens=2048, temperature=0.5)
        except Exception as error:  # noqa: BLE001 - shots can still be planned without it
            print(f"[CustomPlanner] Scene design failed, planning shots directly: {error}")
            return []
        scenes, seen = [], set()
        for item in raw or []:
            if not isinstance(item, dict):
                continue
            scene_id = re.sub(r"[^a-z0-9_]+", "_", str(item.get("scene_id") or "").lower()).strip("_")
            scene_id = scene_id or f"scene_{len(scenes) + 1}"
            if scene_id in seen:
                continue
            seen.add(scene_id)
            scenes.append({
                "scene_id": scene_id,
                **{key: str(item.get(key) or "")[:400] for key in ("place", "time_of_day", "light", "geography")},
                "props": [str(p)[:160] for p in (item.get("props") or [])][:6],
                "beats": [str(b)[:200] for b in (item.get("beats") or [])][:8],
            })
        return scenes[:4]
