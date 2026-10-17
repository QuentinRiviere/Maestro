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
from ..blocking_map import parse_blocking_map
from ..custom_skills import DIALOGUE_DENSITIES
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


# Shot lengths when H3 sequences film a scene's shots back to back, and the
# most shots such a plan may hold (shorter shots, so more of them).
SEQUENCE_SHOT_RANGE = (1.5, 5.0)
SEQUENCE_MAX_SHOTS = 24

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
- Action is a visible transition: the shot's decisive gesture is NOT done yet at the first frame (a hand that
  will grab a wrist starts on the table; a nod starts with the head still) and IS done by the last frame, so the
  render shows the change itself; the action prompt names the start pose, the one move and the end pose, and
  that move falls inside keep_sec.
- Hand contact is a short close, not a reach: a hand that takes a wrist, a cup or another hand starts already
  beside it (fingertips a few centimetres away) and closes that gap slowly, then tightens; never a reach across
  the frame, never two hands crossing (short clips morph the fingers on a long reach). On an insert the camera
  holds still and nothing else moves.
- Action is performance, second by second: a physical intention, the visible expression change it causes, the
  reaction. No filler motion (aimless walking, hair touching, looking around) to fill time.
- Filmable action only: an action beat describes what a viewer can see at the shot's framing in a short, low
  resolution render: a head turn, a posture change, a hand action, a step, a visible expression change (a smile
  fading, a frown, eyes closing, a look away). Never micro-details the frame cannot show, e.g. pupils dilating,
  lips parting a millimeter, a throat swallowing, a pulse in the neck, a single tear forming, eyelashes
  trembling, a jaw muscle twitching.
- Reaction shots are at most about one third of the shots; the scene moves forward through action and dialogue,
  not through a chain of silent looks.
- Camera is static (locked-off) by default. Move only when the story motivates it (a slow push-in on a
  realization, a follow when a character crosses the space) and say why in camera_plan.motivation.
- Framing follows the coverage: establishing or master to place people, over-the-shoulder shot/reverse-shot for
  exchanges, inserts for decisive details (a hand, an object), close-ups for reactions.
- Body-part inserts of gross motion are strong coverage: feet walking in and stopping, legs entering the frame on
  an arrival, a back walking away, a resting hand that tightens once, a door opening. Use them for movement,
  arrivals, departures and expressive details; they also hide the join between two shots.
- Hands: at most ONE simple, decisive hand action per shot, done in one move and then held still (a hand grabs a
  wrist and holds it; a hand sets a cup down and rests on the table). Keep hands out of sustained manipulation:
  no finger-level business, no pouring or liquid in motion, no object passed from hand to hand, no interlocked
  fingers moving, no continuous shaping, kneading or wiping. Show such a moment through its result or the faces.
- Props: every object the action uses is already visible at the first frame; name it with its place in
  opening_blocking and in props. Nothing appears, materializes or is pulled into frame mid-shot.
- Prop state: track each prop through the consecutive shots of a scene: where it is, who holds it, and its
  condition (raw clay stays raw until a shot shows it shaped; a tool roll left on the bench stays on the bench).
  A shot's props open exactly as the previous shot of the scene left them.
- Do not describe style, grade, lens flare or "aesthetic" in shots: the visual style is set once for the whole
  production. Shot fields describe what the camera sees and what people do.
- Dialogue: every line has a speaker, the exact cast name of the character who says it. At most one line per
  shot, short enough to say inside keep_sec (about 2.5 words per second), then
  explicit silence (breath, stillness) until the end of the shot; never a line the model must repeat to fill time.
- Write every field in English, whatever the language of the skill; only spoken_text uses the dialogue
  language the skill or brief asks for.
- keep_sec: the [start, end] seconds of the rendered shot the final edit keeps. Reactions and inserts keep
  1.5-3 s, exchanges 2-4 s, establishing shots 3-5 s; any dialogue and the decisive action fall inside it."""

# The renderer starts a continuing shot from the true last frame of the shot
# before it, so the flag must stay rare: a chained shot cannot reframe.
CONTINUES_PREVIOUS_RULE = """continues_previous: true ONLY when this shot is the same camera setup (same framing, same angle,
same camera movement) carrying on the previous shot's action past what one render can hold (a long run, a long
walk, one unbroken gesture). It is rendered from the previous shot's last frame, so it opens exactly there and
cannot reframe. false for every cut: shot/reverse-shot, a new angle, a close-up after a wide, an insert, a
reaction, and always false for the first shot of a scene. When true, the previous shot's closing_blocking is this
shot's opening_blocking."""

HAND_DETAIL_RULE = """hand_detail: true ONLY when the story beat cannot be told without a delicate hand/object interaction in
this shot (a ring slipped onto a finger, a folded note opened); such a shot gets a slower, finer render. false for
every other shot, including body-part inserts of gross motion (feet, legs, a hand that tightens once)."""


DIALOGUE_DENSITY_RULES = {
    "low": "sparse dialogue: lines only at the scene's turning points, about one line every 4-5 shots; looks and "
           "actions carry the rest.",
    "medium": "in exchanges, about one line every 2-3 shots; the story is told through what people say as much "
              "as through looks.",
    "high": "dialogue-driven: in exchanges about two shots out of three carry a line; silent shots are the "
            "exception.",
}


def dialogue_density(spec: dict, options: dict[str, str]) -> str:
    """The skill's dialogue density: a run option wins over the skill rule, default medium."""
    for value in (options.get("dialogue_density"), (spec.get("rules") or {}).get("dialogue_density")):
        value = str(value or "").strip().lower()
        if value in DIALOGUE_DENSITIES:
            return value
    return "medium"


def dialogue_density_rule(level: str) -> str:
    return (f"DIALOGUE DENSITY ({level}): {DIALOGUE_DENSITY_RULES[level]} Keep the one-line-per-shot and "
            "words-per-second limits; give a talking scene more shots rather than longer lines.")


BLOCKING_MAP_RULE = """blocking_map: a top view of the scene floor at the first frame, consistent with the scene geography and
scene_map. x goes from 0 (the left side as the scene's main camera sees it) to 1 (the right side), y from 0 (the
back wall) to 1 (the front, where the main camera stands). facing_deg: 0 faces the back wall, 90 the right side,
180 the front, 270 the left side. Place the camera where this shot is taken from: close to the subject for a
close-up, far for a wide. A person keeps their spot from shot to shot of a scene unless the action moves them;
the camera moves around them. "subjects" names each person in frame by their cast name; add "moves": true to a
person the action has moved since the scene's previous shot. "props" places the main furniture and set pieces,
the same spots in every shot of the scene. The camera stays on one side of the line between the two main
characters for the whole scene; add "crosses_line": true to the camera only when this shot deliberately crosses it."""

# Planners asked for "the scene's light" write even, balanced light ("soft
# amber glow balanced by cool twilight, natural skin tones") even at night,
# and the image comes out flat like a catalogue photo.
LIGHTING_RULE = """LIGHTING (live-action / photographic looks): the light is motivated and contrasted, consistent with the
scene's time of day. Name the visible light sources (practicals: lamps, pendants, windows, neon, streetlights,
screens) and where the shadows fall on the faces. Night and interior drama scenes are low-key: most of the frame
dark, deep shadows, faces partly in shadow. Never "balanced", "even", "soft natural lighting" or "natural skin
tones" as the whole description."""

SCENE_DESIGN_SYSTEM = """You are the director preparing the scenes of a short production before shooting.
From the brief, the skill rules and the cast, design the scenes. Return ONLY a JSON array, one object per scene:
[{"scene_id": "short_snake_case", "location": "the cast location this scene is shot in, by its exact name",
  "place": "the exact location", "time_of_day": "...",
  "light": "one lighting description kept for every shot of this scene",
  "geography": "the layout of the space: where each character stands or moves, which screen side each one keeps",
  "scene_map": "the space as the main camera sees it: what is on the left, in the center, on the right and in the
    background; doors (hinge side, opening inward or outward), windows, stairs and the light sources",
  "sound": "the room tone and recurring sounds of this place",
  "props": ["objects that matter, which hand or place holds them, and their condition"],
  "beats": ["the dramatic beats of this scene in order, each one action and its reaction"]}]
Use as few scenes as the story needs (usually one or two for a short piece). Time and light only change
between scenes, never inside one. When cast locations are listed, every scene takes place in one of them (its
reference image is the set) and "place" describes that same location; use "" for location only if the story
truly needs a place the cast lacks. Write in English.

""" + LIGHTING_RULE


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
        # H3 sequences render several shots in one generation at their exact
        # length, so a shot is planned at its final screen time instead of
        # being rendered long and cut short.
        sequence_shots = bool(kwargs.get("sequence_shots")) and not clips
        if sequence_shots:
            low, high = SEQUENCE_SHOT_RANGE
        else:
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
            max_shots = int(spec.get("max_shots") or 12)
            if sequence_shots:
                # Shorter shots: as many as the target length needs, within reason.
                max_shots = max(max_shots, min(SEQUENCE_MAX_SHOTS, math.ceil(target_duration / average)))
            target_scenes = max(1, min(max_shots, round(target_duration / average)))
            timing_rule = (f"Output exactly {target_scenes} shot plans totaling about {target_duration}s, "
                           f"each between {low:g} and {high:g} seconds. Every shot is rendered on its own, so give "
                           "each one enough action to fill its full duration.")
            if sequence_shots:
                timing_rule = (f"Output exactly {target_scenes} shot plans totaling about {target_duration}s, "
                               f"each between {low:g} and {high:g} seconds. duration_sec is the shot's final screen "
                               "time: shots of one scene are filmed back to back with hard cuts and nothing is "
                               "trimmed, so keep_sec is always [0, duration_sec]. Reactions and inserts last "
                               "1.5-3 s, exchanges 2-4 s, establishing shots 3-5 s.")

        char_profiles = [
            CharacterProfile(id=f"char_{i}", display_name=c.get("name", ""),
                             physical_description=c.get("description", "person"),
                             wardrobe=c.get("wardrobe") or None)
            for i, c in enumerate(characters or [])
        ]
        narrative = spec.get("workflow", "story") == "story" and not clip_durations
        rules = skill_rules_block(spec, options)
        grammar = scene_grammar_block(spec)
        locations = [loc for loc in kwargs.get("locations") or [] if isinstance(loc, dict) and loc.get("name")]
        scenes = self._design_scenes(concept, rules, grammar, characters or [], locations) if narrative else []
        scene_location = {scene["scene_id"]: scene.get("location", "") for scene in scenes}
        scene_by_id = {scene["scene_id"]: scene for scene in scenes}
        scene_ids = [scene["scene_id"] for scene in scenes]
        scene_block = (
            "SCENE DESIGN (fixed; every shot uses one scene_id and keeps its place, light and geography):\n"
            + json.dumps(scenes, ensure_ascii=False, indent=1)
        ) if scenes else ""
        # A top view per shot: the storyboard draws it so the image model
        # takes the framing from geometry instead of the scene's master image.
        blocking_field = (',\n  "blocking_map": {"camera": {"x": 0.5, "y": 0.95, "facing_deg": 0, "framing": "wide|medium|close-up"},'
                          '\n                   "subjects": [{"name": "...", "x": 0.4, "y": 0.4, "facing_deg": 90}],'
                          '\n                   "props": [{"name": "table", "x": 0.5, "y": 0.4}]}'
                          ) if narrative else ""
        blocking_rule = BLOCKING_MAP_RULE + "\n\n" if narrative else ""
        density_rule = dialogue_density_rule(dialogue_density(spec, options)) if narrative else ""
        cast_names = [str(c.get("name") or "").strip() for c in characters or []
                      if isinstance(c, dict) and str(c.get("name") or "").strip()]
        cast_rule = ("CAST NAMES (use them exactly for subjects_on_screen.name and the dialogue speaker): "
                     + ", ".join(cast_names)) if cast_names else ""
        system_prompt = f"""You are the director of a production written with a specific skill.

{rules}

{grammar}

{"You are given a REFERENCE PHOTO; keep its subjects and look." if has_reference else ""}
{build_character_rules_block(has_reference, char_profiles or None)}

{SCENE_CRAFT if narrative else build_camera_style_block()}

{LIGHTING_RULE}

{density_rule}

{cast_rule}

{scene_block}

OUTPUT FORMAT — respond with ONLY a JSON array of shot objects:
[{{"scene_id": "{scene_ids[0] if scene_ids else 'scene_1'}", "scene_goal": "...", "scene_type": "establishing|exchange|insert|reaction|action",
  "duration_sec": {low}, "keep_sec": [0, {low}],
  "subjects_on_screen": [{{"name": "the cast name of this person, or empty", "visual_description": "...",
                          "position_or_relation": "screen side, distance, facing"}}],
  "opening_blocking": "exact positions and poses at the first frame", "environment": "...",
  "props": ["each object the action uses: what it is, where it is at the first frame, its condition"],
  "lighting": "the scene's light", "mood": "...",
  "action_beats": ["timed, e.g. '0.0-2.0s: ...': visible action, visible expression change, reaction"],
  "performance": "the acting task: the intention, where the eyes go, what stays still, and what the actor never does",
  "locks": ["a fact this shot must hold, in positive form, e.g. 'only two hands in frame, both hers'",
            "'the books stay in her left arm'"],
  "dialogue_beats": [{{"speaker": "exact cast name of who says it", "spoken_text": "...", "delivery": "..."}}],
  "camera_plan": {{"framing": "...", "movement": "static unless motivated", "movement_intensity": "subtle|moderate|dynamic",
                  "motivation": "why the camera moves, or empty"}},
  "audio_plan": {{"mode": "generated_audio", "ambience": "the room tone of this place at this moment",
                 "effects": ["timed sounds, e.g. '2.0s: the books thud on the floor'"]}},
  "closing_blocking": "exact positions and poses at the last frame",
  "continues_previous": false, "hand_detail": false{blocking_field}}}]

{CONTINUES_PREVIOUS_RULE}

{HAND_DETAIL_RULE}

{blocking_rule}{timing_rule} Go:"""
        if kwargs.get("polish_block"):
            system_prompt = f"{system_prompt}\n\n{kwargs['polish_block']}"
        user_prompt = f"Brief: {concept}\nTarget duration: {target_duration}s"

        shot_dicts = self._call_llm_json(
            user_prompt=user_prompt, system_prompt=system_prompt,
            max_tokens=max(2048, target_scenes * (1000 if narrative else 900) + 1024),
            image_paths=[reference_image_path] if has_reference else None,
        )
        # A truncated answer can end on an empty shell; never render one.
        shot_dicts = [raw for raw in shot_dicts or [] if isinstance(raw, dict)
                      and any(raw.get(key) for key in ("action_beats", "scene_goal", "dialogue_beats",
                                                       "opening_blocking", "closing_blocking", "subjects_on_screen"))]
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
            # Same camera carrying the previous shot's action on: render it
            # from that shot's last frame. Ordinary same-scene cuts stay
            # independent renders with their own start image.
            continues_previous = same_scene and str(raw.get("continues_previous")).strip().lower() == "true"
            closing = str(raw.get("closing_blocking") or raw.get("ending_beat") or "")
            scene = scene_by_id.get(scene_id) or {}
            performance = raw.get("performance")
            locks = [str(lock).strip() for lock in raw.get("locks") or [] if str(lock).strip()]
            effects = [str(e).strip() for e in audio_raw.get("effects") or [] if str(e).strip()]
            subjects = _subjects(raw.get("subjects_on_screen"), cast_names)
            props = [str(p).strip()[:160] for p in (raw.get("props") if isinstance(raw.get("props"), list) else [])
                     if str(p).strip()][:6]
            hand_detail = raw.get("hand_detail") is True or str(raw.get("hand_detail")).strip().lower() == "true"
            shots.append(ShotPlan(
                shot_id=self._make_shot_id(i, "cs"),
                index=i,
                duration_sec=duration,
                skill_type=self.skill_type,
                scene_goal=raw.get("scene_goal", f"Shot {i + 1}"),
                scene_type=raw.get("scene_type", "scene"),
                source_mode_preference="i2v" if has_reference else "t2v",
                image_strategy="reference_edit" if has_reference else "fresh_generation",
                continuity_strategy=(
                    "extend_previous" if continues_previous else "continuous" if same_scene else "independent"
                ),
                subjects_on_screen=subjects,
                spatial_setup=raw.get("opening_blocking") or raw.get("spatial_setup", ""),
                environment=raw.get("environment", ""),
                # The style contract carries the look; per-shot style text only
                # repeats it and crowds out the action in the model prompt.
                visual_style="" if narrative else raw.get("visual_style", ""),
                lighting=raw.get("lighting", ""),
                mood=raw.get("mood", ""),
                action_beats=raw.get("action_beats", []),
                performance_beats=[str(performance).strip()] if isinstance(performance, str) and performance.strip()
                else None,
                constraints=locks or None,
                dialogue_beats=_dialogue_beats(raw.get("dialogue_beats"), subjects, cast_names, i) or None,
                camera_plan=CameraPlan(framing=cam.get("framing", "medium shot"), angle=cam.get("angle"),
                                       movement=cam.get("movement"),
                                       movement_intensity=cam.get("movement_intensity", "moderate")),
                audio_plan=AudioPlan(mode="audio_driven" if clip_durations else audio_raw.get("mode", "generated_audio"),
                                     ambience=audio_raw.get("ambience") or scene.get("sound") or None,
                                     effects=effects or None,
                                     timing_anchor="audio" if clip_durations else "video"),
                ending_beat=closing,
                metadata={
                    "skill": spec.get("id"), "skill_version": spec.get("version"), **options,
                    **({"continuity_group": scene_id} if scene_id else {}),
                    **({"location": scene_location[scene_id]} if scene_location.get(scene_id) else {}),
                    **({"scene_map": str(scene["scene_map"])} if scene.get("scene_map") else {}),
                    **({"closing_blocking": closing} if closing else {}),
                    **({"props": props} if props else {}),
                    **({"hand_detail": True} if hand_detail else {}),
                    **({"blocking_map": blocking} if (blocking := parse_blocking_map(
                        raw.get("blocking_map"), cam.get("framing"))) else {}),
                    **({"keep_sec": keep} if not sequence_shots
                       and (keep := normalize_keep(raw.get("keep_sec"), duration)) else {}),
                },
            ))
            if continues_previous:
                # The handoff frame is the previous shot's last frame, so the
                # edit keeps that shot to its end and this one from its start.
                for shot, edge in ((shots[-2], 1), (shots[-1], 0)):
                    keep = shot.metadata.pop("keep_sec", None)
                    if keep:
                        keep = list(keep)
                        keep[edge] = shot.duration_sec if edge else 0.0
                        if (keep := normalize_keep(keep, shot.duration_sec)):
                            shot.metadata["keep_sec"] = keep
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

    def _design_scenes(self, concept: str, rules: str, grammar: str, characters: list[dict],
                       locations: list[dict] | None = None) -> list[dict]:
        """First pass: place, light, geography and beats of each scene."""
        cast = "\n".join(
            f"- {c.get('name')}: {c.get('description', '')}" for c in characters if isinstance(c, dict)
        )
        sets = "\n".join(f"- {loc['name']}: {loc.get('description', '')}" for loc in locations or [])
        location_names = {str(loc["name"]) for loc in locations or []}
        style = str(self.spec.get("style_hint") or "").strip()
        prompt = "\n\n".join(filter(None, [
            f"Brief: {concept}", rules, grammar,
            f"Visual style (the light you choose must respect it): {style}" if style else "",
            f"Cast:\n{cast}" if cast else "",
            f"Cast locations:\n{sets}" if sets else "",
        ]))
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
            location = str(item.get("location") or "").strip()
            scenes.append({
                "scene_id": scene_id,
                "location": location if location in location_names else "",
                **{key: str(item.get(key) or "")[:400] for key in ("place", "time_of_day", "light", "geography",
                                                                    "scene_map", "sound")},
                "props": [str(p)[:160] for p in (item.get("props") or [])][:6],
                "beats": [str(b)[:200] for b in (item.get("beats") or [])][:8],
            })
        return scenes[:4]


def _cast_name(value: Any, cast_names: list[str]) -> str:
    """The cast name ``value`` refers to (case-insensitive), or ""."""
    key = str(value or "").strip().casefold()
    return next((name for name in cast_names if key and name.casefold() == key), "")


def _subjects(raw: Any, cast_names: list[str]) -> list[SubjectRef]:
    """Subjects in frame; a cast member carries the planner's char id and cast name."""
    subjects = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            subjects.append(SubjectRef(visual_description=str(item)))
            continue
        name = _cast_name(item.get("name") or item.get("speaker_name"), cast_names)
        if not name:
            # A subject without a usable name is still a cast member when its
            # description names exactly one of them.
            text = str(item.get("visual_description") or "")
            named = [n for n in cast_names if re.search(rf"\b{re.escape(n)}\b", text, re.IGNORECASE)]
            name = named[0] if len(named) == 1 else ""
        subject = SubjectRef.from_dict({**item, "visual_description": str(item.get("visual_description") or name)})
        if name:
            subject.speaker_name = name
            subject.character_id = subject.character_id or f"char_{cast_names.index(name)}"
        subjects.append(subject)
    return subjects


def _dialogue_beats(raw: Any, subjects: list[SubjectRef], cast_names: list[str], index: int) -> list[DialogueBeat]:
    """Dialogue beats whose ``speaker_id`` is the exact cast name of who speaks.

    An unknown or missing speaker falls back to the cast member in frame (the
    first one when several are), and the substitution is logged.
    """
    beats = []
    in_frame = [s.speaker_name for s in subjects if s.speaker_name]
    for item in raw if isinstance(raw, list) else []:
        beat = DialogueBeat.from_dict(item)
        if not beat.spoken_text:
            continue
        given = str((item.get("speaker") or item.get("speaker_id") or "") if isinstance(item, dict) else "").strip()
        if cast_names:
            speaker = _cast_name(given, cast_names)
            if not speaker and in_frame:
                speaker = in_frame[0]
                where = "alone" if len(set(in_frame)) == 1 else "first"
                print(f"[CustomPlanner] Shot {index + 1}: speaker {given or '(none)'!r} is not in the cast; "
                      f"attributed to {speaker} ({where} in frame)")
            elif not speaker:
                print(f"[CustomPlanner] Shot {index + 1}: speaker {given or '(none)'!r} is not in the cast "
                      "and no cast member is in frame")
            beat.speaker_id = speaker or given or None
        else:
            beat.speaker_id = given or None
        beats.append(beat)
    return beats
