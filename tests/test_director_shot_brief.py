"""A custom-skill shot carries a full brief to the video model: map, acting, light, sound and locks."""
from __future__ import annotations

import json
import os
import sys

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director.custom_skills import validate_spec  # noqa: E402
from services.director.planners.custom import CustomSkillPlanner  # noqa: E402
from services.director.renderers.ltx_i2v import LtxI2VRenderer  # noqa: E402
from services.director.renderers.ltx_t2v import LtxT2VRenderer  # noqa: E402

SCENE = {
    "scene_id": "hallway", "place": "school corridor", "time_of_day": "morning", "light": "overcast daylight",
    "geography": "she enters screen left, he waits screen right",
    "scene_map": "left: tall windows over pine trees; center: polished floor running to a wooden door; "
                 "right: blue lockers; the door at the back opens inward",
    "sound": "quiet corridor room tone, distant classroom murmur",
}
SHOT = {
    "scene_id": "hallway", "scene_goal": "he stops her", "duration_sec": 6,
    "subjects_on_screen": [{"visual_description": "the teen girl", "position_or_relation": "screen left"}],
    "opening_blocking": "she walks screen left to right", "lighting": "overcast daylight",
    "action_beats": ["0.0-2.0s: she walks past", "2.0-4.0s: his hand closes on her wrist"],
    "performance": "she keeps her eyes down; surprise shows only in a held breath; she never looks at the camera",
    "locks": ["only two hands in frame", "the books stay in her left arm"],
    "audio_plan": {"mode": "generated_audio", "effects": ["2.0s: a sharp intake of breath"]},
    "camera_plan": {"framing": "medium two-shot", "movement": "static"},
    "closing_blocking": "his hand holds her wrist between them",
}


def _plan():
    def generate(**call):
        if "design the scenes" in call.get("system_prompt", ""):
            return json.dumps([SCENE])
        return json.dumps([SHOT])

    spec = validate_spec({"name": "K-drama", "workflow": "story", "shot_duration": [4, 8], "rules": {"goal": "first love"}})
    return CustomSkillPlanner(spec, llm_generate=generate).plan(concept="a first meeting", target_duration=6)


def test_the_planner_keeps_acting_locks_sound_and_the_scene_map():
    shot = _plan().shots[0]
    assert shot.performance_beats == [SHOT["performance"]]
    assert shot.constraints == SHOT["locks"]
    # The shot gave no room tone: the scene's sound fills it.
    assert shot.audio_plan.ambience == SCENE["sound"]
    assert shot.audio_plan.effects == ["2.0s: a sharp intake of breath"]
    assert shot.metadata["scene_map"] == SCENE["scene_map"]


def test_the_video_prompt_is_a_brief_in_fixed_sections():
    shot = _plan().shots[0]
    prompt = LtxT2VRenderer().render(shot)
    labels = [part.split(":", 1)[0] for part in prompt.split(" | ")]
    assert labels == ["SHOT", "SETTING", "SCENE MAP", "SUBJECTS", "ACTION", "PERFORMANCE", "CAMERA", "LIGHT",
                      "AUDIO", "LOCKS", "ENDING"]
    assert "AUDIO: quiet corridor room tone, distant classroom murmur; 2.0s: a sharp intake of breath" in prompt
    assert ("LOCKS: only two hands in frame; the books stay in her left arm; "
            "wardrobe, hair and props stay exactly as in the first frame") in prompt
    assert "LIGHT: constant for the whole shot" in prompt


def test_an_image_to_video_prompt_keeps_the_locks_but_not_the_map():
    prompt = LtxI2VRenderer().render(_plan().shots[0])
    assert "LOCKS: only two hands in frame" in prompt
    assert "SCENE MAP" not in prompt


def test_the_brief_survives_the_h3_compiler():
    from services.director.h3_dialogue import compile_h3_official_prompt, validate_h3_prompt_contract

    shot = _plan().shots[0]
    body = LtxT2VRenderer().render(shot)
    prompt, _ = compile_h3_official_prompt(
        body, [], [], mode="i2va", duration_seconds=shot.duration_sec,
        audio_plan={"ambience": shot.audio_plan.ambience, "effects": shot.audio_plan.effects},
    )
    validate_h3_prompt_contract(prompt, mode="i2va")
    for section in ("SCENE MAP:", "PERFORMANCE:", "LIGHT:", "LOCKS:"):
        assert section in prompt
    assert "overall_soundscape: quiet corridor room tone" in prompt


def test_the_acting_is_written_once():
    prompt = LtxT2VRenderer().render(_plan().shots[0])
    assert prompt.count("she never looks at the camera") == 1
