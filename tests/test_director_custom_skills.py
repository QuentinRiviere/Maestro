"""Declarative Director skills: validation, storage, planner and creator."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director import custom_skills, skill_creator  # noqa: E402
from services.director.planners.custom import CustomSkillPlanner, skill_rules_block  # noqa: E402

SPEC = {
    "name": "Recipe Reel", "description": "Fast overhead cooking reels.", "icon": "heart", "workflow": "story",
    "default_duration": 20, "shot_duration": [2, 4], "max_shots": 6, "style_hint": "bright top-down food look",
    "rules": {"goal": "Make the dish look effortless.", "structure": "Ingredients, steps, reveal.",
              "camera": "Overhead, locked off.", "do": ["Show hands only"], "dont": ["No faces"]},
    "options": [{"key": "Cuisine", "label": "Cuisine", "values": ["italian", "thai"], "default": "spanish"}],
    "references": [{"kind": "video", "name": "ref.mp4", "notes": "Cuts every 2 s on the beat."}],
}


class SpecTests(unittest.TestCase):
    def test_validation_bounds_and_normalises(self):
        spec = custom_skills.validate_spec({**SPEC, "shot_duration": [9, 2], "icon": "nope", "evil": "code()",
                                            "max_shots": 999})
        self.assertEqual(spec["id"], "recipe_reel")
        self.assertEqual(spec["icon"], "sparkles")
        self.assertEqual(spec["shot_duration"], [9, 9])
        self.assertEqual(spec["max_shots"], 60)
        self.assertNotIn("evil", spec)
        self.assertEqual(spec["options"][0], {"key": "cuisine", "label": "Cuisine", "values": ["italian", "thai"],
                                              "default": "italian"})

    def test_rejections(self):
        for raw in ([], {"rules": {"goal": "x"}}, {"name": "x", "rules": {}}):
            with self.assertRaises(custom_skills.SkillError):
                custom_skills.validate_spec(raw)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        self.original = custom_skills.SKILLS_DIR
        custom_skills.SKILLS_DIR = self.dir
        self.addCleanup(setattr, custom_skills, "SKILLS_DIR", self.original)

    def test_save_versions_catalog_and_delete(self):
        first = custom_skills.save_custom(SPEC)
        second = custom_skills.save_custom(SPEC)  # same name → new id
        self.assertEqual((first["id"], second["id"]), ("recipe_reel", "recipe_reel_2"))
        edited = custom_skills.save_custom({**SPEC, "description": "v2"}, skill_id="recipe_reel")
        self.assertEqual((edited["version"], edited["created_at"]), (2, first["created_at"]))
        ids = [entry["id"] for entry in custom_skills.catalog()]
        self.assertEqual(ids[:3], ["music_video", "short_film", "viral_video"])
        self.assertIn("custom:recipe_reel", ids)
        self.assertEqual(custom_skills.get_custom("custom:recipe_reel")["description"], "v2")
        self.assertTrue(custom_skills.delete_custom("recipe_reel"))
        self.assertIsNone(custom_skills.get_custom("recipe_reel"))
        with self.assertRaises(custom_skills.SkillError):
            custom_skills.delete_custom("../etc")

    def test_builtin_inspiration(self):
        base = custom_skills.inspiration("viral_video")
        self.assertTrue(base["builtin"])
        self.assertIn("Hook", base["rules"]["structure"])


class FakePlanner(CustomSkillPlanner):
    def __init__(self, spec, answer):
        super().__init__(spec)
        self.answer, self.prompts = answer, []

    def _call_llm_json(self, **kwargs):
        self.prompts.append(kwargs)
        return list(self.answer)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.spec = {**custom_skills.validate_spec(SPEC), "version": 3}

    def test_rules_options_and_duration_bounds(self):
        planner = FakePlanner(self.spec, [{"scene_goal": "a", "duration_sec": 10}, {"scene_goal": "b", "duration_sec": 1}])
        plan = planner.plan(scene_description="Carbonara in 20 seconds", skill_options={"cuisine": "thai"})
        prompt = planner.prompts[0]["system_prompt"]
        self.assertIn("Never: No faces", prompt)
        self.assertIn("Cuisine: thai", prompt)
        self.assertIn("Cuts every 2 s on the beat.", prompt)
        self.assertIn("Carbonara", planner.prompts[0]["user_prompt"])
        self.assertEqual([s.duration_sec for s in plan.shots], [4.0, 2.0])
        self.assertEqual(plan.skill_type, "custom:recipe_reel")
        self.assertEqual(plan.shots[0].metadata["skill_version"], 3)

    def test_audio_timeline_wins(self):
        plan = FakePlanner(self.spec, [{"scene_goal": "only"}]).plan(
            concept="x", clips=[{"duration_sec": 1.5}, {"start": 1.5, "end": 5}])
        self.assertEqual([s.duration_sec for s in plan.shots], [1.5, 3.5])
        self.assertEqual(plan.shots[0].audio_plan.mode, "audio_driven")

    def test_block_without_options(self):
        self.assertIn("SKILL: Recipe Reel", skill_rules_block(self.spec, {}))


class CreatorTests(unittest.TestCase):
    def test_draft_validates_and_keeps_reference_notes(self):
        answers = iter(["not json", json.dumps(SPEC)])
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return next(answers)
        analyses = [{"kind": "video", "name": "a.mp4", "notes": "Whip pans between steps."},
                    {"kind": "audio", "name": "b.wav", "error": "Reference file not found"}]
        spec = skill_creator.draft_skill("Cooking reels", generate=generate,
                                         base=custom_skills.inspiration("viral_video"), analyses=analyses)
        self.assertEqual(len(calls), 2)
        self.assertIn("Whip pans", calls[0]["prompt"])
        self.assertIn("Base skill for inspiration", calls[0]["prompt"])
        self.assertEqual(spec["references"], [{"kind": "video", "name": "a.mp4", "notes": "Whip pans between steps."}])
        self.assertEqual(spec["based_on"], "viral_video")

    def test_editing_keeps_the_skill_id(self):
        spec = skill_creator.draft_skill("shorter shots", generate=lambda **k: json.dumps({**SPEC, "name": "Other"}),
                                         base={"id": "recipe_reel", "name": "Recipe Reel"}, editing=True)
        self.assertEqual(spec["id"], "recipe_reel")

    def test_reference_kinds_and_analysis_errors(self):
        self.assertEqual([skill_creator.reference_kind(p) for p in ("a.PNG", "b.mp3", "c.webm", "d.txt")],
                         ["image", "audio", "video", None])
        analyses = skill_creator.analyze_references(["/missing.mp4", "/x.txt"], fal_key="k", publish=lambda p: p,
                                                    ask=lambda *a: ("unused", 0))
        self.assertEqual([a.get("error") for a in analyses], ["Reference file not found", "Unsupported reference type"])


if __name__ == "__main__":
    unittest.main()


def test_custom_planner_writes_shots_the_video_model_renders_alone():
    from services.director.custom_skills import validate_spec
    from services.director.planners.custom import CustomSkillPlanner, fit_shot_range, model_shot_bounds

    bounds = model_shot_bounds({"fps": 24, "frames_minimum": 124, "frames_maximum": 192})
    assert fit_shot_range(2, 5, bounds) == (5.2, 5.2)
    assert fit_shot_range(2, 12, bounds) == (5.2, 8.0)
    assert fit_shot_range(2, 5, None) == (2.0, 5.0)

    seen = {}

    def generate(**call):
        seen["system"] = call.get("system_prompt", "")
        return json.dumps([{"scene_goal": "a", "duration_sec": 2}, {"scene_goal": "b", "duration_sec": 30}])

    spec = validate_spec({"name": "Drama", "shot_duration": [2, 5], "rules": {"goal": "tension"}})
    plan = CustomSkillPlanner(spec, llm_generate=generate).plan(
        concept="a meeting", target_duration=12, fps=24, frames_minimum=124, frames_maximum=192)
    assert [shot.duration_sec for shot in plan.shots] == [5.2, 5.2]
    assert "between 5.2 and 5.2 seconds" in seen["system"]
