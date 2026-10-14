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


GRAMMAR = {"coverage": "Master, then shot-reverse-shot singles, reaction close-ups.",
           "cutting_rhythm": "Reactions 2-3 s, establishing 3-4 s.", "camera_default": "Locked off; push-in on reveals.",
           "blocking": "Keep the 180-degree line.", "lighting_continuity": "One light per scene.",
           "acting": "Restraint, micro-expressions."}


class SceneGrammarTests(unittest.TestCase):
    def test_absent_scene_grammar_is_backward_compatible(self):
        spec = custom_skills.validate_spec(SPEC)
        self.assertEqual(spec["scene_grammar"], {field: "" for field in custom_skills.SCENE_GRAMMAR_FIELDS})
        self.assertEqual(custom_skills.validate_spec({**SPEC, "scene_grammar": "junk"})["scene_grammar"],
                         spec["scene_grammar"])

    def test_scene_grammar_is_bounded_and_known_fields_only(self):
        spec = custom_skills.validate_spec({**SPEC, "scene_grammar": {
            "coverage": "x" * 2000, "acting": "  Restraint  ", "evil": "code()", "blocking": None}})
        grammar = spec["scene_grammar"]
        self.assertEqual(set(grammar), set(custom_skills.SCENE_GRAMMAR_FIELDS))
        self.assertEqual(len(grammar["coverage"]), custom_skills.SCENE_GRAMMAR_LIMIT)
        self.assertEqual((grammar["acting"], grammar["blocking"]), ("Restraint", ""))


class LintTests(unittest.TestCase):
    def lint(self, rules, grammar=GRAMMAR, shot=(2, 4), **extra):
        return custom_skills.lint_spec(custom_skills.validate_spec(
            {"name": "K", "shot_duration": list(shot), "rules": rules, "scene_grammar": grammar, **extra}))

    def codes(self, warnings):
        return sorted(w["code"] for w in warnings)

    def test_clean_skill_has_no_warnings(self):
        self.assertEqual(self.lint({"goal": "End on a held close-up.", "dont": ["No freeze frames", "No crowds"]}), [])

    def test_kdrama_freeze_frame_contradiction_english(self):
        warnings = self.lint({"goal": "Culminate on a freeze frame.", "dont": ["Freeze frames"]})
        self.assertEqual(self.codes(warnings), ["contradiction", "unrenderable"])
        self.assertEqual(warnings[0]["fields"], ["rules.goal", "rules.dont[0]"])

    def test_french_contradictions_and_negations(self):
        warnings = self.lint({"structure": "Le climax se joue au ralenti dans la foule.",
                              "pacing": "Pas d'arrêt sur image, jamais.",
                              "dont": ["Pas de ralenti", "Éviter les foules", "Pas d'arrêt sur image"]})
        contradictions = [w for w in warnings if w["code"] == "contradiction"]
        self.assertEqual(len(contradictions), 2)  # slow motion and crowds; the negated freeze frame is fine
        self.assertTrue(any("slow motion" in w["message"] for w in contradictions))
        self.assertTrue(any("crowds" in w["message"] for w in contradictions))
        self.assertFalse(any("freeze frame" in w["message"] for w in warnings))

    def test_unrenderable_techniques_in_any_positive_field(self):
        warnings = self.lint({"goal": "Tension.", "do": ["Split screen for phone calls", "Texte à l'écran pour l'heure"],
                              "camera": "Speed ramps on every punch."},
                             style_hint="slow-mo glow")
        messages = " ".join(w["message"] for w in warnings if w["code"] == "unrenderable")
        for label in ("split screen", "on-screen text", "speed ramp", "slow motion"):
            self.assertIn(label, messages)

    def test_rhythm_mismatch(self):
        rhythm = {**GRAMMAR, "cutting_rhythm": "Réactions 1,5 à 3 s, plans d'ensemble 3-5 secondes."}
        # Cuts shorter than the rendered shot are the tight edit's job.
        self.assertEqual(self.lint({"goal": "x"}, grammar=rhythm, shot=(5, 8)), [])
        warnings = self.lint({"goal": "x"}, grammar=rhythm, shot=(2, 4))
        self.assertEqual(self.codes(warnings), ["rhythm_mismatch"])
        self.assertIn("up to 5 s", warnings[0]["message"])

    def test_missing_scene_grammar_is_flagged(self):
        self.assertEqual(self.codes(self.lint({"goal": "x"}, grammar=None)), ["no_scene_grammar"])
        self.assertEqual(custom_skills.lint_spec(None), [])


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
        self.assertIn("warnings", edited)
        self.assertNotIn("warnings", json.loads((self.dir / "recipe_reel.json").read_text(encoding="utf-8")))
        self.assertEqual(custom_skills.inspiration("recipe_reel")["warnings"][0]["code"], "no_scene_grammar")
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
        # Story skills first design their scenes, then plan the shots.
        if kwargs["system_prompt"].startswith("You are the director preparing the scenes"):
            return list(getattr(self, "scenes", []))
        return list(self.answer)


class PlannerTests(unittest.TestCase):
    def setUp(self):
        self.spec = {**custom_skills.validate_spec(SPEC), "version": 3}

    def test_rules_options_and_duration_bounds(self):
        planner = FakePlanner(self.spec, [{"scene_goal": "a", "duration_sec": 10}, {"scene_goal": "b", "duration_sec": 1}])
        plan = planner.plan(scene_description="Carbonara in 20 seconds", skill_options={"cuisine": "thai"})
        prompt = planner.prompts[-1]["system_prompt"]
        self.assertIn("Never: No faces", prompt)
        self.assertIn("Cuisine: thai", prompt)
        self.assertIn("Cuts every 2 s on the beat.", prompt)
        self.assertIn("Carbonara", planner.prompts[-1]["user_prompt"])
        self.assertEqual([s.duration_sec for s in plan.shots], [4.0, 2.0])
        self.assertEqual(plan.skill_type, "custom:recipe_reel")
        self.assertEqual(plan.shots[0].metadata["skill_version"], 3)

    def test_audio_timeline_wins(self):
        plan = FakePlanner(self.spec, [{"scene_goal": "only"}]).plan(
            concept="x", clips=[{"duration_sec": 1.5}, {"start": 1.5, "end": 5}])
        self.assertEqual([s.duration_sec for s in plan.shots], [1.5, 3.5])
        self.assertEqual(plan.shots[0].audio_plan.mode, "audio_driven")

    def test_story_shots_follow_the_scene_design(self):
        spec = {**self.spec, "workflow": "story", "shot_duration": [5, 8],
                "scene_grammar": {"camera_default": "Locked-off; push in only on a realization."}}
        planner = FakePlanner(spec, [
            {"scene_id": "hallway", "duration_sec": 6, "keep_sec": [0.5, 3.5], "visual_style": "lens flare",
             "opening_blocking": "She walks screen-left to right", "closing_blocking": "She stops by the window"},
            {"scene_id": "unknown", "duration_sec": 6, "keep_sec": [0, 6], "closing_blocking": "He holds her wrist"},
        ])
        planner.scenes = [{"scene_id": "Hallway", "place": "school corridor", "time_of_day": "afternoon",
                           "light": "warm side light", "geography": "she keeps screen-left, he screen-right"}]
        plan = planner.plan(scene_description="A new student meets a mysterious boy")
        shot_prompt = planner.prompts[-1]["system_prompt"]
        self.assertIn("SCENE CRAFT", shot_prompt)
        self.assertIn("she keeps screen-left", shot_prompt)
        self.assertIn("Locked-off; push in only on a realization.", shot_prompt)
        first, second = plan.shots
        self.assertEqual(first.metadata["continuity_group"], "hallway")
        self.assertEqual(second.metadata["continuity_group"], "hallway")
        self.assertEqual((first.continuity_strategy, second.continuity_strategy), ("independent", "continuous"))
        self.assertEqual(first.spatial_setup, "She walks screen-left to right")
        self.assertEqual(first.metadata["closing_blocking"], "She stops by the window")
        self.assertEqual(first.metadata["keep_sec"], (0.5, 3.5))
        self.assertNotIn("keep_sec", second.metadata)  # the whole shot is kept
        self.assertEqual(first.visual_style, "")

    def test_story_shots_carry_a_top_view_blocking_map(self):
        spec = {**self.spec, "workflow": "story", "shot_duration": [5, 8]}
        planner = FakePlanner(spec, [
            {"scene_id": "hallway", "scene_goal": "a", "camera_plan": {"framing": "medium close-up"},
             "blocking_map": {"camera": {"x": 0.5, "y": 0.9, "facing_deg": 0},
                              "subjects": [{"name": "Mina", "x": 0.4, "y": 0.4, "facing_deg": 180}]}},
            {"scene_id": "hallway", "scene_goal": "b", "blocking_map": {"camera": {"x": "far"}}},
        ])
        planner.scenes = [{"scene_id": "hallway", "place": "school corridor"}]
        plan = planner.plan(scene_description="A new student meets a mysterious boy")
        self.assertIn("blocking_map: a top view of the scene floor", planner.prompts[-1]["system_prompt"])
        self.assertIn('"blocking_map": {"camera"', planner.prompts[-1]["system_prompt"])
        first, second = plan.shots
        self.assertEqual(first.metadata["blocking_map"], {
            "camera": {"x": 0.5, "y": 0.9, "facing_deg": 0.0, "framing": "close-up"},
            "subjects": [{"name": "Mina", "x": 0.4, "y": 0.4, "facing_deg": 180.0}]})
        self.assertNotIn("blocking_map", second.metadata)

    def test_block_without_options(self):
        self.assertIn("SKILL: Recipe Reel", skill_rules_block(self.spec, {}))


class CreatorTests(unittest.TestCase):
    def test_draft_validates_and_keeps_reference_notes(self):
        answers = iter(["not json", json.dumps({**SPEC, "scene_grammar": GRAMMAR})])
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

    def test_draft_prompt_asks_for_scene_grammar_and_clean_draft_skips_repair(self):
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return json.dumps({**SPEC, "scene_grammar": GRAMMAR})
        spec = skill_creator.draft_skill("K-drama episodes", generate=generate)
        self.assertEqual(len(calls), 1)
        self.assertIn("scene_grammar", calls[0]["system_prompt"])
        self.assertIn("never aesthetic adjectives", calls[0]["system_prompt"])
        self.assertEqual(spec["scene_grammar"]["coverage"], GRAMMAR["coverage"])
        self.assertEqual(spec["warnings"], [])

    def test_contradictory_draft_gets_one_repair_pass(self):
        broken = {**SPEC, "icon": "film", "scene_grammar": GRAMMAR,
                  "rules": {"goal": "Culminer sur un arrêt sur image.", "dont": ["Pas d'arrêt sur image"]}}
        fixed = {**broken, "rules": {"goal": "Culminer sur un gros plan tenu.", "dont": ["Pas d'arrêt sur image"]}}
        answers = iter([json.dumps(broken), json.dumps(fixed)])
        calls = []

        def generate(**kwargs):
            calls.append(kwargs)
            return next(answers)
        spec = skill_creator.draft_skill("K-drama", generate=generate,
                                         analyses=[{"kind": "video", "name": "a.mp4", "notes": "Static two-shots."}])
        self.assertEqual(len(calls), 2)
        self.assertIs(calls[1]["system_prompt"], skill_creator.REPAIR_SYSTEM)
        self.assertIn("forbids it", calls[1]["prompt"])
        self.assertEqual(spec["rules"]["goal"], "Culminer sur un gros plan tenu.")
        self.assertEqual((spec["warnings"], spec["icon"]), ([], "film"))
        self.assertEqual(spec["references"][0]["notes"], "Static two-shots.")

    def test_failed_repair_keeps_the_draft_and_its_warnings(self):
        broken = {**SPEC, "scene_grammar": GRAMMAR, "rules": {"goal": "End on a freeze frame.", "dont": ["Freeze frames"]}}
        answers = iter([json.dumps(broken), "sorry, no JSON"])
        spec = skill_creator.draft_skill("x", generate=lambda **k: next(answers))
        self.assertEqual(spec["rules"]["goal"], "End on a freeze frame.")
        self.assertEqual(sorted(w["code"] for w in spec["warnings"]), ["contradiction", "unrenderable"])
        self.assertNotIn("warnings", custom_skills.validate_spec(spec))

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
