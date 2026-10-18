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
from services.director.blocking_map import subjects_in_view  # noqa: E402
from services.director.planners.custom import (CustomSkillPlanner, keep_contact_off_screen,  # noqa: E402
                                               skill_rules_block)

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
        blocking = first.metadata["blocking_map"]
        self.assertEqual(blocking["subjects"], [{"name": "Mina", "x": 0.4, "y": 0.4, "facing_deg": 180.0}])
        # The example facing missed Mina; the close-up camera turns to her and comes closer.
        self.assertEqual(blocking["camera"]["framing"], "close-up")
        self.assertTrue(subjects_in_view(blocking))
        self.assertLess(blocking["camera"]["y"], 0.9)
        self.assertNotIn("blocking_map", second.metadata)

    CAST = [{"name": "Mina", "description": "a new student"}, {"name": "Joon", "description": "a quiet boy"}]

    def test_dialogue_speakers_are_required_and_mapped_to_cast_names(self):
        spec = {**self.spec, "workflow": "story", "shot_duration": [2, 5]}
        planner = FakePlanner(spec, [
            {"scene_goal": "a", "subjects_on_screen": [{"name": "mina", "visual_description": "a girl"}],
             "dialogue_beats": [{"speaker": "MINA", "spoken_text": "Who are you?", "delivery": "wary"}]},
            {"scene_goal": "b", "subjects_on_screen": [{"visual_description": "Joon by the window"}],
             "dialogue_beats": [{"speaker": "the boy", "spoken_text": "Nobody."}]},
            {"scene_goal": "c", "subjects_on_screen": [{"name": "Joon", "visual_description": "x"},
                                                      {"name": "Mina", "visual_description": "y"}],
             "dialogue_beats": [{"spoken_text": "Wait."}, {"speaker_id": "Mina", "spoken_text": "Why?"}]},
            {"scene_goal": "d", "subjects_on_screen": [{"visual_description": "an empty hallway"}],
             "dialogue_beats": [{"speaker": "Joon", "spoken_text": "Here."}, {"speaker": "ghost", "spoken_text": "Boo"}]},
        ])
        plan = planner.plan(scene_description="A new student meets a mysterious boy", characters=self.CAST)
        prompt = planner.prompts[-1]["system_prompt"]
        self.assertIn('"speaker": "exact cast name of who says it"', prompt)
        self.assertIn("every line has a speaker", prompt)
        self.assertIn("CAST NAMES (use them exactly for subjects_on_screen.name and the dialogue speaker): Mina, Joon",
                      prompt)
        speakers = [[b.speaker_id for b in shot.dialogue_beats] for shot in plan.shots]
        # Exact name, the only one in frame, the first one in frame, and an
        # off-screen cast speaker kept (an unresolvable one is left as given).
        self.assertEqual(speakers, [["Mina"], ["Joon"], ["Joon", "Mina"], ["Joon", "ghost"]])
        first = plan.shots[0].subjects_on_screen[0]
        self.assertEqual((first.speaker_name, first.character_id), ("Mina", "char_0"))
        self.assertEqual(plan.shots[1].subjects_on_screen[0].character_id, "char_1")
        self.assertIsNone(plan.shots[3].subjects_on_screen[0].speaker_name)

    def test_clip_speakers_resolve_from_custom_plan(self):
        from services.director_pipeline import _director_clip_speakers
        spec = {**self.spec, "workflow": "story", "shot_duration": [2, 5]}
        plan = FakePlanner(spec, [
            {"scene_goal": "a", "subjects_on_screen": [{"name": "Joon", "visual_description": "a boy"}],
             "dialogue_beats": [{"speaker": "", "spoken_text": "Nobody."}]},
        ]).plan(scene_description="x", characters=self.CAST)
        shot = plan.shots[0].to_dict()
        clip = {"_director_dialogue_beats": shot["dialogue_beats"],
                "_director_subjects_on_screen": shot["subjects_on_screen"]}
        params = {"characters": self.CAST, "_director_cast": {"characters": self.CAST}}
        self.assertEqual(_director_clip_speakers(clip, params), ["Joon"])

    def test_filmable_action_and_reaction_share_rules(self):
        spec = {**self.spec, "workflow": "story"}
        planner = FakePlanner(spec, [{"scene_goal": "a"}])
        planner.plan(scene_description="x")
        prompt = planner.prompts[-1]["system_prompt"]
        self.assertIn("Filmable action only", prompt)
        self.assertIn("pupils dilating", prompt)
        self.assertIn("Reaction shots are at most about one third of the shots", prompt)
        self.assertNotIn("micro-expression", prompt)

    def test_hand_and_prop_rules_are_in_the_prompt(self):
        spec = {**self.spec, "workflow": "story"}
        planner = FakePlanner(spec, [{"scene_goal": "a"}])
        planner.plan(scene_description="x")
        prompt = planner.prompts[-1]["system_prompt"]
        self.assertIn("Body-part inserts of gross motion are strong coverage", prompt)
        self.assertIn("legs entering the frame on\n  an arrival", prompt)
        self.assertIn("at most ONE simple, decisive hand action per shot", prompt)
        self.assertIn("no pouring or liquid in motion, no object passed from hand to hand", prompt)
        self.assertIn("every object the action uses is already visible at the first frame", prompt)
        self.assertIn("Nothing appears, materializes or is pulled into frame mid-shot", prompt)
        self.assertIn("Prop state: track each prop through the consecutive shots of a scene", prompt)
        self.assertIn('"hand_detail": false', prompt)
        self.assertIn("hand_detail: true ONLY when the story beat", prompt)
        self.assertIn('"props": ["each object the action uses', prompt)

    def test_scene_and_shot_prompts_carry_the_lighting_rule(self):
        spec = {**self.spec, "workflow": "story"}
        planner = FakePlanner(spec, [{"scene_goal": "a"}])
        planner.plan(scene_description="x")
        scene_prompt, shot_prompt = planner.prompts[0]["system_prompt"], planner.prompts[-1]["system_prompt"]
        for prompt in (scene_prompt, shot_prompt):
            self.assertIn("LIGHTING (live-action / photographic looks)", prompt)
            self.assertIn("Night and interior drama scenes are low-key", prompt)
            self.assertIn("lamps, pendants, windows, neon, streetlights", prompt)
            self.assertIn('Never "balanced", "even", "soft natural lighting"', prompt)

    def test_props_and_hand_detail_reach_shot_metadata(self):
        spec = {**self.spec, "workflow": "story"}
        plan = FakePlanner(spec, [
            {"scene_goal": "a", "hand_detail": True, "props": ["a lump of raw clay on the wheel", " ", 3]},
            {"scene_goal": "b", "hand_detail": "false", "props": "not a list"},
            {"scene_goal": "c", "hand_detail": "true"},
        ]).plan(scene_description="x")
        meta = [shot.metadata for shot in plan.shots]
        self.assertTrue(meta[0]["hand_detail"])
        self.assertEqual(meta[0]["props"], ["a lump of raw clay on the wheel", "3"])
        self.assertNotIn("hand_detail", meta[1])
        self.assertNotIn("props", meta[1])
        self.assertTrue(meta[2]["hand_detail"])

    def test_dialogue_density_rule_option_and_default(self):
        spec = {**self.spec, "workflow": "story"}
        planner = FakePlanner(spec, [{"scene_goal": "a"}])
        planner.plan(scene_description="x")
        self.assertIn("DIALOGUE DENSITY (medium): in exchanges, about one line every 2-3 shots",
                      planner.prompts[-1]["system_prompt"])
        dense = custom_skills.validate_spec({**SPEC, "workflow": "story",
                                             "rules": {**SPEC["rules"], "dialogue_density": " High "}})
        self.assertEqual(dense["rules"]["dialogue_density"], "high")
        self.assertNotIn("dialogue_density", custom_skills.validate_spec(
            {**SPEC, "rules": {**SPEC["rules"], "dialogue_density": "chatty"}})["rules"])
        planner = FakePlanner(dense, [{"scene_goal": "a"}])
        planner.plan(scene_description="x")
        self.assertIn("DIALOGUE DENSITY (high)", planner.prompts[-1]["system_prompt"])
        planner.plan(scene_description="x", skill_options={"dialogue_density": "low"})
        self.assertIn("DIALOGUE DENSITY (low)", planner.prompts[-1]["system_prompt"])
        self.assertIn("one-line-per-shot", planner.prompts[-1]["system_prompt"])

    def test_sequence_shots_keep_their_range_with_speakers(self):
        spec = {**self.spec, "workflow": "story"}
        planner = FakePlanner(spec, [
            {"scene_goal": "a", "duration_sec": 9, "keep_sec": [0, 2],
             "dialogue_beats": [{"speaker": "Mina", "spoken_text": "Hi."}]},
            {"scene_goal": "b", "duration_sec": 1, "continues_previous": True},
        ])
        plan = planner.plan(scene_description="x", characters=self.CAST, sequence_shots=True)
        self.assertEqual([s.duration_sec for s in plan.shots], [5.0, 1.5])
        self.assertNotIn("keep_sec", plan.shots[0].metadata)
        self.assertEqual(plan.shots[0].dialogue_beats[0].speaker_id, "Mina")

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


class InsertContactTests(unittest.TestCase):
    def shots(self):
        return [
            {"scene_id": "cafe", "scene_type": "exchange", "duration_sec": 6,
             "opening_blocking": "Jin-woo sits screen-left, his right hand flat on the table.",
             "closing_blocking": "Jin-woo's right hand rests inches from her wrist on the tabletop.",
             "action_beats": ["0.0-3.0s: he leans in", "3.0-6.0s: his hand slides toward her wrist"]},
            {"scene_id": "cafe", "scene_type": "insert", "duration_sec": 4,
             "opening_blocking": "Hae-won's hands clutch the mug. Jin-woo's hand rests open on the table inches away from her wrist.",
             "closing_blocking": "Jin-woo's hand securely encircles Hae-won's wrist; her fingers stay around the mug.",
             "action_beats": ["0.0-4.0s: his hand crosses and closes around her wrist"], "locks": ["only two pairs of hands"]},
            {"scene_id": "cafe", "scene_type": "reaction", "duration_sec": 5,
             "opening_blocking": "Hae-won looks up.", "closing_blocking": "She nods."},
        ]

    def test_a_hold_on_a_person_goes_below_the_frame_and_the_shot_before_stops_short(self):
        shots = self.shots()
        self.assertEqual(keep_contact_off_screen(shots), 1)
        insert, previous = shots[1], shots[0]
        self.assertEqual(insert["scene_type"], "reaction")
        self.assertTrue(insert["opening_blocking"].startswith("Framed from the chest up, every hand below the bottom edge"))
        self.assertIn("no hand in sight", insert["closing_blocking"])
        self.assertIn("faces carry the hold", insert["action_beats"][0])
        self.assertEqual(len(insert["action_beats"]), 1)
        self.assertEqual(insert["locks"][0], "only two pairs of hands")
        self.assertIn("the hold (Jin-woo's hand securely encircles Hae-won's wrist; her fingers stay around the mug) "
                      "stays below the bottom edge of the frame", insert["locks"][1])
        self.assertEqual(insert["camera_plan"]["framing"], "medium close-up from the chest up, every hand below the bottom edge of the frame")
        # The touch itself falls between the cuts: the shot before keeps its hand short of the wrist.
        self.assertEqual(previous["closing_blocking"], "Jin-woo's right hand rests inches from her wrist on the tabletop.")
        self.assertEqual(previous["action_beats"][-1], "3.0-6.0s: his hand slides toward her wrist")
        self.assertIn("no contact is made in this shot", previous["locks"][0])
        self.assertEqual(shots[2]["opening_blocking"], "Hae-won looks up.")

    def test_a_two_shot_that_ends_on_the_grab_is_reframed_above_the_hands(self):
        # Run 830926ff: the grab in the two-shot morphed; run e823ac19: the hold already made still melted.
        shots = self.shots()
        shots[1] = {"scene_id": "cafe", "scene_type": "exchange", "duration_sec": 6,
                    "camera_plan": {"framing": "medium two-shot", "movement": "static"},
                    "opening_blocking": "Jin-woo screen-left, his right hand resting on the table short of her wrist.",
                    "closing_blocking": "Jin-woo's right hand is wrapped around Hae-won's left wrist on the table.",
                    "action_beats": ["0.0-2.0s: she says the line and looks away",
                                     "2.0-4.0s: his hand slides across the table and closes around her wrist",
                                     "4.0-6.0s: she looks up at him, lips parted"]}
        self.assertEqual(keep_contact_off_screen(shots), 1)
        two_shot = shots[1]
        self.assertEqual(two_shot["scene_type"], "exchange")
        self.assertEqual(two_shot["camera_plan"]["framing"],
                         "medium two-shot, from the chest up, every hand below the bottom edge of the frame")
        self.assertEqual(two_shot["action_beats"], [
            "0.0-6.0s: nothing moves below the shoulders; the faces carry the hold that exists below the frame: "
            "a caught breath, eyes that lock.",
            "0.0-2.0s: she says the line and looks away",
            "4.0-6.0s: she looks up at him, lips parted"])
        self.assertIn("never shown; no hand in frame", two_shot["locks"][-1])
        self.assertIn("drops below the bottom edge of the frame", shots[0]["locks"][0])

    def test_an_insert_on_an_object_still_takes_its_closing_move_from_the_shot_before(self):
        shots = self.shots()
        shots[0]["closing_blocking"] = "Hae-won's right hand rests flat on the table beside the mug."
        shots[0]["action_beats"] = ["0.0-6.0s: she listens, her hand drifting toward the mug"]
        shots[1]["opening_blocking"] = "Hae-won's hand lies open on the table, fingertips inches from the mug."
        shots[1]["closing_blocking"] = "Her fingers are wrapped around the mug."
        self.assertEqual(keep_contact_off_screen(shots), 1)
        self.assertIn("nothing moves but the fingers", shots[1]["action_beats"][0])
        self.assertEqual(shots[0]["closing_blocking"], "Her fingers are wrapped around the mug.")
        self.assertTrue(shots[0]["action_beats"][-1].endswith("takes hold, slowly; nothing else moves."))
        # The same hold of a cup in an exchange is left alone: a single hand on an object renders.
        shots = self.shots()
        shots[1]["scene_type"] = "exchange"
        shots[1]["closing_blocking"] = "Her fingers are wrapped around the mug."
        self.assertEqual(keep_contact_off_screen(shots), 0)

    def test_an_insert_that_already_holds_or_never_touches_is_left_alone(self):
        shots = self.shots()
        shots[1]["opening_blocking"] = "Jin-woo's hand already holds her wrist on the table."
        self.assertEqual(keep_contact_off_screen(shots), 0)
        shots = self.shots()
        shots[1]["closing_blocking"] = "The mug sits alone on the table, steam rising."
        self.assertEqual(keep_contact_off_screen(shots), 0)
        self.assertEqual(shots[0]["closing_blocking"], "Jin-woo's right hand rests inches from her wrist on the tabletop.")

    def test_every_form_of_taking_hold_counts(self):
        shots = self.shots()
        shots[1]["opening_blocking"] = ("Jin-woo's right hand enters from the left edge of frame, hovering just "
                                        "inches from her left wrist.")
        shots[1]["closing_blocking"] = "Jin-woo's fingers are securely and gently wrapped around Hae-won's left wrist."
        self.assertEqual(keep_contact_off_screen(shots), 1)
        shots = self.shots()
        shots[1]["closing_blocking"] = "His hand rests on her forearm; the cup is still."
        self.assertEqual(keep_contact_off_screen(shots), 1)
        shots = self.shots()
        # Run 7bb9fc86: her hands already hold the cup, his hand is still on its way to the wrist.
        shots[1]["opening_blocking"] = ("Cha Hae-won's slender pale hands tremble while holding the white ceramic cup; "
                                        "Song Jin-woo's right hand enters smoothly from screen left, resting fingertips "
                                        "inches from her right wrist.")
        shots[1]["closing_blocking"] = "Jin-woo's fingers remain wrapped firmly around Hae-won's right wrist."
        self.assertEqual(keep_contact_off_screen(shots), 1)

    def test_a_plan_without_closings_uses_the_last_beat(self):
        shots = self.shots()
        for shot in shots:
            shot.pop("closing_blocking")
        shots[1]["action_beats"] = ["0.0-1.0s: her hands tremble", "1.0-4.0s: his hand closes around her wrist"]
        self.assertEqual(keep_contact_off_screen(shots), 1)
        self.assertEqual(shots[1]["closing_blocking"], "his hand closes around her wrist")
        self.assertTrue(shots[1]["opening_blocking"].startswith("Framed from the chest up"))
        # The previous shot ended "his hand slides toward her wrist" (approach): it stays short of it.
        self.assertNotIn("closing_blocking", shots[0])
        self.assertIn("no contact is made in this shot", shots[0]["locks"][0])


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        self.original = custom_skills.SKILLS_DIR
        custom_skills.SKILLS_DIR = self.dir
        self.addCleanup(setattr, custom_skills, "SKILLS_DIR", self.original)

    def spec(self, goal):
        return {"id": "k_drama", "name": "K-drama", "description": "d", "workflow": "both",
                "rules": {"goal": goal, "structure": "s", "pacing": "p", "camera": "c", "dialogue": "d",
                          "sound": "s", "do": [], "dont": []}}

    def test_each_save_keeps_the_replaced_version(self):
        first = custom_skills.save_custom(self.spec("tension"))
        self.assertEqual(first["version"], 1)
        second = custom_skills.save_custom(self.spec("tension, then a cliffhanger"), skill_id="k_drama")
        self.assertEqual(second["version"], 2)
        revisions = custom_skills.list_revisions("custom:k_drama")
        self.assertEqual([(r["version"], r["current"]) for r in revisions], [(1, False), (2, True)])
        self.assertEqual(custom_skills.get_revision("k_drama", 1)["rules"]["goal"], "tension")
        self.assertEqual(custom_skills.current_version("custom:k_drama"), 2)
        self.assertIsNone(custom_skills.current_version("music_video"))

    def test_restoring_an_older_version_makes_a_new_current_one(self):
        custom_skills.save_custom(self.spec("tension"))
        custom_skills.save_custom(self.spec("worse"), skill_id="k_drama")
        restored = custom_skills.restore_revision("k_drama", 1, note="version 2 scored lower")
        self.assertEqual((restored["version"], restored["restored_from"]), (3, 1))
        self.assertEqual(custom_skills.get_custom("k_drama")["rules"]["goal"], "tension")
        self.assertEqual(custom_skills.get_custom("k_drama")["revision_note"], "version 2 scored lower")
        self.assertEqual([r["version"] for r in custom_skills.list_revisions("k_drama")], [1, 2, 3])
        with self.assertRaises(custom_skills.SkillError):
            custom_skills.restore_revision("k_drama", 9)
