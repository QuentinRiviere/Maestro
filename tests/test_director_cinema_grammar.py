import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.director import cinema_grammar as cg  # noqa: E402
from services.director.cinema_grammar import render, vocabulary  # noqa: E402


class PlannerBlockTests(unittest.TestCase):
    def test_block_is_bounded_and_complete(self):
        for target, shot_range in ((None, None), (45, (1.5, 5.0)), (90, (3.0, 8.0)), (60, None)):
            block = cg.planner_block(target_duration=target, shot_range=shot_range)
            self.assertTrue(block.startswith("CINEMA GRAMMAR"))
            self.assertLessEqual(len(block), cg.MAX_CHARS)
            self.assertGreater(len(block), 0.8 * cg.TARGET_CHARS)
            for section in ("UNIT:", "CAMERA:", "RHYTHM", "SCALE:", "INSERTS:", "K-DRAMA FRAMING:", "ACTING:"):
                self.assertIn(section, block)
            self.assertIn("ONE move per shot", block)

    def test_hard_cap_drops_whole_rules_from_the_end(self):
        block = cg.planner_block(target_duration=60, max_chars=900)
        self.assertLessEqual(len(block), 900)
        self.assertIn("UNIT:", block)
        self.assertNotIn("LIGHT:", block)
        self.assertTrue(all(line.startswith(("CINEMA", "- ")) for line in block.splitlines()))

    def test_rhythm_targets_follow_the_stats_and_the_model_range(self):
        free = cg.rhythm_targets(60)
        self.assertEqual(free["reference_episodes"], 2825)
        self.assertLess(free["asl_sec"], 3)
        self.assertLess(free["shots_min"], free["shots_max"])
        self.assertLess(free["silent_shot_sec"][0], free["line_shot_sec"][0])
        bounded = cg.rhythm_targets(60, (3.0, 8.0))
        self.assertEqual(bounded["asl_sec"], 3.0)
        self.assertEqual(bounded["shots_min"], bounded["shots_max"], 20)
        self.assertIn("about 20 shots", cg.planner_block(60, (3.0, 8.0)))

    def test_block_is_english_only(self):
        self.assertIsNone(re.search(r"[一-鿿]", cg.planner_block(60)))
        grammar = (render.DATA_DIR / "grammar.json").read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"[一-鿿]", grammar))


class GrammarDataTests(unittest.TestCase):
    def test_grammar_sections(self):
        grammar = cg.load_grammar()
        for key in ("shot_sizes", "angles", "camera_moves", "lenses", "lighting", "action_reaction", "dialogue_coverage",
                    "editing", "kdrama_motifs", "acting", "bad_acting", "planner_rules"):
            self.assertTrue(grammar[key], key)
        sources = set(grammar["sources"])
        for section in ("shot_sizes", "angles", "camera_moves", "kdrama_motifs"):
            for entry in grammar[section]:
                self.assertTrue(set(entry["source"]) <= sources, entry)
        self.assertGreaterEqual(len(grammar["action_reaction"]["variants"]), 8)

    def test_rhythm_stats_schema(self):
        stats = json.loads(render.RHYTHM_PATH.read_text(encoding="utf-8"))
        self.assertEqual(stats["schema_version"], 1)
        self.assertGreater(stats["source"]["shots"], 50000)
        for block in ("duration_sec", "shots", "asl_sec"):
            self.assertTrue({"n", "mean", "median", "p10", "p25", "p75", "p90"} <= set(stats["episode"][block]))
        durations = stats["shot_duration_sec"]
        self.assertEqual(set(durations), {"overall", "by_scale", "by_motion", "by_dialogue", "by_emotion"})
        self.assertEqual(set(durations["by_dialogue"]), {"with_dialogue", "silent"})
        for name, dist in stats["distribution"].items():
            self.assertAlmostEqual(sum(dist.values()), 1.0, delta=0.01, msg=name)
        self.assertTrue(set(stats["scale_vocabulary"]) <= set(durations["by_scale"]) | {"other"})
        for nexts in stats["transitions_scale_to_next"].values():
            self.assertGreater(nexts["n"], 0)
        coverage = stats["dialogue_coverage"]
        self.assertAlmostEqual(sum(coverage["shots_with_line"].values()), 1.0, delta=0.01)
        # Aggregates only: no free text from the annotations.
        self.assertLess(len(render.RHYTHM_PATH.read_bytes()), 20000)
        self.assertIsNone(re.search(r"[一-鿿]", render.RHYTHM_PATH.read_text(encoding="utf-8")))


class CameraVocabularyTests(unittest.TestCase):
    def test_synonyms_map_to_canonical_moves_keeping_grammar(self):
        cases = {
            "Wide two-shot, slow dolly in toward her": "Wide two-shot, slow push-in toward her",
            "The camera zooms in on his fist": "The camera pushes in on his fist",
            "medium shot, zoom out to reveal the room": "medium shot, pull-out to reveal the room",
            "steadicam follow down the corridor": "tracking shot down the corridor",
            "locked-off camera on the doorway": "static camera on the doorway",
        }
        for raw, expected in cases.items():
            self.assertEqual(vocabulary.canonical_camera_text(raw), expected)

    def test_vague_motion_becomes_static_unless_a_move_is_named(self):
        self.assertEqual(vocabulary.canonical_camera_text("the camera drifts slowly around her"), "static camera")
        self.assertEqual(vocabulary.canonical_camera_text("medium shot, cinematic camera movement"),
                         "medium shot, static camera")
        self.assertEqual(vocabulary.canonical_camera_text("slow push-in on her face, digital drift"),
                         "slow push-in on her face")

    def test_unrecognised_prose_is_kept(self):
        text = "A medium view follows the folder as she crosses the threshold, then moves sideways."
        self.assertEqual(vocabulary.canonical_camera_text(text), text)
        self.assertEqual(vocabulary.canonical_camera_text(""), "")

    def test_moves_detected_and_bracket_commands_off_by_default(self):
        self.assertFalse(vocabulary.MINIMAX_BRACKET_COMMANDS)
        self.assertEqual(vocabulary.camera_moves_in("handheld, then a slow push-in"), ["handheld", "push_in"])
        self.assertNotIn("[", vocabulary.canonical_camera_text("slow dolly in"))
        self.assertEqual(vocabulary.canonical_camera_text("slow dolly in", bracket_commands=True),
                         "slow push-in [Push in]")
        self.assertEqual(set(vocabulary.CAMERA_MOVES), set(vocabulary.CANONICAL_MOVES))

    def test_compiled_h3_prompt_uses_the_vocabulary(self):
        from services.director.h3_dialogue import compile_h3_official_prompt

        prompt, _ = compile_h3_official_prompt(
            "A cafe at night. Action: She sets the cup down. Camera: medium close-up, slow dolly in. "
            "Final beat: She looks at the door.",
            [], [], duration_seconds=6.0,
        )
        self.assertIn("Camera: medium close-up, slow push-in.", prompt)
        self.assertNotIn("dolly in", prompt)


class InjectionTests(unittest.TestCase):
    def setUp(self):
        from services import director_pipeline

        self.pipeline = director_pipeline

    def test_on_by_default_for_k_drama_skills(self):
        params = {"target_duration": 60}
        block = self.pipeline._apply_cinema_grammar("p1", params, "custom:k_drama_mini",
                                                    {"sequence_shots": True})
        self.assertIn("CINEMA GRAMMAR", block)
        self.assertEqual(params["director_cinema_grammar"], "on")
        self.assertEqual(params["_director_cinema_grammar"]["chars"], len(block))
        self.assertEqual(params["_director_cinema_grammar"]["shot_range"], [1.5, 5.0])

    def test_off_by_default_elsewhere_and_on_request(self):
        params = {}
        self.assertEqual(self.pipeline._apply_cinema_grammar("p1", params, "short_film"), "")
        self.assertEqual(params["director_cinema_grammar"], "off")
        self.assertEqual(params["_director_cinema_grammar"], {})
        params = {"director_cinema_grammar": "on", "target_duration": 45}
        self.assertIn("CINEMA GRAMMAR", self.pipeline._apply_cinema_grammar("p1", params, "short_film"))

    def test_off_wins_for_k_drama_skills(self):
        params = {"director_cinema_grammar": "off"}
        self.assertEqual(self.pipeline._apply_cinema_grammar("p1", params, "custom:k_drama_mini"), "")
        self.assertEqual(params["director_cinema_grammar"], "off")

    def test_mode_resolution(self):
        self.assertEqual(cg.cinema_grammar_mode({}, "custom:kdrama"), "on")
        self.assertEqual(cg.cinema_grammar_mode({}, "custom:romance"), "off")
        self.assertEqual(cg.cinema_grammar_mode({"director_cinema_grammar": "ON"}, "music_video"), "on")


if __name__ == "__main__":
    unittest.main()
