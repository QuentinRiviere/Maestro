import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services.director import cinema_grammar as cg  # noqa: E402
from services.director.cinema_grammar import profiles, render, vocabulary  # noqa: E402

CJK = re.compile(r"[一-鿿]")


def rhythm_line(block):
    return next(line for line in block.splitlines() if line.startswith("- RHYTHM"))


class PlannerBlockTests(unittest.TestCase):
    def test_block_is_bounded_and_complete_for_every_layer(self):
        for genre in cg.genre_ids():
            for fmt in cg.format_ids():
                for target, shot_range in ((None, None), (45, (1.5, 5.0)), (90, (3.0, 8.0)), (60, None)):
                    block = cg.planner_block(target, shot_range, format_id=fmt, genre_id=genre)
                    self.assertTrue(block.startswith("CINEMA GRAMMAR"))
                    self.assertLessEqual(len(block), cg.MAX_CHARS, (genre, fmt))
                    self.assertGreater(len(block), 0.8 * cg.TARGET_CHARS, (genre, fmt))
                    for section in ("UNIT:", "CAMERA:", "RHYTHM", "FORMAT", "CONTINUITY:", "AI VIDEO:"):
                        self.assertIn(section, block, (genre, fmt))
                    self.assertIn("ONE move per shot", block)
                    self.assertIsNone(CJK.search(block))

    def test_k_drama_block_carries_its_genre_layer(self):
        block = cg.planner_block(60, (1.5, 10.0), format_id="vertical_short", genre_id="kdrama")
        self.assertIn("(K-drama, 9:16 vertical short;", block)
        for section in ("K-DRAMA CAMERA:", "K-DRAMA ACTING:", "K-DRAMA MOTIFS:", "COVERAGE:", "frame-within-frame"):
            self.assertIn(section, block)
        self.assertIn("slow push-in on a realisation", block)

    def test_hard_cap_drops_whole_lines_from_the_lowest_priority(self):
        block = cg.planner_block(60, max_chars=900, genre_id="kdrama")
        self.assertLessEqual(len(block), 900)
        self.assertIn("UNIT:", block)
        self.assertNotIn("K-DRAMA LOOK:", block)
        self.assertTrue(all(line.startswith(("CINEMA", "- ")) for line in block.splitlines()))

    def test_k_drama_no_longer_gets_micro_drama_pacing(self):
        kdrama = cg.rhythm_targets(60, genre_id="kdrama")
        micro = cg.rhythm_targets(60, genre_id="micro_drama")
        self.assertGreaterEqual(kdrama["median_sec"], 3.0)
        self.assertGreaterEqual(kdrama["asl_band_sec"][0], 3.0)
        self.assertLessEqual(kdrama["asl_band_sec"][1], 6.0)
        self.assertGreaterEqual(kdrama["silent_sec"][1], 5.0)
        self.assertLess(micro["median_sec"], 2.0)
        self.assertGreater(micro["shots_max"], kdrama["shots_max"])
        line = rhythm_line(cg.planner_block(60, genre_id="kdrama", format_id="vertical_short"))
        self.assertIn("K-drama default", line)
        self.assertNotIn("DramaBoard", line)
        self.assertNotIn("1.7", line)

    def test_micro_drama_rhythm_follows_dramaboard_and_the_model_range(self):
        free = cg.rhythm_targets(60, genre_id="micro_drama")
        self.assertEqual(free["reference_episodes"], 2825)
        self.assertLess(free["asl_sec"], 3)
        self.assertLess(free["shots_min"], free["shots_max"])
        self.assertLess(free["silent_sec"][0], free["line_sec"][0])
        bounded = cg.rhythm_targets(60, (3.0, 8.0), genre_id="micro_drama")
        self.assertEqual(bounded["asl_sec"], 3.0)
        self.assertEqual(bounded["shots_min"], bounded["shots_max"], 20)
        self.assertIn("about 20 shots", cg.planner_block(60, (3.0, 8.0), genre_id="micro_drama"))
        self.assertIn("DramaBoard, 2,825 micro-drama episodes", rhythm_line(cg.planner_block(60, genre_id="micro_drama")))

    def test_every_length_stays_inside_the_model_range(self):
        for genre in cg.genre_ids():
            targets = cg.rhythm_targets(60, (2.0, 4.0), genre_id=genre)
            values = [targets["median_sec"], targets["asl_sec"], *targets["asl_band_sec"], *targets["silent_sec"],
                      *targets["line_sec"]]
            self.assertTrue(all(2.0 <= v <= 4.0 for v in values), (genre, values))

    def test_format_mix_comes_from_the_format_and_the_genre_bias(self):
        vertical, label = profiles.biased_mix("vertical_short", "micro_drama")
        self.assertEqual(label, "DramaBoard 9:16 mix")
        self.assertAlmostEqual(vertical["medium"], 26.2, delta=0.5)
        horizontal, _ = profiles.format_mix("horizontal")
        self.assertGreater(horizontal["wide"], vertical["wide"] * 1.5)
        kdrama, label = profiles.biased_mix("vertical_short", "kdrama")
        self.assertIn("biased for K-drama", label)
        self.assertGreater(kdrama["wide"], vertical["wide"])
        for mix in (vertical, horizontal, kdrama):
            self.assertAlmostEqual(sum(mix.values()), 100, delta=0.5)

    def test_vlog_has_no_shot_reverse_coverage(self):
        block = cg.planner_block(60, genre_id="vlog", format_id="vertical_short")
        self.assertNotIn("COVERAGE:", block)
        self.assertIn("VLOG CAMERA: handheld", block)


class CalibrationTests(unittest.TestCase):
    MEASURES = [
        {"id": "a", "weight": 30, "asl_sec": 2.6, "median_sec": 2.1, "cuts_per_minute": 23.0, "static_share": 0.6,
         "shot_mix": {"close_up": 45.0, "medium": 30.0, "two_shot": 25.0}, "line_sec": [2.4, 3.0]},
        {"id": "b", "weight": 10, "asl_sec": 3.4, "shot_mix": {"wide": 50.0, "medium": 50.0}},
    ]

    def test_weighted_average_across_references(self):
        calibration = cg.calibrate(self.MEASURES)
        self.assertEqual(calibration["references"], ["a", "b"])
        self.assertAlmostEqual(calibration["asl_sec"], 2.8)
        self.assertEqual(calibration["median_sec"], 2.1)
        self.assertEqual(calibration["static_share"], 0.6)
        self.assertEqual(calibration["line_sec"], [2.4, 3.0])
        self.assertNotIn("silent_sec", calibration)
        mix = calibration["shot_mix"]
        self.assertAlmostEqual(sum(mix.values()), 100, delta=0.5)
        self.assertEqual(next(iter(mix)), "medium")
        self.assertEqual(cg.calibrate([]), {})
        self.assertEqual(cg.calibrate([{}]), {})

    def test_references_override_genre_numbers_and_name_their_source(self):
        calibration = cg.calibrate(self.MEASURES)
        targets = cg.rhythm_targets(60, genre_id="kdrama", calibration=calibration)
        self.assertEqual(targets["asl_sec"], 2.8)
        self.assertEqual(targets["median_sec"], 2.1)
        self.assertEqual(targets["static_share"], 0.6)
        self.assertEqual(targets["line_sec"], [2.4, 3.0])
        self.assertEqual(targets["sources"]["asl_sec"], "references")
        # No silent split measured: the genre's holds scaled to the references' pace.
        self.assertEqual(targets["sources"]["silent_sec"], "scaled")
        self.assertLess(targets["silent_sec"][1], 6.0)
        block = cg.planner_block(60, genre_id="kdrama", format_id="vertical_short", calibration=calibration)
        line = rhythm_line(block)
        self.assertIn("from your references: median shot ~2.1s, average 2.2-3.5s (23 cuts/min), "
                      "shots under a line ~2.4-3s, 60% static;", line)
        self.assertIn("K-drama default scaled to your references: silent reactions", line)
        self.assertIn("Shot mix (from your references): medium", block)
        self.assertIn("two-shot", block)
        self.assertLessEqual(len(block), cg.MAX_CHARS)

    def test_calibrated_lengths_are_clamped_to_the_model_range(self):
        calibration = cg.calibrate([{"id": "fast", "asl_sec": 1.0, "median_sec": 0.8, "line_sec": [0.9, 1.2]}])
        targets = cg.rhythm_targets(30, (3.0, 8.0), genre_id="kdrama", calibration=calibration)
        self.assertEqual(targets["asl_sec"], 3.0)
        self.assertEqual(targets["median_sec"], 3.0)
        self.assertEqual(targets["line_sec"], [3.0, 3.0])
        self.assertEqual(targets["shots_max"], 10)

    def test_genre_defaults_without_calibration(self):
        targets = cg.rhythm_targets(60, genre_id="kdrama")
        self.assertEqual(set(targets["sources"].values()), {"genre"})
        self.assertNotIn("cuts_per_minute", targets)


class LayerSelectionTests(unittest.TestCase):
    def test_format_from_aspect_ratio_and_override(self):
        self.assertEqual(cg.select_format({"director_aspect_ratio": "9:16"}), ("vertical_short", "aspect_ratio"))
        self.assertEqual(cg.select_format({"director_aspect_ratio": "3:4"}), ("vertical_short", "aspect_ratio"))
        self.assertEqual(cg.select_format({"director_aspect_ratio": "1:1"}), ("vertical_short", "aspect_ratio"))
        self.assertEqual(cg.select_format({"director_aspect_ratio": "16:9"}), ("horizontal", "aspect_ratio"))
        self.assertEqual(cg.select_format({"director_aspect_ratio": "21:9"}), ("horizontal", "aspect_ratio"))
        self.assertEqual(cg.select_format({"director_aspect_ratio": "auto"}), ("horizontal", "default"))
        self.assertEqual(cg.select_format({}), ("horizontal", "default"))
        self.assertEqual(cg.select_format({"director_aspect_ratio": "16:9", "director_format_profile": "vertical_short"}),
                         ("vertical_short", "param"))
        self.assertEqual(cg.select_format({"director_aspect_ratio": "9:16", "director_format_profile": "nope"}),
                         ("vertical_short", "aspect_ratio"))

    def test_genre_from_skill_and_explicit_param(self):
        self.assertEqual(cg.select_genre({}, "custom:k_drama_mini"), ("kdrama", "skill"))
        self.assertEqual(cg.select_genre({}, "custom:KDrama"), ("kdrama", "skill"))
        self.assertEqual(cg.select_genre({}, "custom:my_vlog"), ("vlog", "skill"))
        self.assertEqual(cg.select_genre({}, "short_film"), ("neutral", "default"))
        self.assertEqual(cg.select_genre({"director_genre_profile": "auto"}, "custom:romance"), ("neutral", "default"))
        self.assertEqual(cg.select_genre({"director_genre_profile": "anime_mini_series"}, "custom:k_drama_mini"),
                         ("anime_mini_series", "param"))
        self.assertEqual(cg.select_genre({"director_genre_profile": "unknown"}, "custom:k_drama_mini"),
                         ("kdrama", "skill"))

    def test_genre_files_are_complete(self):
        self.assertEqual(set(cg.genre_ids()), {"kdrama", "micro_drama", "anime_mini_series", "vlog", "neutral"})
        for genre_id in cg.genre_ids():
            profile = profiles.genre_profile(genre_id)
            for key in ("label", "tag", "summary", "camera", "look", "acting", "motifs", "do", "dont", "source"):
                self.assertIn(key, profile, (genre_id, key))
            rhythm = profiles.genre_rhythm(genre_id)
            for key in ("median_sec", "asl_sec", "asl_band_sec", "silent_sec", "line_sec", "static_share"):
                self.assertIn(key, rhythm, (genre_id, key))
            self.assertLessEqual(rhythm["asl_band_sec"][0], rhythm["asl_sec"])
            self.assertLessEqual(rhythm["asl_sec"], rhythm["asl_band_sec"][1])
            for path in profiles.GENRES_DIR.glob("*.json"):
                self.assertIsNone(CJK.search(path.read_text(encoding="utf-8")))

    def test_universal_layer_has_no_genre_numbers(self):
        text = json.dumps(cg.load_universal())
        self.assertNotRegex(text, r"\d+(?:\.\d+)?\s*s\b")
        self.assertNotIn("K-drama", text)
        self.assertIn("180-degree", text)

    def test_block_is_english_only(self):
        self.assertIsNone(CJK.search(cg.planner_block(60)))
        grammar = (render.DATA_DIR / "grammar.json").read_text(encoding="utf-8")
        self.assertIsNone(CJK.search(grammar))


class GrammarDataTests(unittest.TestCase):
    def test_grammar_sections(self):
        grammar = cg.load_grammar()
        for key in ("shot_sizes", "angles", "camera_moves", "lenses", "lighting", "action_reaction", "dialogue_coverage",
                    "editing", "kdrama_motifs", "acting", "bad_acting"):
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

    def test_on_by_default_for_k_drama_skills_with_k_drama_pacing(self):
        params = {"target_duration": 60, "director_aspect_ratio": "9:16"}
        block = self.pipeline._apply_cinema_grammar("p1", params, "custom:k_drama_mini",
                                                    {"sequence_shots": True})
        self.assertIn("CINEMA GRAMMAR (K-drama, 9:16 vertical short;", block)
        self.assertEqual(params["director_cinema_grammar"], "on")
        self.assertEqual(params["director_genre_profile"], "auto")
        self.assertEqual(params["director_format_profile"], "auto")
        snapshot = params["_director_cinema_grammar"]
        self.assertEqual(snapshot["chars"], len(block))
        self.assertEqual(snapshot["shot_range"], [1.5, 5.0])
        self.assertEqual((snapshot["genre"], snapshot["genre_source"]), ("kdrama", "skill"))
        self.assertEqual((snapshot["format"], snapshot["format_source"]), ("vertical_short", "aspect_ratio"))
        self.assertEqual(snapshot["calibration"], "genre_default")
        self.assertEqual(snapshot["calibration_references"], [])
        self.assertGreaterEqual(snapshot["rhythm"]["median_sec"], 3.0)
        self.assertNotIn("DramaBoard, 2,825", block)

    def test_explicit_layers_win(self):
        params = {"director_aspect_ratio": "9:16", "director_format_profile": "horizontal",
                  "director_genre_profile": "micro_drama", "target_duration": 60}
        block = self.pipeline._apply_cinema_grammar("p1", params, "custom:k_drama_mini")
        self.assertIn("(micro-drama, 16:9 horizontal;", block)
        snapshot = params["_director_cinema_grammar"]
        self.assertEqual((snapshot["genre"], snapshot["genre_source"]), ("micro_drama", "param"))
        self.assertEqual((snapshot["format"], snapshot["format_source"]), ("horizontal", "param"))
        self.assertLess(snapshot["rhythm"]["median_sec"], 2.0)

    def test_off_by_default_elsewhere_and_on_request(self):
        params = {}
        self.assertEqual(self.pipeline._apply_cinema_grammar("p1", params, "short_film"), "")
        self.assertEqual(params["director_cinema_grammar"], "off")
        self.assertEqual(params["_director_cinema_grammar"], {})
        params = {"director_cinema_grammar": "on", "target_duration": 45}
        self.assertIn("CINEMA GRAMMAR (neutral, 16:9 horizontal;", self.pipeline._apply_cinema_grammar(
            "p1", params, "short_film"))
        params = {"director_genre_profile": "vlog"}
        self.assertIn("(vlog,", self.pipeline._apply_cinema_grammar("p1", params, "short_film"))
        self.assertEqual(params["director_cinema_grammar"], "on")

    def test_off_wins_for_k_drama_skills(self):
        params = {"director_cinema_grammar": "off", "director_genre_profile": "kdrama"}
        self.assertEqual(self.pipeline._apply_cinema_grammar("p1", params, "custom:k_drama_mini"), "")
        self.assertEqual(params["director_cinema_grammar"], "off")

    def test_mode_resolution(self):
        self.assertEqual(cg.cinema_grammar_mode({}, "custom:kdrama"), "on")
        self.assertEqual(cg.cinema_grammar_mode({}, "custom:romance"), "off")
        self.assertEqual(cg.cinema_grammar_mode({}, "kdrama"), "off")
        self.assertEqual(cg.cinema_grammar_mode({"director_cinema_grammar": "ON"}, "music_video"), "on")
        self.assertEqual(cg.cinema_grammar_mode({"director_genre_profile": "anime_mini_series"}, "music_video"), "on")
        self.assertEqual(cg.cinema_grammar_mode({"director_genre_profile": "auto"}, "music_video"), "off")


class ReferenceCalibrationInjectionTests(unittest.TestCase):
    """Style references of the run calibrate the cinema block; the reference block drops its numbers."""

    def setUp(self):
        import tempfile
        from unittest.mock import patch

        from services import director_pipeline
        from services.director import bibles
        from services.director import style_reference as sr

        self.pipeline, self.sr = director_pipeline, sr
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bible = bibles.save_bible({"name": "Seoul Nights"})
        video = Path(tmp.name) / "ref.mp4"
        video.write_bytes(b"fake video")
        self.refs = []
        for label, asl, count, static in (("Quarrel", 2.0, 30, 70.0), ("Confession", 4.0, 10, 100.0)):
            ref = sr.copy_into_bible(self.bible["id"], str(video), label=label)
            sr._set(self.bible["id"], ref["id"], status="done", stats={
                "asl_sec": asl, "median_sec": asl - 0.4, "cuts_per_minute": round(60 / asl, 1), "shot_count": count,
                "shot_mix": {"close_up": 60.0, "medium": 40.0},
                "motion_mix": {"static": static, "push_in": 100.0 - static} if static < 100 else {"static": 100.0},
                "audio_lengths": {"line_sec": [asl, asl + 1.0]}},
                grammar={"rhythm": {"target_asl_sec": asl}, "shot_mix": {"close_up": 60.0, "medium": 40.0},
                         "rules": [f"{label} rule: hold the reaction."], "patterns": ["Reaction after each line"]})
            self.refs.append(ref)

    def test_references_calibrate_the_cinema_block(self):
        params = {"bible_id": self.bible["id"], "target_duration": 60, "director_aspect_ratio": "9:16"}
        block = self.pipeline._apply_cinema_grammar("p1", params, "custom:k_drama_mini")
        snapshot = params["_director_cinema_grammar"]
        self.assertEqual(snapshot["calibration"], "references")
        self.assertEqual(snapshot["calibration_references"], [r["id"] for r in self.refs])
        # Weighted by shot count: (30 * 2.0 + 10 * 4.0) / 40.
        self.assertEqual(snapshot["rhythm"]["asl_sec"], 2.5)
        self.assertEqual(snapshot["rhythm"]["static_share"], 0.78)
        self.assertEqual(snapshot["rhythm"]["sources"]["asl_sec"], "references")
        line = rhythm_line(block)
        self.assertIn("from your references: median shot ~2.1s, average 2-3.1s", line)
        self.assertIn("78% static", line)
        self.assertIn("Shot mix (from your references): close-up 60%, medium 40%", block)
        self.assertLessEqual(len(block), cg.MAX_CHARS)

    def test_reference_block_leaves_the_numbers_to_the_cinema_block(self):
        params = {"bible_id": self.bible["id"], "scene_description": "A quarrel."}
        with_cinema = self.pipeline._apply_style_references("p1", params, "custom:k_drama_mini")
        self.assertIn("Quarrel rule: hold the reaction.", with_cinema)
        self.assertIn("are in the CINEMA GRAMMAR block", with_cinema)
        self.assertNotIn("- Rhythm:", with_cinema)
        self.assertNotIn("- Shot mix:", with_cinema)
        params = {"bible_id": self.bible["id"], "scene_description": "A quarrel."}
        alone = self.pipeline._apply_style_references("p1", params, "short_film")
        self.assertIn("- Rhythm: target average shot 2.0 s", alone)
        self.assertIn("- Shot mix:", alone)

    def test_references_off_means_genre_defaults(self):
        params = {"bible_id": self.bible["id"], "director_style_reference": "off", "target_duration": 60}
        self.pipeline._apply_cinema_grammar("p1", params, "custom:k_drama_mini")
        self.assertEqual(params["_director_cinema_grammar"]["calibration"], "genre_default")

    def test_older_analyses_read_motion_from_their_shots(self):
        reference = {"id": "old", "stats": {"asl_sec": 3.0, "shot_count": 4}}
        shots = [{"movement": "static", "audio": "dialogue", "duration": d} for d in (2.0, 3.0, 4.0)] + [
            {"movement": "push_in", "audio": "silence", "duration": 5.0}]
        measures = self.sr.reference_measures(reference, shots)
        self.assertEqual(measures["static_share"], 0.75)
        self.assertEqual(measures["line_sec"], [3.0, 4.0])
        self.assertNotIn("silent_sec", measures)
        self.assertEqual(measures["weight"], 4)
        self.assertEqual(self.sr.reference_measures({"id": "empty"}), {})


if __name__ == "__main__":
    unittest.main()
