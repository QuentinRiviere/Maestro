"""Shot prompt and reference regressions found on bible k_love, run fcd1e37f (18 shots).

Fixtures are the run's own plans: the scene-locked blocking of shots 5, 13
and 18, and shot 10's saved (planning-time, LLM-polished) insert prompt.
"""

from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services.director.blocking_map import (  # noqa: E402
    gaze_targets, ots_foreground, parse_blocking_map, staging_line,
)
from services.director.policies import visible_wardrobe  # noqa: E402

HAE = {"name": "Cha Hae-won", "x": 0.35, "y": 0.4, "facing_deg": 90.0}
SONG_OFF = {"name": "Song Jin-woo", "x": 0.65, "y": 0.4, "facing_deg": 270.0}
# Shot 5: over Hae-won's left shoulder toward Do-hyeon (the scene lock keeps it as planned).
SHOT_5 = {"camera": {"x": 0.1813, "y": 0.6094, "facing_deg": 63.43, "framing": "medium"},
          "subjects": [{"name": "Cha Hae-won", "x": 0.35, "y": 0.45, "facing_deg": 90.0},
                       {"name": "Min Do-hyeon", "x": 0.65, "y": 0.45, "facing_deg": 270.0}]}
SHOT_5_FRAMING = "over-the-shoulder shot over Hae-won's left shoulder toward Do-hyeon"
# Shot 13: close-up of Hae-won looking at Jin-woo, who is off screen.
SHOT_13 = {"camera": {"x": 0.4, "y": 0.55, "facing_deg": 341.57, "framing": "close-up"},
           "subjects": [HAE], "offscreen": [SONG_OFF]}
# Shot 18 after the scene lock: Hae-won keeps facing Jin-woo's seat (90 degrees)
# while the shot has her hold Do-hyeon's gaze through the window.
SHOT_18 = {"camera": {"x": 0.425, "y": 1.2468, "facing_deg": 0.0, "framing": "medium"},
           "subjects": [HAE, {"name": "Min Do-hyeon", "x": 0.5, "y": 0.75, "facing_deg": 0.0}],
           "offscreen": [SONG_OFF]}
SHOT_18_FRAMING = "medium shot capturing both faces on either side of the rain-streaked glass pane"

CAST = {"characters": [
    {"name": "Cha Hae-won",
     "description": "22 years old, South Korean. Oval face with soft features, watchful dark brown almond eyes.",
     "wardrobe": "Oversized knitted cream-colored wool sweater over a crisp white cotton collared shirt, charcoal "
                 "pleated wide-leg trousers, and white canvas lace-up sneakers. Wears a slim silver chain with a "
                 "plain round disc pendant."},
    {"name": "Min Do-hyeon", "description": "23 years old, South Korean. Chiseled jawline.",
     "wardrobe": "Tailored midnight-blue wool trench coat over a black cashmere turtleneck sweater."},
    {"name": "Song Jin-woo",
     "description": "23 years old, South Korean. Round friendly face, warm crinkling dark eyes.",
     "wardrobe": "Forest green corduroy overshirt over a beige graphic t-shirt, relaxed olive cargo pants, "
                 "wire-rimmed round tortoiseshell glasses, and brown leather ankle boots."},
]}
# Shot 10 as run fcd1e37f saved it: planned before inserts dropped the face, then polished.
SHOT_10 = {
    "_director_scene_type": "action",
    "_director_camera_framing": "close-up insert",
    "_director_opening_blocking": "Hae-won's right hand shakes beside the mug; Jin-woo's hand enters "
                                  "screen-right swiftly.",
    "_director_subjects_on_screen": [
        {"visual_description": "the 22-year-old South Korean woman's right wrist and hand in cream knit sweater "
                               "sleeve", "character_id": "char_0", "position_or_relation": "screen-left over the "
                               "table", "speaker_name": "Cha Hae-won"},
        {"visual_description": "the 23-year-old South Korean man's warm-toned right hand in forest green sleeve",
         "character_id": "char_2", "position_or_relation": "screen-right reaching across the table",
         "speaker_name": "Song Jin-woo"}],
    "image_prompt": "SHOT: close-up insert. SETTING: Café Haru, small round oak table surface. SUBJECT: A "
                    "22-year-old South Korean woman with an oval face, soft features, watchful dark brown almond "
                    "eyes, positioned screen-left over the table. Her right hand shakes beside the mug as his hand "
                    "enters screen-right swiftly. MOOD: Decisive, protective intervention. Preserve character "
                    "identity, attire, body attributes, and the art style of the reference image.",
}


class InsertRerunTest(unittest.TestCase):
    """Bug 1: a rerun rebuilds an insert's SUBJECT; it does not reuse the planning-time face."""

    def test_saved_insert_prompt_gets_a_hands_only_subject(self):
        prompt = pipeline.director_insert_prompt(SHOT_10, SHOT_10["image_prompt"], CAST)
        self.assertNotIn("almond eyes", prompt)
        self.assertNotIn("oval face", prompt)
        self.assertIn("SUBJECT: hands of a person in Oversized knitted cream-colored wool sweater over a crisp "
                      "white cotton collared shirt, screen-left over the table and hands of a person in Forest "
                      "green corduroy overshirt over a beige graphic t-shirt, screen-right reaching across the "
                      "table, Hae-won's right hand shakes beside the mug; Jin-woo's hand enters screen-right "
                      "swiftly. MOOD: Decisive", prompt)
        # The setting and the mood stay as planned.
        self.assertTrue(prompt.startswith("SHOT: close-up insert. SETTING: Café Haru, small round oak table"))

    def test_pipe_separated_fields_and_free_form_prompts(self):
        plan = {**SHOT_10, "_director_subjects_on_screen": SHOT_10["_director_subjects_on_screen"][:1]}
        piped = "SHOT: insert | SUBJECT: 22 years old, almond eyes, hands on a mug | MOOD: fragile"
        self.assertEqual(pipeline.director_insert_prompt(plan, piped, CAST),
                         "SHOT: insert | SUBJECT: hands of a person in Oversized knitted cream-colored wool "
                         "sweater over a crisp white cotton collared shirt, screen-left over the table, Hae-won's "
                         "right hand shakes beside the mug; Jin-woo's hand enters screen-right swiftly. | MOOD: "
                         "fragile")
        prose = "Close-up insert on a café table. A 22-year-old woman with almond eyes and black hair. Steam rises."
        rebuilt = pipeline.director_insert_prompt(plan, prose, CAST)
        self.assertNotIn("almond", rebuilt)
        self.assertTrue(rebuilt.startswith("Close-up insert on a café table. Steam rises. SUBJECT: hands of"))

    def test_other_shots_are_untouched(self):
        plan = {**SHOT_10, "_director_scene_type": "dialogue", "_director_camera_framing": "medium close-up"}
        self.assertEqual(pipeline.director_insert_prompt(plan, SHOT_10["image_prompt"], CAST),
                         SHOT_10["image_prompt"])

    def test_rerun_sends_the_rebuilt_subject(self):
        with tempfile.TemporaryDirectory() as folder:
            state = {
                "image_model": "qwen_image_21_7B",
                "image_params": {"num_inference_steps": 6, "guidance_scale": 1.0, "sample_solver": "viggle_v021",
                                 "custom_settings": {"qwen21_kv_cache": "Enabled"}},
                "_params_snapshot": {"_director_cast": CAST},
                "clips": [copy.deepcopy(SHOT_10)],
            }
            sent = []

            def submit(params, **_kwargs):
                sent.append(params)
                raise RuntimeError("captured")

            with patch.object(pipeline, "load_pipeline_state", return_value=state), \
                    patch.object(pipeline, "_find_pipeline_file", return_value=os.path.join(folder, "p.json")), \
                    patch.object(pipeline, "_validate_director_models"), \
                    patch.object(pipeline, "_wgp", SimpleNamespace(save_path=folder)), \
                    patch.object(pipeline, "_submit_and_wait", side_effect=submit):
                with self.assertRaises(RuntimeError):
                    pipeline._rerun_clip_image_impl(folder, "fcd1e37f", 0)
        job = sent[0]
        self.assertNotIn("almond eyes", job["prompt"])
        self.assertIn("SUBJECT: hands of a person in Oversized knitted", job["prompt"])
        self.assertNotIn("Preserve character identity", job["prompt"])


class InsertWardrobeTest(unittest.TestCase):
    """Bug 2: a hands insert names what hands and forearms show, not glasses or necklaces."""

    def test_detail_reach_drops_head_and_neck_items(self):
        song = CAST["characters"][2]["wardrobe"]
        self.assertEqual(visible_wardrobe(song, "detail"),
                         "Forest green corduroy overshirt over a beige graphic t-shirt")
        hae = CAST["characters"][0]["wardrobe"]
        self.assertEqual(visible_wardrobe(hae, "detail"),
                         "Oversized knitted cream-colored wool sweater over a crisp white cotton collared shirt")

    def test_detail_reach_keeps_hand_and_wrist_items(self):
        wardrobe = ("black leather jacket, silver chain bracelet, gold wedding ring, wristwatch, "
                    "pearl earrings, red beret and a silk scarf")
        self.assertEqual(visible_wardrobe(wardrobe, "detail"),
                         "black leather jacket, silver chain bracelet, gold wedding ring, wristwatch")

    def test_upper_reach_keeps_the_necklace_of_its_own_sentence(self):
        hae = CAST["characters"][0]["wardrobe"]
        upper = visible_wardrobe(hae, "upper")
        self.assertIn("slim silver chain", upper)
        self.assertNotIn("sneakers", upper)
        self.assertNotIn("trousers", upper)


class OffscreenGazeTest(unittest.TestCase):
    """Bug 3: a person in frame who looks back wins over an off-screen one."""

    def test_shot_18_looks_at_the_person_in_frame(self):
        blocking = parse_blocking_map(SHOT_18)
        self.assertEqual(gaze_targets(blocking), {"Cha Hae-won": "Min Do-hyeon", "Min Do-hyeon": "Cha Hae-won"})
        line = staging_line(blocking, SHOT_18_FRAMING)
        self.assertIn("Cha Hae-won looks at Min Do-hyeon; Min Do-hyeon looks at Cha Hae-won.", line)
        self.assertNotIn("Song Jin-woo", line)

    def test_shot_18_as_planned_before_the_lock(self):
        planned = {**SHOT_18, "camera": {**SHOT_18["camera"], "x": 0.5},
                   "subjects": [HAE, {"name": "Min Do-hyeon", "x": 0.65, "y": 0.75, "facing_deg": 0.0}]}
        self.assertEqual(gaze_targets(parse_blocking_map(planned))["Cha Hae-won"], "Min Do-hyeon")

    def test_off_screen_eyeline_stays_when_nobody_in_frame_looks_back(self):
        # Shot 13: alone in frame, she looks toward Jin-woo off screen.
        line = staging_line(parse_blocking_map(SHOT_13), "close-up")
        self.assertIn("Cha Hae-won looks toward frame right at Song Jin-woo, who is out of frame.", line)
        # Someone in frame who does not look back does not steal an exact off-screen eyeline.
        turned = {**SHOT_18, "subjects": [HAE, {"name": "Min Do-hyeon", "x": 0.5, "y": 0.75, "facing_deg": 180.0}]}
        self.assertEqual(gaze_targets(parse_blocking_map(turned))["Cha Hae-won"], "Song Jin-woo")

    def test_in_frame_wins_inside_the_same_cone(self):
        both = {"camera": {"x": 0.5, "y": 1.0, "facing_deg": 0.0, "framing": "medium"},
                "subjects": [{"name": "A", "x": 0.3, "y": 0.5, "facing_deg": 90.0},
                             {"name": "B", "x": 0.6, "y": 0.62, "facing_deg": 90.0}],
                "offscreen": [{"name": "C", "x": 0.9, "y": 0.5, "facing_deg": 270.0}]}
        # C is dead ahead of A, B 22 degrees off: B is in frame and inside the cone.
        self.assertEqual(gaze_targets(parse_blocking_map(both))["A"], "B")


class OffscreenPortraitTest(unittest.TestCase):
    """Bug 4: a person only named as out of frame gets no reference image."""

    PARAMS = {"character_ref_labels": ["Cha Hae-won", "Min Do-hyeon", "Song Jin-woo"],
              "character_ref_paths": ["/b/hae.png", "/b/min.png", "/b/song.png"]}

    def test_shot_13_drops_the_off_screen_portrait(self):
        plan = {"_director_blocking_map": SHOT_13, "_director_camera_framing": "close-up"}
        self.assertEqual(pipeline.director_out_of_frame(plan), {"song jin-woo"})
        refs = pipeline.shot_identity_refs(self.PARAMS, plan, ["/b/hae.png", "/b/song.png", "/b/cafe.png"])
        self.assertEqual(refs, ["/b/hae.png", "/b/cafe.png"])

    def test_people_in_frame_keep_their_portraits(self):
        plan = {"_director_blocking_map": SHOT_18, "_director_camera_framing": SHOT_18_FRAMING}
        refs = pipeline.shot_identity_refs(self.PARAMS, plan, ["/b/hae.png", "/b/min.png", "/b/song.png"])
        self.assertEqual(refs, ["/b/hae.png", "/b/min.png"])
        # Without a blocking map nobody is known to be out of frame.
        self.assertEqual(pipeline.shot_identity_refs(self.PARAMS, {}, ["/b/hae.png", "/b/song.png"]),
                         ["/b/hae.png", "/b/song.png"])


class OverTheShoulderListTest(unittest.TestCase):
    """Bug 5: the OTS foreground person stays in the left-to-right list (shot 5 drew Hae-won twice)."""

    def test_shot_5_lists_both_people_once(self):
        blocking = parse_blocking_map(SHOT_5)
        self.assertEqual(ots_foreground(blocking, SHOT_5_FRAMING), "Cha Hae-won")
        line = staging_line(blocking, SHOT_5_FRAMING)
        self.assertIn("In frame, left to right: Cha Hae-won (back to camera, the out-of-focus back of the head "
                      "and shoulder in the left foreground, cropped by the frame edge, face not visible), "
                      "Min Do-hyeon (facing camera).", line)
        self.assertIn("Exactly 2 people in frame, each shown once and never duplicated. "
                      "Cha Hae-won is seen only from behind.", line)
        self.assertIn("Cha Hae-won looks at Min Do-hyeon; Min Do-hyeon looks at Cha Hae-won.", line)
        self.assertNotIn("In frame: Min Do-hyeon", line)
        self.assertNotIn("Foreground:", line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
