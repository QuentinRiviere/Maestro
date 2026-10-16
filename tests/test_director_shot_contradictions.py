"""Shot prompts must not contradict their framing: inserts without faces, face shots without backs."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director.blocking_map import (  # noqa: E402
    face_the_camera, faces_wanted, ots_foreground, parse_blocking_map, staging_line, subject_view,
)
from services.director.policies import resolve_subjects_text  # noqa: E402
from services.director_pipeline import (  # noqa: E402
    director_is_insert, director_shot_layout, scene_locked_plan, shot_identity_refs,
)

# Shot 16 of run fcd1e37f: the camera stands behind the only person of a medium close-up.
LONE_BACK = {"camera": {"x": 0.5, "y": 0.85, "facing_deg": 0.0, "framing": "medium"},
             "subjects": [{"name": "Min Do-hyeon", "x": 0.5, "y": 0.75, "facing_deg": 0.0}]}
# Shot 18: "both faces" through the glass, with the nearest person's back to the camera.
BOTH_FACES = {"camera": {"x": 0.5, "y": 1.2468, "facing_deg": 0.0, "framing": "medium"},
              "subjects": [{"name": "Cha Hae-won", "x": 0.35, "y": 0.4, "facing_deg": 90.0},
                           {"name": "Min Do-hyeon", "x": 0.65, "y": 0.75, "facing_deg": 0.0}]}


class InsertTest(unittest.TestCase):
    def test_unresolved_person_in_an_insert_loses_the_face(self):
        shot = SimpleNamespace(scene_type="", camera_plan=SimpleNamespace(framing="close-up insert"),
                               subjects_on_screen=[SimpleNamespace(
                                   visual_description="22 years old, oval face, almond eyes", character_id=None,
                                   position_or_relation="Hae-won's trembling hands hold the mug")])
        text = resolve_subjects_text(shot, None)
        self.assertNotIn("almond eyes", text)
        self.assertTrue(text.startswith("hands, Hae-won's trembling hands"))

    def test_insert_by_framing_drops_portraits_and_identity_line(self):
        plan = {"_director_camera_framing": "close-up insert", "_director_scene_type": "dialogue"}
        self.assertTrue(director_is_insert(plan))
        params = {"character_ref_labels": ["Cha Hae-won"], "character_ref_paths": ["/b/hae.png"]}
        self.assertEqual(shot_identity_refs(params, plan, ["/b/hae.png", "/b/cafe.png"]), ["/b/cafe.png"])
        _path, prompt = director_shot_layout("p", 0, plan, "Hands on a mug. Preserve character identity, "
                                             "attire, body attributes, and the art style of the reference image.",
                                             "qwen_image_21_7B", "/tmp")
        self.assertNotIn("Preserve character identity", prompt)
        self.assertIn("Only the hands", prompt)


class FaceShotTest(unittest.TestCase):
    def test_faces_wanted(self):
        self.assertTrue(faces_wanted("medium shot capturing both faces on either side of the glass"))
        self.assertTrue(faces_wanted("reaction close-up"))
        self.assertFalse(faces_wanted("medium close-up"))
        self.assertFalse(faces_wanted("close-up from behind, her face hidden"))
        self.assertFalse(faces_wanted("over-the-shoulder on his face"))

    def test_lone_close_shot_camera_moves_in_front(self):
        blocking = parse_blocking_map(LONE_BACK)
        faced = face_the_camera(blocking, "medium close-up")
        self.assertIsNot(faced, blocking)
        self.assertIn("(facing camera)", staging_line(faced, "medium close-up"))
        self.assertEqual(subject_view(faced, "Min Do-hyeon"), "face_front")

    def test_back_view_and_wide_shots_keep_their_camera(self):
        blocking = parse_blocking_map(LONE_BACK)
        self.assertIs(face_the_camera(blocking, "close-up from behind as he walks away"), blocking)
        self.assertIs(face_the_camera(blocking, "medium shot"), blocking)

    def test_scene_locked_plan_faces_the_camera(self):
        plan = scene_locked_plan({"_director_blocking_map": LONE_BACK,
                                  "_director_camera_framing": "medium close-up"}, None)
        self.assertTrue(plan["_director_camera_faced"])
        self.assertIn("(facing camera)", staging_line(parse_blocking_map(plan["_director_blocking_map"]),
                                                       "medium close-up"))

    def test_both_faces_is_not_an_over_the_shoulder(self):
        framing = "medium shot capturing both faces on either side of the rain-streaked glass pane"
        blocking = parse_blocking_map(BOTH_FACES)
        self.assertEqual(ots_foreground(blocking, framing), "")
        self.assertEqual(ots_foreground(blocking, "medium shot"), "Min Do-hyeon")
        line = staging_line(blocking, framing)
        self.assertNotIn("seen only from behind", line)
        self.assertNotIn("back to camera", line)
        self.assertIn("face visible", line)
        self.assertEqual(subject_view(blocking, "Min Do-hyeon", faces=True), "face_three_quarter")


if __name__ == "__main__":
    unittest.main(verbosity=2)
