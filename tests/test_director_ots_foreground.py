"""Over-the-shoulder shots: the foreground person gets no reference and is drawn once, as a blurred shoulder."""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402

MAN = {"name": "Man", "x": 0.5, "y": 0.3, "facing_deg": 180}
WOMAN = {"name": "Woman", "x": 0.5, "y": 0.7, "facing_deg": 0}
# Over the man's shoulder toward the woman.
OVER_MAN = {"camera": {"x": 0.6, "y": 0.1, "facing_deg": 170, "framing": "medium"},
            "subjects": [MAN, WOMAN]}
CAST = {"characters": [
    {"name": "Jin-woo", "description": "a man in his thirties, short black hair",
     "wardrobe": "a charcoal wool overcoat over a white shirt, a silver pendant, black leather shoes"},
    {"name": "Hae-won", "description": "a woman in her twenties", "wardrobe": "a red knit sweater"},
]}
# The planner's words only (no blocking map): Jin-woo's back fills the left foreground.
OTS_ON_HAE_WON = {
    "_director_camera_framing": "medium close-up over-the-shoulder on Hae-won",
    "_director_subjects_on_screen": [
        {"speaker_name": "Jin-woo", "visual_description": "a man",
         "position_or_relation": "foreground left, back to camera"},
        {"speaker_name": "Hae-won", "visual_description": "a woman", "position_or_relation": "facing camera"},
    ],
}


class OtsReferenceTests(unittest.TestCase):
    PARAMS = {"character_ref_labels": ["Jin-woo", "Hae-won"],
              "character_ref_paths": ["/p/jinwoo.png", "/p/haewon.png"]}

    def test_only_the_person_facing_the_camera_keeps_a_reference(self):
        views = {"/p/jinwoo.png": {"face_front": "/p/jinwoo_face.png", "back": "/p/jinwoo_back.png"}}
        params = {**self.PARAMS, "_director_image_views": views}
        refs = pipeline.shot_identity_refs(params, OTS_ON_HAE_WON, ["/p/jinwoo.png", "/p/haewon.png"])
        self.assertEqual(refs, ["/p/haewon.png"])
        self.assertEqual(pipeline.without_ots_front_refs(params, OTS_ON_HAE_WON,
                                                         ["/p/jinwoo.png", "/p/haewon.png", "/p/cafe.png"]),
                         ["/p/haewon.png", "/p/cafe.png"])

    def test_a_two_shot_keeps_both_people(self):
        plan = {**OTS_ON_HAE_WON, "_director_camera_framing": "medium two-shot"}
        self.assertEqual(pipeline.shot_identity_refs(self.PARAMS, plan, ["/p/jinwoo.png", "/p/haewon.png"]),
                         ["/p/jinwoo.png", "/p/haewon.png"])


class OtsPromptTests(unittest.TestCase):
    def test_the_foreground_person_is_a_blurred_shoulder_known_by_their_clothes(self):
        prompt = pipeline.director_ots_prompt(OTS_ON_HAE_WON, "Hae-won listens.", CAST)
        self.assertTrue(prompt.startswith("Hae-won listens. "))
        self.assertIn("Jin-woo appears only once in this shot, as that blurred foreground shape", prompt)
        self.assertIn("dark, out-of-focus shoulder and the back of the head at the left edge of the frame",
                      prompt)
        self.assertIn("recognisable only by their clothes (a charcoal wool overcoat over a white shirt)", prompt)
        # Front-only items and what the framing cannot see stay out.
        self.assertNotIn("pendant", prompt)
        self.assertNotIn("shoes", prompt)
        self.assertIn("Jin-woo's face is never shown", prompt)
        self.assertIn("neither seated nor facing the camera", prompt)
        # Applied once.
        self.assertEqual(pipeline.director_ots_prompt(OTS_ON_HAE_WON, prompt, CAST), prompt)

    def test_the_side_follows_the_planner_words_or_the_blocking_map(self):
        right = {**OTS_ON_HAE_WON, "_director_subjects_on_screen": [
            {**OTS_ON_HAE_WON["_director_subjects_on_screen"][0], "position_or_relation": "foreground right"},
            OTS_ON_HAE_WON["_director_subjects_on_screen"][1]]}
        self.assertIn("at the right edge of the frame", pipeline.director_ots_prompt(right, "Shot.", CAST))
        mapped = pipeline.director_ots_prompt({"_director_blocking_map": OVER_MAN}, "Shot.", None)
        self.assertIn("Man appears only once", mapped)
        self.assertIn("at the right edge of the frame", mapped)
        self.assertIn("recognisable only by the clothes on that shoulder", mapped)

    def test_other_shots_are_unchanged(self):
        plan = {**OTS_ON_HAE_WON, "_director_camera_framing": "medium two-shot"}
        self.assertEqual(pipeline.director_ots_prompt(plan, "Both talk.", CAST), "Both talk.")
        self.assertEqual(pipeline.director_ots_prompt({}, "Alone.", CAST), "Alone.")


class OtsImageJobTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        saved = (pipeline._pipelines, pipeline._wgp)
        self.addCleanup(lambda: setattr(pipeline, "_pipelines", saved[0]) or setattr(pipeline, "_wgp", saved[1]))
        pipeline._pipelines = {}
        pipeline._wgp = SimpleNamespace(save_path=self.tmp.name)
        self.pid = "pipe-ots"
        pipeline._pipelines[self.pid] = {
            "id": self.pid, "status": "running", "phase": "generating_images", "progress": {},
            "clip_plans": [], "clip_images": [], "output_files": [], "created_at": time.time(),
            "params": {}, "pause_reason": None, "out_dir": self.tmp.name,
        }

    def _file(self, name):
        path = os.path.join(self.tmp.name, name)
        with open(path, "wb") as handle:
            handle.write(b"image")
        return path

    def test_the_start_image_job_sends_one_portrait_and_the_shoulder_in_words(self):
        anchor, jinwoo, haewon = (self._file(n) for n in ("anchor.jpg", "jinwoo.jpg", "haewon.jpg"))
        params = {"reference_image_path": anchor, "character_ref_paths": [jinwoo, haewon],
                  "character_ref_labels": ["Jin-woo", "Hae-won"], "image_model": "flux2_klein_9b",
                  "_director_cast": CAST}
        plans = [{**OTS_ON_HAE_WON, "image_prompt": "Hae-won listens to Jin-woo across the table."}]
        requests = []

        def submit(gen_params, **_kwargs):
            requests.append(gen_params)
            self._file("shot1.jpg")
            return ["shot1.jpg"]

        with patch.object(pipeline, "_submit_and_wait", side_effect=submit):
            pipeline._run_image_generation(self.pid, params, plans, out_dir=self.tmp.name)
        refs = [os.path.basename(r) for r in requests[0]["image_refs"]]
        self.assertEqual(refs, ["anchor.jpg", "haewon.jpg"])
        self.assertIn("Jin-woo appears only once in this shot", requests[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
