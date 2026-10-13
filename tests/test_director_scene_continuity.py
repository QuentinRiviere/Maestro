"""Storyboard images keep a scene's set, wardrobe and blocking from shot to shot."""
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
from services.director import casting  # noqa: E402
from services.director.planners.custom import CustomSkillPlanner  # noqa: E402


class SceneContinuityImageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        saved = (pipeline._pipelines, pipeline._wgp)
        self.addCleanup(lambda: setattr(pipeline, "_pipelines", saved[0]) or setattr(pipeline, "_wgp", saved[1]))
        pipeline._pipelines = {}
        pipeline._wgp = SimpleNamespace(save_path=self.tmp.name)
        self.pid = "pipe-scenes"
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

    def test_later_shots_of_a_scene_are_edited_from_its_master(self):
        anchor, girl, hall, roof = (self._file(n) for n in ("anchor.jpg", "girl.jpg", "hall.jpg", "roof.jpg"))
        params = {
            "reference_image_path": anchor,
            "character_ref_paths": [girl],
            "location_ref_paths": [hall, roof],
            "location_ref_labels": ["Hallway", "Rooftop"],
        }
        plans = [
            {"image_prompt": "wide of the hallway", "_director_continuity_group": "meet",
             "_director_location": "Hallway"},
            {"image_prompt": "close-up on her hands", "_director_continuity_group": "meet",
             "_director_location": "Hallway", "_director_opening_blocking": "she stands screen left by the lockers"},
            {"image_prompt": "reaction of the boy", "_director_continuity_group": "meet",
             "_director_location": "Hallway"},
            {"image_prompt": "wide of the rooftop at dusk", "_director_continuity_group": "after",
             "_director_location": "Rooftop"},
        ]
        names = iter(["shot1.jpg", "shot2.jpg", "shot3.jpg", "shot4.jpg"])
        requests = []

        def submit(gen_params, **_kwargs):
            requests.append({"prompt": gen_params["prompt"],
                             "refs": [os.path.basename(r) for r in gen_params["image_refs"]]})
            name = next(names)
            self._file(name)
            return [name]

        with patch.object(pipeline, "_submit_and_wait", side_effect=submit):
            images, _ = pipeline._run_image_generation(self.pid, params, plans, out_dir=self.tmp.name)

        self.assertEqual(images, ["shot1.jpg", "shot2.jpg", "shot3.jpg", "shot4.jpg"])
        # First shot of the scene: the anchor, its own set only, the cast.
        self.assertEqual(requests[0]["refs"], ["anchor.jpg", "hall.jpg", "girl.jpg"])
        # Later shots: the scene master first, then the shot before, then the rest.
        self.assertEqual(requests[1]["refs"], ["shot1.jpg", "anchor.jpg", "girl.jpg"])
        self.assertTrue(requests[1]["prompt"].startswith(pipeline.SCENE_CONTINUITY_PREFIX))
        self.assertIn("screen left by the lockers", requests[1]["prompt"])
        self.assertEqual(requests[2]["refs"], ["shot1.jpg", "shot2.jpg", "anchor.jpg", "girl.jpg"])
        # A new scene starts from its own set, not from the first scene's anchor.
        self.assertEqual(requests[3]["refs"], ["roof.jpg", "girl.jpg"])
        self.assertNotIn(pipeline.SCENE_CONTINUITY_PREFIX, requests[3]["prompt"])


class ImageRefProfileTests(unittest.TestCase):
    def test_qwen_gets_three_refs_named_pictures(self):
        master, refs = pipeline.scene_refs("qwen_image_edit_2511_20B_fp8_lightning_8step", "m.jpg",
                                           ["girl.jpg", "boy.jpg"], ["prev.jpg", "anchor.jpg"])
        self.assertEqual((master, refs), ("m.jpg", ["girl.jpg", "boy.jpg"]))
        self.assertIn("Picture 1", pipeline.scene_continuity_prefix("qwen_image_edit_plus_20B"))

    def test_seedream_keeps_the_previous_shot_when_room_is_left(self):
        _, refs = pipeline.scene_refs("cloud_fal_seedream_4", "m.jpg", ["girl.jpg", "boy.jpg"],
                                      ["prev.jpg", "anchor.jpg"])
        self.assertEqual(refs, ["prev.jpg", "girl.jpg", "boy.jpg"])

    def test_flux_keeps_every_reference(self):
        _, refs = pipeline.scene_refs("flux2_klein_9b", "m.jpg", ["girl.jpg"], ["prev.jpg", "anchor.jpg"])
        self.assertEqual(refs, ["prev.jpg", "anchor.jpg", "girl.jpg"])
        self.assertIn("reference image 1", pipeline.scene_continuity_prefix("flux2_klein_9b"))


class CastLocationTests(unittest.TestCase):
    def test_cast_keeps_a_short_descriptor_for_prompts(self):
        cast = casting.normalize_cast({"characters": [{
            "name": "Eun-seo", "descriptor": "the teen girl with see-through bangs",
            "description": "About 17, Korean, oval face, long black hair with see-through bangs."}]})
        self.assertEqual(casting.planner_characters(cast)[0]["descriptor"], "the teen girl with see-through bangs")

    def test_scenes_are_set_in_cast_locations(self):
        planner = CustomSkillPlanner.__new__(CustomSkillPlanner)
        planner.spec = {"style_hint": ""}
        prompts = []

        def call(**kwargs):
            prompts.append(kwargs["user_prompt"])
            return [{"scene_id": "meet", "location": "Hallway", "place": "school hallway"},
                    {"scene_id": "dream", "location": "Moon base", "place": "a moon base"}]
        planner._call_llm_json = call
        scenes = planner._design_scenes("brief", "", "", [], [{"name": "Hallway", "description": "lockers"}])
        self.assertIn("Cast locations:\n- Hallway: lockers", prompts[0])
        self.assertEqual([s["location"] for s in scenes], ["Hallway", ""])


if __name__ == "__main__":
    unittest.main()
