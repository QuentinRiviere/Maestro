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
from services.director.blocking_map import framing_line, parse_blocking_map, render_blocking_map  # noqa: E402
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

    def test_the_bible_look_ends_every_image_prompt(self):
        anchor = self._file("anchor.jpg")
        look = "Soft pastel highlights, luminous skin."
        params = {"reference_image_path": anchor, "_director_bible_look": look}
        plans = [{"image_prompt": "wide of the hallway", "_director_continuity_group": "meet"},
                 {"image_prompt": "close-up on her hands", "_director_continuity_group": "meet"}]
        names = iter(["shot1.jpg", "shot2.jpg"])
        prompts = []

        def submit(gen_params, **_kwargs):
            prompts.append(gen_params["prompt"])
            name = next(names)
            self._file(name)
            return [name]

        with patch.object(pipeline, "_submit_and_wait", side_effect=submit):
            pipeline._run_image_generation(self.pid, params, plans, out_dir=self.tmp.name)
        self.assertTrue(all(p.endswith(look) for p in prompts))
        self.assertTrue(prompts[1].startswith(pipeline.SCENE_CONTINUITY_PREFIX))


    def _two_shot_scene(self, image_model, blocking_map):
        anchor, girl = self._file("anchor.jpg"), self._file("girl.jpg")
        params = {"reference_image_path": anchor, "character_ref_paths": [girl], "image_model": image_model}
        plans = [
            {"image_prompt": "wide of the hallway", "_director_continuity_group": "meet"},
            {"image_prompt": "close-up on her face", "_director_continuity_group": "meet",
             "_director_blocking_map": blocking_map},
        ]
        names = iter(["shot1.jpg", "shot2.jpg"])
        requests = []

        def submit(gen_params, **_kwargs):
            requests.append({"prompt": gen_params["prompt"], "refs": list(gen_params["image_refs"])})
            name = next(names)
            self._file(name)
            return [name]

        with patch.object(pipeline, "_submit_and_wait", side_effect=submit):
            pipeline._run_image_generation(self.pid, params, plans, out_dir=self.tmp.name)
        return requests

    def test_a_layout_diagram_gives_the_framing_and_the_master_only_the_look(self):
        requests = self._two_shot_scene("qwen_image_edit_plus_20B", CLOSE_UP)
        shot = requests[1]
        # Qwen takes three refs: the diagram takes the last slot, the master stays first.
        self.assertEqual([os.path.basename(r) for r in shot["refs"][:2]], ["shot1.jpg", "girl.jpg"])
        layout = shot["refs"][2]
        self.assertTrue(layout.endswith("_layout_shot02.png") and os.path.isfile(layout))
        self.assertEqual(os.path.dirname(layout), self.tmp.name)
        self.assertTrue(shot["prompt"].startswith(pipeline.scene_layout_prefix("qwen_image_edit_plus_20B")))
        self.assertIn("not its framing", shot["prompt"])
        self.assertIn("Picture 3 is a top-view layout diagram", shot["prompt"])
        self.assertIn("do not draw it", shot["prompt"])
        self.assertNotIn(pipeline.SCENE_CONTINUITY_PREFIX.replace("reference image 1", "Picture 1"), shot["prompt"])
        # The master itself is drawn from the photos alone.
        self.assertEqual(len(requests[0]["refs"]), 2)

    def test_an_editor_that_reads_no_image_gets_the_layout_as_text(self):
        with patch.object(pipeline, "director_image_reads_refs", return_value=False):
            requests = self._two_shot_scene("qwen_image_edit_plus_20B", CLOSE_UP)
        shot = requests[1]
        self.assertFalse(any(r.endswith(".png") for r in shot["refs"]))
        self.assertIn("Framing: close-up shot. In frame, left to right: Mina (facing camera), Joon", shot["prompt"])
        self.assertNotIn("layout diagram", shot["prompt"])
        self.assertTrue(shot["prompt"].startswith(pipeline.SCENE_CONTINUITY_PREFIX.replace(
            "reference image 1", "Picture 1")))

    def test_a_bad_blocking_map_changes_nothing(self):
        requests = self._two_shot_scene("flux2_klein_9b", {"camera": {"x": "near"}, "subjects": []})
        self.assertFalse(any(r.endswith(".png") for r in requests[1]["refs"]))
        self.assertTrue(requests[1]["prompt"].startswith(pipeline.SCENE_CONTINUITY_PREFIX))
        self.assertNotIn("Framing:", requests[1]["prompt"])

# Camera at the front facing the back wall, Mina screen left facing the camera,
# Joon screen right turned toward her.
CLOSE_UP = {"camera": {"x": 0.45, "y": 0.7, "facing_deg": 0, "framing": "close-up"},
            "subjects": [{"name": "Mina", "x": 0.4, "y": 0.4, "facing_deg": 180},
                         {"name": "Joon", "x": 0.6, "y": 0.4, "facing_deg": 270}]}

class BlockingMapTests(unittest.TestCase):
    def test_parse_normalizes_and_rejects_garbage(self):
        parsed = parse_blocking_map({"camera": {"x": 1.1, "y": "0.9", "facing_deg": -90},
                                     "subjects": [{"name": "Mina", "x": 0.2, "y": 0.3, "facing_deg": "left"},
                                                  {"name": "", "x": 0.5, "y": 0.5},
                                                  {"name": "Ghost", "x": 9, "y": 0.5}]},
                                    fallback_framing="Medium close-up")
        self.assertEqual(parsed["camera"], {"x": 1.0, "y": 0.9, "facing_deg": 270.0, "framing": "close-up"})
        self.assertEqual(parsed["subjects"], [{"name": "Mina", "x": 0.2, "y": 0.3, "facing_deg": None}])
        for bad in (None, "map", {"camera": {"x": 0.5, "y": 0.5}}, {"camera": {"x": 0.5, "y": 0.5, "facing_deg": 0}},
                    {"camera": {"x": 0.5, "y": 0.5, "facing_deg": 0}, "subjects": [{"name": "A"}]}):
            self.assertIsNone(parse_blocking_map(bad))

    def test_framing_line_reads_left_and_right_from_the_camera(self):
        parsed = parse_blocking_map(CLOSE_UP)
        self.assertEqual(framing_line(parsed),
                         "Framing: close-up shot. In frame, left to right: Mina (facing camera), "
                         "Joon (facing frame left).")
        # The reverse angle swaps the screen sides.
        reverse = parse_blocking_map({**CLOSE_UP, "camera": {"x": 0.5, "y": 0.1, "facing_deg": 180,
                                                              "framing": "wide"}})
        self.assertTrue(framing_line(reverse).startswith("Framing: wide shot. In frame, left to right: Joon"))
        self.assertIn("Mina (back to camera)", framing_line(reverse))

    def test_render_writes_a_png_diagram(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            path = render_blocking_map(parse_blocking_map(CLOSE_UP), os.path.join(folder, "layout.png"), size=512)
            self.assertTrue(os.path.getsize(path) > 1000)
            with Image.open(path) as image:
                self.assertEqual((image.format, image.size), ("PNG", (512, 512)))
                # Not a blank page: the floor, camera and people are drawn.
                self.assertGreater(len(image.convert("L").getcolors(512 * 512)), 4)

class ImageRefProfileTests(unittest.TestCase):
    def test_layout_needs_a_slot_beside_the_source(self):
        self.assertTrue(pipeline.layout_diagram_fits("qwen_image_edit_plus_20B"))
        with patch.object(pipeline, "director_image_ref_budget", return_value=1):
            self.assertFalse(pipeline.layout_diagram_fits("qwen_image_edit_plus_20B"))
        self.assertIn("reference image 4 is a top-view layout diagram",
                      pipeline.layout_role_text("cloud_fal_flux_2_edit", 4))

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
