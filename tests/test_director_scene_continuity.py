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
from services.director.blocking_map import (  # noqa: E402
    framing_line, parse_blocking_map, render_blocking_map, subjects_in_view,
)
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

    def _hallway_scene(self, image_model="flux2_klein_9b"):
        anchor, girl, hall, roof = (self._file(n) for n in ("anchor.jpg", "girl.jpg", "hall.jpg", "roof.jpg"))
        params = {
            "reference_image_path": anchor,
            "character_ref_paths": [girl],
            "location_ref_paths": [hall, roof],
            "location_ref_labels": ["Hallway", "Rooftop"],
            "image_model": image_model,
        }
        plans = [
            {"image_prompt": "wide of the hallway", "_director_continuity_group": "meet",
             "_director_location": "Hallway", "_director_camera_framing": "wide establishing shot"},
            {"image_prompt": "close-up on her hands", "_director_continuity_group": "meet",
             "_director_location": "Hallway", "_director_camera_framing": "close-up",
             "_director_opening_blocking": "she stands screen left by the lockers"},
            {"image_prompt": "reaction of the boy", "_director_continuity_group": "meet",
             "_director_location": "Hallway", "_director_camera_framing": "medium shot"},
            {"image_prompt": "wide of the rooftop at dusk", "_director_continuity_group": "after",
             "_director_location": "Rooftop", "_director_camera_framing": "wide"},
        ]
        return params, plans

    def _run(self, params, plans, names):
        names = iter(names)
        requests = []

        def submit(gen_params, **_kwargs):
            requests.append({"prompt": gen_params["prompt"], "type": gen_params["video_prompt_type"],
                             "refs": [os.path.basename(r) for r in gen_params["image_refs"]]})
            name = next(names)
            self._file(name)
            return [name]

        with patch.object(pipeline, "_submit_and_wait", side_effect=submit):
            images, _ = pipeline._run_image_generation(self.pid, params, plans, out_dir=self.tmp.name)
        return images, requests

    def test_later_shots_of_a_scene_take_the_people_first_and_never_the_master_as_canvas(self):
        params, plans = self._hallway_scene()
        images, requests = self._run(params, plans, ["shot1.jpg", "shot2.jpg", "shot3.jpg", "shot4.jpg"])

        self.assertEqual(images, ["shot1.jpg", "shot2.jpg", "shot3.jpg", "shot4.jpg"])
        # The establishing shot is unchanged: the anchor as canvas, its own set, the cast.
        self.assertEqual(requests[0]["refs"], ["anchor.jpg", "hall.jpg", "girl.jpg"])
        self.assertEqual(requests[0]["type"], "KI")
        self.assertNotIn(pipeline.SCENE_SHOT_PREFIX, requests[0]["prompt"])
        # A close-up: the character sheet only, the set in words, no canvas.
        close = requests[1]
        self.assertEqual(close["refs"], ["girl.jpg"])
        self.assertEqual(close["type"], "I")
        self.assertTrue(close["prompt"].startswith(pipeline.SCENE_SHOT_PREFIX))
        self.assertIn("reference image 1 shows who and what is in this shot", close["prompt"])
        self.assertIn("Setting: Hallway.", close["prompt"])
        self.assertIn("screen left by the lockers", close["prompt"])
        self.assertNotIn("set's surfaces", close["prompt"])
        # A medium shot: the people first, the master after them for the set only.
        medium = requests[2]
        self.assertEqual(medium["refs"], ["girl.jpg", "shot1.jpg"])
        self.assertEqual(medium["type"], "I")
        self.assertIn("reference image 2 only shows the set's surfaces and light; do not copy its framing "
                      "or camera position", medium["prompt"])
        # A new scene starts from its own set, not from the first scene's anchor.
        self.assertEqual(requests[3]["refs"], ["roof.jpg", "girl.jpg"])
        self.assertEqual(requests[3]["type"], "KI")
        self.assertNotIn(pipeline.SCENE_SHOT_PREFIX, requests[3]["prompt"])

    def test_an_insert_drops_the_master_whatever_its_framing_text(self):
        params, plans = self._hallway_scene()
        plans[2] = {**plans[2], "_director_scene_type": "insert"}
        _, requests = self._run(params, plans[:3], ["shot1.jpg", "shot2.jpg", "shot3.jpg"])
        self.assertEqual(requests[2]["refs"], ["girl.jpg"])

    def test_a_rerun_rebuilds_the_references_of_the_run(self):
        params, plans = self._hallway_scene("qwen_image_edit_2511_20B_fp8_lightning_8step")
        plans[2]["_director_blocking_map"] = MEDIUM
        images, run = self._run(params, plans, ["shot1.jpg", "shot2.jpg", "shot3.jpg", "shot4.jpg"])
        record = pipeline._pipelines[self.pid]
        record.update(params=params, clip_plans=plans, clip_images=images)
        self.assertTrue(pipeline._save_pipeline_state(self.pid))

        reruns = []

        def submit(gen_params, **_kwargs):
            reruns.append({"prompt": gen_params["prompt"], "type": gen_params["video_prompt_type"],
                           "refs": [os.path.basename(r) for r in gen_params["image_refs"]]})
            name = f"rerun{len(reruns)}.jpg"
            self._file(name)
            return [name]

        with patch.object(pipeline, "_submit_and_wait", side_effect=submit):
            # Last shot first: rerunning the master would change the later shots' set.
            for index in reversed(range(4)):
                pipeline._rerun_clip_image_impl(self.tmp.name, self.pid, index)
        reruns.reverse()

        # Same set, people, diagram, roles and canvas mode as the run, shot by shot.
        self.assertEqual(reruns, run)
        self.assertEqual(reruns[2]["refs"], ["girl.jpg", "shot1.jpg", f"{self.pid}_layout_shot03.png"])
        self.assertIn("Picture 3 is a top-view layout diagram", reruns[2]["prompt"])
        self.assertEqual(reruns[0]["refs"], ["anchor.jpg", "hall.jpg", "girl.jpg"])

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
        self.assertTrue(prompts[1].startswith(pipeline.SCENE_SHOT_PREFIX))


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
            requests.append({"prompt": gen_params["prompt"], "refs": list(gen_params["image_refs"]),
                             "type": gen_params["video_prompt_type"]})
            name = next(names)
            self._file(name)
            return [name]

        with patch.object(pipeline, "_submit_and_wait", side_effect=submit):
            pipeline._run_image_generation(self.pid, params, plans, out_dir=self.tmp.name)
        return requests

    def test_a_close_up_gets_the_people_and_the_diagram_but_not_the_master(self):
        requests = self._two_shot_scene("qwen_image_edit_plus_20B", CLOSE_UP)
        shot = requests[1]
        self.assertEqual([os.path.basename(r) for r in shot["refs"][:1]], ["girl.jpg"])
        self.assertEqual(len(shot["refs"]), 2)
        layout = shot["refs"][1]
        self.assertTrue(layout.endswith("_layout_shot02.png") and os.path.isfile(layout))
        self.assertEqual(os.path.dirname(layout), self.tmp.name)
        self.assertEqual(shot["type"], "I")
        self.assertTrue(shot["prompt"].startswith(pipeline.SCENE_SHOT_PREFIX))
        self.assertIn("Picture 1 shows who and what is in this shot", shot["prompt"])
        self.assertIn("Picture 2 is a top-view layout diagram", shot["prompt"])
        self.assertIn("do not draw it", shot["prompt"])
        self.assertNotIn("shot1", " ".join(shot["refs"]))
        # The master itself is drawn from the photos alone.
        self.assertEqual(len(requests[0]["refs"]), 2)
        self.assertEqual(requests[0]["type"], "KI")

    def test_a_medium_shot_keeps_the_master_after_the_people_with_its_role(self):
        requests = self._two_shot_scene("qwen_image_edit_plus_20B", MEDIUM)
        shot = requests[1]
        # Qwen takes three refs: the person, the master for the set, the diagram last.
        self.assertEqual([os.path.basename(r) for r in shot["refs"][:2]], ["girl.jpg", "shot1.jpg"])
        self.assertTrue(shot["refs"][2].endswith("_layout_shot02.png"))
        self.assertIn("Picture 2 only shows the set's surfaces and light; do not copy its framing or camera "
                      "position", shot["prompt"])
        self.assertIn("Picture 3 is a top-view layout diagram", shot["prompt"])
        self.assertEqual(shot["type"], "I")

    def test_an_editor_that_reads_no_image_gets_the_layout_as_text(self):
        with patch.object(pipeline, "director_image_reads_refs", return_value=False):
            requests = self._two_shot_scene("qwen_image_edit_plus_20B", CLOSE_UP)
        shot = requests[1]
        self.assertFalse(any(r.endswith(".png") for r in shot["refs"]))
        self.assertIn("Framing: close-up shot. In frame, left to right: Mina (facing camera), Joon", shot["prompt"])
        self.assertNotIn("layout diagram", shot["prompt"])
        self.assertTrue(shot["prompt"].startswith(pipeline.SCENE_SHOT_PREFIX))

    def test_a_bad_blocking_map_changes_nothing(self):
        requests = self._two_shot_scene("flux2_klein_9b", {"camera": {"x": "near"}, "subjects": []})
        self.assertFalse(any(r.endswith(".png") for r in requests[1]["refs"]))
        self.assertTrue(requests[1]["prompt"].startswith(pipeline.SCENE_SHOT_PREFIX))
        self.assertNotIn("Framing:", requests[1]["prompt"])

# Camera at the front facing the back wall, Mina screen left facing the camera,
# Joon screen right turned toward her.
CLOSE_UP = {"camera": {"x": 0.45, "y": 0.7, "facing_deg": 0, "framing": "close-up"},
            "subjects": [{"name": "Mina", "x": 0.4, "y": 0.4, "facing_deg": 180},
                         {"name": "Joon", "x": 0.6, "y": 0.4, "facing_deg": 270}]}
MEDIUM = {**CLOSE_UP, "camera": {**CLOSE_UP["camera"], "framing": "medium"}}

class BlockingMapTests(unittest.TestCase):
    def test_parse_normalizes_and_rejects_garbage(self):
        parsed = parse_blocking_map({"camera": {"x": 1.1, "y": "0.9", "facing_deg": -90},
                                     "subjects": [{"name": "Mina", "x": 0.2, "y": 0.3, "facing_deg": "left"},
                                                  {"name": "", "x": 0.5, "y": 0.5},
                                                  {"name": "Ghost", "x": 9, "y": 0.5}]},
                                    fallback_framing="Medium close-up")
        self.assertEqual(parsed["camera"]["framing"], "close-up")
        self.assertEqual(parsed["subjects"], [{"name": "Mina", "x": 0.2, "y": 0.3, "facing_deg": None}])
        # Facing 270 from there would miss Mina: the camera turns to her.
        self.assertTrue(subjects_in_view(parsed))
        for bad in (None, "map", {"camera": {"x": 0.5, "y": 0.5}}, {"camera": {"x": 0.5, "y": 0.5, "facing_deg": 0}},
                    {"camera": {"x": 0.5, "y": 0.5, "facing_deg": 0}, "subjects": [{"name": "A"}]}):
            self.assertIsNone(parse_blocking_map(bad))

    def test_a_close_up_left_at_the_example_facing_is_aimed_at_its_subject(self):
        raw = {"camera": {"x": 0.5, "y": 0.95, "facing_deg": 0, "framing": "close-up"},
               "subjects": [{"name": "Mina", "x": 0.15, "y": 0.5, "facing_deg": 90}]}
        self.assertFalse(subjects_in_view({"camera": dict(raw["camera"]), "subjects": raw["subjects"]}))
        parsed = parse_blocking_map(raw)
        self.assertTrue(subjects_in_view(parsed))
        camera = parsed["camera"]
        # Close enough for a close-up, on the line from where it stood.
        self.assertLessEqual(((camera["x"] - 0.15) ** 2 + (camera["y"] - 0.5) ** 2) ** 0.5, 0.31)
        self.assertGreater(camera["y"], 0.5)
        self.assertEqual(framing_line(parsed), "Framing: close-up shot. In frame: Mina (facing frame right).")
        # Parsing a saved map again keeps it.
        self.assertEqual(parse_blocking_map(parsed), parsed)

    def test_a_camera_that_already_frames_everyone_is_kept(self):
        raw = {"camera": {"x": 0.5, "y": 0.95, "facing_deg": 0, "framing": "wide"},
               "subjects": [{"name": "Mina", "x": 0.4, "y": 0.4}, {"name": "Joon", "x": 0.6, "y": 0.45}]}
        parsed = parse_blocking_map(raw)
        self.assertEqual(parsed["camera"], {"x": 0.5, "y": 0.95, "facing_deg": 0.0, "framing": "wide"})

    def test_people_far_apart_in_a_tight_shot_push_the_camera_back(self):
        raw = {"camera": {"x": 0.5, "y": 0.6, "facing_deg": 0, "framing": "close-up"},
               "subjects": [{"name": "Mina", "x": 0.3, "y": 0.4}, {"name": "Joon", "x": 0.7, "y": 0.4}]}
        parsed = parse_blocking_map(raw)
        self.assertTrue(subjects_in_view(parsed))
        self.assertAlmostEqual(parsed["camera"]["x"], 0.5, places=3)
        self.assertGreater(parsed["camera"]["y"], 1.0)
        self.assertAlmostEqual(parsed["camera"]["facing_deg"], 0.0, places=1)

    def test_a_camera_without_a_position_stands_in_front_of_its_subjects(self):
        parsed = parse_blocking_map({"camera": {"x": "near", "facing_deg": "up", "framing": "medium"},
                                     "subjects": [{"name": "Mina", "x": 0.5, "y": 0.3}]})
        self.assertTrue(subjects_in_view(parsed))
        self.assertEqual((parsed["camera"]["x"], parsed["camera"]["facing_deg"]), (0.5, 0.0))
        self.assertGreater(parsed["camera"]["y"], 0.3)

    def test_framed_subjects_always_end_up_in_the_wedge(self):
        import random

        rng = random.Random(7)
        for _ in range(2000):
            raw = {"camera": {"x": rng.uniform(-0.5, 1.5), "y": rng.uniform(-0.5, 1.5),
                              "facing_deg": rng.choice([0, rng.uniform(0, 360)]),
                              "framing": rng.choice(["wide", "medium", "close-up", "insert"])},
                   "subjects": [{"name": f"S{n}", "x": rng.random(), "y": rng.random()}
                                for n in range(rng.randint(1, 4))]}
            parsed = parse_blocking_map(raw)
            self.assertTrue(subjects_in_view(parsed), raw)
            self.assertTrue(subjects_in_view(parse_blocking_map(parsed)), raw)

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

    def test_qwen_keeps_one_person_and_the_master_beside_the_diagram(self):
        plan = {"_director_camera_framing": "medium shot", "_director_location": "Hallway"}
        refs, prompt = pipeline.scene_shot_refs("qwen_image_edit_2511_20B_fp8_lightning_8step", plan, "two-shot",
                                                master="m.jpg", identity=["girl.jpg", "boy.jpg"], layout=True)
        self.assertEqual(refs, ["girl.jpg", "m.jpg"])
        self.assertIn("Picture 2 only shows the set's surfaces and light", prompt)
        self.assertTrue(prompt.endswith("The camera setup of this shot is: two-shot"))

    def test_a_close_up_spends_the_master_slot_on_people(self):
        plan = {"_director_camera_framing": "medium close-up", "_director_environment": "a dim school hallway"}
        refs, prompt = pipeline.scene_shot_refs("qwen_image_edit_2511_20B_fp8_lightning_8step", plan, "her face",
                                                master="m.jpg", identity=["girl.jpg", "boy.jpg"], layout=True)
        self.assertEqual(refs, ["girl.jpg", "boy.jpg"])
        self.assertIn("Picture 1, Picture 2 show who and what is in this shot", prompt)
        self.assertIn("Setting: a dim school hallway.", prompt)
        self.assertNotIn("surfaces and light", prompt)

    def test_seedream_takes_every_person_then_the_master(self):
        refs, prompt = pipeline.scene_shot_refs("cloud_fal_seedream_4", {"_director_camera_framing": "wide"}, "x",
                                                master="m.jpg", identity=["girl.jpg", "boy.jpg"], layout=True)
        self.assertEqual(refs, ["girl.jpg", "boy.jpg", "m.jpg"])
        self.assertIn("reference image 3 only shows the set's surfaces", prompt)

    def test_the_blocking_framing_marks_a_tight_shot_without_camera_text(self):
        self.assertTrue(pipeline.shot_is_tight({"_director_blocking_map": CLOSE_UP}))
        self.assertTrue(pipeline.shot_is_tight({"_director_camera_framing": "medium", "_director_scene_type": "insert"}))
        self.assertFalse(pipeline.shot_is_tight({"_director_camera_framing": "over-the-shoulder"}))

    def test_a_shot_without_canvas_reads_every_ref_as_a_reference(self):
        with patch.object(pipeline, "_director_image_prompt_type", return_value="KI"):
            self.assertEqual(pipeline.director_shot_prompt_type("qwen_image_edit_plus_20B", ["a"], canvas=False), "I")
            self.assertEqual(pipeline.director_shot_prompt_type("qwen_image_edit_plus_20B", ["a"]), "KI")
        with patch.object(pipeline, "_director_image_prompt_type", return_value=""):
            self.assertEqual(pipeline.director_shot_prompt_type("flux", [], canvas=False), "")


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
