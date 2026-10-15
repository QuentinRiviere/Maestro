"""The 3D blockout: location layouts, shot cameras from the scene lock, the ray caster and the off-by-default hook."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services.director import bibles  # noqa: E402
from services.director.blocking_map import lock_scene_blockings, parse_blocking_map, screen_sides  # noqa: E402
from services.director.blockout import guide  # noqa: E402
from services.director.blockout.layout import (  # noqa: E402
    default_layout, merge_props, normalize_layout, suggest_layout, validate_layout,
)
from services.director.blockout.render import PALETTE, palette_table, render  # noqa: E402
from services.director.blockout.scene import (  # noqa: E402
    LABELS, build_scene, camera_path, character_label, infer_postures, parse_movement, shot_camera,
)

# The scene_lock fixture: two people face to face along the room's depth, a table between them.
MAN = {"name": "Man", "x": 0.5, "y": 0.3, "facing_deg": 180}
WOMAN = {"name": "Woman", "x": 0.5, "y": 0.7, "facing_deg": 0}
TABLE = {"name": "table", "x": 0.5, "y": 0.5}
MASTER = {"camera": {"x": 0.9, "y": 0.9, "facing_deg": 315, "framing": "wide"},
          "subjects": [MAN, WOMAN], "props": [TABLE]}
OVER_WOMAN = {"camera": {"x": 0.6, "y": 0.9, "facing_deg": 350, "framing": "medium"}, "subjects": [MAN, WOMAN]}
OVER_MAN = {"camera": {"x": 0.6, "y": 0.1, "facing_deg": 170, "framing": "medium"}, "subjects": [MAN, WOMAN]}

# Side by side across the room's width; the window is on the left wall.
LEFT = {"name": "Hae-won", "x": 0.35, "y": 0.5, "facing_deg": 90}
RIGHT = {"name": "Jin-woo", "x": 0.65, "y": 0.5, "facing_deg": 270}
WINDOW_LEFT = {"room": {"width": 6, "depth": 6, "height": 2.8},
               "openings": [{"wall": "left", "kind": "window", "center": 0.5, "width": 4.0, "bottom": 0.4,
                             "height": 2.0}]}
FRONT_WIDE = {"camera": {"x": 0.5, "y": 1.0, "facing_deg": 0, "framing": "wide"}, "subjects": [LEFT, RIGHT]}
BACK_WIDE = {"camera": {"x": 0.5, "y": 0.0, "facing_deg": 180, "framing": "wide"}, "subjects": [LEFT, RIGHT]}


def _frame(layout, raw_blocking, *, framing_text="", size=(256, 144)):
    blocking = parse_blocking_map(raw_blocking)
    scene = build_scene(layout, blocking, character_slots={"Man": 0, "Woman": 1, "Hae-won": 0, "Jin-woo": 1})
    camera = shot_camera(scene, blocking)
    return scene, render(scene, camera, *size)


class LayoutTests(unittest.TestCase):
    def test_a_minimal_layout_takes_the_default_room(self):
        layout, problems = validate_layout({})
        self.assertEqual(layout["room"], {"width": 6.0, "depth": 5.0, "height": 2.8})
        self.assertEqual(layout["openings"], [])
        self.assertTrue(problems)
        self.assertIsNone(normalize_layout("a cafe"))

    def test_openings_are_validated_and_clamped(self):
        layout, problems = validate_layout({
            "room": {"width": 4, "depth": 100, "height": 3},
            "openings": [
                {"wall": "left", "kind": "window", "center": 0.0, "width": 2.0},
                {"wall": "ceiling", "kind": "window"},
                {"wall": "back", "kind": "hatch"},
                {"wall": "back", "kind": "door", "height": 9},
            ],
        })
        self.assertEqual(layout["room"]["depth"], 40.0)
        window, door = layout["openings"]
        # The window is pushed inside its wall: its centre leaves half its width to the corner.
        self.assertAlmostEqual(window["center"], 1.0 / 40.0)
        self.assertEqual((door["bottom"], door["height"]), (0.0, 3.0))
        self.assertTrue(any("unknown wall" in p for p in problems))
        self.assertTrue(any("unknown kind" in p for p in problems))

    def test_overlapping_openings_are_reported(self):
        _layout, problems = validate_layout({"openings": [
            {"wall": "back", "kind": "window", "center": 0.5, "width": 2},
            {"wall": "back", "kind": "door", "center": 0.55, "width": 1}]})
        self.assertTrue(any("overlap" in p for p in problems))

    def test_furniture_gets_a_class_and_default_size(self):
        layout, problems = validate_layout({"furniture": [
            {"name": "espresso bar", "x": 0.9, "y": 0.4},
            {"name": "small round table", "x": 0.5, "y": 0.5},
            {"name": "ghost", "x": 3, "y": 0.5},
        ]})
        bar, table = layout["furniture"]
        self.assertEqual((bar["class"], bar["height"]), ("counter", 1.05))
        self.assertEqual((table["class"], table["shape"]), ("table", "cylinder"))
        self.assertTrue(any("outside the floor" in p for p in problems))

    def test_keywords_place_the_window_on_the_named_wall(self):
        left = default_layout("A cosy cafe with a big window on the left wall and an espresso bar.")
        self.assertEqual([(o["wall"], o["kind"]) for o in left["openings"]], [("left", "window")])
        self.assertEqual([f["class"] for f in left["furniture"]], ["counter"])
        # A window the text does not place faces the master camera (the place's main image).
        self.assertEqual(default_layout("A diner with windows.")["openings"][0]["wall"], "back")
        self.assertEqual(default_layout("")["openings"], [])

    def test_the_llm_layout_is_validated_and_falls_back_to_keywords(self):
        answer = {"room": {"width": 7, "depth": 5}, "openings": [{"wall": "right", "kind": "window"}],
                  "furniture": [{"name": "bar", "class": "counter", "x": 0.1, "y": 0.5}]}
        result = suggest_layout("Cafe", "a cafe", generate=lambda **_kw: "Here: " + json.dumps(answer))
        self.assertEqual(result["source"], "llm")
        self.assertEqual(result["layout"]["openings"][0]["wall"], "right")
        failed = suggest_layout("Cafe", "a cafe with a window on the left", generate=lambda **_kw: "no idea")
        self.assertEqual(failed["source"], "keywords")
        self.assertEqual(failed["layout"]["openings"][0]["wall"], "left")
        self.assertEqual(suggest_layout("Cafe", "a cafe")["source"], "keywords")

    def test_blocking_props_move_the_layout_furniture(self):
        layout = normalize_layout({"furniture": [{"name": "table", "x": 0.2, "y": 0.2}]})
        merged = merge_props(layout, [{"name": "round table", "x": 0.5, "y": 0.6}, {"name": "cup", "x": 0.5, "y": 0.6},
                                      {"name": "counter", "x": 0.9, "y": 0.5}])
        self.assertEqual([(f["class"], f["x"], f["y"]) for f in merged["furniture"]],
                         [("table", 0.5, 0.6), ("counter", 0.9, 0.5)])
        self.assertEqual(layout["furniture"][0]["x"], 0.2)


class BibleLayoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_location_keeps_its_layout_through_saves_and_into_the_cast(self):
        cafe = {"name": "Cafe Haru", "description": "A cafe.", "layout": WINDOW_LEFT}
        bible = bibles.save_bible({"name": "K-love", "locations": [cafe]})
        stored = bible["locations"][0]
        self.assertEqual(stored["layout"]["openings"][0]["wall"], "left")
        # A client that does not know layouts keeps it; an explicit null clears it.
        without = {k: v for k, v in stored.items() if k != "layout"}
        bible = bibles.save_bible({**bible, "locations": [without]}, bible_id=bible["id"])
        self.assertEqual(bibles.get_bible(bible["id"])["locations"][0]["layout"]["openings"][0]["wall"], "left")
        cast = bibles.project_cast(bibles.get_bible(bible["id"]), os.path.join(self.tmp.name, "project"))
        self.assertEqual(cast["locations"][0]["layout"]["room"]["width"], 6.0)
        bible = bibles.save_bible({**bible, "locations": [{**without, "layout": None}]}, bible_id=bible["id"])
        self.assertIsNone(bibles.get_bible(bible["id"])["locations"][0]["layout"])


class CameraMappingTests(unittest.TestCase):
    def test_the_reverse_wide_puts_the_window_on_the_other_side(self):
        _scene, front = _frame(WINDOW_LEFT, FRONT_WIDE)
        _scene, back = _frame(WINDOW_LEFT, BACK_WIDE)
        self.assertLess(front.class_coverage()["window"]["center_x"], 0.4)
        self.assertGreater(back.class_coverage()["window"]["center_x"], 0.6)

    def test_people_sit_on_the_screen_side_the_blocking_says(self):
        locks = lock_scene_blockings([MASTER, OVER_WOMAN, OVER_MAN])
        for lock in locks:
            blocking = lock["blocking"]
            scene = build_scene(None, blocking, character_slots={"Man": 0, "Woman": 1})
            frame = render(scene, shot_camera(scene, blocking), 256, 144)
            coverage = frame.class_coverage(scene)
            sides = screen_sides(blocking)
            with self.subTest(camera=blocking["camera"]):
                left, right = ("Man", "Woman") if coverage["Man"]["center_x"] < coverage["Woman"]["center_x"] \
                    else ("Woman", "Man")
                self.assertNotEqual(sides[left], "right")
                self.assertNotEqual(sides[right], "left")
        # The two over-the-shoulder shots swap the screen sides of the two people.
        over_woman, over_man = locks[1]["blocking"], locks[2]["blocking"]
        self.assertNotEqual(screen_sides(over_woman)["Man"], screen_sides(over_man)["Man"])

    def test_the_framing_sets_the_lens_and_the_angle_the_height(self):
        blocking = parse_blocking_map(OVER_WOMAN)
        scene = build_scene(None, blocking)
        medium = shot_camera(scene, blocking)
        high = shot_camera(scene, blocking, angle="high angle")
        low = shot_camera(scene, blocking, angle="low angle")
        self.assertEqual(medium.hfov_deg, 42.0)
        self.assertGreater(high.position[1], medium.position[1])
        self.assertLess(low.position[1], medium.position[1])
        close = parse_blocking_map({**OVER_WOMAN, "camera": {**OVER_WOMAN["camera"], "framing": "close-up"}})
        self.assertGreater(shot_camera(build_scene(None, close), close).focal_mm(1280, 720), medium.focal_mm(1280, 720))

    def test_the_camera_never_stands_inside_a_person(self):
        raw = {"camera": {"x": 0.5, "y": 0.72, "facing_deg": 0, "framing": "medium"}, "subjects": [MAN, WOMAN]}
        blocking = parse_blocking_map(raw)
        scene = build_scene(None, blocking)
        camera = shot_camera(scene, blocking, foreground="Woman")
        woman = scene.person("Woman")
        self.assertGreaterEqual(np.hypot(camera.position[0] - woman.x, camera.position[2] - woman.z), 0.34)

    def test_postures_come_from_the_shot_words(self):
        postures = infer_postures(["Hae-won", "Jin-woo"], "Hae-won sits at the table. Jin-woo stands by the door.")
        self.assertEqual(postures, {"Hae-won": "seated", "Jin-woo": "standing"})
        self.assertEqual(infer_postures(["A", "B"], "They sit across a table."), {"A": "seated", "B": "seated"})
        self.assertEqual(infer_postures(["A"], ""), {"A": "standing"})


class RendererTests(unittest.TestCase):
    def test_known_pixels_get_their_class(self):
        layout = {"room": {"width": 6, "depth": 6, "height": 3},
                  "openings": [{"wall": "back", "kind": "window", "center": 0.5, "width": 2, "bottom": 1.2,
                                "height": 1.0}]}
        raw = {"camera": {"x": 0.5, "y": 1.0, "facing_deg": 0, "framing": "wide"},
               "subjects": [{"name": "A", "x": 0.2, "y": 0.5, "facing_deg": 180}]}
        blocking = parse_blocking_map(raw)
        scene = build_scene(layout, blocking, character_slots={"A": 0})
        camera = shot_camera(scene, blocking)
        # Look straight at the window's centre from the front wall at its height.
        camera = type(camera)(position=(3.0, 1.7, 6.0), target=(3.0, 1.7, 0.0), hfov_deg=70.0)
        frame = render(scene, camera, 64, 64)
        self.assertEqual(frame.labels[32, 32], LABELS["window"])
        self.assertEqual(frame.labels[63, 32], LABELS["floor"])
        self.assertEqual(frame.labels[0, 32], LABELS["ceiling"])
        self.assertEqual(frame.labels[32, 0], LABELS["wall"])
        self.assertAlmostEqual(float(frame.depth[32, 32]), 6.0, places=3)
        self.assertEqual(tuple(frame.semantic_image()[32, 32]), PALETTE["window"])
        self.assertIn(character_label(0), set(np.unique(frame.labels)))

    def test_rendering_is_deterministic_and_fast(self):
        blocking = lock_scene_blockings([MASTER])[0]["blocking"]
        scene = build_scene(default_layout("A cafe with a window on the left, an espresso bar and a door."),
                            blocking, postures={"Man": "seated", "Woman": "seated"})
        camera = shot_camera(scene, blocking)
        started = time.perf_counter()
        first = render(scene, camera, 512, 288)
        elapsed = time.perf_counter() - started
        second = render(scene, camera, 512, 288)
        self.assertTrue(np.array_equal(first.labels, second.labels))
        self.assertTrue(np.array_equal(first.depth_image(), second.depth_image()))
        self.assertTrue(np.array_equal(first.normal_image(), second.normal_image()))
        self.assertLess(elapsed, 2.0)
        self.assertEqual(first.edge_image().shape, (288, 512, 3))
        self.assertEqual(palette_table().shape[1], 3)

    def test_a_push_in_drives_the_camera_closer(self):
        blocking = parse_blocking_map(OVER_WOMAN)
        scene = build_scene(None, blocking)
        camera = shot_camera(scene, blocking)
        path = camera_path(camera, "slow push-in", intensity="moderate", frames=5)
        self.assertEqual(len(path), 5)
        self.assertEqual(path[0], camera)
        # The camera nears what the first frame aims at, frame after frame.
        distances = [float(np.linalg.norm(np.subtract(camera.target, c.position))) for c in path]
        self.assertEqual(distances, sorted(distances, reverse=True))
        self.assertGreater(np.linalg.norm(np.subtract(path[-1].position, camera.position)), 0.1)
        self.assertEqual(camera_path(camera, "static", frames=3), [camera] * 3)
        self.assertEqual(parse_movement("pan left then tilt up"), [("pan", -1.0), ("tilt", 1.0)])
        pan = camera_path(camera, "pan right", frames=2)[-1]
        _f, right, _u = camera.basis()
        self.assertGreater(float(np.subtract(pan.target, camera.target) @ right), 0)


class GuideHookTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.plan = {"_director_continuity_group": "s1", "_director_location": "Cafe Haru",
                     "_director_blocking_map": lock_scene_blockings([OVER_MAN])[0]["blocking"],
                     "_director_camera_framing": "over-the-shoulder medium",
                     "image_prompt": "They sit at the table."}

    def test_the_guide_is_off_by_default(self):
        self.assertEqual(guide.guide_mode({}), "off")
        self.assertEqual(guide.guide_mode({"director_blockout_guide": "bogus"}), "off")
        self.assertEqual(guide.guide_mode({"director_blockout_guide": "Semantic"}), "semantic")
        self.assertIsNone(guide.render_shot_guide(self.plan, {}, out_dir=self.tmp.name, name="p", resolution="1280x720"))
        self.assertIsNone(pipeline.director_blockout_guide("p", 0, self.plan, {}, self.tmp.name, "1280x720", {}))
        self.assertEqual(os.listdir(self.tmp.name), [])
        prompt = "Same place. The camera setup of this shot is: reference image 1 shows Jin-woo."
        self.assertEqual(guide.guide_prompt(prompt, None, control=False), prompt)
        job = {"video_prompt_type": "KI", "prompt": prompt}
        self.assertIs(guide.apply_control(job, ""), job)

    def test_the_guide_is_rendered_and_recorded_when_on(self):
        params = {"director_blockout_guide": "semantic",
                  "_director_cast": {"locations": [{"name": "Cafe Haru", "layout": WINDOW_LEFT}]}}
        record = pipeline.director_blockout_guide("p1", 2, self.plan, params, self.tmp.name, "864x1536",
                                                  {"s1": {"Man": 0, "Woman": 1}})
        self.assertEqual(record["mode"], "semantic")
        self.assertEqual(record["image"], "p1_shot03_blockout_semantic.png")
        self.assertEqual(sorted(os.listdir(self.tmp.name)), sorted(record["files"].values()))
        self.assertEqual(record["size"], [432, 768])
        self.assertEqual(record["layout_source"], "bible")
        self.assertEqual(record["foreground"], "Man")
        self.assertEqual(record["people"]["Woman"], {"slot": 1, "posture": "seated"})
        self.assertIn("focal_mm", record["camera"])

    def test_the_layout_falls_back_from_the_description_to_the_default_room(self):
        params = {"_director_cast": {"locations": [{"name": "Cafe Haru", "description": "A cafe, door on the left."}]}}
        layout, source = guide.location_layout(params, "cafe haru")
        self.assertEqual((source, layout["openings"][0]["wall"]), ("description", "left"))
        self.assertEqual(guide.location_layout({}, "Nowhere")[1], "default")
        override = {**params, "director_location_layouts": {"Cafe Haru": WINDOW_LEFT}}
        self.assertEqual(guide.location_layout(override, "Cafe Haru")[1], "params")

    def test_the_control_image_takes_slot_one(self):
        self.assertEqual(guide.apply_control({"video_prompt_type": "KI"}, "/g.png"),
                         {"video_prompt_type": "VI", "image_guide": "/g.png"})
        self.assertEqual(guide.apply_control({"video_prompt_type": ""}, "/g.png")["video_prompt_type"], "V")
        self.assertEqual(guide.shift_reference_numbers("reference image 1 and reference image 2, <image3>"),
                         "reference image 2 and reference image 3, <image4>")
        record = {"mode": "depth", "set_sides": {"window": "frame left"}}
        prompt = guide.guide_prompt("reference image 1 shows Hae-won.", record, control=True)
        self.assertTrue(prompt.startswith("Reference image 1 is a depth map"))
        self.assertIn("reference image 2 shows Hae-won.", prompt)
        self.assertTrue(prompt.endswith("Set layout: the window on frame left."))
        self.assertEqual(guide.guide_prompt("x", record, control=False), "x Set layout: the window on frame left.")
        self.assertTrue(guide.control_supported("qwen_image_21_7B_turbo_8step"))
        self.assertFalse(guide.control_supported("cloud_fal_nano_banana"))
        self.assertEqual(guide.scene_character_slots([self.plan, {**self.plan, "_director_continuity_group": ""}]),
                         {"s1": {"Man": 0, "Woman": 1}})


if __name__ == "__main__":
    unittest.main()
