"""A scene keeps one floor plan: people stay put, the camera keeps its side of the line, reverses get their wall."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services.director.blocking_map import (  # noqa: E402
    camera_faces_reverse, framing_line, is_over_the_shoulder, lock_scene_blockings, ots_foreground,
    parse_blocking_map, scene_floor_layout, screen_sides, staging_line,
)

# Two people face to face across a table, along the room's depth: the man at
# the back, the woman at the front. The master stands front right.
MAN = {"name": "Man", "x": 0.5, "y": 0.3, "facing_deg": 180}
WOMAN = {"name": "Woman", "x": 0.5, "y": 0.7, "facing_deg": 0}
TABLE = {"name": "table", "x": 0.5, "y": 0.5}
MASTER = {"camera": {"x": 0.9, "y": 0.9, "facing_deg": 315, "framing": "wide"},
          "subjects": [MAN, WOMAN], "props": [TABLE]}
# Over the woman's shoulder toward the man (looks at the back wall like the master).
OVER_WOMAN = {"camera": {"x": 0.6, "y": 0.9, "facing_deg": 350, "framing": "medium"},
              "subjects": [MAN, WOMAN]}
# Over the man's shoulder toward the woman: the camera looks at the front wall.
OVER_MAN = {"camera": {"x": 0.6, "y": 0.1, "facing_deg": 170, "framing": "medium"},
            "subjects": [MAN, WOMAN]}
# The same reverse planned from the other side of the line.
OVER_MAN_CROSSED = {"camera": {"x": 0.4, "y": 0.1, "facing_deg": 190, "framing": "medium"},
                    "subjects": [MAN, WOMAN]}


def _order(blocking):
    """Names in frame, left to right."""
    line = framing_line(blocking)
    names = line.split("left to right: ", 1)[1].rstrip(".").split(", ")
    return [name.split(" (")[0] for name in names]


class SceneLayoutTests(unittest.TestCase):
    def test_the_layout_comes_from_the_first_shot(self):
        layout = scene_floor_layout(parse_blocking_map(MASTER))
        self.assertEqual([p["name"] for p in layout["people"]], ["Man", "Woman"])
        self.assertEqual(layout["props"], [TABLE])
        self.assertEqual(layout["camera"], {"x": 0.9, "y": 0.9, "facing_deg": 315.0})
        self.assertEqual(layout["line"], ["Man", "Woman"])
        self.assertNotEqual(layout["side"], 0)

    def test_a_structured_scene_map_wins_over_the_first_shot(self):
        scene_map = {"people": [{"name": "Man", "x": 0.2, "y": 0.5, "facing_deg": 90}],
                     "props": [{"name": "sofa", "x": 0.5, "y": 0.2}],
                     "camera": {"x": 0.5, "y": 0.95, "facing_deg": 0}}
        layout = scene_floor_layout(parse_blocking_map(MASTER), scene_map)
        self.assertEqual((layout["people"][0]["x"], layout["people"][0]["y"]), (0.2, 0.5))
        # The first shot adds who the map does not place.
        self.assertEqual([p["name"] for p in layout["people"]], ["Man", "Woman"])
        self.assertEqual([p["name"] for p in layout["props"]], ["sofa", "table"])
        self.assertEqual(layout["camera"]["facing_deg"], 0.0)

    def test_later_shots_keep_the_people_where_the_scene_put_them(self):
        # The planner drifted a little (0.16 of the room) and turned the woman away.
        drifted = {"camera": {"x": 0.5, "y": 0.95, "facing_deg": 0, "framing": "medium"},
                   "subjects": [{"name": "Man", "x": 0.4, "y": 0.42, "facing_deg": 180},
                                {"name": "Woman", "x": 0.6, "y": 0.58, "facing_deg": 180}]}
        master, shot = lock_scene_blockings([MASTER, drifted])
        self.assertFalse(master["corrected"])
        self.assertEqual(master["notes"], [])
        self.assertTrue(shot["corrected"])
        people = {s["name"]: s for s in shot["blocking"]["subjects"]}
        self.assertEqual((people["Man"]["x"], people["Man"]["y"], people["Man"]["facing_deg"]), (0.5, 0.3, 180.0))
        self.assertEqual((people["Woman"]["x"], people["Woman"]["y"], people["Woman"]["facing_deg"]), (0.5, 0.7, 0.0))
        self.assertTrue(any("Woman kept facing 0 degrees" in note for note in shot["notes"]))
        self.assertTrue(any("kept at the scene position" in note for note in shot["notes"]))
        # They still face each other, and the props stay where the master put them.
        self.assertIn("Man looks at Woman", staging_line(shot["blocking"]))
        self.assertEqual(shot["blocking"]["props"], [TABLE])

    def test_a_planned_move_updates_the_layout_for_the_next_shots(self):
        moved = {"camera": {"x": 0.5, "y": 0.95, "facing_deg": 0, "framing": "medium"},
                 "subjects": [{"name": "Man", "x": 0.2, "y": 0.6, "facing_deg": 90, "moves": True}]}
        later = {"camera": {"x": 0.5, "y": 0.95, "facing_deg": 0, "framing": "medium"},
                 "subjects": [{"name": "Man", "x": 0.3, "y": 0.5, "facing_deg": 180}]}
        _, move, after = lock_scene_blockings([MASTER, moved, later])
        self.assertFalse(move["corrected"])
        self.assertEqual(move["notes"], ["Man moves to a new place, as planned"])
        man = after["blocking"]["subjects"][0]
        self.assertEqual((man["x"], man["y"], man["facing_deg"]), (0.2, 0.6, 90.0))

    def test_a_far_move_follows_the_shot_instead_of_the_old_place(self):
        # Past 0.2 of the room this is a move, not drift: forcing the old
        # place would contradict the shot's own prose.
        walked = {"camera": {"x": 0.5, "y": 0.95, "facing_deg": 0, "framing": "medium"},
                  "subjects": [{"name": "Man", "x": 0.15, "y": 0.6, "facing_deg": 90}, WOMAN]}
        later = {"camera": {"x": 0.5, "y": 0.95, "facing_deg": 0, "framing": "medium"},
                 "subjects": [{"name": "Man", "x": 0.2, "y": 0.62, "facing_deg": 90}]}
        _, walk, after = lock_scene_blockings([MASTER, walked, later])
        self.assertFalse(walk["corrected"])
        self.assertEqual(walk["notes"], ["Man moves to a new place, as the shot places them"])
        man = next(s for s in walk["blocking"]["subjects"] if s["name"] == "Man")
        self.assertEqual((man["x"], man["y"], man["facing_deg"]), (0.15, 0.6, 90.0))
        # The scene layout now holds the new place: the next shot's small drift snaps to it.
        man = after["blocking"]["subjects"][0]
        self.assertEqual((man["x"], man["y"]), (0.15, 0.6))
        self.assertFalse(after["corrected"])

    def test_a_far_move_still_gets_the_facing_of_the_others_corrected(self):
        walked = {"camera": {"x": 0.5, "y": 0.95, "facing_deg": 0, "framing": "medium"},
                  "subjects": [{"name": "Man", "x": 0.15, "y": 0.6, "facing_deg": 90},
                               {"name": "Woman", "x": 0.5, "y": 0.7, "facing_deg": 180}]}
        _, walk = lock_scene_blockings([MASTER, walked])
        self.assertTrue(walk["corrected"])
        self.assertTrue(any("Woman kept facing 0 degrees" in note for note in walk["notes"]))

    def test_someone_new_joins_the_layout_where_they_first_appear(self):
        newcomer = {"camera": {"x": 0.5, "y": 0.95, "facing_deg": 0, "framing": "wide"},
                    "subjects": [{"name": "Waiter", "x": 0.8, "y": 0.4, "facing_deg": 270}, MAN]}
        again = {"camera": {"x": 0.5, "y": 0.95, "facing_deg": 0, "framing": "wide"},
                 "subjects": [{"name": "Waiter", "x": 0.65, "y": 0.4, "facing_deg": 270}]}
        locks = lock_scene_blockings([MASTER, newcomer, again])
        self.assertEqual(locks[2]["blocking"]["subjects"][0]["x"], 0.8)
        self.assertTrue(locks[2]["corrected"])

    def test_shots_without_a_usable_map_are_left_alone(self):
        locks = lock_scene_blockings([None, {"camera": {}}, MASTER])
        self.assertEqual([lock["blocking"] is None for lock in locks], [True, True, False])
        self.assertEqual(locks[2]["notes"], [])


class ScreenDirectionTests(unittest.TestCase):
    def test_the_reverse_keeps_each_person_on_their_screen_side(self):
        master, over_woman, over_man = lock_scene_blockings([MASTER, OVER_WOMAN, OVER_MAN])
        self.assertEqual(_order(master["blocking"]), ["Woman", "Man"])
        self.assertEqual(_order(over_woman["blocking"]), ["Woman", "Man"])
        self.assertEqual(_order(over_man["blocking"]), ["Woman", "Man"])
        self.assertEqual(over_man["notes"], [])

    def test_a_reverse_planned_across_the_line_is_brought_back(self):
        master, crossed = lock_scene_blockings([MASTER, OVER_MAN_CROSSED])
        # Taken as planned it would swap the two people on screen.
        self.assertEqual(_order(parse_blocking_map(OVER_MAN_CROSSED)), ["Man", "Woman"])
        self.assertEqual(_order(crossed["blocking"]), ["Woman", "Man"])
        self.assertTrue(crossed["corrected"])
        self.assertIn("180-degree rule", " ".join(crossed["notes"]))
        self.assertGreater(crossed["blocking"]["camera"]["x"], 0.5)

    def test_the_line_is_crossed_only_when_the_plan_says_so(self):
        planned = {**OVER_MAN_CROSSED, "camera": {**OVER_MAN_CROSSED["camera"], "crosses_line": True}}
        after = {**OVER_WOMAN, "camera": {**OVER_WOMAN["camera"], "x": 0.6}}
        _, crossed, following = lock_scene_blockings([MASTER, planned, after])
        self.assertEqual(_order(crossed["blocking"]), ["Man", "Woman"])
        self.assertFalse(crossed["corrected"])
        self.assertEqual(crossed["notes"], ["the camera crosses the action line, as planned"])
        # The scene now keeps the new side: a later shot from the old side is mirrored.
        self.assertLess(following["blocking"]["camera"]["x"], 0.5)

    def test_a_camera_on_the_line_moves_to_the_scenes_side(self):
        on_line = {"camera": {"x": 0.5, "y": 0.05, "facing_deg": 180, "framing": "medium"},
                   "subjects": [MAN, WOMAN]}
        _, shot = lock_scene_blockings([MASTER, on_line])
        self.assertGreater(shot["blocking"]["camera"]["x"], 0.5)
        self.assertEqual(_order(shot["blocking"]), ["Woman", "Man"])

    def test_a_single_close_up_looks_toward_the_off_screen_partner(self):
        close = {"camera": {"x": 0.6, "y": 0.1, "facing_deg": 175, "framing": "close-up"},
                 "subjects": [WOMAN]}
        master, shot = lock_scene_blockings([MASTER, close])
        blocking = shot["blocking"]
        self.assertEqual([p["name"] for p in blocking["offscreen"]], ["Man"])
        line = staging_line(blocking)
        self.assertIn("Exactly 1 person in frame", line)
        side = screen_sides(blocking)["Man"]
        self.assertIn(f"Woman looks toward frame {side} at Man, who is out of frame", line)
        # Over the man's shoulder from the master's side, he is on the right of the frame, as in the master.
        self.assertEqual(side, "right")
        self.assertEqual(screen_sides(master["blocking"]), {"Man": "right", "Woman": "left"})
        # A saved locked map parses to itself.
        self.assertEqual(parse_blocking_map(blocking), blocking)

    def test_props_are_placed_in_the_frame(self):
        line = staging_line(parse_blocking_map({**MASTER, "camera": {"x": 0.5, "y": 0.95, "facing_deg": 0,
                                                                     "framing": "wide"}}))
        self.assertIn("Set: the table frame center.", line)


class ReverseAngleTests(unittest.TestCase):
    def test_a_camera_facing_the_opposite_wall_is_a_reverse(self):
        locks = lock_scene_blockings([MASTER, OVER_WOMAN, OVER_MAN])
        self.assertEqual([lock["reverse"] for lock in locks], [False, False, True])
        layout = scene_floor_layout(parse_blocking_map(MASTER))
        self.assertFalse(camera_faces_reverse(layout, None))

    def _plans(self):
        return [
            {"_director_continuity_group": "cafe", "_director_blocking_map": MASTER},
            {"_director_continuity_group": "cafe", "_director_blocking_map": OVER_MAN_CROSSED,
             "_director_opening_blocking": "she sits on his left"},
            {"_director_continuity_group": "street", "_director_blocking_map": OVER_MAN},
            {"_director_blocking_map": OVER_MAN},
        ]

    def test_the_pipeline_locks_each_scene_on_its_own(self):
        plans = self._plans()
        locks = pipeline.director_scene_locks(plans)
        self.assertIsNone(locks[3])  # outside any scene
        self.assertTrue(locks[1]["reverse"] and locks[1]["corrected"])
        # The street scene starts its own layout: its first shot is its master.
        self.assertFalse(locks[2]["reverse"])
        plan = pipeline.scene_locked_plan(plans[1], locks[1])
        self.assertTrue(plan["_director_camera_reverse"])
        self.assertTrue(plan["_director_layout_corrected"])
        self.assertIs(pipeline.scene_locked_plan(plans[3], None), plans[3])
        self.assertEqual(plans[1]["_director_blocking_map"], OVER_MAN_CROSSED)  # the saved plan is untouched

    def test_a_reverse_shot_takes_the_places_reverse_view(self):
        plans = self._plans()
        plan = pipeline.scene_locked_plan(plans[1], pipeline.director_scene_locks(plans)[1])
        request = pipeline.director_shot_image_request(
            "cloud_fal_seedream_4", plan, "They talk.", scene_master="/o/master.png",
            scene_location="/o/cafe.png", scene_location_reverse="/o/cafe_reverse.png",
            subject_refs=["/o/man.png", "/o/woman.png"])
        refs = [request["source"], *request["supplemental"]]
        self.assertIn("/o/cafe_reverse.png", refs)
        self.assertNotIn("/o/cafe.png", refs)
        self.assertIn("seen from its opposite wall", request["prompt"])
        self.assertNotIn("faces the opposite side", request["prompt"])
        # The planner's words for the layout the lock corrected are not repeated.
        self.assertNotIn("she sits on his left", request["prompt"])

    def test_without_a_reverse_view_the_main_image_stays_with_a_note(self):
        plans = self._plans()
        plan = pipeline.scene_locked_plan(plans[1], pipeline.director_scene_locks(plans)[1])
        request = pipeline.director_shot_image_request(
            "cloud_fal_seedream_4", plan, "They talk.", scene_master="/o/master.png",
            scene_location="/o/cafe.png", subject_refs=["/o/man.png", "/o/woman.png"])
        refs = [request["source"], *request["supplemental"]]
        self.assertIn("/o/cafe.png", refs)
        self.assertIn("This camera faces the opposite side of the room", request["prompt"])
        self.assertIn("shows this place empty", request["prompt"])

    def test_a_shot_facing_the_masters_wall_keeps_the_main_image(self):
        plans = self._plans()
        request = pipeline.director_shot_image_request(
            "cloud_fal_seedream_4", pipeline.scene_locked_plan(plans[0], pipeline.director_scene_locks(plans)[0]),
            "They talk.", scene_master="/o/master.png", scene_location="/o/cafe.png",
            scene_location_reverse="/o/cafe_reverse.png", subject_refs=["/o/man.png"])
        refs = [request["source"], *request["supplemental"]]
        self.assertIn("/o/cafe.png", refs)
        self.assertNotIn("/o/cafe_reverse.png", refs)
        self.assertNotIn("opposite", request["prompt"])

    def test_the_reverse_view_is_found_among_the_cast_views(self):
        with tempfile.TemporaryDirectory() as folder:
            place, reverse = os.path.join(folder, "cafe.png"), os.path.join(folder, "cafe_reverse.png")
            with open(reverse, "wb") as handle:
                handle.write(b"png")
            params = {"_director_image_views": {place: {"reverse": reverse}}}
            self.assertEqual(pipeline.location_reverse_view(params, place), reverse)
            self.assertEqual(pipeline.location_reverse_view(params, ""), "")
            self.assertEqual(pipeline.location_reverse_view({}, place), "")
            os.remove(reverse)
            self.assertEqual(pipeline.location_reverse_view(params, place), "")

    def test_the_layout_text_comes_from_the_locked_map(self):
        plans = self._plans()
        plans[1]["_director_subjects_on_screen"] = [{"name": "Man"}, {"name": "Woman"}]
        plan = pipeline.scene_locked_plan(plans[1], pipeline.director_scene_locks(plans)[1])
        with tempfile.TemporaryDirectory() as folder:
            _, prompt = pipeline.director_shot_layout("p", 1, plan, "They talk.", "cloud_fal_seedream_4", folder)
        # Over the man's shoulder: he is the cropped back on the right, she is in frame.
        self.assertIn("In frame, left to right: Woman (facing camera), Man (back to camera, the out-of-focus "
                      "back of the head and shoulder in the right foreground", prompt)
        self.assertIn("Exactly 2 people in frame", prompt)


class OverTheShoulderTests(unittest.TestCase):
    """The foreground back of an OTS shot: no front portrait, no face, same people count."""

    PARAMS = {"character_ref_labels": ["Man", "Woman"], "character_ref_paths": ["/p/man.png", "/p/woman.png"]}

    def test_the_nearest_back_to_camera_person_is_the_foreground(self):
        self.assertEqual(ots_foreground(parse_blocking_map(OVER_MAN)), "Man")
        self.assertEqual(ots_foreground(parse_blocking_map(OVER_WOMAN)), "Woman")
        # A wide master is not an OTS, and one person alone is not either.
        self.assertEqual(ots_foreground(parse_blocking_map(MASTER)), "")
        self.assertEqual(ots_foreground(parse_blocking_map({**OVER_MAN, "subjects": [WOMAN]})), "")

    def test_an_ots_framing_names_the_nearest_person(self):
        # Both face the camera in this map, but the framing says over-the-shoulder.
        facing = {"camera": {"x": 0.5, "y": 0.95, "facing_deg": 0, "framing": "medium"},
                  "subjects": [{"name": "Man", "x": 0.4, "y": 0.75, "facing_deg": 180},
                               {"name": "Woman", "x": 0.55, "y": 0.4, "facing_deg": 180}]}
        blocking = parse_blocking_map(facing)
        self.assertEqual(ots_foreground(blocking), "")
        self.assertEqual(ots_foreground(blocking, "OTS on Woman"), "Man")
        self.assertEqual(ots_foreground(blocking, "medium over-the-shoulder"), "Man")
        self.assertTrue(is_over_the_shoulder("Over the shoulder, eye level"))
        self.assertFalse(is_over_the_shoulder("medium shot of the shoulders and hands"))

    def test_the_layout_describes_the_back_and_keeps_the_count(self):
        line = staging_line(parse_blocking_map(OVER_MAN))
        # The foreground back stays in the list, so the list matches the count:
        # listed apart, editors drew the person twice (shot 5 of run fcd1e37f).
        self.assertIn("In frame, left to right: Woman (facing camera), Man (back to camera, the out-of-focus "
                      "back of the head and shoulder in the right foreground, cropped by the frame edge, "
                      "face not visible).", line)
        self.assertIn("Exactly 2 people in frame", line)
        self.assertIn("Man is seen only from behind.", line)
        self.assertNotIn("Foreground:", line)
        self.assertIn("Man looks at Woman", line)

    def test_the_foreground_person_gets_no_reference(self):
        plan = {"_director_blocking_map": OVER_MAN}
        views = {"/p/man.png": {"face_front": "/p/man_face.png", "back": "/p/man_back.png"},
                 "/p/woman.png": {"face_front": "/p/woman_face.png"}}
        # Not even the back view: given any image of the man, the editor drew him a second time.
        refs = pipeline.shot_identity_refs({**self.PARAMS, "_director_image_views": views}, plan,
                                           ["/p/man.png", "/p/woman.png"])
        self.assertEqual(refs, ["/p/woman_face.png"])
        no_back = {"/p/man.png": {"face_front": "/p/man_face.png"}}
        refs = pipeline.shot_identity_refs({**self.PARAMS, "_director_image_views": no_back}, plan,
                                           ["/p/man.png", "/p/woman.png"])
        self.assertEqual(refs, ["/p/woman.png"])
        # Without any bible views, the front portrait is still dropped.
        self.assertEqual(pipeline.shot_identity_refs(self.PARAMS, plan, ["/p/man.png", "/p/woman.png"]),
                         ["/p/woman.png"])
        # The fallback references (no scene set) drop the front portrait too.
        self.assertEqual(pipeline.without_ots_front_refs(self.PARAMS, plan, ["/p/man.png", "/p/woman.png", "/p/bar.png"]),
                         ["/p/woman.png", "/p/bar.png"])
        # A shot that is not an OTS keeps both people.
        self.assertEqual(pipeline.shot_identity_refs(self.PARAMS, {"_director_blocking_map": MASTER},
                                                     ["/p/man.png", "/p/woman.png"]),
                         ["/p/man.png", "/p/woman.png"])

    def test_without_a_map_the_planner_words_name_the_foreground(self):
        plan = {"_director_camera_framing": "Over-the-shoulder medium shot",
                "_director_subjects_on_screen": [
                    {"speaker_name": "Man", "visual_description": "a man",
                     "position_or_relation": "foreground right, back to camera"},
                    {"speaker_name": "Woman", "visual_description": "a woman"}]}
        self.assertEqual(pipeline.director_ots_foreground(plan), "Man")
        self.assertEqual(pipeline.shot_identity_refs(self.PARAMS, plan, ["/p/man.png", "/p/woman.png"]),
                         ["/p/woman.png"])
        _, prompt = pipeline.director_shot_layout("p", 0, plan, "They talk.", "flux2_klein_9b", "/tmp")
        self.assertIn("Exactly 2 people in frame (Man, Woman)", prompt)
        self.assertIn("the out-of-focus back of Man's head and shoulder in the right third", prompt)
        plan["_director_camera_framing"] = "medium two-shot"
        self.assertEqual(pipeline.director_ots_foreground(plan), "")


if __name__ == "__main__":
    unittest.main()
