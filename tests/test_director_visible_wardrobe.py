"""A closer framing names only the garments it can see."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from types import SimpleNamespace  # noqa: E402

from services.director.policies import resolve_subjects_text, shot_reach, visible_wardrobe  # noqa: E402

OUTFIT = (
    "an oversized knitted cream wool cardigan over a crisp white cotton shirt, "
    "charcoal pleated wide-leg trousers, white canvas lace-up sneakers, and a slim silver chain."
)


class VisibleWardrobeTest(unittest.TestCase):
    def test_reach_by_framing(self):
        self.assertEqual(shot_reach("Wide master shot at eye level"), "full")
        self.assertEqual(shot_reach("Medium wide shot"), "full")
        self.assertEqual(shot_reach("Medium close-up on Hae-won"), "upper")
        self.assertEqual(shot_reach("Over-the-shoulder shot on Jin-woo"), "upper")
        self.assertEqual(shot_reach("Extreme close-up insert on hand and tool"), "detail")
        self.assertEqual(shot_reach("extreme close-up"), "detail")
        self.assertEqual(shot_reach("Extreme close-up on her eyes"), "upper")
        self.assertEqual(shot_reach(""), "full")

    def test_waist_up_drops_legs_and_shoes(self):
        upper = visible_wardrobe(OUTFIT, "upper")
        self.assertIn("cardigan", upper)
        self.assertIn("silver chain", upper)
        self.assertNotIn("trousers", upper)
        self.assertNotIn("sneakers", upper)

    def test_full_keeps_everything_and_detail_keeps_the_sleeves(self):
        self.assertEqual(visible_wardrobe(OUTFIT, "full"), OUTFIT)
        self.assertIn("cardigan", visible_wardrobe(OUTFIT, "detail"))
        self.assertNotIn("sneakers", visible_wardrobe(OUTFIT, "detail"))
        self.assertIsNone(visible_wardrobe(None, "upper"))

    def test_an_insert_names_hands_without_the_face_or_the_clothes(self):
        char = SimpleNamespace(physical_description="22, oval face, almond eyes, a young woman", wardrobe=OUTFIT)
        plan = SimpleNamespace(get_character=lambda _id: char)
        shot = SimpleNamespace(scene_type="insert", camera_plan=SimpleNamespace(framing="close-up"),
                               subjects_on_screen=[SimpleNamespace(visual_description="", character_id="c1",
                                                                   position_or_relation="screen right")])
        self.assertEqual(resolve_subjects_text(shot, plan), "a woman's hands, screen right")
        # A person the plan cannot resolve: the planner's own words give the gender, nothing else.
        shot.subjects_on_screen = [SimpleNamespace(visual_description="the man's hand in a green sleeve",
                                                   character_id="", position_or_relation="")]
        self.assertEqual(resolve_subjects_text(shot, None), "a man's hands")
        shot.subjects_on_screen = [SimpleNamespace(visual_description="", character_id="", position_or_relation="")]
        self.assertEqual(resolve_subjects_text(shot, None), "hands")


if __name__ == "__main__":
    unittest.main(verbosity=2)
