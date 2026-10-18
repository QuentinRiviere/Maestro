"""A hands insert names no face and no clothes, in the draft, the scene map and the polish."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director.planners.custom import SCENE_CRAFT  # noqa: E402
from services.director.policies import hands_subject  # noqa: E402
from services.director.prompt_polish import (  # noqa: E402
    INSERT_POLISH_RULE,
    _is_insert_plan,
    insert_person,
)
from services.director.renderers.base import BaseRenderer  # noqa: E402


class HandsSubjectTest(unittest.TestCase):
    def test_gender_word_only(self):
        self.assertEqual(hands_subject("22 years old, a slender young Korean woman", "cream sweater"),
                         "a woman's hands")
        self.assertEqual(hands_subject("", "the man's hand in a green sleeve"), "a man's hands")
        self.assertEqual(hands_subject("23 years old, South Korean, chiseled jawline"), "hands")

    def test_polish_names_a_person_without_the_face(self):
        self.assertEqual(insert_person("A slender young Korean woman with guarded eyes"), "the woman")
        self.assertEqual(insert_person("A friendly Korean young man with round spectacles"), "the man")
        self.assertEqual(insert_person("the white unicorn"), "the person")

    def test_insert_detection_reads_both_keys(self):
        self.assertTrue(_is_insert_plan({"_director_scene_type": "Insert"}))
        self.assertTrue(_is_insert_plan({"scene_type": "insert"}))
        self.assertFalse(_is_insert_plan({"_director_scene_type": "exchange"}))
        self.assertFalse(_is_insert_plan({}))

    def test_polish_rule_forbids_people(self):
        self.assertIn("no face", INSERT_POLISH_RULE)
        self.assertIn("no clothes", INSERT_POLISH_RULE)
        self.assertIn("'the woman' / 'the man'", INSERT_POLISH_RULE)


class SceneMapTest(unittest.TestCase):
    def test_an_insert_gets_no_scene_map(self):
        metadata = {"scene_map": "Left: Hae-won seated. Right: Jin-woo leaning in."}
        self.assertEqual(BaseRenderer._scene_map_text(SimpleNamespace(scene_type="insert", metadata=metadata)), "")
        self.assertEqual(BaseRenderer._scene_map_text(SimpleNamespace(scene_type="exchange", metadata=metadata)),
                         metadata["scene_map"])


class PlannerRuleTest(unittest.TestCase):
    def test_contact_between_two_people_is_never_filmed(self):
        self.assertIn("Contact between two people is never in frame", SCENE_CRAFT)
        self.assertIn("every hand below the bottom edge of the frame", SCENE_CRAFT)
        self.assertIn("never the subject of an insert", SCENE_CRAFT)
        self.assertIn("An insert shows ONE person's hand", SCENE_CRAFT)
        self.assertIn("An insert names no face", SCENE_CRAFT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
