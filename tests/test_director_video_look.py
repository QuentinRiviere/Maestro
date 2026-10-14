"""The bible's look reaches every video prompt verbatim, without breaking H3's format contract."""
from __future__ import annotations

import os
import sys
import unittest

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director import h3_dialogue  # noqa: E402
from services.director_pipeline import _video_prompt_with_look  # noqa: E402

LOOK = "Soft anamorphic 40mm, Fuji Eterna 500T, warm pastel highlights"


class VideoLookTests(unittest.TestCase):
    def test_a_brief_gets_the_look_once_and_each_window_gets_it(self):
        brief = "SHOT: close-up | LIGHT: constant"
        once = _video_prompt_with_look(brief, LOOK)
        self.assertEqual(once, f"{brief} | STYLE: {LOOK}")
        self.assertEqual(_video_prompt_with_look(once, LOOK), once)
        windows = _video_prompt_with_look("first window\nsecond window", LOOK, windowed=True).splitlines()
        self.assertEqual(len(windows), 2)
        self.assertTrue(all(w.endswith(LOOK) for w in windows))
        self.assertEqual(_video_prompt_with_look(brief, "  "), brief)

    def test_h3_keeps_its_contract_with_the_look(self):
        subjects = [{"visual_description": "a young woman in a school uniform"}]
        prompt, _ = h3_dialogue.compile_h3_official_prompt(
            "A young woman stops in the rainy corridor and looks back.", subjects, [],
            mode="t2va", duration_seconds=5, look=LOOK,
        )
        self.assertIn(f"Visual style: {LOOK}", prompt)
        self.assertEqual(h3_dialogue.validate_h3_prompt_contract(prompt, [], mode="t2va", subjects=subjects), [])


if __name__ == "__main__":
    unittest.main()
