"""A place's room sheet and the text-only prompts of its angles."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director import bibles  # noqa: E402
from services.director import room_spec as rs  # noqa: E402

ANSWER = json.dumps({
    "summary": "A small brick coffee shop at dusk.",
    "materials": "polished concrete floor, pale red brick walls, honey-oak panelling, white ceiling",
    "light": "warm 2700K pendants and table lamps, blue dusk through the window",
    "palette": ["honey oak", "pale red brick", "brass", "dusk blue"],
    "walls": {"front": "the picture window onto the lane, ferns on the sill, a bentwood chair",
              "left": "brick wall with oak panelling, two round tables with lamps, the entrance door at the near end",
              "right": "the dark espresso bar with the brass machine and the pastry case",
              "back": "a wall of shelves with cups and bags of beans, the till"},
})
ELEMENT = {"id": "cafe", "name": "Café Haru", "description": "A cozy coffee shop in a brick alley.",
           "notes": "Street window on the south side."}


class RoomSpecTests(unittest.TestCase):
    def test_parse_and_wall_prompts(self):
        spec = rs.parse_spec("Here you go:\n" + ANSWER)
        self.assertEqual(spec["walls"]["back"][:16], "a wall of shelve")
        self.assertEqual(spec["palette"][0], "honey oak")
        left = rs.wall_prompt("left", ELEMENT, spec)
        self.assertIn("looks straight at the left wall", left)
        self.assertIn("That wall holds: brick wall with oak panelling", left)
        self.assertIn("At the left edge of the frame, the start of the back wall", left)
        self.assertIn("At the right edge of the frame, the start of the front wall", left)
        self.assertIn("Materials: polished concrete", left)
        reverse = rs.wall_prompt("reverse", ELEMENT, spec, state="rain on the window")
        self.assertIn("looks straight at the back wall", reverse)
        self.assertIn("State: rain on the window", reverse)
        self.assertIn("nobody in it", reverse)
        # Without a sheet the description and layout notes stand in.
        bare = rs.wall_prompt("right", ELEMENT, None)
        self.assertIn("Place: A cozy coffee shop", bare)
        self.assertIn("Layout: Street window", bare)
        with self.assertRaises(rs.RoomSpecError):
            rs.wall_prompt("top", ELEMENT, spec)
        with self.assertRaises(rs.RoomSpecError):
            rs.parse_spec('{"walls": {"front": "x"}}')

    def test_ensure_spec_asks_once_and_stores_a_sidecar(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        with patch.object(bibles, "BIBLES_DIR", Path(tmp)):
            bible = bibles.save_bible({"name": "Spring"})
            calls = []

            class Backend:
                provider = "openai"
                available = True
                model = "analyst"

                def ask(self, http, key, model, system, prompt, media, sleep, **kw):
                    calls.append((model, prompt, media))
                    return ANSWER, 0.0

            plate = Path(tmp, bible["id"], "plate.jpg")
            plate.write_bytes(b"jpg")
            spec = rs.ensure_spec(bible["id"], ELEMENT, str(plate), backend=Backend())
            self.assertEqual(spec["walls"]["front"][:19], "the picture window ")
            self.assertEqual(calls[0][0], "analyst")
            self.assertEqual(calls[0][2], [str(plate)])
            self.assertTrue(rs.spec_path(bible["id"], "cafe").is_file())
            rs.ensure_spec(bible["id"], ELEMENT, str(plate), backend=Backend())
            self.assertEqual(len(calls), 1)
            # No analyst: no sheet, no error.
            self.assertIsNone(rs.ensure_spec(bible["id"], {"id": "other", "name": "X"}, str(plate), backend=None))


if __name__ == "__main__":
    unittest.main()
