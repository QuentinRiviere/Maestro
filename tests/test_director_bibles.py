"""Bibles keep a series' cast, style and world rules for every project made from them."""
from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services.director import bibles  # noqa: E402

GIRL = {"name": "Eun-seo", "role": "lead", "descriptor": "the teen girl with see-through bangs",
        "description": "About 17, Korean, long black hair with see-through bangs.",
        "wardrobe": "navy blazer, grey pleated skirt"}
HALL = {"name": "Hallway", "description": "School corridor, lockers on the right, windows on the left."}


class BibleStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.project = os.path.join(self.tmp.name, "project")
        os.makedirs(self.project)

    def _image(self, name):
        with open(os.path.join(self.project, name), "wb") as handle:
            handle.write(b"jpeg")
        return name

    def test_a_project_cast_becomes_a_bible_with_its_own_images(self):
        cast = {"characters": [{**GIRL, "image": self._image("girl.jpg")}],
                "locations": [{**HALL, "image": self._image("hall.jpg")}]}
        bible = bibles.add_cast(cast, [self.project], name="Spring Rain", style={"medium": "live action"})

        self.assertEqual(bible["id"], "spring_rain")
        self.assertEqual(bible["revision"], 2)
        self.assertEqual(bible["style"], {"medium": "live action"})
        image = bible["characters"][0]["image"]
        self.assertTrue(image.startswith("char_eun_seo_"))
        self.assertIsNotNone(bibles.image_path("spring_rain", image))
        # The bible no longer depends on the project's folder.
        os.remove(os.path.join(self.project, "girl.jpg"))
        self.assertIsNotNone(bibles.image_path("spring_rain", image))
        self.assertEqual([b["id"] for b in bibles.list_bibles()], ["spring_rain"])

    def test_saving_a_later_cast_replaces_entries_by_name_and_keeps_the_others(self):
        bibles.add_cast({"characters": [GIRL], "locations": [HALL]}, [self.project], name="Spring Rain")
        boy = {"name": "Ji-ho", "description": "About 18, Korean, short black hair."}
        older_girl = {**GIRL, "wardrobe": "winter coat"}
        bible = bibles.add_cast({"characters": [older_girl, boy]}, [self.project], bible_id="spring_rain")

        self.assertEqual([c["name"] for c in bible["characters"]], ["Eun-seo", "Ji-ho"])
        self.assertEqual(bible["characters"][0]["wardrobe"], "winter coat")
        self.assertEqual([l["name"] for l in bible["locations"]], ["Hallway"])

    def test_an_episode_picks_entries_and_gets_copies_in_its_folder(self):
        cast = {"characters": [{**GIRL, "image": self._image("girl.jpg")}, {"name": "Ji-ho", "description": "boy"}],
                "locations": [{**HALL, "image": self._image("hall.jpg")}]}
        bible = bibles.add_cast(cast, [self.project], name="Spring Rain")
        episode = os.path.join(self.tmp.name, "episode2")

        picked = bibles.project_cast(bible, episode, characters=["eun-seo"])

        self.assertEqual([c["name"] for c in picked["characters"]], ["Eun-seo"])
        self.assertTrue(os.path.isfile(os.path.join(episode, picked["characters"][0]["image"])))
        self.assertEqual(len(picked["locations"]), 1)

    def test_ids_and_image_names_cannot_leave_the_bible_folder(self):
        with self.assertRaises(bibles.BibleError):
            bibles.get_bible("../secrets") or bibles._folder("../secrets")
        bibles.save_bible({"name": "Spring Rain"})
        self.assertIsNone(bibles.image_path("spring_rain", "../../bible.json"))
        self.assertIsNone(bibles.image_path("spring_rain", "bible.json"))

    def test_world_rules_reach_the_planner(self):
        bible = bibles.save_bible({"name": "Spring Rain", "world": {"logline": "First love in a Busan high school",
                                                                    "forbidden": "phones before episode 3"}})
        block = bibles.world_block(bible)
        self.assertIn("Series premise: First love in a Busan high school", block)
        self.assertIn("Never: phones before episode 3", block)
        self.assertEqual(bibles.world_block(bibles.save_bible({"name": "Empty"})), "")


class BiblePipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        saved = (pipeline._pipelines, pipeline._wgp)
        self.addCleanup(lambda: setattr(pipeline, "_pipelines", saved[0]) or setattr(pipeline, "_wgp", saved[1]))
        pipeline._pipelines = {}
        pipeline._wgp = SimpleNamespace(save_path=self.tmp.name)
        self.out_dir = os.path.join(self.tmp.name, "out")
        pipeline._pipelines["p1"] = {"id": "p1", "status": "running", "progress": {}, "created_at": time.time(),
                                     "params": {}, "out_dir": self.out_dir}

    def test_a_project_starts_from_its_bible(self):
        bibles.save_bible({"name": "Spring Rain", "style": {"medium": "live action"},
                           "characters": [GIRL], "locations": [HALL], "world": {"tone": "tender, rainy"}})
        params = {"bible_id": "spring_rain", "bible_locations": ["Hallway"]}

        block = pipeline._apply_bible("p1", params)

        self.assertEqual([c["name"] for c in params["_director_cast"]["characters"]], ["Eun-seo"])
        self.assertEqual(params["_director_style_contract"], {"medium": "live action"})
        self.assertEqual(params["director_casting"], "on")
        self.assertEqual(params["_director_bible_revision"], 1)
        self.assertIn("Tone: tender, rainy", block)
        self.assertEqual(pipeline._pipelines["p1"]["bible"]["id"], "spring_rain")

    def test_a_revised_project_keeps_its_own_cast(self):
        bibles.save_bible({"name": "Spring Rain", "characters": [GIRL]})
        own = {"characters": [], "locations": []}
        params = {"bible_id": "spring_rain", "_director_cast": own}
        pipeline._apply_bible("p1", params)
        self.assertIs(params["_director_cast"], own)

    def test_an_unknown_bible_stops_the_project(self):
        with self.assertRaises(ValueError):
            pipeline._apply_bible("p1", {"bible_id": "missing"})


if __name__ == "__main__":
    unittest.main()
