"""An insert's draft that shows faces is cut back to the band below them."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from PIL import Image  # noqa: E402

from services.director import insert_crop  # noqa: E402


class BandTests(unittest.TestCase):
    def test_the_band_keeps_the_aspect_and_centres_under_the_faces(self):
        # 432x768 draft, two faces in the upper third, mean x = 216.
        band = insert_crop.band_below_faces((432, 768), [(60, 100, 90, 90), (282, 120, 90, 90)])
        left, top, width, height = band
        self.assertEqual((width, height), (297, 528))  # 768 - (210 + 30) = 528 high at 9:16 -> 297 wide
        self.assertEqual(top, 768 - 528)
        self.assertEqual(left, round(216 - 297 / 2))
        self.assertAlmostEqual(width / height, 432 / 768, places=2)

    def test_the_band_is_clamped_inside_the_image(self):
        left, _, width, _ = insert_crop.band_below_faces((432, 768), [(0, 100, 60, 60)])
        self.assertEqual(left, 0)
        left, _, width, _ = insert_crop.band_below_faces((432, 768), [(380, 100, 52, 60)])
        self.assertEqual(left + width, 432)

    def test_no_band_when_the_faces_leave_too_little_room(self):
        self.assertIsNone(insert_crop.band_below_faces((432, 768), [(100, 500, 100, 100)]))
        self.assertIsNone(insert_crop.band_below_faces((432, 768), []))


class CropTests(unittest.TestCase):
    def test_the_crop_is_written_at_the_band(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "draft.png")
            Image.new("RGB", (432, 768), "white").save(src)
            dest = insert_crop.crop_below_faces(src, os.path.join(tmp, "band.png"),
                                                detect=lambda path: [(170, 100, 92, 92)])
            self.assertTrue(dest and os.path.isfile(dest))
            with Image.open(dest) as image:
                self.assertAlmostEqual(image.width / image.height, 432 / 768, places=2)
                self.assertLess(image.height, 768 - 192)

    def test_nothing_to_cut_without_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "draft.png")
            Image.new("RGB", (432, 768), "white").save(src)
            self.assertIsNone(insert_crop.crop_below_faces(src, os.path.join(tmp, "band.png"), detect=lambda p: []))

    def test_the_haar_detector_finds_nothing_on_a_blank_image(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "draft.png")
            Image.new("RGB", (432, 768), "white").save(src)
            self.assertEqual(insert_crop.detect_faces(src), [])
            self.assertEqual(insert_crop.detect_faces(os.path.join(tmp, "missing.png")), [])


if __name__ == "__main__":
    unittest.main()
