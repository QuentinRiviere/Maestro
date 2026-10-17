"""TextEncoderCache: keyed hits and misses, the entry bound, clearing, and the reference image key."""
from __future__ import annotations

import os
import sys
import unittest

import torch
from PIL import Image

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from shared.utils.text_encoder_cache import TextEncoderCache, reference_images_key  # noqa: E402


def picture(colour, size=(8, 8)):
    return Image.new("RGB", size, colour)


class ReferenceImagesKeyTests(unittest.TestCase):
    def test_same_pixels_same_key_and_any_difference_another(self):
        red, blue = picture((255, 0, 0)), picture((0, 0, 255))
        self.assertEqual(reference_images_key([red, blue]), reference_images_key([picture((255, 0, 0)), blue]))
        self.assertNotEqual(reference_images_key([red, blue]), reference_images_key([blue, red]))
        self.assertNotEqual(reference_images_key([red]), reference_images_key([red, blue]))
        self.assertNotEqual(reference_images_key([red]), reference_images_key([picture((255, 0, 0), (16, 16))]))
        self.assertEqual(reference_images_key(None), ())
        self.assertEqual(reference_images_key(red), reference_images_key([red]))
        hash(reference_images_key([red, blue]))


class KeyedCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = TextEncoderCache(max_entries=4)
        self.calls = []

    def encode(self, prompts):
        self.calls.append(list(prompts))
        return [(torch.full((3, 2), float(len(self.calls))), torch.ones(3)) for _ in prompts]

    def test_hits_on_the_same_prompt_and_references_misses_on_either_changing(self):
        refs = reference_images_key([picture((1, 2, 3))])
        first = self.cache.encode(self.encode, ["a shot"], cache_keys=[("a shot", refs)])
        again = self.cache.encode(self.encode, ["a shot"], cache_keys=[("a shot", refs)])
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(torch.equal(first[0][0], again[0][0]))
        self.cache.encode(self.encode, ["a shot"], cache_keys=[("a shot", reference_images_key([picture((9, 9, 9))]))])
        self.cache.encode(self.encode, ["another shot"], cache_keys=[("another shot", refs)])
        self.assertEqual(len(self.calls), 3)
        self.assertEqual(len(self.cache), 3)

    def test_the_entry_bound_drops_the_least_recently_used(self):
        for number in range(5):
            self.cache.encode(self.encode, [f"p{number}"], cache_keys=[("p", number)])
        self.assertEqual(len(self.cache), 4)
        self.cache.encode(self.encode, ["p0"], cache_keys=[("p", 0)])
        self.assertEqual(len(self.calls), 6)  # p0 was evicted and encoded again
        self.cache.encode(self.encode, ["p4"], cache_keys=[("p", 4)])
        self.assertEqual(len(self.calls), 6)  # p4 is still there

    def test_clear_forgets_everything(self):
        self.cache.encode(self.encode, ["p"], cache_keys=[("p", 1)])
        self.cache.clear()
        self.assertEqual(len(self.cache), 0)
        self.cache.encode(self.encode, ["p"], cache_keys=[("p", 1)])
        self.assertEqual(len(self.calls), 2)

    def test_unbounded_by_default(self):
        cache = TextEncoderCache()
        for number in range(6):
            cache.encode(self.encode, [f"p{number}"])
        self.assertEqual(len(cache), 6)


if __name__ == "__main__":
    unittest.main()
