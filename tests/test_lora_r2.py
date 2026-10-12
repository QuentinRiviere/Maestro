"""LoRAs kept in R2: pointers, listing, upload and on-demand download."""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import unittest

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import lora_r2  # noqa: E402


class FakeStore:
    bucket = "bucket"

    def __init__(self):
        self.objects = {}

    def asset_key(self, sha256, name):
        return f"maestro/assets/{sha256}/{name}"

    def upload(self, path, key):
        with open(path, "rb") as handle:
            self.objects[key] = handle.read()

    def download(self, key, path):
        with open(path, "wb") as handle:
            handle.write(self.objects[key])


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


class LoraR2Tests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.weights = os.path.join(self.dir, "style.safetensors")
        with open(self.weights, "wb") as handle:
            handle.write(b"weights")
        self.store = FakeStore()

    def test_r2_only_moves_the_file_and_keeps_it_listed(self):
        messages = []
        pointer = lora_r2.store_in_r2(self.store, self.weights, keep_local=False, sha256=sha256,
                                      update=messages.append)
        self.assertFalse(os.path.exists(self.weights))
        self.assertEqual(self.store.objects[pointer["key"]], b"weights")
        self.assertEqual(pointer["size"], 7)
        self.assertEqual(lora_r2.pointer_weight_names(self.dir), ["style.safetensors"])
        self.assertEqual(lora_r2.storage_of(self.weights), "r2")
        self.assertEqual(messages, ["Uploading style.safetensors to R2…"])

    def test_both_keeps_the_local_file_out_of_pointer_listings(self):
        lora_r2.store_in_r2(self.store, self.weights, keep_local=True, sha256=sha256)
        self.assertTrue(os.path.isfile(self.weights))
        self.assertEqual(lora_r2.pointer_weight_names(self.dir), [])  # the real file lists it
        self.assertEqual(lora_r2.storage_of(self.weights), "both")

    def test_materialize_downloads_once(self):
        lora_r2.store_in_r2(self.store, self.weights, keep_local=False, sha256=sha256)
        self.assertTrue(lora_r2.materialize(self.store, self.weights))
        with open(self.weights, "rb") as handle:
            self.assertEqual(handle.read(), b"weights")
        self.assertFalse(lora_r2.materialize(self.store, self.weights))

    def test_storage_choices_and_bad_pointers(self):
        self.assertEqual(lora_r2.normalize_storage(None), "local")
        self.assertEqual(lora_r2.normalize_storage("R2"), "r2")
        with self.assertRaises(ValueError):
            lora_r2.normalize_storage("s3")
        with open(os.path.join(self.dir, "broken.r2.json"), "w") as handle:
            handle.write("{not json")
        self.assertIsNone(lora_r2.read_pointer(os.path.join(self.dir, "broken.safetensors")))
        self.assertEqual(lora_r2.storage_of(self.weights), "local")


if __name__ == "__main__":
    unittest.main()
