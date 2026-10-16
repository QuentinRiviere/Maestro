import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'app'))

from app.services import output_annotations
from app.services.gallery_library import GalleryLibrary


class NormalizeTests(unittest.TestCase):
    def test_trims_note_and_dedupes_tags(self):
        self.assertEqual(
            output_annotations.normalize({"note": "  hand morphs  ", "tags": ["Hand  morph", "hand morph", " ", "keep"]}),
            {"note": "hand morphs", "tags": ["Hand morph", "keep"]},
        )

    def test_comma_separated_tags(self):
        self.assertEqual(output_annotations.normalize({"tags": "keep, redo ,"})["tags"], ["keep", "redo"])

    def test_rejects_bad_bodies(self):
        for body in ({"note": 3}, {"tags": [1]}, {"tags": {"a": 1}},
                     {"note": "x" * (output_annotations.MAX_NOTE_LENGTH + 1)},
                     {"tags": ["x" * (output_annotations.MAX_TAG_LENGTH + 1)]},
                     {"tags": [f"t{i}" for i in range(output_annotations.MAX_TAGS + 1)]}):
            with self.assertRaises(ValueError, msg=body):
                output_annotations.normalize(body)


class WriteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_keeps_existing_sidecar_fields(self):
        meta = self.root / "shot.meta.json"
        meta.write_text(json.dumps({"params": {"prompt": "cafe"}, "provenance": {"model": "x"}}), encoding="utf-8")
        record = output_annotations.write(str(meta), {"note": "cup handle appears", "tags": ["redo"]})
        stored = json.loads(meta.read_text(encoding="utf-8"))
        self.assertEqual(stored["params"], {"prompt": "cafe"})
        self.assertEqual(stored["provenance"], {"model": "x"})
        self.assertEqual(stored["annotation"]["note"], "cup handle appears")
        self.assertEqual(stored["annotation"], record)
        self.assertEqual(list(self.root.iterdir()), [meta])

    def test_creates_minimal_sidecar(self):
        meta = self.root / "upload.meta.json"
        output_annotations.write(str(meta), {"note": "good"})
        stored = json.loads(meta.read_text(encoding="utf-8"))
        self.assertIn("created_at", stored)
        self.assertNotIn("params", stored)

    def test_empty_annotation_clears_the_key(self):
        meta = self.root / "shot.meta.json"
        meta.write_text(json.dumps({"params": {}}), encoding="utf-8")
        output_annotations.write(str(meta), {"note": "x"})
        self.assertIsNone(output_annotations.write(str(meta), {"note": " ", "tags": []}))
        self.assertNotIn("annotation", json.loads(meta.read_text(encoding="utf-8")))

    def test_clearing_without_sidecar_writes_nothing(self):
        meta = self.root / "none.meta.json"
        self.assertIsNone(output_annotations.write(str(meta), {}))
        self.assertFalse(meta.exists())


class GalleryAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name) / "default"
        self.folder.mkdir()
        self.folders = [("default", str(self.folder))]
        for name in ("a.png", "b.png"):
            (self.folder / name).write_bytes(b"img")
            (self.folder / name).with_suffix(".meta.json").write_text(
                json.dumps({"params": {"prompt": "studio"}}), encoding="utf-8")
        output_annotations.write(str(self.folder / "a.meta.json"), {"note": "Hand morphs on cup", "tags": ["redo"]})

    def test_listing_flags_filters_and_searches_annotations(self):
        library = GalleryLibrary()
        items = {item["name"]: item for item in library.list(self.folders)["outputs"]}
        self.assertTrue(items["a.png"]["annotated"])
        self.assertFalse(items["b.png"]["annotated"])
        annotated = library.list(self.folders, media_filter="annotated")["outputs"]
        self.assertEqual([item["name"] for item in annotated], ["a.png"])
        found = library.list(self.folders, search="hand morphs")["outputs"]
        self.assertEqual([item["name"] for item in found], ["a.png"])
        self.assertEqual([i["name"] for i in library.list(self.folders, search="redo")["outputs"]], ["a.png"])


if __name__ == "__main__":
    unittest.main()
