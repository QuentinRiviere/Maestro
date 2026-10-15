"""Post-analysis annotations on Director runs: storage, validation, routes."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_annotations as notes  # noqa: E402
from services import director_pipeline as pipeline  # noqa: E402


def _write_state(directory: str, pid: str, **state) -> str:
    path = os.path.join(directory, f"{pipeline._PIPELINE_FILE_PREFIX}{pid}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({
            "pipeline_id": pid,
            "status": "completed",
            "scene_description": "A lighthouse keeper finds a message in a bottle.",
            "clips": [
                {"index": 0, "video_prompt": "Wide shot of the lighthouse at dusk.", "image_prompt": "lighthouse"},
                {"index": 1, "video_prompt": "", "image_prompt": "Close-up of the bottle."},
            ],
            "output_files": [],
            **state,
        }, handle)
    return path


def _note(**overrides):
    body = {
        "target": {"kind": "shot", "ref": "0", "node": "clip-0"},
        "category": "video_motion",
        "severity": 3,
        "text": "Camera drifts left although the prompt asks for a locked-off shot.",
        "timecode_sec": 2.5,
    }
    body.update(overrides)
    return body


class AnnotationCrudTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name
        _write_state(self.out, "run1")

    def tearDown(self):
        self._tmp.cleanup()

    def test_create_list_update_delete(self):
        created = notes.create_annotation(self.out, "run1", _note())
        self.assertEqual(created["category"], "video_motion")
        self.assertEqual(created["severity"], 3)
        self.assertEqual(created["timecode_sec"], 2.5)
        self.assertFalse(created["resolved"])
        self.assertEqual(created["target"], {"kind": "shot", "ref": "0", "node": "clip-0"})
        self.assertTrue(created["id"])
        self.assertEqual(created["created_at"], created["updated_at"])

        listed = notes.list_annotations(self.out, "run1")
        self.assertEqual([item["id"] for item in listed], [created["id"]])
        self.assertTrue(os.path.isfile(os.path.join(self.out, "_director_annotations_run1.json")))

        updated = notes.update_annotation(self.out, "run1", created["id"], {"resolved": True, "severity": 1})
        self.assertTrue(updated["resolved"])
        self.assertEqual(updated["severity"], 1)
        self.assertEqual(updated["text"], created["text"])
        self.assertGreaterEqual(updated["updated_at"], created["updated_at"])

        notes.delete_annotation(self.out, "run1", created["id"])
        self.assertEqual(notes.list_annotations(self.out, "run1"), [])
        with self.assertRaises(notes.AnnotationNotFoundError):
            notes.delete_annotation(self.out, "run1", created["id"])

    def test_defaults_for_a_run_note(self):
        created = notes.create_annotation(self.out, "run1", {"target": {"kind": "run"}, "text": " Too slow overall "})
        self.assertEqual(created["target"], {"kind": "run", "ref": ""})
        self.assertEqual(created["category"], "other")
        self.assertEqual(created["severity"], 2)
        self.assertEqual(created["text"], "Too slow overall")
        self.assertIsNone(created["timecode_sec"])

    def test_unknown_run_and_annotation(self):
        with self.assertRaises(notes.AnnotationNotFoundError):
            notes.create_annotation(self.out, "missing", _note())
        with self.assertRaises(notes.AnnotationNotFoundError):
            notes.list_annotations(self.out, "../run1")
        with self.assertRaises(notes.AnnotationNotFoundError):
            notes.update_annotation(self.out, "run1", "nope", {"text": "x"})

    def test_run_in_a_workspace_folder_keeps_its_notes_beside_its_state(self):
        workspace = os.path.join(self.out, "ws")
        os.makedirs(workspace)
        _write_state(workspace, "run2")
        notes.create_annotation(self.out, "run2", _note())
        self.assertTrue(os.path.isfile(os.path.join(workspace, "_director_annotations_run2.json")))

    def test_deleting_the_run_removes_its_notes(self):
        notes.create_annotation(self.out, "run1", _note())
        result = pipeline.delete_pipeline(self.out, "run1")
        self.assertTrue(result["ok"])
        self.assertFalse(os.path.exists(os.path.join(self.out, "_director_annotations_run1.json")))


class AnnotationValidationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name
        _write_state(self.out, "run1")

    def tearDown(self):
        self._tmp.cleanup()

    def assertRejected(self, body, partial=False):
        with self.assertRaises(notes.AnnotationValidationError):
            notes.validate_annotation_fields(body, partial=partial)

    def test_rejects_bad_fields(self):
        self.assertRejected("text")
        self.assertRejected(_note(target=None))
        self.assertRejected(_note(target={"kind": "galaxy", "ref": "1"}))
        self.assertRejected(_note(target={"kind": "scene", "ref": ""}))
        self.assertRejected(_note(target={"kind": "shot", "ref": "first"}))
        self.assertRejected(_note(category="vibes"))
        self.assertRejected(_note(severity=0))
        self.assertRejected(_note(severity=4))
        self.assertRejected(_note(severity=True))
        self.assertRejected(_note(severity=2.5))
        self.assertRejected(_note(text="   "))
        self.assertRejected(_note(text=12))
        self.assertRejected(_note(text="x" * (notes.MAX_TEXT_CHARS + 1)))
        self.assertRejected(_note(timecode_sec=-1))
        self.assertRejected(_note(timecode_sec="soon"))
        self.assertRejected(_note(timecode_sec=float("nan")))
        self.assertRejected(_note(resolved="yes"))
        self.assertRejected({}, partial=True)

    def test_accepts_the_text_cap_and_integer_shot_refs(self):
        fields = notes.validate_annotation_fields(
            _note(text="x" * notes.MAX_TEXT_CHARS, target={"kind": "shot", "ref": 4}), partial=False,
        )
        self.assertEqual(fields["target"], {"kind": "shot", "ref": "4"})
        self.assertEqual(len(fields["text"]), notes.MAX_TEXT_CHARS)

    def test_update_can_clear_the_timecode(self):
        created = notes.create_annotation(self.out, "run1", _note())
        updated = notes.update_annotation(self.out, "run1", created["id"], {"timecode_sec": None})
        self.assertIsNone(updated["timecode_sec"])

    def test_invalid_update_leaves_the_note_untouched(self):
        created = notes.create_annotation(self.out, "run1", _note())
        with self.assertRaises(notes.AnnotationValidationError):
            notes.update_annotation(self.out, "run1", created["id"], {"severity": 9})
        self.assertEqual(notes.list_annotations(self.out, "run1")[0]["severity"], 3)


class AnnotationPersistenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()
        with pipeline._pipeline_lock:
            pipeline._pipelines.pop("live1", None)

    def test_notes_survive_a_running_pipeline_state_save(self):
        with pipeline._pipeline_lock:
            pipeline._pipelines["live1"] = {
                "id": "live1",
                "status": "running",
                "phase": "generating_video",
                "out_dir": self.out,
                "params": {"scene_description": "Live run"},
                "clip_plans": [{"video_prompt": "A shot", "image_prompt": "A frame"}],
                "clip_images": [],
                "output_files": [],
            }
        # A note written before the run's first save lands beside the state file.
        first = notes.create_annotation(self.out, "live1", _note())
        self.assertTrue(pipeline._save_pipeline_state("live1"))
        second = notes.create_annotation(self.out, "live1", _note(text="Second note"))
        # The run keeps rewriting its state from memory.
        self.assertTrue(pipeline._save_pipeline_state("live1"))
        self.assertTrue(pipeline._save_pipeline_state("live1"))

        ids = [item["id"] for item in notes.list_annotations(self.out, "live1")]
        self.assertEqual(ids, [first["id"], second["id"]])
        with open(os.path.join(self.out, "_director_pipeline_live1.json"), encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["pipeline_id"], "live1")

    def test_writes_are_allowed_while_a_dashboard_operation_holds_the_run(self):
        _write_state(self.out, "run1")
        self.assertTrue(pipeline._claim_pipeline_operation("run1"))
        try:
            created = notes.create_annotation(self.out, "run1", _note())
        finally:
            pipeline._release_pipeline_operation("run1")
        self.assertEqual(len(notes.list_annotations(self.out, "run1")), 1)
        self.assertTrue(created["id"])

    def test_state_listing_ignores_annotation_files(self):
        _write_state(self.out, "run1")
        notes.create_annotation(self.out, "run1", _note())
        self.assertEqual([item["id"] for item in pipeline.list_pipeline_states(self.out)], ["run1"])


class AnnotationAggregationTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name
        _write_state(self.out, "runA", created_at=100.0)
        workspace = os.path.join(self.out, "ws")
        os.makedirs(workspace)
        _write_state(workspace, "runB", scene_description="Second film", created_at=200.0)

    def tearDown(self):
        self._tmp.cleanup()

    def test_aggregates_every_run_with_context(self):
        shot = notes.create_annotation(self.out, "runA", _note())
        image_only = notes.create_annotation(
            self.out, "runA", _note(target={"kind": "shot", "ref": "1"}, category="image"),
        )
        scene = notes.create_annotation(
            self.out, "runB", _note(target={"kind": "scene", "ref": "scene-1"}, category="continuity", severity=1),
        )
        notes.update_annotation(self.out, "runB", scene["id"], {"resolved": True})

        rows = notes.aggregate_annotations(self.out)
        by_id = {row["id"]: row for row in rows}
        self.assertEqual(set(by_id), {shot["id"], image_only["id"], scene["id"]})
        self.assertEqual(by_id[shot["id"]]["pid"], "runA")
        self.assertEqual(by_id[shot["id"]]["workspace"], "default")
        self.assertIn("lighthouse keeper", by_id[shot["id"]]["run_description"])
        self.assertEqual(by_id[shot["id"]]["shot_prompt_excerpt"], "Wide shot of the lighthouse at dusk.")
        self.assertEqual(by_id[image_only["id"]]["shot_prompt_excerpt"], "Close-up of the bottle.")
        self.assertEqual(by_id[scene["id"]]["workspace"], "ws")
        self.assertEqual(by_id[scene["id"]]["run_description"], "Second film")
        self.assertIsNone(by_id[scene["id"]]["shot_prompt_excerpt"])
        self.assertTrue(by_id[scene["id"]]["resolved"])

        summary = notes.summarize(rows)
        self.assertEqual(summary["total"], 3)
        self.assertEqual(summary["open"], 2)
        self.assertEqual(summary["runs"], 2)
        self.assertEqual(summary["by_category"], {"video_motion": 1, "image": 1, "continuity": 1})


class AnnotationRouteTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name
        _write_state(self.out, "run1")
        api = FastAPI()
        notes.register_routes(api, get_base_dir=lambda: self.out)
        self.client = TestClient(api)

    def tearDown(self):
        self._tmp.cleanup()

    def test_crud_over_http(self):
        base = "/api/v1/director/pipelines/run1/annotations"
        response = self.client.post(base, json=_note())
        self.assertEqual(response.status_code, 200)
        aid = response.json()["annotation"]["id"]

        self.assertEqual([a["id"] for a in self.client.get(base).json()["annotations"]], [aid])

        response = self.client.put(f"{base}/{aid}", json={"text": "Edited"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["annotation"]["text"], "Edited")

        self.assertEqual(self.client.post(base, json=_note(severity=7)).status_code, 400)
        self.assertEqual(
            self.client.post(base, content=b"not json", headers={"content-type": "application/json"}).status_code,
            400,
        )
        self.assertEqual(self.client.post("/api/v1/director/pipelines/nope/annotations", json=_note()).status_code, 404)
        self.assertEqual(self.client.put(f"{base}/missing", json={"text": "x"}).status_code, 404)

        all_rows = self.client.get("/api/v1/director/annotations").json()
        self.assertEqual(all_rows["summary"]["total"], 1)
        self.assertEqual(all_rows["annotations"][0]["pid"], "run1")

        export = self.client.get("/api/v1/director/annotations/export")
        self.assertEqual(export.status_code, 200)
        self.assertIn("attachment", export.headers["content-disposition"])
        self.assertEqual(json.loads(export.content)["annotations"][0]["text"], "Edited")

        self.assertEqual(self.client.delete(f"{base}/{aid}").status_code, 200)
        self.assertEqual(self.client.delete(f"{base}/{aid}").status_code, 404)
        self.assertEqual(self.client.get(base).json()["annotations"], [])


if __name__ == "__main__":
    unittest.main()
