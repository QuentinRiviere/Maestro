"""Paginated, searchable Director runs list and its newest-first order."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services import director_runs_list as runs_list  # noqa: E402


def _write(directory: str, pid: str, *, mtime: float | None = None, **state) -> str:
    path = os.path.join(directory, f"{pipeline._PIPELINE_FILE_PREFIX}{pid}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"pipeline_id": pid, "clips": [], "output_files": [], **state}, handle)
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


class TimestampTests(unittest.TestCase):
    def test_coerce_timestamp(self):
        coerce = pipeline._coerce_timestamp
        self.assertEqual(coerce(1000), 1000.0)
        self.assertEqual(coerce("1500.5"), 1500.5)
        self.assertEqual(coerce(1_700_000_000_000), 1_700_000_000.0)
        self.assertEqual(coerce("1970-01-01T00:16:40Z"), 1000.0)
        for bad in (None, True, "soon", "", 0, -5, float("nan"), float("inf"), [], {}):
            self.assertIsNone(coerce(bad), bad)

    def test_activity_is_the_latest_known_time(self):
        self.assertEqual(pipeline.pipeline_activity_time({"created_at": 10, "completed_at": 30, "updated_at": 20}), 30)
        self.assertEqual(pipeline.pipeline_activity_time({"created_at": "x"}), 0.0)


class ListOrderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def test_mixed_timestamp_types_sort_instead_of_crashing(self):
        # A string created_at next to numbers used to raise TypeError in
        # sort(), failing the whole endpoint.
        _write(self.out, "num", status="completed", created_at=2000.0, mtime=2000)
        _write(self.out, "text", status="completed", created_at="3000", mtime=1000)
        _write(self.out, "junk", status="completed", created_at="soon", mtime=500)
        _write(self.out, "none", status="completed", mtime=400)
        ids = [item["id"] for item in pipeline.list_pipeline_states(self.out)]
        self.assertEqual(ids, ["text", "num", "junk", "none"])

    def test_cloned_runs_with_the_same_created_at_sort_by_activity(self):
        # Hand-cloned state files kept the source run's created_at; the file
        # write time (updated_at) and completed_at still tell them apart.
        _write(self.out, "source", status="completed", created_at=1000.0, completed_at=1100.0, mtime=1100)
        _write(self.out, "clone_a", status="completed", created_at=1000.0, mtime=5000)
        _write(self.out, "clone_b", status="completed", created_at=1000.0, completed_at=6000.0, mtime=4000)
        _write(self.out, "tie_b", status="completed", created_at=1000.0, mtime=1100)
        rows = pipeline.list_pipeline_states(self.out)
        self.assertEqual([r["id"] for r in rows], ["clone_b", "clone_a", "source", "tie_b"])
        by_id = {r["id"]: r for r in rows}
        self.assertAlmostEqual(by_id["clone_a"]["updated_at"], 5000.0, places=3)
        self.assertEqual(by_id["clone_b"]["activity_at"], 6000.0)

    def test_workspace_subdirectories_are_listed(self):
        os.mkdir(os.path.join(self.out, "ws"))
        _write(os.path.join(self.out, "ws"), "inner", status="completed", created_at=10.0)
        rows = pipeline.list_pipeline_states(self.out)
        self.assertEqual([(r["id"], r["workspace"]) for r in rows], [("inner", "ws")])

    def test_unchanged_files_are_served_from_the_summary_cache(self):
        _write(self.out, "a", status="completed", created_at=10.0, mtime=10)
        pipeline.list_pipeline_states(self.out)
        with patch.object(pipeline, "repair_payload", side_effect=AssertionError("re-read")):
            rows = pipeline.list_pipeline_states(self.out)
        self.assertEqual([r["id"] for r in rows], ["a"])

        # A rewrite invalidates the cached summary.
        _write(self.out, "a", status="failed", created_at=10.0, mtime=20)
        self.assertEqual(pipeline.list_pipeline_states(self.out)[0]["status"], "failed")

        os.remove(os.path.join(self.out, f"{pipeline._PIPELINE_FILE_PREFIX}a.json"))
        self.assertEqual(pipeline.list_pipeline_states(self.out), [])
        self.assertFalse(any(path.startswith(self.out) for path in pipeline._pipeline_summary_cache))

    def test_crash_normalisation_keeps_the_run_in_place(self):
        _write(self.out, "stale", status="running", created_at=10.0, mtime=50)
        _write(self.out, "newer", status="completed", created_at=100.0, mtime=100)
        rows = pipeline.list_pipeline_states(self.out)
        self.assertEqual([(r["id"], r["status"]) for r in rows], [("newer", "completed"), ("stale", "crashed")])
        self.assertAlmostEqual(rows[1]["updated_at"], 50.0, places=3)
        with open(os.path.join(self.out, f"{pipeline._PIPELINE_FILE_PREFIX}stale.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["status"], "crashed")


class PageTests(unittest.TestCase):
    def _rows(self):
        return [
            {"id": f"run{i:02d}", "status": "completed" if i % 2 else "failed",
             "created_at": 1000.0 + i, "updated_at": 1000.0 + i,
             "_search_text": f"run{i:02d} {'lighthouse' if i % 3 == 0 else 'forest'} keeper"}
            for i in range(10)
        ]

    def test_pages_cover_every_row_once_in_order(self):
        rows = self._rows()
        seen = []
        cursor = None
        while True:
            page = runs_list.page_pipeline_summaries(rows, limit=3, cursor=cursor)
            self.assertEqual(page["total"], 10)
            seen.extend(item["id"] for item in page["pipelines"])
            cursor = page["next_cursor"]
            if not cursor:
                break
        self.assertEqual(seen, [f"run{i:02d}" for i in range(9, -1, -1)])

    def test_rows_added_between_pages_do_not_shift_the_next_page(self):
        rows = self._rows()
        first = runs_list.page_pipeline_summaries(rows, limit=4)
        rows.append({"id": "brand_new", "status": "running", "created_at": 9999.0})
        second = runs_list.page_pipeline_summaries(rows, limit=4, cursor=first["next_cursor"])
        self.assertEqual([r["id"] for r in second["pipelines"]], ["run05", "run04", "run03", "run02"])

    def test_search_and_status_filters(self):
        rows = self._rows()
        page = runs_list.page_pipeline_summaries(rows, q="  Lighthouse ", status="failed")
        self.assertEqual([r["id"] for r in page["pipelines"]], ["run06", "run00"])
        self.assertIsNone(page["next_cursor"])
        page = runs_list.page_pipeline_summaries(rows, q="run07")
        self.assertEqual([r["id"] for r in page["pipelines"]], ["run07"])
        repairing = runs_list.page_pipeline_summaries(
            [{"id": "a", "status": "completed", "repair_status": "running"},
             {"id": "b", "status": "running"}, {"id": "c", "status": "completed"}],
            status="running,repairing",
        )
        self.assertEqual(sorted(r["id"] for r in repairing["pipelines"]), ["a", "b"])

    def test_private_keys_are_dropped_and_thumbnails_added(self):
        page = runs_list.page_pipeline_summaries(
            [{"id": "a b", "clip_count": 2, "_filepath": "/x", "_search_text": "a"}])
        item = page["pipelines"][0]
        self.assertNotIn("_filepath", item)
        self.assertNotIn("_search_text", item)
        self.assertEqual(item["thumbnail_url"], "/api/v1/director/pipelines/a%20b/thumbnail")

    def test_bad_cursor_is_rejected(self):
        with self.assertRaises(runs_list.InvalidCursorError):
            runs_list.decode_cursor("not-a-cursor")


class RouteTests(unittest.TestCase):
    def setUp(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        self._tmp = tempfile.TemporaryDirectory()
        self.out = self._tmp.name
        for i in range(5):
            _write(self.out, f"r{i}", status="completed", created_at=100.0 + i,
                   scene_description=f"Scene {i} by the sea" if i % 2 else f"Scene {i} in town",
                   mtime=100 + i)
        api = FastAPI()
        runs_list.register_routes(api, get_base_dir=lambda: self.out)
        self.client = TestClient(api)

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_params_returns_every_run_newest_first(self):
        body = self.client.get("/api/v1/director/pipelines").json()
        self.assertEqual([r["id"] for r in body["pipelines"]], ["r4", "r3", "r2", "r1", "r0"])
        self.assertIsNone(body["next_cursor"])
        self.assertIn("updated_at", body["pipelines"][0])
        self.assertNotIn("_filepath", body["pipelines"][0])

    def test_paged_search_over_http(self):
        first = self.client.get("/api/v1/director/pipelines", params={"limit": 1, "q": "sea"}).json()
        self.assertEqual([r["id"] for r in first["pipelines"]], ["r3"])
        self.assertEqual(first["total"], 2)
        second = self.client.get(
            "/api/v1/director/pipelines",
            params={"limit": 1, "q": "sea", "cursor": first["next_cursor"]},
        ).json()
        self.assertEqual([r["id"] for r in second["pipelines"]], ["r1"])
        self.assertIsNone(second["next_cursor"])

    def test_invalid_cursor_is_a_400(self):
        response = self.client.get("/api/v1/director/pipelines", params={"cursor": "%%%"})
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
