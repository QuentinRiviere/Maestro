"""The Director runs list carries each run's completion time and duration."""
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

from services import director_pipeline as pipeline  # noqa: E402


class TestDirectorRunsList(unittest.TestCase):
    def _write(self, directory: str, pid: str, **state) -> None:
        path = os.path.join(directory, f"{pipeline._PIPELINE_FILE_PREFIX}{pid}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"pipeline_id": pid, "clips": [], "output_files": [], **state}, handle)

    def test_summaries_include_completion_and_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, "done", status="completed", created_at=1000.0, completed_at=1125.5)
            self._write(directory, "failed", status="failed", created_at=2000.0, total_time_sec=42.04)
            self._write(directory, "legacy", status="cancelled", created_at=3000.0)

            summaries = pipeline.list_pipeline_states(directory)

        by_id = {item["id"]: item for item in summaries}
        self.assertEqual([item["id"] for item in summaries], ["legacy", "failed", "done"])
        self.assertEqual(by_id["done"]["completed_at"], 1125.5)
        self.assertEqual(by_id["done"]["duration_sec"], 125.5)
        self.assertIsNone(by_id["failed"]["completed_at"])
        self.assertEqual(by_id["failed"]["duration_sec"], 42.0)
        self.assertIsNone(by_id["legacy"]["duration_sec"])

    def test_bad_timestamps_do_not_break_the_list(self):
        with tempfile.TemporaryDirectory() as directory:
            self._write(directory, "odd", status="completed", created_at="soon", completed_at="later")
            summaries = pipeline.list_pipeline_states(directory)
        self.assertEqual(len(summaries), 1)
        self.assertIsNone(summaries[0]["duration_sec"])


if __name__ == "__main__":
    unittest.main()
