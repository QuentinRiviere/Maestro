"""A Director project's spend: pod runs over the project, cloud calls, $ per final minute."""
from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import cloud_image  # noqa: E402
from services.director import cost_ledger  # noqa: E402


class CostLedgerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(cost_ledger, "COSTS_DIR", Path(self.tmp.name))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_pod_started_for_the_project_is_billed_from_zero_with_boot_as_overhead(self):
        start = 1_000_000.0
        with patch.object(cost_ledger.time, "time", return_value=start + 600):
            # Pod submitted after the project queued its first job: its boot is this project's.
            cost_ledger.record("p1", "gpu", stage="image", run="r1", submitted_at=start + 10, queued_at=start,
                               price_per_hour=1.8, cost_start=0.15, cost_end=0.16, work_seconds=20)
            cost_ledger.record("p1", "gpu", stage="video", run="r1", submitted_at=start + 10, queued_at=start + 30,
                               price_per_hour=1.8, cost_start=0.16, cost_end=0.30, work_seconds=280)
            cost_ledger.record("p1", "cloud", stage="cast", usd=0.06)
            cost_ledger.record("p1", "cloud", stage="cast", usd=None)
        report = cost_ledger.summary("p1", final_seconds=30, rendered_seconds=45)
        self.assertAlmostEqual(report["gpu_usd"], 0.30)
        self.assertAlmostEqual(report["usd"], 0.36)
        self.assertAlmostEqual(report["stages"]["video"]["usd"], 0.14)
        self.assertAlmostEqual(report["stages"]["image"]["usd"], 0.01)
        self.assertAlmostEqual(report["stages"]["overhead"]["usd"], 0.15)
        self.assertEqual(report["unpriced_calls"], 1)
        self.assertAlmostEqual(report["usd_per_final_minute"], 0.72)

    def test_a_pod_already_running_is_billed_only_for_its_growth(self):
        cost_ledger.record("p2", "gpu", stage="video", run="r9", submitted_at=1.0, queued_at=5_000.0,
                           price_per_hour=1.8, cost_start=2.0, cost_end=2.5, work_seconds=900)
        self.assertAlmostEqual(cost_ledger.summary("p2")["gpu_usd"], 0.5)
        self.assertIsNone(cost_ledger.summary("p2")["usd_per_final_minute"])

    def test_lengths_count_what_the_edit_keeps(self):
        final, rendered = cost_ledger.project_lengths([
            {"_director_duration_sec": 6, "_director_keep_sec": [1, 4]},
            {"_director_duration_sec": 5},
        ])
        self.assertEqual((final, rendered), (8.0, 11.0))


class CloudPriceTests(unittest.TestCase):
    def _png(self, width, height):
        from PIL import Image
        buffer = io.BytesIO()
        Image.new("RGB", (width, height)).save(buffer, "PNG")
        return cloud_image.CloudImage(buffer.getvalue(), "image/png")

    def test_megapixel_and_per_image_prices(self):
        spec = {"image_field": "image_urls", "price": {"per_megapixel": 0.02, "edit_per_megapixel": 0.03}}
        image = self._png(1000, 2000)
        self.assertAlmostEqual(cloud_image.estimate_cost(spec, {}, [image]), 0.04)
        with patch.object(cloud_image, "input_images", return_value=["ref.png"]):
            self.assertAlmostEqual(cloud_image.estimate_cost(spec, {}, [image]), 0.06)
        self.assertAlmostEqual(cloud_image.estimate_cost({"price": {"per_image": 0.04}}, {}, [image, image]), 0.08)
        self.assertIsNone(cloud_image.estimate_cost({}, {}, [image]))


if __name__ == "__main__":
    unittest.main()
