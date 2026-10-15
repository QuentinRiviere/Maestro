"""Model-free tests: each Director shot shows its clip as soon as it renders."""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services.job_lifecycle import (  # noqa: E402
    finish_job,
    record_job_outputs,
    record_live_clip_output,
    try_start,
)


class LiveClipVideoTests(unittest.TestCase):
    def setUp(self):
        self.originals = {
            "pipelines": pipeline._pipelines,
            "jobs": pipeline._jobs,
            "child_jobs": pipeline._pipeline_child_jobs,
            "run_generation": pipeline._run_generation,
            "wgp": pipeline._wgp,
        }
        self.temp_dir = tempfile.TemporaryDirectory()
        pipeline._pipelines = {}
        pipeline._jobs = {}
        pipeline._pipeline_child_jobs = {}
        pipeline._run_generation = None
        pipeline._wgp = SimpleNamespace(save_path=self.temp_dir.name)

    def tearDown(self):
        pipeline._pipelines = self.originals["pipelines"]
        pipeline._jobs = self.originals["jobs"]
        pipeline._pipeline_child_jobs = self.originals["child_jobs"]
        pipeline._run_generation = self.originals["run_generation"]
        pipeline._wgp = self.originals["wgp"]
        self.temp_dir.cleanup()

    def _add_pipeline(self, pid: str, clip_count: int, status: str = "running") -> dict:
        record = {
            "id": pid,
            "status": status,
            "phase": "generating_video",
            "progress": {"current": 0, "total": 1, "message": "Generating video...",
                         "step": 0, "total_steps": 0},
            "clip_plans": [{"video_prompt": f"shot {i}"} for i in range(clip_count)],
            "clip_images": [],
            "output_files": [],
            "created_at": time.time(),
            "params": {"pipeline_type": "short_film_story", "seamless": False},
            "pause_reason": None,
            "out_dir": self.temp_dir.name,
        }
        pipeline._pipelines[pid] = record
        return record

    def _saved_clips(self, pid: str) -> list[dict]:
        state = pipeline.load_pipeline_state(self.temp_dir.name, pid)
        return state["clips"]

    def test_wait_loop_reports_each_clip_while_the_job_runs(self):
        pid = "live-wait"
        self._add_pipeline(pid, 2)
        seen: list[dict] = []
        first_seen = threading.Event()

        def fake_generation(job_id: str):
            job = pipeline._jobs[job_id]
            if not try_start(job):
                return
            record_live_clip_output(job, 0, "clip_0.mp4")
            # The waiter shows clip 0 before clip 1 exists.
            self.assertTrue(first_seen.wait(timeout=5))
            record_live_clip_output(job, 1, "clip_1.mp4")
            time.sleep(1.5)
            record_job_outputs(
                job, ["clip_0.mp4", "clip_1.mp4", "join_multiclip.mp4"],
                clip_output_files={0: "clip_0.mp4", 1: "clip_1.mp4"},
                join_output_file="join_multiclip.mp4",
            )
            finish_job(job, "completed", message="Done")

        def on_clip_output(files: dict) -> None:
            seen.append(files)
            first_seen.set()

        pipeline._run_generation = fake_generation
        outputs = pipeline._submit_and_wait(
            {"_director_pipeline_id": pid}, timeout_s=10,
            out_dir=self.temp_dir.name, on_clip_output=on_clip_output,
        )
        self.assertEqual(seen[0], {"0": "clip_0.mp4"})
        self.assertEqual(seen[-1], {"0": "clip_0.mp4", "1": "clip_1.mp4"})
        # The batch result itself is unchanged.
        self.assertEqual(outputs, ["clip_0.mp4", "clip_1.mp4", "join_multiclip.mp4"])
        self.assertEqual(outputs.clip_output_files, {0: "clip_0.mp4", 1: "clip_1.mp4"})

    def test_a_failing_callback_never_fails_the_generation(self):
        pid = "live-callback-error"
        self._add_pipeline(pid, 1)

        def fake_generation(job_id: str):
            job = pipeline._jobs[job_id]
            if not try_start(job):
                return
            record_live_clip_output(job, 0, "clip_0.mp4")
            time.sleep(1.5)
            record_job_outputs(job, ["clip_0.mp4"], clip_output_files={0: "clip_0.mp4"})
            finish_job(job, "completed", message="Done")

        def broken(files: dict) -> None:
            raise RuntimeError("display failure")

        pipeline._run_generation = fake_generation
        outputs = pipeline._submit_and_wait(
            {"_director_pipeline_id": pid}, timeout_s=10,
            out_dir=self.temp_dir.name, on_clip_output=broken,
        )
        self.assertEqual(outputs, ["clip_0.mp4"])

    def test_published_clips_are_saved_as_provisional_until_the_batch_settles(self):
        pid = "live-save"
        self._add_pipeline(pid, 3)
        self.assertTrue(pipeline._publish_live_clip_videos(pid, {"0": "clip_0.mp4"}, 3))
        # Same map again: nothing to save.
        self.assertFalse(pipeline._publish_live_clip_videos(pid, {"0": "clip_0.mp4"}, 3))
        clips = self._saved_clips(pid)
        self.assertEqual([c["video_filename"] for c in clips], ["clip_0.mp4", None, None])
        self.assertTrue(clips[0]["video_provisional"])
        self.assertNotIn("video_provisional", clips[1])

        # The batch result replaces the provisional entries.
        pipeline._update_pipeline(
            pid, status="completed", phase="completed",
            output_files=["final_0.mp4", "final_1.mp4", "final_2.mp4"],
            _clip_video_files=["final_0.mp4", "final_1.mp4", "final_2.mp4"],
        )
        self.assertTrue(pipeline._save_pipeline_state(pid))
        clips = self._saved_clips(pid)
        self.assertEqual(
            [c["video_filename"] for c in clips],
            ["final_0.mp4", "final_1.mp4", "final_2.mp4"],
        )
        self.assertFalse(any(c.get("video_provisional") for c in clips))

    def test_a_settled_run_never_keeps_provisional_clips(self):
        for status in ("failed", "cancelled"):
            with self.subTest(status=status):
                pid = f"live-{status}"
                record = self._add_pipeline(pid, 2)
                pipeline._publish_live_clip_videos(pid, {"0": "clip_0.mp4"}, 2)
                record["status"] = status
                self.assertTrue(pipeline._save_pipeline_state(pid))
                clips = self._saved_clips(pid)
                self.assertEqual([c["video_filename"] for c in clips], [None, None])
                # Nothing is published once the run settled.
                self.assertFalse(
                    pipeline._publish_live_clip_videos(pid, {"1": "clip_1.mp4"}, 2)
                )

    def test_h3_sequence_units_publish_only_single_shot_units(self):
        pid = "live-units"
        self._add_pipeline(pid, 4)
        # Units: shots 0+1 render together, shot 2 alone, shot 3 alone.
        units = [[0, 1], [2], [3]]
        changed = pipeline._publish_live_clip_videos(
            pid, {"0": "seq_01.mp4", "1": "unit_1.mp4", "2": "unit_2.mp4", "3": "tail.mp4"},
            4, units,
        )
        self.assertTrue(changed)
        clips = self._saved_clips(pid)
        self.assertEqual(
            [c["video_filename"] for c in clips],
            [None, None, "unit_1.mp4", "unit_2.mp4"],
        )

    def test_indexes_outside_the_shot_list_are_ignored(self):
        pid = "live-tail"
        self._add_pipeline(pid, 2)
        # Index 2 is a trim-compensation tail clip, not a shot.
        self.assertFalse(pipeline._publish_live_clip_videos(pid, {"2": "tail.mp4", "x": "y.mp4"}, 2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
