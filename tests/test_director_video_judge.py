"""Read-only Director video judge: segments, verdict parsing, review loop."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

import httpx

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director import video_judge  # noqa: E402


def clip(start, end, **extra):
    return {"planned_clip": {"start": start, "end": end, "duration_sec": end - start, "label": "hook"},
            "video_prompt": "a barista pours milk", **extra}


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def touch(self, name):
        open(os.path.join(self.dir, name), "wb").close()
        return name

    def test_joined_video_is_cut_on_plan_boundaries(self):
        joined = self.touch("final.mp4")
        clips = [clip(10, 14), clip(14, 19.5), clip(19.5, 25)]  # audio timeline offset
        segments = video_judge.plan_segments(clips, [joined], self.dir)
        self.assertEqual([(s["start"], s["end"]) for s in segments], [(0, 4), (4, 9.5), (9.5, 15)])

    def test_per_clip_files_are_used_whole(self):
        clips = [clip(0, 4, video_filename=self.touch("a.mp4")), clip(4, 8, video_filename=self.touch("b.mp4"))]
        segments = video_judge.plan_segments(clips, [], self.dir)
        self.assertEqual([os.path.basename(s["path"]) for s in segments], ["a.mp4", "b.mp4"])
        self.assertIsNone(segments[0]["start"])

    def test_nothing_rendered(self):
        self.assertEqual(video_judge.plan_segments([clip(0, 4)], ["missing.mp4"], self.dir), [])


class VerdictTests(unittest.TestCase):
    def test_fenced_json_is_normalised(self):
        text = """```json
{"score": 6.6, "scores": {"plan_adherence": 12, "anatomy_faces_hands": "4"},
 "issues": [{"at_sec": 9, "severity": "huge", "fixable_by": "magic", "description": "mouth warps"},
            {"description": ""}],
 "action": "retake_segment", "retake_range_sec": [1, 30], "summary": "ok"}
```"""
        verdict = video_judge.parse_verdict(text, duration=4)
        self.assertEqual(verdict["score"], 7)
        self.assertEqual(verdict["scores"]["plan_adherence"], 10)
        self.assertEqual(verdict["scores"]["anatomy_faces_hands"], 4)
        self.assertIsNone(verdict["scores"]["continuity"])
        self.assertEqual(verdict["issues"], [{"at_sec": 4.0, "severity": "minor", "fixable_by": "none_model_limit",
                                              "description": "mouth warps"}])
        self.assertEqual(verdict["retake_range_sec"], [1.0, 4.0])

    def test_unusable_answers_raise(self):
        for text in ("no json here", '{"issues": []}', "{bad json}"):
            with self.assertRaises(video_judge.JudgeError):
                video_judge.parse_verdict(text, duration=4)


class JudgeCallTests(unittest.TestCase):
    def test_request_shape_and_retry_on_bad_json(self):
        outputs = iter(["sorry, no json", '{"score": 8, "action": "accept"}'])
        requests = []

        def handler(request):
            requests.append(request)
            url = str(request.url)
            if request.method == "POST":
                return httpx.Response(200, json={"status_url": "https://q/s", "response_url": "https://q/r"})
            if url.endswith("/s"):
                return httpx.Response(200, json={"status": "COMPLETED"})
            return httpx.Response(200, json={"output": next(outputs), "usage": {"cost": 0.006}})

        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            verdict = video_judge.judge_shot("https://r2/clip.mp4", {"duration_sec": 4}, index=0, count=3,
                                             previous=None, fal_key="k", http=http, sleep=lambda _: None)
        self.assertEqual((verdict["score"], verdict["cost_usd"]), (8, 0.006))
        body = json.loads(requests[0].content)
        self.assertEqual(str(requests[0].url), "https://queue.fal.run/openrouter/router/video")
        self.assertTrue(body["reasoning"])
        self.assertEqual(body["video_urls"], ["https://r2/clip.mp4"])
        self.assertIn("Shot 1 of 3", body["prompt"])
        self.assertEqual(sum(1 for r in requests if r.method == "POST"), 2)


class ReviewLoopTests(unittest.TestCase):
    def test_failures_are_recorded_per_shot(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        clips = [clip(0, 4, video_filename="a.mp4"), clip(4, 8, video_filename="b.mp4")]
        for name in ("a.mp4", "b.mp4"):
            open(os.path.join(tmp, name), "wb").close()
        original = video_judge.cut_segment
        video_judge.cut_segment = lambda src, start, end, dest: src
        calls = []

        def judge(url, brief, **kwargs):
            calls.append((url, kwargs["index"], kwargs["previous"]))
            if kwargs["index"] == 1:
                raise video_judge.JudgeError("timeout")
            return {"score": 7, "cost_usd": 0.01}
        try:
            review = video_judge.review_clips(clips, [], tmp, fal_key="k", publish=lambda p: "url:" + p,
                                              judge=judge)
        finally:
            video_judge.cut_segment = original
        self.assertEqual(review["status"], "completed")
        self.assertEqual(review["score"], 7)
        self.assertEqual(review["shots"][1]["error"], "timeout")
        self.assertIsNone(calls[0][2])


if __name__ == "__main__":
    unittest.main()
