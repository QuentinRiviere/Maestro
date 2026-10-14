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
{"score": 6.6, "scores": {"plan_adherence": 12, "temporal_stability": "4"},
 "issues": [{"at_sec": 9, "severity": "huge", "category": "morphing", "fixable_by": "magic",
             "description": "mouth warps"},
            {"description": ""}],
 "action": "retake_segment", "retake_range_sec": [1, 30], "summary": "ok"}
```"""
        verdict = video_judge.parse_verdict(text, duration=4)
        self.assertEqual(verdict["score"], 7)
        self.assertEqual(verdict["scores"]["plan_adherence"], 10)
        self.assertEqual(verdict["scores"]["temporal_stability"], 4)
        self.assertIsNone(verdict["scores"]["audio_fit"])
        self.assertEqual(verdict["issues"], [{"at_sec": 4.0, "severity": "minor", "category": "morphing",
                                              "fixable_by": "none_model_limit", "description": "mouth warps"}])
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


def fake_fal(outputs):
    """MockTransport answering the fal queue with scripted outputs."""
    answers = iter(outputs)
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == "POST":
            return httpx.Response(200, json={"status_url": "https://q/s", "response_url": "https://q/r"})
        if str(request.url).endswith("/s"):
            return httpx.Response(200, json={"status": "COMPLETED"})
        return httpx.Response(200, json={"output": next(answers), "usage": {"cost": 0.002}})
    return handler, requests


class CompareTakesTests(unittest.TestCase):
    def run_compare(self, first, second):
        handler, requests = fake_fal([first, second])
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            result = video_judge.compare_takes("https://a", "https://b", {"label": "hook"}, fal_key="k",
                                               style_contract={"medium": "2D anime"}, http=http,
                                               sleep=lambda _: None)
        return result, requests

    def test_consistent_preference_across_orders_wins(self):
        result, requests = self.run_compare('{"winner": "B", "confidence": 0.8}', '{"winner": "A", "confidence": 0.6}')
        self.assertEqual((result["winner"], result["confidence"]), ("B", 0.7))
        bodies = [json.loads(r.content) for r in requests if r.method == "POST"]
        self.assertEqual([b["video_urls"] for b in bodies], [["https://a", "https://b"], ["https://b", "https://a"]])
        self.assertIn("2D anime", bodies[0]["prompt"])
        self.assertEqual(result["cost_usd"], 0.004)

    def test_position_bias_gives_a_tie(self):
        result, _ = self.run_compare('{"winner": "A", "confidence": 0.9}', '{"winner": "A", "confidence": 0.9}')
        self.assertEqual((result["winner"], result["confidence"]), ("tie", 0.0))


class SelectTakesTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.original_cut = video_judge.cut_segment
        video_judge.cut_segment = lambda src, start, end, dest: src
        self.addCleanup(setattr, video_judge, "cut_segment", self.original_cut)

    def test_challenger_must_win_confidently_and_loser_is_restored(self):
        clips = [clip(0, 4, video_filename="a1.mp4"), clip(4, 8, video_filename="b1.mp4"), clip(8, 9)]
        renders = iter(["a2.mp4", "b2.mp4"])
        verdicts = iter([{"winner": "B", "confidence": 0.8}, {"winner": "B", "confidence": 0.3}])
        restored = []
        results = video_judge.select_takes(
            clips, self.dir, extra_takes=1, rerun=lambda i: next(renders),
            restore=lambda i, name: restored.append((i, name)), publish=lambda p: p,
            compare=lambda a, b, brief: next(verdicts))
        self.assertEqual([r["kept"] for r in results], ["a2.mp4", "b1.mp4", None])
        self.assertEqual(restored, [(1, "b1.mp4")])  # low-confidence win keeps the incumbent
        self.assertIn("no video of its own", results[2]["error"])

    def test_render_failure_keeps_the_original(self):
        def boom(index):
            raise RuntimeError("worker down")
        results = video_judge.select_takes(
            [clip(0, 4, video_filename="a1.mp4")], self.dir, extra_takes=2, rerun=boom,
            restore=lambda i, n: None, publish=lambda p: p, compare=lambda *a: {})
        self.assertEqual((results[0]["kept"], results[0]["error"]), ("a1.mp4", "worker down"))

    def test_per_shot_counts_are_honored(self):
        clips = [clip(0, 4, video_filename="a1.mp4"), clip(4, 8, video_filename="b1.mp4"),
                 clip(8, 9, video_filename="c1.mp4")]
        rendered = []

        def rerun(index):
            rendered.append(index)
            return f"s{index}_{len(rendered)}.mp4"
        results = video_judge.select_takes(
            clips, self.dir, extra_takes=[2, 0, 1], rerun=rerun, restore=lambda i, n: None,
            publish=lambda p: p, compare=lambda *a: {"winner": "A", "confidence": 0.9})
        self.assertEqual(rendered, [0, 0, 2])
        self.assertEqual([len(r["takes"]) for r in results], [3, 1, 2])
        self.assertEqual([r["kept"] for r in results], ["a1.mp4", "b1.mp4", "c1.mp4"])


class ShotTakesTests(unittest.TestCase):
    def shot(self, scene_type="", framing="", dialogue=None, strategy="independent"):
        return {"_director_scene_type": scene_type, "_director_camera_framing": framing,
                "_director_dialogue_beats": dialogue or [], "_director_continuity_strategy": strategy}

    def test_shot_type_moves_the_baseline(self):
        line = [{"spoken_text": "We leave tonight.", "speaker_id": "mara"}]
        clips = [
            self.shot("establishing", "wide shot"),
            self.shot("exchange", "medium shot", dialogue=line),
            self.shot("reaction", "close-up"),
            self.shot("insert", "extreme close-up"),
            self.shot("action", "close-up"),
            self.shot("exchange", "medium shot", dialogue=[{"spoken_text": "  "}]),
        ]
        self.assertEqual(video_judge.shot_takes(clips, 2), [2, 3, 3, 1, 2, 2])
        self.assertEqual(video_judge.shot_takes(clips, 3), [3, 3, 3, 1, 3, 3])

    def test_shot_chained_into_the_next_keeps_one_take(self):
        line = [{"spoken_text": "Now."}]
        clips = [self.shot("exchange", "close-up", dialogue=line),
                 self.shot("exchange", "close-up", dialogue=line, strategy="extend_previous"),
                 self.shot("action", "wide shot")]
        self.assertEqual(video_judge.shot_takes(clips, 2), [1, 3, 2])

    def test_legacy_clips_without_plan_fields_use_the_default(self):
        self.assertEqual(video_judge.shot_takes([clip(0, 4), clip(4, 8)], 2), [2, 2])

    def test_cap_limits_performance_shots(self):
        clips = [self.shot("reaction", "CU")]
        self.assertEqual(video_judge.shot_takes(clips, 2, maximum=2), [2])
        self.assertEqual(video_judge.shot_takes(clips, 2), [3])


class StyleContractTests(unittest.TestCase):
    def test_contract_parsing_and_planner_block(self):
        from services.director.style_contract import contract_block, derive_style_contract, parse_contract

        contract = parse_contract('{"medium": "3D animated feature", "allowed_stylization": ["squash and stretch"],'
                                  ' "must_avoid": ["melting faces"], "finishing": {"film_grain": false}}')
        self.assertEqual(contract["medium"], "3D animated feature")
        block = contract_block(contract)
        self.assertIn("Intended stylisation: squash and stretch", block)
        self.assertIn("melting faces", block)
        self.assertEqual(contract_block({"medium": "unspecified"}), "")

        def broken(**kwargs):
            raise RuntimeError("llm offline")
        self.assertEqual(derive_style_contract("A cat", broken)["medium"], "unspecified")


class EpisodeSourceTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def touch(self, name):
        open(os.path.join(self.dir, name), "wb").close()
        return name

    def test_stitched_multiclip_is_preferred(self):
        clips = [clip(0, 4, video_filename=self.touch("a.mp4")), clip(4, 8, video_filename=self.touch("b.mp4"))]
        outputs = ["a.mp4", "b.mp4", self.touch("old_multiclip.mp4"), self.touch("1_rejoin_multiclip.mp4"),
                   "gone_multiclip.mp4"]
        paths, kind = video_judge.episode_source(clips, outputs, self.dir)
        self.assertEqual((kind, [os.path.basename(p) for p in paths]), ("multiclip", ["1_rejoin_multiclip.mp4"]))

    def test_shots_are_concatenated_without_a_join(self):
        clips = [clip(0, 4, video_filename=self.touch("a.mp4")), clip(4, 8, video_filename=self.touch("b.mp4"))]
        paths, kind = video_judge.episode_source(clips, ["a.mp4", "b.mp4"], self.dir)
        self.assertEqual((kind, [os.path.basename(p) for p in paths]), ("concat", ["a.mp4", "b.mp4"]))

    def test_seamless_render_and_incomplete_projects(self):
        joined = self.touch("seamless.mp4")
        self.assertEqual(video_judge.episode_source([clip(0, 4), clip(4, 8)], [joined], self.dir)[1], "joined")
        partial = [clip(0, 4, video_filename=self.touch("a.mp4")), clip(4, 8)]
        self.assertIsNone(video_judge.episode_source(partial, ["a.mp4"], self.dir))

    def test_timeline_uses_real_lengths_when_known(self):
        clips = [clip(10, 14, video_filename=self.touch("a.mp4")), clip(14, 20)]
        timeline = video_judge.episode_timeline(clips, self.dir, probe=lambda path: 4.5)
        self.assertEqual(timeline, [{"shot": 1, "start_sec": 0.0, "end_sec": 4.5},
                                    {"shot": 2, "start_sec": 4.5, "end_sec": 10.5}])


TIMELINE = [{"shot": 1, "start_sec": 0.0, "end_sec": 4.0}, {"shot": 2, "start_sec": 4.0, "end_sec": 9.0}]


class EpisodeVerdictTests(unittest.TestCase):
    def test_defects_and_fixes_are_normalised(self):
        text = """Here is my review:
{"score": 7, "verdict": "needs_polish", "summary": "Coverage is thin.",
 "scores": {"scene_geography": 6, "authenticity": 7, "acting": "11"},
 "defects": [{"at_sec": 5.2, "shot": 9, "severity": "major", "criterion": "cut_continuity",
              "root_cause": "planner", "description": "cup jumps hands across the cut"},
             {"at_sec": 50, "severity": "nope", "criterion": "vibes", "root_cause": "gremlins",
              "description": "orbiting camera without motivation"},
             {"description": "  "}],
 "skill_fixes": [{"field": "rules.camera", "change": "Static frames by default."},
                 {"field": "rules.colour", "change": "x"}, {"field": "rules.pacing", "change": ""}],
 "planner_fixes": ["Add a reverse shot for the reply", "", 3],
 "pipeline_fixes": "not a list"}"""
        verdict = video_judge.parse_episode_verdict(text, 9.0, TIMELINE)
        self.assertEqual((verdict["score"], verdict["verdict"]), (7, "needs_polish"))
        self.assertEqual(verdict["scores"]["acting"], 10)
        self.assertIsNone(verdict["scores"]["coverage"])
        self.assertEqual(verdict["defects"], [
            {"at_sec": 5.2, "shot": 2, "severity": "major", "criterion": "cut_continuity", "root_cause": "planner",
             "description": "cup jumps hands across the cut"},
            {"at_sec": 9.0, "shot": 2, "severity": "minor", "criterion": "authenticity",
             "root_cause": "model_limit", "description": "orbiting camera without motivation"}])
        self.assertEqual(verdict["skill_fixes"], [{"field": "rules.camera", "change": "Static frames by default."}])
        self.assertEqual(verdict["planner_fixes"], ["Add a reverse shot for the reply", "3"])
        self.assertEqual(verdict["pipeline_fixes"], [])

    def test_generic_ai_footage_is_capped(self):
        capped = video_judge.parse_episode_verdict('{"score": 8, "verdict": "generic_ai"}', 9, TIMELINE)
        self.assertEqual(capped["score"], 5)
        tells = video_judge.parse_episode_verdict('{"score": 7, "scores": {"authenticity": 3}}', 9, TIMELINE)
        self.assertEqual((tells["score"], tells["verdict"]), (5, "generic_ai"))
        broken = video_judge.parse_episode_verdict('{"score": 6, "verdict": "broken"}', 9, TIMELINE)
        self.assertEqual(broken["score"], 3)
        inferred = video_judge.parse_episode_verdict('{"score": 9, "verdict": "great"}', 9, TIMELINE)
        self.assertEqual(inferred["verdict"], "broadcast_ready")

    def test_unusable_answers_raise(self):
        for text in ("no json", '{"verdict": "generic_ai"}', "{oops}"):
            with self.assertRaises(video_judge.JudgeError):
                video_judge.parse_episode_verdict(text, 9, TIMELINE)


class EpisodeJudgeTests(unittest.TestCase):
    def test_request_uses_the_episode_prompt_and_logs_cost(self):
        handler, requests = fake_fal(["not json", '{"score": 4, "verdict": "generic_ai"}'])
        with httpx.Client(transport=httpx.MockTransport(handler)) as http:
            verdict = video_judge.judge_episode(
                "https://r2/episode.mp4", {"skill_id": "custom:kdrama", "brief": "A breakup at a cafe"},
                TIMELINE, [{"label": "hook"}, {"label": "reply"}], fal_key="k", http=http, sleep=lambda _: None)
        bodies = [json.loads(r.content) for r in requests if r.method == "POST"]
        self.assertEqual(len(bodies), 2)
        self.assertEqual(bodies[0]["model"], video_judge.DEFAULT_EPISODE_MODEL)
        self.assertEqual(bodies[0]["system_prompt"], video_judge.EPISODE_PROMPT)
        self.assertIn('"start_sec": 4.0', bodies[0]["prompt"])
        self.assertIn("custom:kdrama", bodies[0]["prompt"])
        self.assertEqual((verdict["score"], verdict["cost_usd"]), (4, 0.004))

    def test_review_episode_records_failures(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        open(os.path.join(tmp, "x_multiclip.mp4"), "wb").close()
        original = video_judge.cut_segment
        video_judge.cut_segment = lambda src, start, end, dest: src
        self.addCleanup(setattr, video_judge, "cut_segment", original)
        clips = [clip(0, 4), clip(4, 9)]

        def judge(url, context, timeline, shots, **kwargs):
            self.assertTrue(url.endswith("x_multiclip.mp4"))
            self.assertEqual(len(shots), 2)
            return {"score": 5, "verdict": "generic_ai", "cost_usd": 0.03}
        review = video_judge.review_episode(clips, ["x_multiclip.mp4"], tmp, context={}, fal_key="k",
                                            publish=lambda p: "url:" + p, judge=judge, probe=lambda p: None)
        self.assertEqual((review["status"], review["source"], review["score"]), ("completed", "multiclip", 5))
        self.assertEqual(review["timeline"], TIMELINE)

        def boom(*args, **kwargs):
            raise video_judge.JudgeError("timeout")
        failed = video_judge.review_episode(clips, ["x_multiclip.mp4"], tmp, context={}, fal_key="k",
                                            publish=lambda p: p, judge=boom, probe=lambda p: None)
        self.assertEqual((failed["status"], failed["error"]), ("failed", "timeout"))
        missing = video_judge.review_episode(clips, [], tmp, context={}, fal_key="k", publish=lambda p: p)
        self.assertEqual(missing["status"], "failed")


if __name__ == "__main__":
    unittest.main()
