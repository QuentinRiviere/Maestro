"""Storyboard judge: duplicate shots, verdict flow, cost records, disabled path."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director import cost_ledger, storyboard_judge  # noqa: E402


def draw(path, kind, *, shift=0):
    image = Image.new("RGB", (320, 180), (30, 30, 30))
    pen = ImageDraw.Draw(image)
    if kind == "wide":
        pen.rectangle((20 + shift, 100, 300 + shift, 170), fill=(200, 180, 120))
        pen.ellipse((140, 20, 180, 60), fill=(240, 240, 240))
    elif kind == "close":
        pen.ellipse((60, 10, 260, 175), fill=(220, 170, 140))
    elif kind == "insert":
        for x in range(0, 320, 40):
            pen.rectangle((x, 0, x + 18, 180), fill=(90, 160, 220))
    image.save(path)
    return path


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def shot(self, name, kind, *, group="s1", scene_type="exchange", framing="medium", strategy=""):
        return {"path": draw(os.path.join(self.dir, name), kind), "image_prompt": f"prompt {name}",
                "group": group, "scene_type": scene_type, "framing": framing, "strategy": strategy}

    def rerun_to(self, kinds):
        """A rerun that draws the next kind of picture, recording its prompts."""
        calls = []

        def rerun(index, prompt):
            kind = kinds[len(calls)]
            calls.append((index, prompt))
            return draw(os.path.join(self.dir, f"rerun_{len(calls)}.png"), kind)
        return rerun, calls


class DuplicateTests(Base):
    def test_hash_matches_copies_and_separates_different_frames(self):
        a = storyboard_judge.image_hashes(draw(os.path.join(self.dir, "a.png"), "wide"))
        a_shifted = storyboard_judge.image_hashes(draw(os.path.join(self.dir, "b.png"), "wide", shift=2))
        close = storyboard_judge.image_hashes(draw(os.path.join(self.dir, "c.png"), "close"))
        self.assertTrue(storyboard_judge.near_identical(a, a_shifted))
        self.assertFalse(storyboard_judge.near_identical(a, close))

    def test_later_non_establishing_shot_copies_and_other_scenes_are_ignored(self):
        shots = [self.shot("1.png", "wide", scene_type="establishing"),
                 self.shot("2.png", "close"),
                 self.shot("3.png", "wide"),
                 self.shot("4.png", "wide", group="s2"),
                 self.shot("5.png", "close", strategy="extend_previous")]
        hashes = storyboard_judge._safe_hashes(shots)
        found = [storyboard_judge.duplicate_of(i, shots, hashes) for i in range(len(shots))]
        self.assertEqual(found, [None, None, 0, None, None])

    def test_duplicate_only_mode_regenerates_once_with_a_framing_hint(self):
        shots = [self.shot("1.png", "wide", scene_type="establishing"), self.shot("2.png", "wide", framing="close-up")]
        rerun, calls = self.rerun_to(["close"])
        report = storyboard_judge.judge_storyboard(shots, rerun=rerun)
        self.assertEqual(report["mode"], "duplicates_only")
        self.assertEqual(len(calls), 1)
        self.assertIn("close-up", calls[0][1])
        self.assertIn("shot 1", calls[0][1])
        self.assertEqual(report["shots"][1]["verdict"], "regenerated")
        self.assertEqual(report["shots"][1]["history"][0]["issues"], ["copies shot 1"])
        self.assertEqual(report["regenerated"], [1])
        self.assertEqual(report["cost_usd"], 0.0)

    def test_a_duplicate_that_stays_a_copy_is_flagged_after_one_try(self):
        shots = [self.shot("1.png", "wide"), self.shot("2.png", "wide")]
        rerun, calls = self.rerun_to(["wide", "wide"])
        report = storyboard_judge.judge_storyboard(shots, rerun=rerun)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["flagged"], [1])
        self.assertEqual(report["shots"][0]["verdict"], "pass")


class VerdictFlowTests(Base):
    def test_pass_regenerate_then_pass_and_flag(self):
        shots = [self.shot("1.png", "wide", scene_type="establishing"),
                 self.shot("2.png", "close", group="s2"),
                 self.shot("3.png", "insert", group="s3")]
        answers = {
            "1.png": [{"pass": True, "issues": [], "fix_hint": ""}],
            "2.png": [{"pass": False, "issues": ["two people, plan has one"], "fix_hint": "Show only the girl."}],
            "rerun_1.png": [{"pass": True, "issues": [], "fix_hint": ""}],
            "3.png": [{"pass": False, "issues": ["wrong set"], "fix_hint": "Set it in the kitchen."}],
            "rerun_2.png": [{"pass": False, "issues": ["still wrong set"], "fix_hint": "Kitchen counter."}],
            "rerun_3.png": [{"pass": False, "issues": ["still wrong"], "fix_hint": "Kitchen."}],
        }
        checked = []

        def check(shot):
            name = os.path.basename(shot["path"])
            checked.append(name)
            return {**answers[name].pop(0), "cost_usd": 0.002}

        recorded = []
        rerun, calls = self.rerun_to(["close", "insert", "insert"])
        report = storyboard_judge.judge_storyboard(
            shots, rerun=rerun, check=check, model="m", workers=1,
            record_cost=lambda index, verdict: recorded.append((index, verdict["cost_usd"])))
        verdicts = [s["verdict"] for s in report["shots"]]
        self.assertEqual(verdicts, ["pass", "regenerated", "flagged"])
        self.assertEqual(calls[0], (1, "prompt 2.png Show only the girl."))
        self.assertEqual(calls[1], (2, "prompt 3.png Set it in the kitchen."))
        self.assertEqual(calls[2], (2, "prompt 3.png Kitchen counter."))
        self.assertEqual(report["shots"][2]["attempts"], 2)
        self.assertEqual(report["shots"][2]["issues"], ["still wrong"])
        self.assertEqual(report["flagged"], [2])
        self.assertEqual(shots[1]["path"], os.path.join(self.dir, "rerun_1.png"))
        self.assertEqual(len(recorded), 6)
        self.assertAlmostEqual(report["cost_usd"], 0.012)
        self.assertEqual(report["status"], "completed")

    def test_vision_errors_never_fail_a_shot_and_copies_skip_the_first_vision_call(self):
        shots = [self.shot("1.png", "wide"), self.shot("2.png", "wide")]
        checked = []

        def check(shot):
            checked.append(os.path.basename(shot["path"]))
            raise storyboard_judge.JudgeError("fal down")

        rerun, _ = self.rerun_to(["close"])
        report = storyboard_judge.judge_storyboard(shots, rerun=rerun, check=check, workers=1)
        self.assertEqual(checked, ["1.png", "rerun_1.png"])
        self.assertEqual([s["verdict"] for s in report["shots"]], ["pass", "regenerated"])
        self.assertIn("fal down", report["shots"][0]["error"])

    def test_failed_regeneration_keeps_the_image_and_flags(self):
        shots = [self.shot("1.png", "wide")]

        def rerun(index, prompt):
            raise RuntimeError("GPU gone")

        report = storyboard_judge.judge_storyboard(
            shots, rerun=rerun, check=lambda shot: {"pass": False, "issues": ["x"], "fix_hint": "y"})
        self.assertEqual(report["flagged"], [0])
        self.assertTrue(shots[0]["path"].endswith("1.png"))
        self.assertIn("GPU gone", report["shots"][0]["error"])

    def test_parse_check(self):
        verdict = storyboard_judge.parse_check('```json\n{"pass": "false", "issues": ["a"], "fix_hint": "b"}\n```')
        self.assertEqual(verdict, {"pass": False, "issues": ["a"], "fix_hint": "b"})
        with self.assertRaises(storyboard_judge.JudgeError):
            storyboard_judge.parse_check('{"issues": []}')

    def test_check_shot_sends_the_still_and_cast_sheets(self):
        shot = self.shot("1.png", "close")
        shot["cast_sheets"] = [{"name": "Mina", "path": draw(os.path.join(self.dir, "mina.png"), "close")}]
        seen = {}

        def ask(http, key, model, system, prompt, urls, sleep, **kwargs):
            seen.update(key=key, model=model, prompt=prompt, urls=urls)
            return '{"pass": true, "issues": [], "fix_hint": ""}', 0.0015

        with patch.object(storyboard_judge, "still_as_video", side_effect=lambda src, dest: src):
            verdict = storyboard_judge.check_shot(shot, fal_key="k", model="m", ask=ask, http=object())
        self.assertTrue(verdict["pass"])
        self.assertEqual(verdict["cost_usd"], 0.0015)
        self.assertEqual(len(seen["urls"]), 2)
        self.assertIn("video 2 = Mina", seen["prompt"])
        self.assertIn('"framing": "medium"', seen["prompt"])


class SettingsTests(unittest.TestCase):
    def test_default_on_with_a_key_and_duplicates_only_without(self):
        self.assertEqual(storyboard_judge.settings({}, {}, "key")["vision"], True)
        no_key = storyboard_judge.settings({}, {}, "")
        self.assertTrue(no_key["enabled"])
        self.assertFalse(no_key["vision"])

    def test_project_param_and_service_flag(self):
        self.assertFalse(storyboard_judge.settings({}, {"director_storyboard_judge": "off"}, "key")["enabled"])
        self.assertFalse(storyboard_judge.settings({"director_storyboard_judge": False}, {}, "key")["enabled"])
        self.assertTrue(storyboard_judge.settings({"director_storyboard_judge": False},
                                                  {"director_storyboard_judge": "on"}, "key")["vision"])
        self.assertEqual(storyboard_judge.settings({"director_storyboard_judge_model": "x"}, {}, "k")["model"], "x")


class StageTests(Base):
    """The pipeline hook, with the pipeline module stubbed."""

    def setUp(self):
        super().setUp()
        from services import director_pipeline as pipeline

        self.pipeline = pipeline
        self.updates = []
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.patches = [
            patch.object(cost_ledger, "COSTS_DIR", Path(tmp.name)),
            patch.object(pipeline, "_update_pipeline", lambda pid, **kw: self.updates.append(kw) or True),
            patch.object(pipeline, "_save_pipeline_state", lambda pid: True),
            patch.object(pipeline, "_pipelines", {"p1": {"status": "running"}}),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def plans(self):
        draw(os.path.join(self.dir, "a.png"), "wide")
        draw(os.path.join(self.dir, "b.png"), "wide")
        return ([{"image_prompt": "wide room", "_director_continuity_group": "s1", "_director_scene_type": "establishing"},
                 {"image_prompt": "her face", "_director_continuity_group": "s1", "_director_camera_framing": "close-up"}],
                ["a.png", "b.png"])

    def services(self, services):
        wgp = type("W", (), {"server_config": {"services": services}})
        return patch.object(self.pipeline, "_wgp", wgp)

    def test_disabled_project_does_nothing(self):
        plans, images = self.plans()
        with self.services({"fal_api_key": "k"}):
            result = storyboard_judge.run_stage("p1", {"director_storyboard_judge": "off"}, plans, images, self.dir)
        self.assertIsNone(result)
        self.assertEqual(self.updates, [])

    def test_vision_stage_records_costs_reruns_and_publishes(self):
        plans, images = self.plans()

        def rerun(out_dir, pid, index, prompt_override=None):
            draw(os.path.join(self.dir, "b2.png"), "close")
            return {"filename": "b2.png", "clip_index": index}

        def check(shot, **kwargs):
            return {"pass": True, "issues": [], "fix_hint": "", "cost_usd": 0.003}

        with self.services({}), \
                patch("services.cloud_image.provider_api_key", return_value="key"), \
                patch.object(self.pipeline, "_rerun_clip_image_impl", rerun), \
                patch.object(storyboard_judge, "check_shot", check):
            report = storyboard_judge.run_stage("p1", {}, plans, images, self.dir)
        self.assertEqual(report["regenerated"], [1])
        self.assertEqual(images, ["a.png", "b2.png"])
        self.assertEqual(report["mode"], "vision")
        ledger = cost_ledger.entries("p1")
        self.assertEqual({e["stage"] for e in ledger}, {"storyboard_judge"})
        self.assertEqual(len(ledger), 2)  # shot 1 once, shot 2 after its rerun (the copy skips the first call)
        self.assertAlmostEqual(cost_ledger.summary("p1")["stages"]["storyboard_judge"]["usd"], 0.006)
        published = [u["storyboard_judge"] for u in self.updates if "storyboard_judge" in u]
        self.assertEqual(published[-1]["status"], "completed")
        saved = storyboard_judge.load_report(self.dir, "p1")
        self.assertEqual(saved["shots"][1]["verdict"], "regenerated")
        json.dumps(saved)


if __name__ == "__main__":
    unittest.main()
