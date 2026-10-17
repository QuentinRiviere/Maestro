"""Draft best-of-N storyboard: candidates, judge ranking, rounds, no-key fallback, final pass, override."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image, ImageDraw, PngImagePlugin

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services.director import storyboard_drafts, storyboard_judge  # noqa: E402
from services.director.image_refine import RefineSettings  # noqa: E402
from services.director.video_judge import JudgeError  # noqa: E402


def picture(path, size=(432, 768), kind=0, settings=None):
    image = Image.new("RGB", size, (30, 30, 30))
    pen = ImageDraw.Draw(image)
    if kind % 3 == 0:
        pen.rectangle((10, size[1] // 2, size[0] - 10, size[1] - 10), fill=(200, 180, 120))
    elif kind % 3 == 1:
        pen.ellipse((20, 20, size[0] - 20, size[1] // 2), fill=(220, 170, 140))
    else:
        for x in range(0, size[0], 40):
            pen.rectangle((x, 0, x + 18, size[1]), fill=(90, 160, 220))
    if settings is None:
        image.save(path)
    else:  # the engine embeds the generation settings as the image's comment
        info = PngImagePlugin.PngInfo()
        info.add_text("comment", json.dumps(settings))
        image.save(path, pnginfo=info)
    return path


def ranking(best, verdicts, fix_hint=""):
    return {"best": best, "fix_hint": fix_hint, "cost_usd": 0.01,
            "verdicts": [{"index": i + 1, "verdict": v, "issues": [] if v == "pass" else ["wrong framing"] * (i + 1)}
                         for i, v in enumerate(verdicts)]}


class SettingsTests(unittest.TestCase):
    def test_draft_resolution_scales_the_long_side_to_768_on_a_16_grid(self):
        self.assertEqual(storyboard_drafts.draft_resolution("720x1280"), "432x768")
        self.assertEqual(storyboard_drafts.draft_resolution("1280x720"), "768x432")
        self.assertEqual(storyboard_drafts.draft_resolution("480x848"), "432x768")
        self.assertEqual(storyboard_drafts.draft_resolution("1920x1080"), "768x432")
        # Never larger than the run.
        self.assertEqual(storyboard_drafts.draft_resolution("512x512"), "512x512")
        self.assertEqual(storyboard_drafts.draft_resolution("bogus"), "bogus")

    def test_defaults_keep_the_mode_off(self):
        settings = storyboard_drafts.draft_settings({}, "720x1280")
        self.assertFalse(settings.enabled)
        self.assertFalse(storyboard_drafts.draft_settings({"director_storyboard_drafts": 1}, "720x1280").enabled)
        on = storyboard_drafts.draft_settings({"director_storyboard_drafts": "3"}, "720x1280")
        self.assertTrue(on.enabled)
        self.assertEqual((on.count, on.rounds, on.resolution, on.final_strength), (3, 2, "432x768", 0.55))

    def test_explicit_values_are_honoured_and_clamped(self):
        settings = storyboard_drafts.draft_settings(
            {"director_storyboard_drafts": 4, "director_draft_resolution": "384x672", "director_draft_rounds": 9,
             "director_draft_final_strength": 2}, "720x1280")
        self.assertEqual((settings.count, settings.resolution, settings.rounds, settings.final_strength),
                         (4, "384x672", storyboard_drafts.MAX_ROUNDS, storyboard_drafts.MAX_FINAL_STRENGTH))
        bad = storyboard_drafts.draft_settings(
            {"director_storyboard_drafts": 3, "director_draft_resolution": "big", "director_draft_final_strength": "x"},
            "1280x720")
        self.assertEqual((bad.resolution, bad.final_strength), ("768x432", 0.55))

    def test_candidate_seeds_are_distinct(self):
        seeds = storyboard_drafts.candidate_seeds(41, 4)
        self.assertEqual(seeds, [41, 42, 43, 44])
        self.assertEqual(len(set(storyboard_drafts.candidate_seeds(2 ** 31 - 2, 4))), 4)

    def test_output_seed_reads_the_sidecar_then_the_image_comment_then_the_name(self):
        with tempfile.TemporaryDirectory() as folder:
            embedded = picture(os.path.join(folder, "a.png"), settings={"seed": 77, "prompt": "p"})
            self.assertEqual(storyboard_drafts.output_seed(embedded), 77)
            with open(os.path.join(folder, "a.json"), "w", encoding="utf-8") as handle:
                json.dump({"seed": 78}, handle)
            self.assertEqual(storyboard_drafts.output_seed(embedded), 78)
            self.assertEqual(storyboard_drafts.output_seed(picture(os.path.join(folder, "x_seed91_shot.png"))), 91)
            self.assertIsNone(storyboard_drafts.output_seed(picture(os.path.join(folder, "plain.png"))))
            self.assertIsNone(storyboard_drafts.output_seed(os.path.join(folder, "missing.png")))
            self.assertIsNone(storyboard_drafts.output_seed(picture(os.path.join(folder, "neg.png"),
                                                                    settings={"seed": -1})))


class RankParsingTests(unittest.TestCase):
    def test_a_valid_ranking(self):
        text = ('Here: {"best": 2, "verdicts": [{"index": 1, "verdict": "flagged", "issues": ["Man drawn twice"]},'
                ' {"index": 2, "verdict": "pass", "issues": []}, {"index": 3, "verdict": "pass"}],'
                ' "fix_hint": "Show Man only as a shoulder."}')
        parsed = storyboard_judge.parse_rank(text, 3)
        self.assertEqual(parsed["best"], 2)
        self.assertFalse(parsed["best_fallback"])
        self.assertEqual([v["verdict"] for v in parsed["verdicts"]], ["flagged", "pass", "pass"])
        self.assertEqual(parsed["verdicts"][0]["issues"], ["Man drawn twice"])
        self.assertEqual(parsed["fix_hint"], "Show Man only as a shoulder.")

    def test_malformed_rankings_raise(self):
        with self.assertRaises(JudgeError):
            storyboard_judge.parse_rank("no json here", 3)
        with self.assertRaises(JudgeError):
            storyboard_judge.parse_rank('{"best": 1, "verdicts": [}', 3)

    def test_an_out_of_range_index_falls_back(self):
        text = json.dumps({"best": 7, "verdicts": [{"index": 1, "verdict": "flagged"}, {"index": 2, "verdict": "pass"},
                                                   {"index": 9, "verdict": "pass"}]})
        parsed = storyboard_judge.parse_rank(text, 3)
        # The first passing candidate; the verdict for candidate 9 is dropped, candidate 3 was skipped.
        self.assertEqual(parsed["best"], 2)
        self.assertTrue(parsed["best_fallback"])
        self.assertEqual(parsed["verdicts"][2], {"index": 3, "verdict": "flagged",
                                                 "issues": ["not ranked by the judge"]})
        nothing_passes = storyboard_judge.parse_rank('{"best": 0, "verdicts": []}', 2)
        self.assertEqual(nothing_passes["best"], 1)

    def test_one_call_ranks_every_candidate_and_retries_bad_json(self):
        replies = iter([("not json", 0.002), ('{"best": 1, "verdicts": [{"index": 1, "verdict": "pass"}]}', 0.003)])
        calls = []

        def ask(http, key, model, system, prompt, urls, sleep, **_kwargs):
            calls.append((system, prompt, list(urls)))
            return next(replies)

        with patch.object(storyboard_judge, "still_as_video", side_effect=lambda path, dest: path), \
                patch.object(storyboard_judge, "data_uri", side_effect=lambda path: f"uri:{os.path.basename(path)}"):
            result = storyboard_judge.rank_candidates({"framing": "medium", "image_prompt": "p"},
                                                      ["/x/a.png", "/x/b.png"], fal_key="k", ask=ask, http=object())
        self.assertEqual(len(calls), 2)
        system, prompt, urls = calls[0]
        self.assertIs(system, storyboard_judge.RANK_PROMPT)
        self.assertIn("duplicated person", system)
        self.assertIn("video 1 = candidate 1, video 2 = candidate 2", prompt)
        self.assertEqual(urls, ["uri:a.png", "uri:b.png"])
        self.assertEqual(result["best"], 1)
        self.assertEqual(result["cost_usd"], 0.005)


class SelectDraftTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)
        self.renders = []

    def render(self, prompt, seed):
        self.renders.append((prompt, seed))
        name = f"draft_{len(self.renders)}.png"
        picture(os.path.join(self.dir, name), kind=len(self.renders))
        return name

    def select(self, rank, count=3, rounds=2, **kwargs):
        settings = storyboard_drafts.DraftSettings(count=count, rounds=rounds, resolution="432x768")
        seeds = iter([100, 200, 300])
        kwargs.setdefault("log", lambda message: None)
        return storyboard_drafts.select_draft(shot_label="shot 1/1", base_prompt="A shot.", settings=settings,
                                              render=self.render, out_dir=self.dir, rank=rank,
                                              seed_base=lambda: next(seeds), **kwargs)

    def test_n_candidates_with_their_own_seeds_and_a_pass_stops(self):
        ranked = []
        messages = []

        def rank(paths):
            ranked.append([os.path.basename(p) for p in paths])
            return ranking(3, ["flagged", "pass", "pass"])

        record = self.select(rank, on_progress=messages.append)
        self.assertEqual([seed for _, seed in self.renders], [100, 101, 102])
        self.assertEqual({prompt for prompt, _ in self.renders}, {"A shot."})
        self.assertEqual(ranked, [["draft_1.png", "draft_2.png", "draft_3.png"]])
        self.assertEqual(record["chosen"], {"round": 1, "candidate": 3, "file": "draft_3.png"})
        self.assertEqual(record["verdict"], "pass")
        self.assertEqual(len(record["rounds"]), 1)
        self.assertEqual([c["verdict"] for c in record["candidates"]], ["flagged", "pass", "pass"])
        self.assertIn("Storyboard drafts: shot 1/1, round 1, candidate 3/3", messages)

    def batch(self, prompt, seed, count, files=None):
        self.batches.append((prompt, seed, count))
        names = []
        for k in range(count if files is None else files):
            name = f"batch_{len(self.batches)}_{k + 1}.png"
            # The engine's own repeat seeds (seed, seed + 1, ...) embedded as the image's settings.
            picture(os.path.join(self.dir, name), kind=k, settings={"seed": seed + k})
            names.append(name)
        return names

    def test_a_batch_renders_the_round_in_one_job_with_the_engine_seeds(self):
        self.batches = []
        messages = []
        record = self.select(lambda paths: ranking(2, ["flagged", "pass", "pass"]), render_batch=self.batch,
                             on_progress=messages.append)
        self.assertEqual(self.batches, [("A shot.", 100, 3)])
        self.assertEqual(self.renders, [])
        self.assertEqual([c["file"] for c in record["candidates"]], ["batch_1_1.png", "batch_1_2.png", "batch_1_3.png"])
        self.assertEqual([c["seed"] for c in record["candidates"]], [100, 101, 102])
        self.assertTrue(all(c["batched"] for c in record["candidates"]))
        self.assertEqual(record["chosen"], {"round": 1, "candidate": 2, "file": "batch_1_2.png"})
        self.assertIn("Storyboard drafts: shot 1/1, round 1, 3 candidates in one job", messages)

    def test_the_engine_seeds_win_over_the_planned_ones(self):
        self.batches = []

        def batch(prompt, seed, count):
            names = self.batch(prompt, seed, count)
            # An engine drawing its own seeds for the repeats records them in each output.
            for k, name in enumerate(names):
                picture(os.path.join(self.dir, name), kind=k, settings={"seed": 5000 + 10 * k})
            return names

        record = self.select(lambda paths: ranking(1, ["pass", "pass", "pass"]), render_batch=batch)
        self.assertEqual([c["seed"] for c in record["candidates"]], [5000, 5010, 5020])

    def test_a_short_batch_falls_back_to_one_job_per_missing_candidate(self):
        self.batches = []
        logs = []
        record = self.select(lambda paths: ranking(3, ["flagged", "flagged", "pass"]),
                             render_batch=lambda prompt, seed, count: self.batch(prompt, seed, count, files=1),
                             log=logs.append)
        self.assertEqual(self.batches, [("A shot.", 100, 3)])
        self.assertEqual(self.renders, [("A shot.", 101), ("A shot.", 102)])
        self.assertEqual([c["file"] for c in record["candidates"]], ["batch_1_1.png", "draft_1.png", "draft_2.png"])
        self.assertEqual([c["seed"] for c in record["candidates"]], [100, 101, 102])
        self.assertEqual([c.get("batched", False) for c in record["candidates"]], [True, False, False])
        self.assertEqual(record["chosen"]["file"], "draft_2.png")
        self.assertTrue(any("returned 1/3 files" in line for line in logs))

    def test_a_failed_batch_renders_every_candidate_one_job_each(self):
        def batch(prompt, seed, count):
            raise RuntimeError("worker refused the repeat")

        logs = []
        record = self.select(lambda paths: ranking(1, ["pass", "pass", "pass"]), render_batch=batch, log=logs.append)
        self.assertEqual([seed for _, seed in self.renders], [100, 101, 102])
        self.assertEqual(len(record["candidates"]), 3)
        self.assertFalse(any(c.get("batched") for c in record["candidates"]))
        self.assertTrue(any("worker refused the repeat" in line for line in logs))

    def test_a_flagged_best_reruns_with_the_fix_hint_and_caps_at_the_rounds(self):
        replies = iter([ranking(1, ["flagged", "flagged"], "Show Man only as a shoulder."),
                        ranking(2, ["flagged", "flagged"], "Wider."),
                        ranking(1, ["pass", "pass"])])
        record = self.select(lambda paths: next(replies), count=2, rounds=2)
        self.assertEqual(len(self.renders), 4)
        self.assertEqual([prompt for prompt, _ in self.renders][2:],
                         ["A shot. Show Man only as a shoulder."] * 2)
        self.assertEqual([seed for _, seed in self.renders], [100, 101, 200, 201])
        self.assertEqual(len(record["rounds"]), 2)
        # Round 1's best had one issue, round 2's two: the best seen overall stays.
        self.assertEqual(record["chosen"], {"round": 1, "candidate": 1, "file": "draft_1.png"})
        self.assertEqual(record["verdict"], "flagged")

    def test_a_later_passing_round_wins(self):
        replies = iter([ranking(1, ["flagged", "flagged"], "Fix it."), ranking(2, ["flagged", "pass"])])
        record = self.select(lambda paths: next(replies), count=2, rounds=3)
        self.assertEqual(record["chosen"], {"round": 2, "candidate": 2, "file": "draft_4.png"})
        self.assertEqual(record["verdict"], "pass")
        self.assertEqual(len(record["rounds"]), 2)

    def test_without_a_judge_candidate_one_is_kept(self):
        logs = []
        settings = storyboard_drafts.DraftSettings(count=3, rounds=2, resolution="432x768")
        record = storyboard_drafts.select_draft(shot_label="shot 2/4", base_prompt="A shot.", settings=settings,
                                                render=self.render, out_dir=self.dir, rank=None, log=logs.append)
        self.assertEqual(len(self.renders), 3)
        self.assertEqual(record["chosen"]["candidate"], 1)
        self.assertEqual(record["chosen"]["file"], "draft_1.png")
        self.assertEqual(record["judge"], "none")
        self.assertTrue(any("keeping candidate 1" in line for line in logs))

    def test_a_ranking_failure_keeps_candidate_one(self):
        def rank(paths):
            raise JudgeError("fal down")

        record = self.select(rank)
        self.assertEqual(record["chosen"]["candidate"], 1)
        self.assertIn("fal down", record["rounds"][0]["error"])

    def test_a_candidate_copying_an_earlier_shot_is_flagged(self):
        earlier = picture(os.path.join(self.dir, "shot1.png"), kind=1)
        # draft_1 is drawn like shot 1 (kind 1); the judge prefers it.
        record = self.select(lambda paths: ranking(1, ["pass", "pass", "pass"]), others=[(0, earlier)])
        self.assertEqual(record["candidates"][0]["verdict"], "flagged")
        self.assertIn("copies shot 1", record["candidates"][0]["issues"])
        self.assertEqual(record["chosen"]["candidate"], 2)


class FinalPassTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.dir)

    def test_the_chosen_draft_is_upscaled_and_re_rendered_at_the_final_strength(self):
        picture(os.path.join(self.dir, "draft_2.png"))
        record = {"chosen": {"round": 1, "candidate": 2, "file": "draft_2.png"}}
        jobs = []

        def submit(job):
            jobs.append(job)
            picture(os.path.join(self.dir, "final.png"), size=(720, 1280))
            return ["final.png"]

        settings = storyboard_drafts.DraftSettings(count=3, final_strength=0.55)
        final = storyboard_drafts.run_final(
            pid="p", index=0, record=record, out_dir=self.dir, image_prompt="A shot.", identity_refs=[],
            settings=settings, refine_settings=RefineSettings(enabled=False), size=(720, 1280), submit=submit)
        self.assertEqual(final, "final.png")
        job = jobs[0]
        self.assertEqual(os.path.basename(job["image_guide"]), "draft_2_upscaled_720x1280.png")
        with Image.open(job["image_guide"]) as upscaled:
            self.assertEqual(upscaled.size, (720, 1280))
        self.assertEqual(job["denoising_strength"], 0.55)
        self.assertEqual(job["resolution"], "720x1280")
        self.assertEqual(job["model_type"], "qwen_image_21_7B")
        self.assertEqual(job["video_prompt_type"], "VAG")
        self.assertEqual(job["_director_cost_stage"], "image_final")
        self.assertTrue(job["prompt"].startswith("A shot. Re-render reference image 1 at full resolution"))
        self.assertEqual(record["final"], "final.png")
        self.assertEqual(record["final_from"], "draft_2.png")


class PipelineDraftTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        saved = (pipeline._pipelines, pipeline._wgp)
        self.addCleanup(lambda: setattr(pipeline, "_pipelines", saved[0]) or setattr(pipeline, "_wgp", saved[1]))
        pipeline._pipelines = {}
        pipeline._wgp = SimpleNamespace(save_path=self.tmp.name)
        self.pid = "pipe-drafts"
        pipeline._pipelines[self.pid] = {
            "id": self.pid, "status": "running", "phase": "generating_images", "progress": {},
            "clip_plans": [], "clip_images": [], "output_files": [], "created_at": time.time(),
            "params": {}, "pause_reason": None, "out_dir": self.tmp.name,
        }
        self.jobs = []
        self.outputs = 0
        cost = patch("services.director.cost_ledger.record")
        cost.start()
        self.addCleanup(cost.stop)

    def submit(self, gen_params, **_kwargs):
        """The engine: one output per repeat, seeds seed, seed + 1, ... embedded as each output's settings."""
        self.jobs.append(gen_params)
        width, height = (int(v) for v in gen_params["resolution"].split("x"))
        names = []
        for k in range(int(gen_params.get("repeat_generation") or 1)):
            self.outputs += 1
            name = f"out_{self.outputs}.png"
            seed = int(gen_params.get("seed", -1))
            picture(os.path.join(self.tmp.name, name), size=(width, height), kind=self.outputs,
                    settings={"seed": (seed if seed >= 0 else 4242) + k})
            names.append(name)
        return names

    def params(self, **extra):
        anchor = picture(os.path.join(self.tmp.name, "anchor.png"))
        return {"reference_image_path": anchor, "image_model": "flux2_klein_9b",
                "image_params": {"resolution": "720x1280"}, **extra}

    def plans(self):
        return [{"image_prompt": "wide of the hallway"}, {"image_prompt": "close-up on her hands"}]

    def generate(self, params, plans, judge=None):
        config = judge or {"enabled": True, "vision": False, "fal_key": "", "model": "m"}
        with patch.object(pipeline, "_submit_and_wait", side_effect=self.submit), \
                patch.object(storyboard_judge, "stage_config", return_value=config):
            return pipeline._run_image_generation(self.pid, params, plans, out_dir=self.tmp.name)

    def test_default_off_renders_one_image_per_shot(self):
        images, _ = self.generate(self.params(), self.plans())
        self.assertEqual(len(self.jobs), 2)
        self.assertEqual({job["seed"] for job in self.jobs}, {-1})
        self.assertEqual({job["resolution"] for job in self.jobs}, {"720x1280"})
        self.assertEqual({job["_director_cost_stage"] for job in self.jobs}, {"image"})
        self.assertNotIn("_clip_storyboard_drafts", pipeline._pipelines[self.pid])

    def test_drafts_without_a_key_keep_candidate_one_then_the_final_pass_re_renders_it(self):
        params = self.params(director_storyboard_drafts=3)
        plans = self.plans()
        images, _ = self.generate(params, plans)
        # One job per shot renders its three candidates (repeat_generation).
        self.assertEqual(len(self.jobs), 2)
        self.assertEqual({job["repeat_generation"] for job in self.jobs}, {3})
        self.assertEqual({job["resolution"] for job in self.jobs}, {"432x768"})
        self.assertEqual({job["_director_cost_stage"] for job in self.jobs}, {"image_draft"})
        self.assertTrue(all(job["seed"] >= 0 for job in self.jobs))
        self.assertEqual(images, ["out_1.png", "out_4.png"])
        records = pipeline._pipelines[self.pid]["_clip_storyboard_drafts"]
        self.assertEqual([c["file"] for c in records[0]["candidates"]], ["out_1.png", "out_2.png", "out_3.png"])
        # Each candidate records the seed its output carries: the job's seed, + 1, + 2.
        base = self.jobs[0]["seed"]
        self.assertEqual([c["seed"] for c in records[0]["candidates"]], [base, base + 1, base + 2])
        self.assertEqual(len({c["seed"] for c in records[1]["candidates"]}), 3)
        report = storyboard_judge.load_report(self.tmp.name, self.pid)
        self.assertEqual(report["mode"], "drafts")

        with patch.object(pipeline, "_submit_and_wait", side_effect=self.submit):
            pipeline._run_draft_finals(self.pid, params, plans, images, out_dir=self.tmp.name)
        finals = self.jobs[2:]
        self.assertEqual(len(finals), 2)
        self.assertEqual({job["repeat_generation"] for job in finals}, {1})
        self.assertEqual(os.path.basename(finals[0]["image_guide"]), "out_1_upscaled_720x1280.png")
        self.assertEqual(finals[0]["denoising_strength"], 0.55)
        self.assertEqual(finals[0]["resolution"], "720x1280")
        self.assertEqual(finals[0]["_director_cost_stage"], "image_final")
        self.assertEqual(images, ["out_7.png", "out_8.png"])
        records = pipeline._pipelines[self.pid]["_clip_storyboard_drafts"]
        self.assertEqual(records[0]["final"], "out_7.png")
        report = storyboard_judge.load_report(self.tmp.name, self.pid)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["shots"][0]["image"], "out_7.png")
        self.assertEqual(report["shots"][0]["drafts"]["chosen"]["file"], "out_1.png")

        # A finished shot is not re-rendered again (resume).
        with patch.object(pipeline, "_submit_and_wait", side_effect=self.submit):
            pipeline._run_draft_finals(self.pid, params, plans, images, out_dir=self.tmp.name)
        self.assertEqual(len(self.jobs), 4)

    def test_an_engine_returning_one_file_per_job_renders_the_other_candidates_one_job_each(self):
        def single(gen_params, **kwargs):
            names = self.submit({**gen_params, "repeat_generation": 1}, **kwargs)
            self.jobs[-1] = gen_params  # the job as submitted (repeat_generation N), one output back
            return names[:1]

        params = self.params(director_storyboard_drafts=3)
        with patch.object(pipeline, "_submit_and_wait", side_effect=single), \
                patch.object(storyboard_judge, "stage_config",
                             return_value={"enabled": True, "vision": False, "fal_key": "", "model": "m"}):
            images, _ = pipeline._run_image_generation(self.pid, params, self.plans(), out_dir=self.tmp.name)
        self.assertEqual(len(self.jobs), 6)
        self.assertEqual([job["repeat_generation"] for job in self.jobs[:3]], [3, 1, 1])
        seeds = [job["seed"] for job in self.jobs[:3]]
        self.assertEqual(seeds, [seeds[0], seeds[0] + 1, seeds[0] + 2])
        records = pipeline._pipelines[self.pid]["_clip_storyboard_drafts"]
        self.assertEqual([c["file"] for c in records[0]["candidates"]], ["out_1.png", "out_2.png", "out_3.png"])
        self.assertEqual([c["seed"] for c in records[0]["candidates"]], seeds)
        self.assertEqual(images, ["out_1.png", "out_4.png"])

    def test_the_judge_ranks_the_candidates_of_each_shot(self):
        params = self.params(director_storyboard_drafts=2)
        judge = {"enabled": True, "vision": True, "fal_key": "k", "model": "m"}
        calls = []

        def rank(shot, paths, **_kwargs):
            calls.append((shot["image_prompt"], [os.path.basename(p) for p in paths]))
            return ranking(2, ["flagged", "pass"])

        with patch.object(storyboard_judge, "rank_candidates", side_effect=rank):
            images, _ = self.generate(params, self.plans(), judge=judge)
        self.assertEqual(calls, [("wide of the hallway", ["out_1.png", "out_2.png"]),
                                 ("close-up on her hands", ["out_3.png", "out_4.png"])])
        self.assertEqual(images, ["out_2.png", "out_4.png"])

    def test_a_human_override_re_runs_the_final_from_another_candidate(self):
        params = self.params(director_storyboard_drafts=3)
        plans = self.plans()
        images, _ = self.generate(params, plans)
        with patch.object(pipeline, "_submit_and_wait", side_effect=self.submit):
            pipeline._run_draft_finals(self.pid, params, plans, images, out_dir=self.tmp.name)
        record = pipeline._pipelines[self.pid]
        record.update(params=params, clip_plans=plans, clip_images=images, status="completed")
        self.assertTrue(pipeline._save_pipeline_state(self.pid))

        with patch.object(pipeline, "_submit_and_wait", side_effect=self.submit):
            result = pipeline._select_storyboard_draft_impl(self.tmp.name, self.pid, 0, "out_3.png")
        job = self.jobs[-1]
        self.assertEqual(os.path.basename(job["image_guide"]), "out_3_upscaled_720x1280.png")
        self.assertEqual(job["_director_cost_stage"], "image_final")
        self.assertEqual(result["chosen"]["file"], "out_3.png")
        state = pipeline.load_pipeline_state(self.tmp.name, self.pid)
        clip = state["clips"][0]
        self.assertEqual(clip["start_image_filename"], result["filename"])
        self.assertTrue(clip["storyboard_drafts"]["chosen"]["override"])
        report = storyboard_judge.load_report(self.tmp.name, self.pid)
        self.assertEqual(report["shots"][0]["image"], result["filename"])
        with self.assertRaises(ValueError):
            pipeline._select_storyboard_draft_impl(self.tmp.name, self.pid, 0, "anchor.png")


if __name__ == "__main__":
    unittest.main()
