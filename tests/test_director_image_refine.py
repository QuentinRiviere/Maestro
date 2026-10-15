"""Director texture refine: settings, job parameters, stage order and per-shot fallback."""

from __future__ import annotations

import ast
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.director import image_refine  # noqa: E402
from services.director.image_refine import RefineSettings, refine_settings  # noqa: E402


def _png(path, size=(64, 48), color=(120, 90, 60), comment=None):
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    if comment:
        info.add_text("comment", comment)
    Image.new("RGB", size, color).save(path, pnginfo=info)
    return path


class RefineSettingsTests(unittest.TestCase):
    def test_defaults_are_off_with_the_base_model(self):
        settings = refine_settings({})
        self.assertEqual(settings, RefineSettings(False, "qwen_image_21_7B", 0.35, 30))
        self.assertEqual(settings.effective_steps, 11)

    def test_enabled_values(self):
        for value in ("on", "ON", True, "true", "1", "enabled"):
            with self.subTest(value=value):
                self.assertTrue(refine_settings({"director_image_refine": value}).enabled)
        for value in ("off", False, "", None, "0", "maybe"):
            with self.subTest(value=value):
                self.assertFalse(refine_settings({"director_image_refine": value}).enabled)

    def test_overrides_are_parsed_and_clamped(self):
        settings = refine_settings({
            "director_image_refine": "on",
            "director_image_refine_model": "qwen_image_21_7B_custom",
            "director_image_refine_strength": "0.4",
            "director_image_refine_steps": "24",
        })
        self.assertEqual(settings, RefineSettings(True, "qwen_image_21_7B_custom", 0.4, 24))
        self.assertEqual(refine_settings({"director_image_refine_strength": 3}).strength, 0.95)
        self.assertEqual(refine_settings({"director_image_refine_strength": -1}).strength, 0.05)
        self.assertEqual(refine_settings({"director_image_refine_strength": "x"}).strength, 0.35)
        self.assertEqual(refine_settings({"director_image_refine_strength": float("nan")}).strength, 0.35)
        self.assertEqual(refine_settings({"director_image_refine_steps": 1000}).steps, 100)
        self.assertEqual(refine_settings({"director_image_refine_steps": "12.0"}).steps, 12)
        self.assertEqual(refine_settings({"director_image_refine_steps": None}).steps, 30)
        self.assertEqual(refine_settings({"director_image_refine_model": "  "}).model, "qwen_image_21_7B")


class RefineJobTests(unittest.TestCase):
    def test_refine_job_is_a_full_frame_masked_edit_of_the_start_image(self):
        settings = refine_settings({"director_image_refine": "on"})
        refs = ["/refs/ana.png", "/refs/ana.png", "/refs/ben.png", "/refs/c.png", "/refs/d.png", "/refs/e.png"]
        job = image_refine.build_refine_job(
            source_path="/out/shot1.png", mask_path="/out/mask.png", prompt="P",
            identity_refs=refs, settings=settings, width=480, height=848, pid="abc")
        self.assertEqual(job["model_type"], "qwen_image_21_7B")
        self.assertEqual(job["image_mode"], 2)
        self.assertEqual(job["video_prompt_type"], "VAGI")
        self.assertEqual(job["image_guide"], "/out/shot1.png")
        self.assertEqual(job["image_mask"], "/out/mask.png")
        self.assertEqual(job["image_refs"], ["/refs/ana.png", "/refs/ben.png", "/refs/c.png", "/refs/d.png"])
        self.assertEqual(job["denoising_strength"], 0.35)
        self.assertEqual(job["masking_strength"], 1.0)
        self.assertEqual(job["model_mode"], 0)
        self.assertEqual(job["num_inference_steps"], 30)
        self.assertEqual(job["guidance_scale"], 4.0)
        self.assertEqual(job["sample_solver"], "default")
        self.assertEqual(job["resolution"], "480x848")
        self.assertEqual(job["generation_mode"], "image")
        self.assertEqual(job["activated_loras"], [])
        self.assertEqual(job["_director_pipeline_id"], "abc")
        self.assertEqual(job["_director_cost_stage"], "image_refine")

        bare = image_refine.build_refine_job(
            source_path="/out/shot1.png", mask_path="/out/mask.png", prompt="P", identity_refs=[],
            settings=settings, width=480, height=848, pid="abc")
        self.assertEqual(bare["video_prompt_type"], "VAG")
        self.assertEqual(bare["image_refs"], [])

    def test_cost_ledger_files_the_refine_under_its_own_stage(self):
        from services.director import cost_ledger

        job = image_refine.build_refine_job(
            source_path="s", mask_path="m", prompt="P", identity_refs=[],
            settings=RefineSettings(True), width=8, height=8, pid="abc")
        self.assertEqual(cost_ledger.stage_of(job), "image_refine")

    def test_prompt_keeps_the_shot_prompt_and_adds_the_texture_clause(self):
        prompt = image_refine.refine_prompt("A woman in a rainy street.", with_identity=True)
        self.assertTrue(prompt.startswith("A woman in a rainy street. "))
        for words in ("skin texture", "pores", "hair strands", "fabric weave", "film grain",
                      "composition", "faces", "clothing"):
            self.assertIn(words, prompt)
        self.assertIn("keep every face identical", prompt)
        self.assertNotIn("keep every face identical",
                         image_refine.refine_prompt("A street.", with_identity=False))

    def test_full_frame_mask_is_white_and_sized(self):
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory)
        path = image_refine.full_frame_mask(directory, 40, 24, pid="p1")
        self.assertEqual(os.path.basename(path), "p1_refine_mask_40x24.png")
        with Image.open(path) as mask:
            self.assertEqual(mask.size, (40, 24))
            self.assertEqual(mask.convert("L").getextrema(), (255, 255))
        self.assertEqual(image_refine.full_frame_mask(directory, 40, 24, pid="p1"), path)


class RefineStageTests(unittest.TestCase):
    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.out)
        self.ref = _png(os.path.join(self.out, "ana_sheet.png"))
        self.images = [_png(os.path.join(self.out, f"shot{i}.png"), size=(64, 48)) for i in range(3)]
        self.names = [os.path.basename(path) for path in self.images]
        self.plans = [{"image_prompt": f"Shot {i} prompt."} for i in range(3)]
        self.jobs = []

    def submit_ok(self, job, size=(64, 32)):
        self.jobs.append(job)
        name = f"refined_{len(self.jobs)}.png"
        _png(os.path.join(self.out, name), size=size, comment='{"seed": 1}')
        return [name]

    def run_stage(self, submit, *, unrefined=None, cancelled=lambda: False, refs=None):
        done = []
        images, originals = image_refine.run_refine_stage(
            pid="p1", clip_images=self.names, clip_plans=self.plans, out_dir=self.out,
            settings=refine_settings({"director_image_refine": "on"}), unrefined=unrefined,
            identity_refs_for=lambda index: refs if refs is not None else [self.ref, "/missing.png"],
            submit=submit, is_cancelled=cancelled,
            on_shot_done=lambda imgs, orig: done.append((list(imgs), list(orig))))
        return images, originals, done

    def test_every_shot_is_refined_once_and_records_its_original(self):
        images, originals, done = self.run_stage(self.submit_ok)
        self.assertIs(images, self.names)  # updated in place for the video stage
        self.assertEqual(images, ["refined_1.png", "refined_2.png", "refined_3.png"])
        self.assertEqual(originals, ["shot0.png", "shot1.png", "shot2.png"])
        self.assertEqual(len(done), 3)
        first = self.jobs[0]
        self.assertEqual(first["image_guide"], self.images[0])
        self.assertEqual(first["image_refs"], [self.ref])  # missing sheets are dropped
        self.assertTrue(first["prompt"].startswith("Shot 0 prompt."))
        with Image.open(first["image_mask"]) as mask:
            self.assertEqual(mask.size, (64, 48))
        # The 32-pixel canvas changed the height; the refined file is brought back.
        with Image.open(os.path.join(self.out, "refined_1.png")) as refined:
            self.assertEqual(refined.size, (64, 48))
            self.assertEqual(refined.info.get("comment"), '{"seed": 1}')

        # A second pass (resume) refines nothing again.
        self.jobs.clear()
        self.run_stage(self.submit_ok, unrefined=originals)
        self.assertEqual(self.jobs, [])

    def test_a_failed_refine_keeps_the_unrefined_image(self):
        def submit(job):
            if job["image_guide"].endswith("shot1.png"):
                raise RuntimeError("worker lost")
            return self.submit_ok(job)

        images, originals, _ = self.run_stage(submit)
        self.assertEqual(images, ["refined_1.png", "shot1.png", "refined_2.png"])
        self.assertEqual(originals, ["shot0.png", "", "shot2.png"])

    def test_a_job_without_output_keeps_the_unrefined_image(self):
        images, originals, _ = self.run_stage(lambda job: [])
        self.assertEqual(images, ["shot0.png", "shot1.png", "shot2.png"])
        self.assertEqual(originals, ["", "", ""])

    def test_missing_shots_are_skipped_and_cancellation_stops(self):
        self.names[1] = ""
        state = {"calls": 0}

        def submit(job):
            state["calls"] += 1
            return self.submit_ok(job)

        images, _, _ = self.run_stage(submit, cancelled=lambda: state["calls"] >= 1)
        self.assertEqual(images, ["refined_1.png", "", "shot2.png"])

    def test_insert_shot_without_people_sends_no_reference(self):
        self.run_stage(self.submit_ok, refs=[])
        self.assertEqual(self.jobs[0]["video_prompt_type"], "VAG")
        self.assertNotIn("keep every face identical", self.jobs[0]["prompt"])


class PipelineRefineTests(unittest.TestCase):
    """The Director pipeline wiring: stage order, identity refs, saved state."""

    @classmethod
    def setUpClass(cls):
        from services import director_pipeline

        cls.pipeline = director_pipeline

    def test_refine_runs_after_review_approval_and_before_video(self):
        path = os.path.join(_APP_DIR, "services", "director_pipeline.py")
        source = open(path, encoding="utf-8").read()
        run = next(node for node in ast.parse(source).body
                   if isinstance(node, ast.FunctionDef) and node.name == "_run_pipeline")
        body = ast.get_source_segment(source, run)
        review = body.index('pause_reason="review_images"')
        resume = body.index("_wait_for_resume(pid)", review)
        refine = body.index("_run_image_refine(pid, params, clip_plans, clip_images")
        video = body.index("output_files = _run_video_generation(pid, params, clip_plans")
        self.assertLess(review, resume)
        self.assertLess(resume, refine)
        self.assertLess(refine, video)
        guard = body.rindex("\n", 0, refine)
        self.assertIn("if requires_shot_images and not _prepared_imgs_ok:", body[body.rindex("if ", 0, guard):refine])
        # Image generation and the storyboard judge come before, never after, the refine.
        self.assertLess(body.index("_run_image_generation(pid, params, clip_plans"), refine)
        self.assertLess(body.index("storyboard_judge.run_stage("), refine)

    def test_run_image_refine_updates_live_state_and_uses_the_shot_cast(self):
        pipeline = self.pipeline
        out = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, out)
        ana = _png(os.path.join(out, "ana.png"))
        ben = _png(os.path.join(out, "ben.png"))
        names = [os.path.basename(_png(os.path.join(out, f"start{i}.png"))) for i in range(2)]
        plans = [{"image_prompt": "Ana smiles at the window."}, {"image_prompt": "Ben pours coffee."}]
        params = {"director_image_refine": "on", "character_ref_paths": [ana, ben],
                  "character_ref_labels": ["Ana", "Ben"]}
        pid = "refinetest"
        pipeline._pipelines[pid] = {"id": pid, "status": "running", "params": params,
                                   "clip_images": list(names), "out_dir": out}
        self.addCleanup(pipeline._pipelines.pop, pid, None)
        jobs = []

        def submit(job, timeout_s=600, workspace=None, out_dir=None):
            jobs.append(job)
            name = f"refined{len(jobs)}.png"
            _png(os.path.join(out_dir, name))
            return [name]

        with mock.patch.object(pipeline, "_submit_and_wait", side_effect=submit), \
                mock.patch.object(pipeline, "_save_pipeline_state", return_value=True) as save:
            pipeline._run_image_refine(pid, params, plans, names, out_dir=out)

        self.assertEqual(names, ["refined1.png", "refined2.png"])
        self.assertEqual([job["image_refs"] for job in jobs], [[ana], [ben]])
        self.assertTrue(all(job["model_type"] == "qwen_image_21_7B" for job in jobs))
        live = pipeline._pipelines[pid]
        self.assertEqual(live["clip_images"], ["refined1.png", "refined2.png"])
        self.assertEqual(live["_clip_unrefined_images"], ["start0.png", "start1.png"])
        self.assertEqual(save.call_count, 2)

    def test_refine_off_submits_nothing(self):
        pipeline = self.pipeline
        with mock.patch.object(pipeline, "_submit_and_wait") as submit:
            pipeline._run_image_refine("nopid", {}, [{"image_prompt": "x"}], ["a.png"], out_dir="/tmp")
        submit.assert_not_called()

    def test_saved_state_round_trips_the_unrefined_filename(self):
        source = open(os.path.join(_APP_DIR, "services", "director_pipeline.py"), encoding="utf-8").read()
        self.assertIn('"start_image_unrefined_filename": (', source)
        self.assertIn('clip_unrefined_images = [c.get("start_image_unrefined_filename") or ""', source)
        self.assertIn('"_clip_unrefined_images": clip_unrefined_images,', source)


if __name__ == "__main__":
    unittest.main()
