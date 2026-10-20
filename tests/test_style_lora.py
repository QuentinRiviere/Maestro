"""Style LoRAs of a bible: dataset, captions, records, training flow with a fake worker, trainer helpers.

No GPU and no model: the worker submit is a fake that writes an adapter file.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from services import style_lora_training as trainer  # noqa: E402
from services.director import bibles  # noqa: E402
from services.director import style_lora as sl  # noqa: E402
from services.director import style_reference as sr  # noqa: E402

HAS_FFMPEG = bool(shutil.which("ffmpeg"))


def _video(path: str, seconds: float = 4.0) -> str:
    import subprocess

    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc=s=320x240:d={seconds}:r=12",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", path], check=True, capture_output=True, timeout=120)
    return path


class TrainerHelpersTests(unittest.TestCase):
    def test_fit_size_keeps_multiples_of_32_and_never_upscales(self):
        self.assertEqual(trainer.fit_size(854, 480, 768), (768, 416))
        self.assertEqual(trainer.fit_size(320, 240, 768), (320, 224))
        self.assertEqual(trainer.fit_size(1920, 1080, 512), (512, 288))

    def test_shift_sigma_follows_the_scheduler(self):
        mu = trainer.calculate_mu(256)
        self.assertAlmostEqual(mu, 0.5)
        # exp(0.5) / (exp(0.5) + 1) at sigma 0.5
        self.assertAlmostEqual(trainer.shift_sigma(0.5, mu), 0.6224593, places=5)
        self.assertLess(trainer.shift_sigma(0.0, mu), 1e-3)
        self.assertGreater(trainer.shift_sigma(1.0, mu), 0.999)

    def test_training_config_bounds(self):
        config = trainer.training_config({"name": "my lora!", "steps": "99999", "rank": 0, "targets": "weird"})
        self.assertEqual(config["name"], "my_lora")
        self.assertEqual(config["steps"], trainer.MAX_STEPS)
        self.assertEqual(config["rank"], 1)
        self.assertEqual(config["targets"], "attention")

    def test_inject_and_state_dict_use_mmgp_keys(self):
        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = nn.Module()
                self.attn.to_q = nn.Linear(8, 8, bias=False)
                self.attn.to_out = nn.ModuleList([nn.Linear(8, 8, bias=False), nn.Dropout(0.0)])
                self.img_mlp = nn.Module()
                self.img_mlp.proj = nn.Linear(8, 16, bias=False)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer_blocks = nn.ModuleList([Block(), Block()])
                self.proj_out = nn.Linear(8, 8, bias=False)

        model = Model()
        before = model.transformer_blocks[0].attn.to_q(torch.ones(1, 8)).clone()
        wrapped = trainer.inject_lora(model, 4, trainer.TARGET_SETS["all"])
        self.assertEqual(sorted(wrapped), ["transformer_blocks.0.attn.to_out.0", "transformer_blocks.0.attn.to_q",
                                           "transformer_blocks.0.img_mlp.proj", "transformer_blocks.1.attn.to_out.0",
                                           "transformer_blocks.1.attn.to_q", "transformer_blocks.1.img_mlp.proj"])
        # B starts at zero: the model is unchanged until it trains.
        torch.testing.assert_close(model.transformer_blocks[0].attn.to_q(torch.ones(1, 8)), before)
        state = trainer.lora_state_dict(wrapped)
        self.assertIn("transformer.transformer_blocks.1.attn.to_q.lora_A.weight", state)
        self.assertIn("transformer.transformer_blocks.1.attn.to_q.lora_B.weight", state)
        self.assertEqual(float(state["transformer.transformer_blocks.1.attn.to_q.alpha"]), 4.0)
        self.assertEqual(state["transformer.transformer_blocks.0.img_mlp.proj.lora_B.weight"].shape, (16, 4))
        self.assertEqual(state["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"].dtype, torch.bfloat16)

    def test_read_manifest_drops_missing_and_unsafe(self):
        with tempfile.TemporaryDirectory() as folder:
            os.makedirs(os.path.join(folder, "images"))
            Path(folder, "images", "a.jpg").write_bytes(b"x")
            Path(folder, "manifest.json").write_text(json.dumps({"items": [
                {"image": "images/a.jpg", "caption": "one"}, {"image": "images/missing.jpg", "caption": "two"},
                {"image": "../etc/passwd", "caption": "bad"}]}))
            items = trainer.read_manifest(folder)
            self.assertEqual([i["caption"] for i in items], ["one"])
            Path(folder, "manifest.json").write_text("[]")
            with self.assertRaises(trainer.TrainingError):
                trainer.read_manifest(folder)


class CaptionTests(unittest.TestCase):
    def test_content_caption_names_the_shot_not_the_look(self):
        frame = {"shot_size": "wide", "prompt": "A rainy street at night, cool grade, 35 mm."}
        shot = {"subject": "a woman in a beige coat", "action": "crossing the street", "angle": "high"}
        self.assertEqual(sl.caption_for(frame, shot, "k_love_style", "content"),
                         "k_love_style, wide shot, a woman in a beige coat, crossing the street, high angle")
        self.assertEqual(sl.caption_for(frame, shot, "k_love_style", "full"),
                         "k_love_style, A rainy street at night, cool grade, 35 mm.")
        self.assertEqual(sl.caption_for(frame, None, "t", "content"), "t, wide shot, A rainy street at night, cool grade, 35 mm.")

    def test_default_trigger_from_the_name(self):
        self.assertEqual(sl.default_trigger({"name": "K Love!"}), "k_love_style")
        self.assertEqual(sl.normalize_trigger("My Look", "x"), "my_look")
        self.assertEqual(sl.normalize_trigger("", "x"), "x")

    def test_training_settings_bounds(self):
        settings = sl.training_settings({"steps": 10, "rank": 999, "caption_mode": "nope", "exclude_close_ups": False},
                                        {"name": "Bible"})
        self.assertEqual(settings["steps"], 50)
        self.assertEqual(settings["rank"], 128)
        self.assertEqual(settings["caption_mode"], "content")
        self.assertFalse(settings["exclude_close_ups"])
        self.assertEqual(settings["trigger"], "bible_style")


class BibleLoraTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.settings = os.path.join(self.tmp, "settings")
        self.loras = os.path.join(self.tmp, "loras", "qwen21")
        os.makedirs(self.loras)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.settings, "director_bibles"))
        patcher.start()
        self.addCleanup(patcher.stop)
        registry = patch.object(sl, "_registry_dir", lambda: self.loras)
        registry.start()
        self.addCleanup(registry.stop)
        self.bible = bibles.save_bible({"name": "K Love"})
        self.bible_id = self.bible["id"]

    def _reference(self, frames: int = 20, close_ups: int = 4) -> dict:
        folder = bibles._folder(self.bible_id)
        video = str(folder / "styleref_test.mp4")
        _video(video)
        reference = {"id": "ref_t", "label": "Ep1", "file": "styleref_test.mp4", "status": "done", "enabled": True,
                     "scene_tags": [], "frames_status": "done", "frames_count": frames}

        def change(bible: dict) -> None:
            bible["style_references"] = [reference]

        bibles._mutate(self.bible_id, change)
        reference = sr.get_reference(self.bible_id, "ref_t")
        shots = [{"index": i, "start": i * 0.2, "end": i * 0.2 + 0.2, "duration": 0.2,
                  "shot_size": "close_up" if i < close_ups else "wide", "subject": f"person {i}", "action": "walks",
                  "credits": i == frames - 1} for i in range(frames)]
        sr.save_shots(self.bible_id, reference, shots)
        sr.save_frames(self.bible_id, reference, [
            {"index": i, "position": "first", "t": i * 0.2 + 0.05, "prompt": f"frame {i} prompt",
             "shot_size": shots[i]["shot_size"]} for i in range(frames)])
        return reference

    def test_candidates_leave_out_close_ups_and_credits(self):
        self._reference(frames=20, close_ups=4)
        bible = bibles.get_bible(self.bible_id)
        settings = sl.training_settings({}, bible)
        found = sl.candidate_frames(self.bible_id, bible, settings)
        self.assertEqual(found["available"], 15)
        self.assertEqual(found["excluded_close_ups"], 4)
        self.assertEqual(found["excluded_credits"], 1)
        self.assertTrue(found["items"][0]["caption"].startswith("k_love_style, wide shot, person 4, walks"))
        settings["exclude_close_ups"] = False
        self.assertEqual(sl.candidate_frames(self.bible_id, bible, settings)["available"], 19)

    def test_records_normalise_and_an_interrupted_training_reads_failed(self):
        lora = sl.create_lora(self.bible_id, {"name": "Look one", "steps": 100})
        self.assertEqual(lora["status"], "queued")
        self.assertEqual(lora["trigger"], "k_love_style")
        self.assertEqual(lora["config"]["steps"], 100)
        self.assertIn("style_loras", bibles.get_bible(self.bible_id))
        # Not in the live set: a restart cut it.
        self.assertEqual(sl.get_lora(self.bible_id, lora["id"])["status"], "failed")
        self.assertEqual(sl.get_lora(self.bible_id, lora["id"])["error"], "Interrupted")
        with self.assertRaises(bibles.BibleError):
            for _ in range(bibles.MAX_STYLE_LORAS + 1):
                sl.create_lora(self.bible_id, {})

    def test_update_enables_one_at_a_time(self):
        first = sl.create_lora(self.bible_id, {"name": "a"})
        second = sl.create_lora(self.bible_id, {"name": "b"})
        for lora in (first, second):
            sl._set(self.bible_id, lora["id"], status="done", file=f"{lora['id']}.safetensors")
        sl.update_lora(self.bible_id, first["id"], {"enabled": True, "scale": 0.5})
        sl.update_lora(self.bible_id, second["id"], {"enabled": True})
        loras = {l["id"]: l for l in sl.list_loras(self.bible_id)}
        self.assertFalse(loras[first["id"]]["enabled"])
        self.assertTrue(loras[second["id"]]["enabled"])
        self.assertEqual(loras[first["id"]]["scale"], 0.5)
        self.assertEqual(sl.enabled_lora(bibles.get_bible(self.bible_id))["id"], second["id"])
        self.assertEqual(sl.enabled_lora(bibles.get_bible(self.bible_id), first["id"])["id"], first["id"])

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg")
    def test_build_dataset_zips_frames_and_captions(self):
        self._reference(frames=16, close_ups=0)
        bible = bibles.get_bible(self.bible_id)
        settings = sl.training_settings({"resolution": 512}, bible)
        archive, facts = sl.build_dataset(self.bible_id, bible, "lora_x", settings)
        self.assertEqual(facts["images"], 15)
        with zipfile.ZipFile(archive) as zipped:
            names = zipped.namelist()
            manifest = json.loads(zipped.read("manifest.json"))
        self.assertIn("images/00001.jpg", names)
        self.assertEqual(len(manifest["items"]), 15)
        self.assertEqual(manifest["trigger"], "k_love_style")
        # The worker reads it back the same way.
        with tempfile.TemporaryDirectory() as folder:
            with zipfile.ZipFile(archive) as zipped:
                zipped.extractall(folder)
            self.assertEqual(len(trainer.read_manifest(folder)), 15)

    @unittest.skipUnless(HAS_FFMPEG, "ffmpeg")
    def test_training_flow_with_a_fake_worker(self):
        self._reference(frames=16, close_ups=0)
        lora = sl.create_lora(self.bible_id, {"name": "Look", "steps": 60})
        submitted = []

        def submit(params, timeout_s, out_dir, job_id=None, **_):
            submitted.append(params)
            os.makedirs(out_dir, exist_ok=True)
            if params.get("lora_training"):
                name = params["lora_training"]["name"] + ".safetensors"
                from safetensors.torch import save_file
                save_file({"transformer.transformer_blocks.0.attn.to_q.lora_A.weight": torch.zeros(2, 4)},
                          os.path.join(out_dir, name))
                Path(out_dir, params["lora_training"]["name"] + ".json").write_text(
                    json.dumps({"final_loss": 0.12, "seconds": 5, "items": 15}))
                return [name]
            Path(out_dir, "sample.png").write_bytes(b"png")
            return ["sample.png"]

        record = sl.start_training(self.bible_id, lora["id"], submit=submit, jobs={}, wait=True)
        self.assertEqual(record["status"], "done", record.get("error"))
        self.assertEqual(record["file"], f"stylelora_{lora['id']}.safetensors")
        self.assertTrue(os.path.isfile(os.path.join(self.loras, record["file"])))
        sidecar = json.loads(Path(self.loras, f"stylelora_{lora['id']}.civitai.json").read_text())
        self.assertEqual(sidecar["trainedWords"], ["k_love_style"])
        self.assertEqual(record["dataset"]["images"], 15)
        self.assertEqual(record["training"]["final_loss"], 0.12)
        self.assertEqual(len(record["samples"]), 3)
        self.assertTrue(record["enabled"])
        training = submitted[0]
        self.assertEqual(training["model_type"], sl.IMAGE_MODEL)
        self.assertEqual(training["lora_training"]["steps"], 60)
        self.assertTrue(training["lora_dataset"].endswith("dataset.zip"))
        sample = submitted[1]
        self.assertEqual(sample["activated_loras"], [record["file"]])
        self.assertEqual(sample["loras_multipliers"], "0.40")
        self.assertTrue(sample["prompt"].endswith("k_love_style"))
        self.assertIsNotNone(sl.sample_path(self.bible_id, lora["id"], 1))
        # Delete removes the registry files too.
        sl.delete_lora(self.bible_id, lora["id"])
        self.assertEqual(os.listdir(self.loras), [])
        self.assertEqual(sl.list_loras(self.bible_id), [])

    def test_trim_letterbox_cuts_black_bars_only(self):
        from PIL import Image

        path = os.path.join(self.tmp, "frame.jpg")
        image = Image.new("RGB", (320, 180), (0, 0, 0))
        image.paste(Image.new("RGB", (320, 134), (120, 90, 80)), (0, 23))
        image.save(path)
        self.assertEqual(sl.trim_letterbox(path), (46, 0))
        with Image.open(path) as trimmed:
            self.assertEqual(trimmed.size, (320, 134))
        self.assertEqual(sl.trim_letterbox(path), (0, 0))
        # A dark picture is not a bar: more than 30 % would go.
        Image.new("RGB", (320, 180), (5, 5, 5)).save(path)
        self.assertEqual(sl.trim_letterbox(path), (0, 0))

    def test_samples_render_again_at_the_current_weight(self):
        lora = sl.create_lora(self.bible_id, {"name": "a"})
        sl._set(self.bible_id, lora["id"], status="done", file="a.safetensors", scale=0.4)
        submitted = []

        def submit(params, timeout_s, out_dir, **_):
            submitted.append(params)
            os.makedirs(out_dir, exist_ok=True)
            Path(out_dir, "s.png").write_bytes(b"png")
            return ["s.png"]

        record = sl.start_samples(self.bible_id, lora["id"], submit=submit, wait=True)
        self.assertEqual(record["status"], "done")
        self.assertEqual(len(record["samples"]), 3)
        self.assertEqual(submitted[0]["loras_multipliers"], "0.40")

    def test_cast_samples_use_the_bible_images(self):
        folder = bibles._folder(self.bible_id)
        for name in ("a.png", "b.png", "l.png"):
            Path(folder, name).write_bytes(b"png")

        def change(bible: dict) -> None:
            bible["characters"] = [
                {"id": "a", "kind": "character", "name": "Ana", "tag": "@ana", "image": "a.png",
                 "variants": [{"id": "default", "label": "Default", "images": [
                     {"id": "i1", "file": "a.png", "status": "approved", "view": ""}]}]},
                {"id": "b", "kind": "character", "name": "Bo", "tag": "@bo", "image": "b.png",
                 "variants": [{"id": "default", "label": "Default", "images": [
                     {"id": "i2", "file": "b.png", "status": "approved", "view": ""}]}]}]
            bible["locations"] = [{"id": "l", "kind": "location", "name": "Loft", "tag": "@loft", "image": "l.png",
                                   "variants": [{"id": "default", "label": "Default", "images": [
                                       {"id": "i3", "file": "l.png", "status": "approved", "view": ""}]}]}]

        bibles._mutate(self.bible_id, change)
        scenes = sl.cast_samples(self.bible_id, bibles.get_bible(self.bible_id))
        self.assertEqual(len(scenes), 3)
        self.assertIn("behind Bo (reference 2)", scenes[0][0])
        self.assertIn("facing Ana (reference 1)", scenes[0][0])
        self.assertEqual([os.path.basename(r) for r in scenes[0][1]], ["a.png", "b.png", "l.png"])
        self.assertEqual(len(scenes[1][1]), 2)
        lora = sl.create_lora(self.bible_id, {"name": "c"})
        sl._set(self.bible_id, lora["id"], status="done", file="c.safetensors")
        submitted = []

        def submit(params, timeout_s, out_dir, **_):
            submitted.append(params)
            os.makedirs(out_dir, exist_ok=True)
            Path(out_dir, "s.png").write_bytes(b"png")
            return ["s.png"]

        record = sl.start_samples(self.bible_id, lora["id"], submit=submit, wait=True, cast=True, scale=0.5)
        self.assertEqual(record["samples_note"], "bible cast, turbo sampler, weight 0.50")
        self.assertEqual(submitted[0]["video_prompt_type"], "I")
        self.assertEqual(len(submitted[0]["image_refs"]), 3)

    def test_training_fails_cleanly_without_frames(self):
        lora = sl.create_lora(self.bible_id, {"name": "Empty"})
        record = sl.start_training(self.bible_id, lora["id"], submit=lambda *a, **k: [], jobs={}, wait=True)
        self.assertEqual(record["status"], "failed")
        self.assertIn("usable frames", record["error"])


if __name__ == "__main__":
    unittest.main()
