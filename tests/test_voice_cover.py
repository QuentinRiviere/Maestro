"""Voice covers: the worker job's settings and mix, the controller's records and run."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from services import voice_cover as worker  # noqa: E402
from services import voice_train as trainer  # noqa: E402
from services.director import bibles  # noqa: E402
from services.director import cover_video as cvid  # noqa: E402
from services.director import persona_clips as pc  # noqa: E402
from services.director import personas as ps  # noqa: E402
from services.director import voice_cover as vc  # noqa: E402
from services.director import voices  # noqa: E402


def _tone(path: str, seconds: float, hz: float, rate: int = 22050, channels: int = 1) -> str:
    t = np.arange(int(seconds * rate)) / rate
    wave = (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    data = np.stack([wave] * channels, axis=1) if channels > 1 else wave
    sf.write(path, data, rate)
    return path


def _output(out_dir: str, name: str) -> None:
    if name.endswith(".wav"):
        _tone(os.path.join(out_dir, name), 0.2, 200)
    else:
        Path(out_dir, name).write_bytes(b"npz")


class WorkerTests(unittest.TestCase):
    def test_config_bounds(self):
        config = worker.cover_config({"steps": 500, "cfg_rate": 2, "semitone_shift": -40, "name": "My Song!"})
        self.assertEqual(config["steps"], worker.MAX_STEPS)
        self.assertEqual(config["cfg_rate"], 1.0)
        self.assertEqual(config["semitone_shift"], -12)
        self.assertEqual(config["name"], "My_Song")
        self.assertEqual(worker.cover_config(None)["steps"], worker.DEFAULT_STEPS)
        rap = worker.cover_config({"mode": "rap", "auto_f0": True})
        self.assertEqual(rap["mode"], "rap")
        self.assertFalse(rap["auto_f0"])
        self.assertEqual(worker.voice_weights(["a", "b"], [3, 1]), [0.75, 0.25])
        self.assertEqual(worker.voice_weights(["a", "b"], None), [0.5, 0.5])
        self.assertEqual(worker.voice_weights(["a", "b"], [0, -1]), [0.5, 0.5])

    def test_mix_resamples_and_limits(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        bed = _tone(os.path.join(tmp, "bed.wav"), 1.0, 220, rate=48000, channels=2)
        voice = _tone(os.path.join(tmp, "voice.wav"), 0.5, 440, rate=44100)
        out = worker.mix(voice, bed, os.path.join(tmp, "mix.wav"), vocal_gain_db=12)
        data, rate = sf.read(out, always_2d=True)
        self.assertEqual(rate, 48000)
        self.assertEqual(data.shape, (48000, 2))
        self.assertLessEqual(float(np.abs(data).max()), 0.99)
        # The voice (louder than the bed once boosted) only covers the first half.
        self.assertGreater(float(np.abs(data[:20000]).max()), float(np.abs(data[30000:]).max()))


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        vc._settings_dir = lambda: os.path.join(self.tmp, "settings")
        self.addCleanup(setattr, vc, "_settings_dir", None)
        voices._settings_dir = vc._settings_dir
        self.addCleanup(setattr, voices, "_settings_dir", None)
        self.song = _tone(os.path.join(self.tmp, "song.wav"), 1.0, 220)
        self.reference = _tone(os.path.join(self.tmp, "ref.wav"), 1.0, 330)

    def test_create_run_and_delete(self):
        record = vc.create_cover("Night song", self.song, self.reference, {"steps": 12}, bible_id="k_love",
                                 character_id="mina")
        self.assertEqual(record["status"], "queued")
        self.assertEqual(record["config"]["steps"], 12)
        submitted = []

        def submit(params, timeout_s, out_dir, job_id=None, **_):
            submitted.append(params)
            os.makedirs(out_dir, exist_ok=True)
            for name in vc.OUTPUTS.values():
                _output(out_dir, name)
            return list(vc.OUTPUTS.values())

        done = vc.start_cover(record["id"], submit=submit, jobs={}, wait=True)
        self.assertEqual(done["status"], "done", done.get("error"))
        self.assertEqual(done["outputs"]["cover"], "cover.wav")
        self.assertEqual(submitted[0]["voice_cover"]["steps"], 12)
        self.assertEqual(submitted[0]["cover_song"], self.song)
        self.assertEqual(submitted[0]["cover_reference"], self.reference)
        self.assertEqual(submitted[0]["voice_cover"]["mode"], "sing")
        self.assertTrue(vc.audio_path(record["id"], "cover").is_file())
        self.assertTrue(vc.audio_path(record["id"], "song").is_file())
        with self.assertRaises(bibles.BibleError):
            vc.audio_path(record["id"], "other")
        vc.delete_cover(record["id"])
        self.assertEqual(vc.list_covers(), [])

    def test_voice_profile_is_kept_and_reused(self):
        second = _tone(os.path.join(self.tmp, "ref2.wav"), 1.0, 440)
        voice = voices.create_voice("Mina", [self.reference, second], [3, 1], mode="rap", bible_id="k_love",
                                    character_id="mina")
        self.assertEqual([r["weight"] for r in voice["references"]], [0.75, 0.25])
        self.assertEqual(voices.job_voice_params(voice)["cover_weights"], [0.75, 0.25])
        record = vc.create_cover("Flow", self.song, voice_id=voice["id"])
        self.assertEqual(record["config"]["mode"], "rap")
        self.assertEqual(record["bible_id"], "k_love")
        submitted = []

        def submit(params, timeout_s, out_dir, job_id=None, **_):
            submitted.append(params)
            os.makedirs(out_dir, exist_ok=True)
            for name in vc.OUTPUTS.values():
                _output(out_dir, name)
            return list(vc.OUTPUTS.values())

        done = vc.start_cover(record["id"], submit=submit, jobs={}, wait=True)
        self.assertEqual(done["status"], "done", done.get("error"))
        self.assertEqual(submitted[0]["cover_references"], [self.reference, second])
        self.assertNotIn("cover_profile", submitted[0])
        voice = voices.get_voice(voice["id"])
        self.assertTrue(voice["profile"].endswith("profile.npz"))
        self.assertEqual(voice["covers"], 1)
        # The next cover sends the stored profile, not the references.
        again = vc.create_cover("Flow 2", self.song, voice_id=voice["id"])
        vc.start_cover(again["id"], submit=submit, jobs={}, wait=True)
        self.assertEqual(submitted[1]["cover_profile"], voice["profile"])
        self.assertNotIn("cover_references", submitted[1])
        with self.assertRaises(bibles.BibleError):
            vc.audio_path(record["id"], "reference")
        # New weights drop the profile; a rename keeps it.
        voices.update_voice(voice["id"], {"name": "Mina v2"})
        self.assertTrue(voices.get_voice(voice["id"])["profile"])
        voices.update_voice(voice["id"], {"weights": [1, 1]})
        self.assertEqual(voices.get_voice(voice["id"])["profile"], "")
        voices.delete_voice(voice["id"])
        self.assertEqual(voices.list_voices(), [])

    def test_failures(self):
        with self.assertRaises(bibles.BibleError):
            vc.create_cover("x", os.path.join(self.tmp, "missing.wav"), self.reference)
        with self.assertRaises(bibles.BibleError):
            vc.create_cover("x", self.song)
        with self.assertRaises(bibles.BibleError):
            voices.create_voice("x", [])
        Path(self.tmp, "notes.txt").write_text("x")
        with self.assertRaises(bibles.BibleError):
            vc.create_cover("x", str(Path(self.tmp, "notes.txt")), self.reference)
        record = vc.create_cover("x", self.song, self.reference)
        done = vc.start_cover(record["id"], submit=lambda *a, **k: [], jobs={}, wait=True)
        self.assertEqual(done["status"], "failed")
        self.assertIn("no cover", done["error"])
        # A record cut by a restart reads as failed.
        vc._set(record["id"], status="running")
        self.assertEqual(vc.get_cover(record["id"])["status"], "failed")
        self.assertEqual(json.loads(vc._records_path().read_text())[-1]["id"], record["id"])


class VideoTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        vc._settings_dir = lambda: os.path.join(self.tmp, "settings")
        self.addCleanup(setattr, vc, "_settings_dir", None)
        voices._settings_dir = vc._settings_dir
        self.addCleanup(setattr, voices, "_settings_dir", None)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp, "settings", "director_bibles"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bible_id = bibles.save_bible({"name": "K Love"})["id"]
        folder = bibles._folder(self.bible_id)
        from PIL import Image
        Image.new("RGB", (64, 64), (200, 100, 50)).save(folder / "mina.png")
        Image.new("RGB", (64, 64), (50, 100, 200)).save(folder / "cafe.png")

        def change(bible):
            def element(key, name, file):
                return {"id": key, "name": name, "variants": [
                    {"id": "default", "label": "Default", "images": [{"file": file, "status": "approved"}]}]}

            bible["characters"] = [element("mina", "Mina", "mina.png")]
            bible["locations"] = [element("cafe", "Café Haru", "cafe.png")]
            bible["world"] = {"look": "Film look."}

        bibles._mutate(self.bible_id, change)
        song = _tone(os.path.join(self.tmp, "song.wav"), 3.0, 220)
        reference = _tone(os.path.join(self.tmp, "ref.wav"), 1.0, 330)
        self.cover = vc.create_cover("Night", song, reference, {}, bible_id=self.bible_id, character_id="mina")

        def submit(params, timeout_s, out_dir, job_id=None, **_):
            os.makedirs(out_dir, exist_ok=True)
            for name in vc.OUTPUTS.values():
                _output(out_dir, name)
            return list(vc.OUTPUTS.values())

        vc.start_cover(self.cover["id"], submit=submit, jobs={}, wait=True)

    def test_frames_and_settings(self):
        self.assertEqual(cvid.frames_for(1.0), 25)
        self.assertEqual(cvid.frames_for(12.0), 289)
        self.assertEqual(cvid.frames_for(40.0), cvid.MAX_FRAMES)
        self.assertEqual(cvid.frames_for(40.0, maximum=10 ** 6), 961)
        self.assertEqual(cvid.window_params(289), {"sliding_window_size": 289})
        self.assertEqual(cvid.window_params(961)["sliding_window_size"], cvid.WINDOW_FRAMES)
        self.assertEqual(cvid.window_params(961)["sliding_window_overlap"], 9)
        long = cvid.video_settings({"seconds": 40})
        self.assertEqual(long["frames"], 961)
        with self.assertRaises(bibles.BibleError):
            cvid.video_settings({"seconds": 500})
        with self.assertRaises(bibles.BibleError):
            pc.clip_settings({"gesture": "wave", "seconds": 40, "speed": 1})
        settings = cvid.video_settings({"seconds": 12, "start_sec": "3.5", "lyrics": "la la"})
        self.assertEqual(settings["frames"], 289)
        self.assertEqual(settings["seconds"], 12.0)
        self.assertEqual(settings["start_sec"], 3.5)
        self.assertEqual(settings["orientation"], "portrait")
        with self.assertRaises(bibles.BibleError):
            cvid.video_settings({"orientation": "square"})
        with self.assertRaises(bibles.BibleError):
            cvid.video_settings({"model_type": "hunyuan"})

    def test_start_image_then_video(self):
        class Pipeline:
            @staticmethod
            def _director_image_prompt_type(model, refs):
                return "KI"

        submitted = []

        def submit(params, timeout_s, out_dir, job_id=None, **_):
            submitted.append(params)
            if params["generation_mode"] == "image":
                from PIL import Image
                Image.new("RGB", (8, 8)).save(os.path.join(out_dir, "frame.png"))
                return ["frame.png"]
            # The worker saves the growing video after each window: the last one is the whole clip.
            Path(out_dir, "window1.mp4").write_bytes(b"one")
            Path(out_dir, "window2.mp4").write_bytes(b"one and two")
            return ["window1.mp4", "window2.mp4"]

        with patch.object(vc, "_director", lambda: Pipeline):
            done = cvid.start_video(self.cover["id"], {"location_id": "cafe", "seconds": 2, "start_sec": 0.5,
                                                       "lyrics": "City lights\nfalling"},
                                    submit=submit, jobs={}, wait=True)
        video = done["video"]
        self.assertEqual(video["status"], "done", video.get("error"))
        self.assertEqual(len(submitted), 2)
        start, clip = submitted
        self.assertEqual(start["model_type"], "qwen_image_21_7B")
        self.assertEqual(len(start["image_refs"]), 2)
        self.assertIn("Mina from reference 1", start["prompt"])
        self.assertIn("Café Haru", start["prompt"])
        self.assertTrue(start["prompt"].endswith("Film look."))
        self.assertEqual(start["resolution"], "720x1280")
        self.assertEqual(clip["model_type"], "ltx2_25")
        self.assertEqual(clip["image_prompt_type"], "S")
        self.assertTrue(clip["image_start"].endswith("start.png"))
        self.assertEqual(clip["audio_prompt_type"], "A")
        self.assertTrue(clip["audio_guide"].endswith("clip_mix.wav"))
        self.assertTrue(clip["audio_conditioning_guide"].endswith("clip_vocals.wav"))
        self.assertEqual(clip["video_length"], 49)
        self.assertEqual(clip["resolution"], "576x1024")
        self.assertIn('Lyrics sung: "City lights / falling"', clip["prompt"])
        self.assertIn("LIP SYNC", clip["prompt"])
        folder = vc.cover_dir(self.cover["id"])
        self.assertEqual((folder / "cover.mp4").read_bytes(), b"one and two")
        self.assertFalse((folder / "window1.mp4").exists())
        data, rate = sf.read(folder / "clip_mix.wav", always_2d=True)
        self.assertEqual(len(data), rate * 2)
        self.assertEqual(cvid.video_path(self.cover["id"], "video").name, "cover.mp4")
        # A second run reuses nothing but the record; an uploaded start image skips the editor.
        submitted.clear()
        from PIL import Image
        Image.new("RGB", (8, 8)).save(folder / "start.png")
        with patch.object(vc, "_director", lambda: Pipeline):
            done = cvid.start_video(self.cover["id"], {"seconds": 1}, start_image=str(folder / "start.png"),
                                    submit=submit, jobs={}, wait=True)
        self.assertEqual(done["video"]["status"], "done", done["video"].get("error"))
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0]["video_length"], 25)

    def test_video_needs_a_done_cover_and_a_character(self):
        record = vc.create_cover("Other", os.path.join(self.tmp, "song.wav"), os.path.join(self.tmp, "ref.wav"), {})
        with self.assertRaises(bibles.BibleError):
            cvid.start_video(record["id"], {"character_id": "mina"})
        with self.assertRaises(bibles.BibleError):
            cvid.start_video(self.cover["id"], {"character_id": "nobody"})


class PersonaClipTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        pc._settings_dir = lambda: os.path.join(self.tmp, "settings")
        self.addCleanup(setattr, pc, "_settings_dir", None)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp, "settings", "director_bibles"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bible_id = bibles.save_bible({"name": "K Love"})["id"]
        folder = bibles._folder(self.bible_id)
        from PIL import Image
        Image.new("RGB", (64, 64), (200, 100, 50)).save(folder / "mina.png")
        Image.new("RGB", (64, 64), (50, 100, 200)).save(folder / "cafe.png")

        def change(bible):
            def element(key, name, file):
                return {"id": key, "name": name, "variants": [
                    {"id": "default", "label": "Default", "images": [{"file": file, "status": "approved"}]}]}

            bible["characters"] = [element("mina", "Mina", "mina.png")]
            bible["locations"] = [element("cafe", "Café Haru", "cafe.png")]
            bible["world"] = {"look": "Film look."}

        bibles._mutate(self.bible_id, change)

    def _submit(self, submitted):
        def submit(params, timeout_s, out_dir, job_id=None, **_):
            submitted.append(params)
            name = "frame.png" if params["generation_mode"] == "image" else "clip.mp4"
            if name.endswith(".png"):
                from PIL import Image
                Image.new("RGB", (8, 8)).save(os.path.join(out_dir, name))
            else:
                Path(out_dir, name).write_bytes(b"mp4")
            return [name]
        return submit

    def test_silent_clip_in_a_place(self):
        class Pipeline:
            @staticmethod
            def _director_image_prompt_type(model, refs):
                return "KI"

        record = pc.create_clip("Mina cafe", {"character_id": "mina", "location_id": "cafe",
                                              "gesture": "a slow finger-gun wink", "seconds": 6, "speed": 1},
                                bible_id=self.bible_id)
        self.assertEqual(record["status"], "queued")
        self.assertEqual(record["settings"]["frames"], 145)
        self.assertEqual(record["settings"]["template"], "single")
        # The default speed renders one and a half times longer, then the clip is sped up to the asked length.
        faster = pc.clip_settings({"gesture": "wave", "seconds": 6})
        self.assertEqual((faster["speed"], faster["frames"], faster["seconds"]), (1.5, 217, 6.0))
        trend = pc.clip_settings({"gesture": "wave", "template": "trend"})
        self.assertEqual([s["frames"] for s in trend["shots"]], [145, 145, 113])
        h3 = pc.clip_settings({"gesture": "wave", "engine": "h3", "seconds": 6})
        self.assertEqual((h3["model_type"], h3["shots"][0]["frames"]), ("minimax_h3", 226))
        self.assertEqual(pc.h3_frames(1), 124)
        job = pc.h3_job(pc.h3_prompt({"name": "Mina"}, h3), "/tmp/s.png", h3, frames=226)
        self.assertEqual((job["resolution"], job["minimax_h3_turbo_preset"], job["video_length"]), ("480x864", pc.H3_TURBO_PRESET, 226))
        self.assertTrue(job["prompt"].startswith("For the target video, at 0.00 seconds"))
        self.assertIn("[Shot 1] Live-action, vertical 9:16", job["prompt"])
        with self.assertRaises(bibles.BibleError):
            pc.clip_settings({"gesture": "wave", "engine": "wan"})
        with self.assertRaises(bibles.BibleError):
            pc.clip_settings({"gesture": "wave", "template": "epic"})
        with self.assertRaises(bibles.BibleError):
            pc.clip_settings({"gesture": "wave", "seconds": 12, "speed": 2})
        self.assertEqual(record["settings"]["audio"], "silent")
        submitted = []
        with patch.object(vc, "_director", lambda: Pipeline):
            done = pc.start_clip(record["id"], submit=self._submit(submitted), jobs={}, wait=True)
        self.assertEqual(done["status"], "done", done.get("error"))
        start, clip = submitted
        self.assertEqual(len(start["image_refs"]), 2)
        self.assertIn("first beat of this move: a slow finger-gun wink", start["prompt"])
        self.assertIn("Café Haru", start["prompt"])
        self.assertEqual(start["resolution"], "720x1280")
        self.assertEqual(clip["model_type"], "ltx2_25")
        self.assertEqual(clip["audio_prompt_type"], "A")
        self.assertNotIn("audio_guide", clip)
        self.assertEqual(clip["video_length"], 145)
        self.assertEqual(clip["temporal_upsampling"], "rife2")
        self.assertEqual(clip["film_grain_intensity"], 0.08)
        self.assertNotIn("temporal_upsampling", pc.clip_job("x", "/tmp/s.png", pc.clip_settings({"gesture": "wave", "smooth": False, "grain": 0})))
        self.assertIn("the whole move in one continuous take: a slow finger-gun wink", clip["prompt"])
        self.assertIn("no slow motion", clip["prompt"])
        self.assertIn("slow motion", clip["negative_prompt"])
        self.assertIn("mouth stays closed", clip["prompt"])
        self.assertEqual(pc.file_path(record["id"], "clip").name, "clip.mp4")
        self.assertEqual(pc.file_path(record["id"], "start").name, "start.png")
        pc.delete_clip(record["id"])
        self.assertEqual(pc.list_clips(), [])

    def test_uploaded_start_image_and_model_audio(self):
        from PIL import Image
        start = os.path.join(self.tmp, "pose.png")
        Image.new("RGB", (8, 8)).save(start)
        record = pc.create_clip("", {"character_id": "mina", "gesture": "thumbs up", "audio": "model",
                                     "place": "a rooftop at dusk", "speed": 1}, bible_id=self.bible_id, start_image=start)
        self.assertEqual(record["name"], "Mina")
        self.assertEqual(record["outputs"], {"start": "start.png"})
        submitted = []
        done = pc.start_clip(record["id"], submit=self._submit(submitted), jobs={}, wait=True)
        self.assertEqual(done["status"], "done", done.get("error"))
        self.assertEqual(len(submitted), 1)
        self.assertEqual(submitted[0]["audio_prompt_type"], "")
        self.assertNotIn("mouth stays closed", submitted[0]["prompt"])
        with self.assertRaises(bibles.BibleError):
            pc.create_clip("x", {"character_id": "mina", "gesture": ""}, bible_id=self.bible_id)
        with self.assertRaises(bibles.BibleError):
            pc.create_clip("x", {"character_id": "cafe", "gesture": "wave"}, bible_id=self.bible_id)


class RvcTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        vc._settings_dir = lambda: os.path.join(self.tmp, "settings")
        self.addCleanup(setattr, vc, "_settings_dir", None)
        voices._settings_dir = vc._settings_dir
        self.addCleanup(setattr, voices, "_settings_dir", None)
        self.song = _tone(os.path.join(self.tmp, "song.wav"), 1.0, 220)
        self.reference = _tone(os.path.join(self.tmp, "ref.wav"), 1.0, 330)

    def test_train_config_and_engine_config(self):
        config = trainer.train_config({"epochs": 5000, "sample_rate": 44100, "batch_size": 0, "name": "My voice!"})
        self.assertEqual(config["epochs"], trainer.MAX_EPOCHS)
        self.assertEqual(config["sample_rate"], 40000)
        self.assertEqual(config["batch_size"], 1)
        self.assertEqual(config["name"], "My_voice")
        self.assertEqual(worker.cover_config({"engine": "rvc"})["engine"], "rvc")
        self.assertEqual(worker.cover_config({"engine": "other"})["engine"], "seedvc")
        self.assertEqual(trainer.infer_config({"semitone_shift": -3, "index_rate": 2})["pitch"], -3)
        self.assertEqual(trainer.infer_config({"index_rate": 2})["index_rate"], 1.0)

    def test_training_then_covers_use_the_model(self):
        voice = voices.create_voice("Mina", [self.reference], None, mode="sing")
        # A sung cover first: its vocals join the training set.
        cover = vc.create_cover("First", self.song, voice_id=voice["id"])
        self.assertEqual(cover["config"]["engine"], "seedvc")

        def sing(params, timeout_s, out_dir, job_id=None, **_):
            os.makedirs(out_dir, exist_ok=True)
            for name in vc.OUTPUTS.values():
                _output(out_dir, name)
            return list(vc.OUTPUTS.values())

        self.assertEqual(vc.start_cover(cover["id"], submit=sing, jobs={}, wait=True)["status"], "done")
        self.assertEqual(len(voices.training_dataset(voices.get_voice(voice["id"]))), 2)
        # A cover whose vocals came out silent is not training material.
        quiet = vc.create_cover("Quiet", self.song, voice_id=voice["id"])

        def mute(params, timeout_s, out_dir, job_id=None, **_):
            sing(params, timeout_s, out_dir, job_id)
            sf.write(os.path.join(out_dir, "vocals.wav"), np.zeros(22050, dtype=np.float32), 22050)
            return list(vc.OUTPUTS.values())

        self.assertEqual(vc.start_cover(quiet["id"], submit=mute, jobs={}, wait=True)["status"], "done")
        self.assertEqual(len(voices.training_dataset(voices.get_voice(voice["id"]))), 2)
        with self.assertRaises(bibles.BibleError):
            voices.resolve_engine(voices.get_voice(voice["id"]), "rvc")
        submitted = []

        def train(params, timeout_s, out_dir, job_id=None, **_):
            submitted.append(params)
            os.makedirs(out_dir, exist_ok=True)
            name = params["voice_train"]["name"]
            Path(out_dir, name + ".pth").write_bytes(b"pth")
            Path(out_dir, name + ".index").write_bytes(b"index")
            return [name + ".pth", name + ".index"]

        trained = voices.start_training(voice["id"], {"epochs": 20}, submit=train, jobs={}, wait=True)
        self.assertEqual(trained["rvc"]["status"], "done", trained["rvc"].get("error"))
        self.assertEqual(trained["rvc"]["config"]["epochs"], 20)
        self.assertEqual(trained["rvc"]["dataset"], 2)
        self.assertEqual(len(submitted[0]["train_dataset"]), 2)
        self.assertTrue(trained["rvc"]["model"].endswith(voice["id"] + ".pth"))
        self.assertTrue(voices.rvc_ready(trained))
        # auto now picks the model; the job carries the pair instead of the profile.
        second = vc.create_cover("Second", self.song, voice_id=voice["id"])
        self.assertEqual(second["config"]["engine"], "rvc")
        forced = vc.create_cover("Third", self.song, voice_id=voice["id"], raw={"engine": "seedvc"})
        self.assertEqual(forced["config"]["engine"], "seedvc")
        jobs = []

        def convert(params, timeout_s, out_dir, job_id=None, **_):
            jobs.append(params)
            os.makedirs(out_dir, exist_ok=True)
            for key, name in vc.OUTPUTS.items():
                if key != "profile":
                    _output(out_dir, name)
            return [n for k, n in vc.OUTPUTS.items() if k != "profile"]

        done = vc.start_cover(second["id"], submit=convert, jobs={}, wait=True)
        self.assertEqual(done["status"], "done", done.get("error"))
        self.assertEqual(jobs[0]["voice_cover"]["engine"], "rvc")
        self.assertTrue(jobs[0]["cover_rvc_model"].endswith(".pth"))
        self.assertNotIn("cover_profile", jobs[0])
        # The RVC cover's vocals are not training material (they would feed the model its own output).
        self.assertEqual(len(voices.training_dataset(voices.get_voice(voice["id"]))), 2)


class PersonaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        for module in (pc, ps):
            module._settings_dir = lambda: os.path.join(self.tmp, "settings")
            self.addCleanup(setattr, module, "_settings_dir", None)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp, "settings", "director_bibles"))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.bible_id = bibles.save_bible({"name": "K Love"})["id"]

    def _submit(self, submitted):
        def submit(params, timeout_s, out_dir, job_id=None, **_):
            submitted.append(params)
            if params["generation_mode"] == "image":
                from PIL import Image
                Image.new("RGB", (8, 16), (120, 40, 40)).save(os.path.join(out_dir, "frame.png"))
                return ["frame.png"]
            Path(out_dir, "clip.mp4").write_bytes(b"mp4")
            return ["clip.mp4"]
        return submit

    def test_persona_from_a_look_then_series_then_bible(self):
        class Pipeline:
            @staticmethod
            def _director_image_prompt_type(model, refs):
                return "KI"

        submitted = []
        with patch.object(vc, "_director", lambda: Pipeline):
            persona = ps.create_persona("Marcel Ventôse", {
                "handle": "@marcel", "look": "a retired lifeguard in a bathrobe", "move": "blows a whistle",
                "places": "a boulangerie\na metro platform\n\na boulangerie", "seconds": 6, "speed": 1},
                submit=self._submit(submitted), jobs={}, wait=True)
        self.assertEqual(persona["status"], "done", persona.get("error"))
        self.assertEqual(persona["handle"], "marcel")
        self.assertEqual(persona["places"], ["a boulangerie", "a metro platform"])
        self.assertTrue(os.path.isfile(persona["portrait"]))
        self.assertIn("retired lifeguard", submitted[0]["prompt"])
        self.assertNotIn("image_refs", submitted[0])
        submitted.clear()
        self.assertEqual(persona["template"], "trend")

        def concat(parts, output):
            Path(output).write_bytes(b"".join(Path(p).read_bytes() for p in parts))
            return output

        with patch.object(vc, "_director", lambda: Pipeline), patch.object(pc, "concat_shots", concat):
            clips = ps.series(persona["id"], count=2, submit=self._submit(submitted), jobs={}, wait=True)
        self.assertEqual([c["status"] for c in clips], ["done", "done"], [c.get("error") for c in clips])
        self.assertEqual(clips[0]["settings"]["persona_id"], persona["id"])
        self.assertEqual(clips[1]["settings"]["place"], "a metro platform")
        # Three shots (wide, side, close): three start images and three LTX jobs per clip, 11 s shown.
        self.assertEqual([s["framing"] for s in clips[0]["settings"]["shots"]], ["wide", "side", "close"])
        self.assertEqual(clips[0]["settings"]["seconds"], 11.0)
        self.assertEqual(len(submitted), 12)
        self.assertEqual(submitted[0]["image_refs"], [persona["portrait"]])
        self.assertIn("wide shot", submitted[0]["prompt"])
        self.assertIn("smartphone", submitted[0]["prompt"])
        self.assertIn("the opening of the move", submitted[1]["prompt"])
        self.assertIn("blows a whistle", submitted[1]["prompt"])
        # Later shots keep the place: the first start image is their second reference.
        self.assertEqual(len(submitted[2]["image_refs"]), 2)
        self.assertIn("same place as reference 2", submitted[2]["prompt"])
        self.assertIn("close-up", submitted[4]["prompt"])
        self.assertEqual((pc.clip_dir(clips[0]["id"]) / "clip.mp4").read_bytes(), b"mp4mp4mp4")
        self.assertEqual(ps.get_persona(persona["id"])["clips"], 2)
        # Joining a bible makes a character with the portrait approved.
        bible = ps.add_to_bible(persona["id"], self.bible_id)
        character = next(c for c in bible["characters"] if c["name"] == "Marcel Ventôse")
        self.assertTrue(character["image"])
        self.assertEqual(ps.get_persona(persona["id"])["character_id"], character["id"])
        with self.assertRaises(bibles.BibleError):
            ps.add_to_bible(persona["id"], self.bible_id)
        # The judge: a fake analyst answers the rubric.
        def ask(http, fal_key, model, system, prompt, media, sleep, **kw):
            self.assertIn("blows a whistle", prompt)
            self.assertEqual(len(media), 2)
            return ('{"identity": 4, "move": 3, "place": 5, "artefacts": "4", "hook": 2, "verdict": "ok", "fix": "closer"}', 0)

        with patch.object(ps, "frame_sheet", lambda video, output, **kw: Path(output).write_bytes(b"jpg") or output), \
             patch("services.director.room_spec._resolve_backend", lambda backend, key, model, ask_fn: (ask_fn, None, "analyst", True)):
            evaluation = ps.evaluate_clip(clips[0]["id"], ask=ask)
        self.assertEqual(evaluation["overall"], 3.6)
        self.assertEqual(evaluation["artefacts"], 4)
        self.assertEqual(pc.get_clip(clips[0]["id"])["evaluation"]["verdict"], "ok")

    def test_persona_needs_a_look_or_a_character(self):
        with self.assertRaises(bibles.BibleError):
            ps.create_persona("Nobody", {"move": "waves"})
        with self.assertRaises(bibles.BibleError):
            ps.create_persona("Nobody", {"look": "a man", "move": ""})
        with self.assertRaises(bibles.BibleError):
            ps.persona_fields({"format": "podcast"})


if __name__ == "__main__":
    unittest.main()
