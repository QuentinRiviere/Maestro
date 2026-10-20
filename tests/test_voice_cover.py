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
from services.director import bibles  # noqa: E402
from services.director import cover_video as cvid  # noqa: E402
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
            name = "frame.png" if params["generation_mode"] == "image" else "clip.mp4"
            if name.endswith(".png"):
                from PIL import Image
                Image.new("RGB", (8, 8)).save(os.path.join(out_dir, name))
            else:
                Path(out_dir, name).write_bytes(b"mp4")
            return [name]

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
        self.assertTrue((folder / "cover.mp4").is_file())
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


if __name__ == "__main__":
    unittest.main()
