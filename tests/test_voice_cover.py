"""Voice covers: the worker job's settings and mix, the controller's records and run."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from services import voice_cover as worker  # noqa: E402
from services.director import bibles  # noqa: E402
from services.director import voice_cover as vc  # noqa: E402


def _tone(path: str, seconds: float, hz: float, rate: int = 22050, channels: int = 1) -> str:
    t = np.arange(int(seconds * rate)) / rate
    wave = (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    data = np.stack([wave] * channels, axis=1) if channels > 1 else wave
    sf.write(path, data, rate)
    return path


class WorkerTests(unittest.TestCase):
    def test_config_bounds(self):
        config = worker.cover_config({"steps": 500, "cfg_rate": 2, "semitone_shift": -40, "name": "My Song!"})
        self.assertEqual(config["steps"], worker.MAX_STEPS)
        self.assertEqual(config["cfg_rate"], 1.0)
        self.assertEqual(config["semitone_shift"], -12)
        self.assertEqual(config["name"], "My_Song")
        self.assertEqual(worker.cover_config(None)["steps"], worker.DEFAULT_STEPS)

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
                _tone(os.path.join(out_dir, name), 0.2, 200)
            return list(vc.OUTPUTS.values())

        done = vc.start_cover(record["id"], submit=submit, jobs={}, wait=True)
        self.assertEqual(done["status"], "done", done.get("error"))
        self.assertEqual(done["outputs"]["cover"], "cover.wav")
        self.assertEqual(submitted[0]["voice_cover"]["steps"], 12)
        self.assertEqual(submitted[0]["cover_song"], self.song)
        self.assertEqual(submitted[0]["cover_reference"], self.reference)
        self.assertTrue(vc.audio_path(record["id"], "cover").is_file())
        self.assertTrue(vc.audio_path(record["id"], "song").is_file())
        with self.assertRaises(bibles.BibleError):
            vc.audio_path(record["id"], "other")
        vc.delete_cover(record["id"])
        self.assertEqual(vc.list_covers(), [])

    def test_failures(self):
        with self.assertRaises(bibles.BibleError):
            vc.create_cover("x", os.path.join(self.tmp, "missing.wav"), self.reference)
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


if __name__ == "__main__":
    unittest.main()
