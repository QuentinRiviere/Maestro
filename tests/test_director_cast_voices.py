"""Each bible character's linked library voice speaks that character's lines."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import character_library  # noqa: E402
from services import director_pipeline as pipeline  # noqa: E402
from services.director import bibles  # noqa: E402


class CastVoiceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.object(bibles, "BIBLES_DIR", Path(self.tmp.name) / "bibles")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.files = {}
        for name in ("mara.wav", "jon.wav", "project.wav", "mara.png", "jon.png", "lea.png"):
            path = os.path.join(self.tmp.name, name)
            with open(path, "wb") as handle:
                handle.write(b"RIFF" if name.endswith(".wav") else b"\x89PNG")
            self.files[name] = path
        library = {
            "lib_mara": {"id": "lib_mara", "voice": {"path": self.files["mara.wav"]}},
            "lib_jon": {"id": "lib_jon", "voice": {"path": self.files["jon.wav"]}},
            "lib_gone": {"id": "lib_gone", "voice": {"path": os.path.join(self.tmp.name, "missing.wav")}},
        }

        def get_character(character_id):
            if character_id not in library:
                raise ValueError("Saved character not found.")
            return library[character_id]

        patcher = patch.object(character_library, "get_character", side_effect=get_character)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _params(self, **extra):
        cast = {"characters": [
            {"name": "Mara", "tag": "@mara", "voice_character_id": "lib_mara"},
            {"name": "Jon", "tag": "@jon", "voice_character_id": "lib_jon"},
            {"name": "Lea", "tag": "@lea", "voice_character_id": "lib_gone"},
        ], "locations": []}
        return {
            "_director_cast": cast,
            "characters": [{"name": c["name"]} for c in cast["characters"]],
            "character_ref_paths": [self.files["mara.png"], self.files["jon.png"], self.files["lea.png"]],
            "character_ref_labels": ["Mara", "Jon", "Lea"],
            "voice_reference": self.files["project.wav"],
            **extra,
        }

    def test_project_cast_keeps_the_linked_voice(self):
        bible = bibles.save_bible({
            "name": "Harbor",
            "characters": [{"name": "Mara", "description": "fisher", "voice_character_id": "lib_mara"},
                           {"name": "Jon", "description": "keeper"}],
        })
        cast = bibles.project_cast(bible, os.path.join(self.tmp.name, "episode"))

        self.assertEqual(cast["characters"][0]["voice_character_id"], "lib_mara")
        self.assertNotIn("voice_character_id", cast["characters"][1])

    def test_missing_voices_are_skipped(self):
        self.assertEqual(pipeline._director_cast_voices(self._params()),
                         {"Mara": self.files["mara.wav"], "Jon": self.files["jon.wav"]})

    def test_h3_manifest_gets_one_voice_per_speaking_character(self):
        manifest = pipeline._director_h3_reference_manifest(
            self._params(), None, out_dir=self.tmp.name, speakers=["Jon", "Mara"],
        )
        voices = [r for r in manifest if r["type"] == "audio"]

        self.assertEqual([(r["role"], r["character_name"]) for r in voices],
                         [("the voice of Jon", "Jon"), ("the voice of Mara", "Mara")])
        self.assertNotIn(self.files["project.wav"], [r["path"] for r in voices])
        named = {r.get("character_name") for r in manifest if r["type"] == "image"}
        self.assertEqual(named, {"Mara", "Jon", None})

    def test_h3_manifest_falls_back_to_the_project_voice(self):
        # Lea speaks but her linked voice file is gone.
        manifest = pipeline._director_h3_reference_manifest(
            self._params(), None, out_dir=self.tmp.name, speakers=["Lea"],
        )
        voices = [r for r in manifest if r["type"] == "audio"]

        self.assertEqual([r["path"] for r in voices], [self.files["project.wav"]])

    def test_ltx_clip_voice_is_its_main_speakers(self):
        params = self._params()
        plans = [
            {"_director_dialogue_beats": [{"speaker_id": "char_1", "spoken_text": "Hold the line."},
                                          {"speaker_id": "Mara", "spoken_text": "Now."}]},
            {"_director_dialogue_beats": [{"speaker_id": "s2", "spoken_text": "Look."}],
             "_director_subjects_on_screen": [{"character_id": "s2", "speaker_name": "Mara",
                                               "visual_description": "a fisher"}]},
            {"_director_dialogue_beats": [{"speaker_id": "Lea", "spoken_text": "Hi."},
                                          {"speaker_id": "@jon", "spoken_text": "Hey."}]},
            {"_director_dialogue_beats": [{"speaker_id": "Lea", "spoken_text": "Alone."}]},
            {"_director_dialogue_beats": []},
        ]

        self.assertEqual(pipeline._director_clip_voice_references(params, plans), [
            self.files["jon.wav"], self.files["mara.wav"], self.files["jon.wav"],
            self.files["project.wav"], self.files["project.wav"],
        ])
        params.pop("voice_reference")
        self.assertEqual(pipeline._director_clip_voice_references(params, plans[3:]), ["", ""])

    def test_ltx_run_sends_one_voice_per_clip(self):
        model_def = {"name": "LTX-2", "architecture": "ltx2_22B", "fps": 24,
                     "frames_minimum": 17, "frames_maximum": 241, "frames_steps": 8, "latent_size": 8,
                     "image_prompt_types_allowed": "TSE", "sliding_window": True, "returns_audio": True}
        original = pipeline._wgp
        self.addCleanup(setattr, pipeline, "_wgp", original)
        pipeline._wgp = SimpleNamespace(
            save_path=self.tmp.name,
            get_model_def=lambda selected: model_def if selected == "ltx2_22B_distilled_1_1" else None,
            get_model_family=lambda selected, for_ui=False: "ltx2",
            get_base_model_type=lambda selected: "ltx2_22B",
            get_model_min_frames_and_step=lambda selected: (17, 8, 8),
            get_default_settings=lambda selected: {},
            get_lora_dir=lambda selected: os.path.join(self.tmp.name, "loras"),
        )
        plans = [
            {"video_prompt": "Jon speaks", "_director_dialogue_beats": [{"speaker_id": "Jon", "spoken_text": "Go."}]},
            {"video_prompt": "Mara speaks", "_director_dialogue_beats": [{"speaker_id": "Mara", "spoken_text": "No."}]},
        ]
        captured = {}
        params = self._params(video_model="ltx2_22B_distilled_1_1", pipeline_type="short_film_story",
                              seamless=False, video_params={})
        with patch.object(pipeline, "_submit_and_wait",
                          side_effect=lambda gen, **kwargs: captured.update(gen) or ["joined.mp4"]):
            pipeline._run_video_generation("cast-voices", params, plans, [{"duration_frames": 97}] * 2,
                                           ["", ""], out_dir=self.tmp.name)

        self.assertEqual(captured["per_clip_voice_references"], [self.files["jon.wav"], self.files["mara.wav"]])
        self.assertEqual(captured["voice_reference"], self.files["project.wav"])


if __name__ == "__main__":
    unittest.main()
