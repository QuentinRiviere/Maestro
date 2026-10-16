"""Photographic references keep identity only; stylized references keep their art style.

With photographic bible references (studio portraits, a daytime location
photo), "...and the art style of the reference image" made the editor copy
their flat studio light. Those runs end with PHOTOGRAPHIC_IDENTITY_ENDING.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402
from services.director.prompt_polish import (  # noqa: E402
    ART_STYLE_IDENTITY_ENDING,
    PHOTOGRAPHIC_IDENTITY_ENDING,
    apply_photographic_identity_ending,
    identity_ending,
    polish_prompts_third_pass,
    sanitize_image_prompt,
)

_GUIDES = Path(_APP_DIR) / "services" / "llm_guides" / "director"
_OLD = "the art style of the reference image"
_SHOT = ("Medium shot of the woman from the reference image at the café table at night. "
         "LIGHTING: amber pendants above the table, cold neon through the window, deep shadows. "
         "Use lighting and color temp from reference image. " + ART_STYLE_IDENTITY_ENDING)


def _polish(prompt: str, **kwargs) -> tuple[str, str]:
    """(polished image prompt, polish system prompt); the LLM echoes its input."""
    calls = []

    def fake_enhance(**call):
        calls.append(call)
        return call["prompt"]

    with patch("services.llm_service.enhance_prompt", side_effect=fake_enhance):
        result = polish_prompts_third_pass(
            [{"image_prompt": prompt}], "ltx2_22B_distilled_1_1", "qwen_image_21_7B",
            polish_video_prompts=False, polish_image_prompts=True, **kwargs,
        )
    return result[0]["image_prompt"], calls[0]["system_override"]


class EndingTests(unittest.TestCase):
    def test_identity_ending_choice(self):
        self.assertEqual(identity_ending(True), PHOTOGRAPHIC_IDENTITY_ENDING)
        self.assertEqual(identity_ending(False), ART_STYLE_IDENTITY_ENDING)
        self.assertNotIn("art style", PHOTOGRAPHIC_IDENTITY_ENDING)

    def test_photographic_replaces_the_art_style_sentence_and_reference_light(self):
        cleaned = apply_photographic_identity_ending(_SHOT)
        self.assertTrue(cleaned.endswith(PHOTOGRAPHIC_IDENTITY_ENDING))
        self.assertNotIn(_OLD, cleaned)
        self.assertNotIn("color temp from reference", cleaned)
        self.assertIn("cold neon through the window, deep shadows.", cleaned)
        self.assertEqual(cleaned.count("Keep each person's face"), 1)
        # Idempotent, and an insert without an identity sentence gets none.
        self.assertEqual(apply_photographic_identity_ending(cleaned), cleaned)
        self.assertEqual(apply_photographic_identity_ending("Hands on a mug."), "Hands on a mug.")

    def test_sanitizer_keeps_the_new_ending(self):
        prompt = "SUBJECT: the girl from the reference image. | MOOD: tense. " + PHOTOGRAPHIC_IDENTITY_ENDING
        self.assertTrue(sanitize_image_prompt(prompt).endswith(PHOTOGRAPHIC_IDENTITY_ENDING))


class PolishTests(unittest.TestCase):
    def test_photographic_polish_asks_for_and_enforces_the_identity_ending(self):
        polished, system = _polish(_SHOT, photographic_references=True)
        self.assertIn(f"ends with: '{PHOTOGRAPHIC_IDENTITY_ENDING}'", system)
        self.assertNotIn(f"ends with: '{ART_STYLE_IDENTITY_ENDING}'", system)
        self.assertTrue(polished.endswith(PHOTOGRAPHIC_IDENTITY_ENDING))
        self.assertNotIn(_OLD, polished)
        self.assertIn("amber pendants above the table", polished)

    def test_stylized_polish_keeps_the_art_style_ending(self):
        polished, system = _polish(_SHOT)
        self.assertIn(f"ends with: '{ART_STYLE_IDENTITY_ENDING}'", system)
        self.assertNotIn(PHOTOGRAPHIC_IDENTITY_ENDING, system)
        self.assertTrue(polished.endswith(ART_STYLE_IDENTITY_ENDING))

    def test_polish_keeps_the_lighting_language(self):
        _, system = _polish(_SHOT)
        self.assertIn("Keep the lighting description complete", system)
        self.assertIn("never soften it into 'balanced', 'even' or 'natural' light", system)


class PipelineTests(unittest.TestCase):
    def test_which_references_are_photographic(self):
        photographic = pipeline.director_references_photographic
        self.assertTrue(photographic({"_director_style_contract": {"medium": "live action K-drama"}}))
        self.assertTrue(photographic({"_director_style_contract": {"medium": "unspecified"},
                                      "_director_bible_look": "Cinematic digital photography, night rain."}))
        self.assertFalse(photographic({"_director_style_contract": {"medium": "2D anime"}}))
        self.assertFalse(photographic({"_reference_style": "hand-drawn pencil sketch",
                                       "_director_style_contract": {"medium": "live action"}}))
        self.assertTrue(photographic({"_reference_style": ""}))
        self.assertFalse(photographic({}))

    def test_generation_time_ending(self):
        self.assertTrue(pipeline._finish_identity_ending(_SHOT, True).endswith(PHOTOGRAPHIC_IDENTITY_ENDING))
        self.assertEqual(pipeline._finish_identity_ending(_SHOT, False), _SHOT)

    def test_insert_drops_either_identity_sentence(self):
        plan = {"_director_camera_framing": "close-up insert", "_director_scene_type": "dialogue"}
        for ending in (ART_STYLE_IDENTITY_ENDING, PHOTOGRAPHIC_IDENTITY_ENDING):
            _path, prompt = pipeline.director_shot_layout("p", 0, plan, f"Hands on a mug. {ending}",
                                                          "qwen_image_21_7B", "/tmp")
            self.assertNotIn("Preserve character identity", prompt)
            self.assertNotIn("Keep each person's face", prompt)


class GuideTests(unittest.TestCase):
    def test_guides_give_photographic_references_the_identity_ending(self):
        for name in ("flux_image_generation.md", "flux_image_edit_pass2.md", "qwen_image_edit_reference.md"):
            text = (_GUIDES / name).read_text(encoding="utf-8")
            self.assertIn(PHOTOGRAPHIC_IDENTITY_ENDING, text, name)
            self.assertIn("Photographic / live-action reference", text, name)
            self.assertIn("Night and interior drama scenes are low-key", text, name)
            self.assertNotIn(f'ALWAYS end every image_prompt with: "{ART_STYLE_IDENTITY_ENDING}"', text, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
