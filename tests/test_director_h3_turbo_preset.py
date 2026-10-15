"""director_h3_turbo_preset: base H3 FL2VA in Turbo mode for live-action Director runs."""

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from services import director_pipeline as pipeline  # noqa: E402

PDD = "alibaba-pai-fl2va-pdd-8step"


def model_definition(model):
    from models.minimax_h3.minimax_h3_handler import family_handler

    defaults = json.loads((ROOT / "app/defaults" / f"{model}.json").read_text(encoding="utf-8"))
    return {
        **defaults["model"],
        **family_handler.query_model_def(defaults["model"]["architecture"], defaults["model"]),
    }


def project(model="minimax_h3", **extra):
    return {"video_model": model, "video_params": {"resolution": "864x480", "num_inference_steps": 20},
            "director_max_shot_frames": 124, **extra}


class DirectorTurboPresetTests(unittest.TestCase):
    def test_default_leaves_base_h3_unaccelerated(self):
        params = project()
        profile = pipeline._create_director_video_execution_profile(
            params, model_def=model_definition("minimax_h3"), hardware={"gpu_vram_gb": 24})
        self.assertFalse(profile["turbo_mode"])
        self.assertEqual(params["video_params"]["num_inference_steps"], 20)
        self.assertNotIn("minimax_h3_turbo_preset", params["video_params"])

    def test_preset_sets_turbo_mode_preset_and_steps_for_every_child_job(self):
        definition = model_definition("minimax_h3")
        params = project(director_h3_turbo_preset=PDD)
        profile = pipeline._create_director_video_execution_profile(
            params, model_def=definition, hardware={"gpu_vram_gb": 24})
        self.assertTrue(profile["turbo_mode"])
        self.assertEqual(profile["turbo_preset"], PDD)
        saved = params["video_params"]  # persisted as the project's video_params (reruns read it)
        self.assertIs(saved["minimax_h3_turbo_mode"], True)
        self.assertEqual(saved["minimax_h3_turbo_preset"], PDD)
        self.assertEqual(saved["num_inference_steps"], 8)
        # A child job (shot, sequence unit or rerun) built from the saved video_params.
        child = {"model_type": "minimax_h3", "video_length": 124,
                 "num_inference_steps": saved["num_inference_steps"]}
        with patch.object(pipeline, "_wgp", SimpleNamespace(get_model_def=lambda _: definition)):
            pipeline._apply_director_h3_optimizations(child, saved, profile)
            pipeline._prepare_director_generation_params(child)
        self.assertIs(child["minimax_h3_turbo_mode"], True)
        self.assertEqual(child["minimax_h3_turbo_preset"], PDD)
        self.assertEqual(child["num_inference_steps"], 8)
        self.assertIn("MiniMax-H3-FL2VA-Acc-8Step.safetensors", child.get("activated_loras") or [])

    def test_aliases_off_values_and_unknown_presets(self):
        definition = model_definition("minimax_h3")
        self.assertEqual(pipeline.director_h3_turbo_preset_id(
            {"director_h3_turbo_preset": "pdd"}, "minimax_h3", definition), PDD)
        for off in (None, "", "off", "default", False):
            self.assertEqual(pipeline.director_h3_turbo_preset_id(
                {"director_h3_turbo_preset": off}, "minimax_h3", definition), "")
        with self.assertRaises(ValueError):
            pipeline.director_h3_turbo_preset_id(
                {"director_h3_turbo_preset": "no-such-preset"}, "minimax_h3", definition)
        # Not an H3 model: ignored.
        self.assertEqual(pipeline.director_h3_turbo_preset_id(
            {"director_h3_turbo_preset": PDD}, "ltx2_22B", {"architecture": "ltx2_22B"}), "")

    def test_fused_turbo_checkpoint_ignores_it(self):
        definition = model_definition("minimax_h3_fused_turbo")
        self.assertEqual(pipeline.director_h3_turbo_preset_id(
            {"director_h3_turbo_preset": PDD}, "minimax_h3_fused_turbo", definition), "")

    def test_hand_detail_never_lowers_the_preset_steps(self):
        plans = [{"_director_hand_detail": True}, {}]
        # Base H3 keeps the preset's 8 steps; the fused 4-step bump never goes below its base.
        self.assertIsNone(pipeline.director_hand_detail_steps("minimax_h3", plans, 8))
        self.assertIsNone(pipeline.director_hand_detail_steps("minimax_h3_fused_turbo", plans, 8))
        self.assertIsNone(pipeline.director_hand_detail_steps("minimax_h3_fused_turbo", plans, 10))
        self.assertEqual(pipeline.director_hand_detail_steps("minimax_h3_fused_turbo", plans, 4), [8, 4])


if __name__ == "__main__":
    unittest.main()
