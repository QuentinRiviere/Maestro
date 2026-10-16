"""The Director's default image model: Qwen-Image 2.1 base + Viggle Turbo v0.2.1 + KV cache."""

from __future__ import annotations

import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services import director_pipeline as pipeline  # noqa: E402


class DefaultImageModelTest(unittest.TestCase):
    """The Director's default image model: Qwen-Image 2.1 base + Viggle Turbo v0.2.1 + KV cache."""

    def test_a_request_without_an_image_model_gets_the_default_and_its_sampler(self):
        params = {"_director_shot_image_policy": "generate", "image_params": {"resolution": "1280x720",
                                                                                "num_inference_steps": 40,
                                                                                "guidance_scale": 4.0}}
        pipeline.apply_director_image_defaults(params)
        self.assertEqual(params["image_model"], "qwen_image_21_7B")
        self.assertEqual(params["image_params"], {
            "resolution": "1280x720", "num_inference_steps": 6, "guidance_scale": 1.0,
            "sample_solver": "viggle_v021", "custom_settings": {"qwen21_kv_cache": "Enabled"},
        })
        self.assertEqual(pipeline.director_image_job_fields(params["image_params"]),
                         {"sample_solver": "viggle_v021", "custom_settings": {"qwen21_kv_cache": "Enabled"}})

    def test_explicit_choices_are_kept(self):
        klein = {"_director_shot_image_policy": "generate", "image_model": "flux2_klein_9b",
                 "image_params": {"num_inference_steps": 4}}
        pipeline.apply_director_image_defaults(klein)
        self.assertEqual((klein["image_model"], klein["image_params"]), ("flux2_klein_9b", {"num_inference_steps": 4}))
        euler = {"image_model": "qwen_image_21_7B", "image_params": {"sample_solver": "default",
                                                                     "num_inference_steps": 40}}
        pipeline.apply_director_image_defaults(euler)
        self.assertEqual(euler["image_params"], {"sample_solver": "default", "num_inference_steps": 40})
        no_cache = pipeline.director_image_params("qwen_image_21_7B",
                                                  {"custom_settings": {"qwen21_kv_cache": "Disabled"}})
        self.assertEqual(no_cache["custom_settings"], {"qwen21_kv_cache": "Disabled"})

    def test_prompt_only_projects_keep_no_image_model(self):
        params = {"_director_shot_image_policy": "prompt_only"}
        pipeline.apply_director_image_defaults(params)
        self.assertFalse(params.get("image_model"))

    def test_saved_projects_without_a_model_rerun_on_klein(self):
        self.assertEqual(pipeline.LEGACY_DIRECTOR_IMAGE_MODEL, "flux2_klein_9b")
        self.assertEqual(pipeline.director_image_job_fields({"num_inference_steps": 8}), {})

    def test_the_viggle_profile_resolves_in_the_qwen_handler(self):
        from models.qwen21.qwen21_handler import apply_acceleration_profile

        preset = pipeline.DIRECTOR_IMAGE_SAMPLERS["qwen_image_21_7B"]
        resolved = apply_acceleration_profile({"sample_solver": preset["sample_solver"], "num_inference_steps": 40,
                                               "guidance_scale": 4.0, "activated_loras": [],
                                               "loras_multipliers": ""})
        self.assertEqual((resolved["num_inference_steps"], resolved["guidance_scale"]),
                         (preset["num_inference_steps"], preset["guidance_scale"]))
        self.assertTrue(resolved["activated_loras"][0].endswith(
            "Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
