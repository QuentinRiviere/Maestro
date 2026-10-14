"""Official Qwen-Image-2.1-Turbo checkpoint: schedule, render size and Director use."""

import json
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from models.qwen21.qwen21_handler import (  # noqa: E402
    MODEL_TYPE,
    apply_acceleration_profile,
    family_handler,
    fixed_sampling_profile,
)
from models.qwen21.render_size import (  # noqa: E402
    downscale_reference,
    render_dimensions,
    resize_uint8_images,
)
from models.qwen21.sampling_profiles import (  # noqa: E402
    QWEN_TURBO_PROFILE,
    QWEN_TURBO_SIGMAS,
    qwen21_sampling_profile,
)

TURBO_TYPE = "qwen_image_21_7B_turbo_8step"
TURBO_DEF_PATH = ROOT / "app" / "defaults" / f"{TURBO_TYPE}.json"
BASE_DEF_PATH = ROOT / "app" / "defaults" / f"{MODEL_TYPE}.json"
# model_index.json "sample_sigmas" of Qwen/Qwen-Image-2.1-Turbo, revision
# fc13ba1986fcb905854dc8de2f52eee395f49b5b.
PUBLISHED_TURBO_SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
TURBO_REVISION = "fc13ba1986fcb905854dc8de2f52eee395f49b5b"
TURBO_PIXELS = 864 * 1536


def _definition(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_model_def(path):
    """Mirror wgp.init_model_def: handler defaults, then the JSON model block."""
    model = _definition(path)["model"]
    resolved = family_handler.query_model_def(MODEL_TYPE, model)
    resolved.update(model)
    return resolved


class TurboSamplingProfileTests(unittest.TestCase):
    def test_profile_uses_the_published_eight_sigmas(self):
        overrides, sigmas = qwen21_sampling_profile(QWEN_TURBO_PROFILE, 8)
        self.assertEqual(list(sigmas), PUBLISHED_TURBO_SIGMAS)
        self.assertEqual(len(sigmas), 8)
        # Same as the checkpoint's scheduler_config.json: no resolution shift.
        self.assertEqual(overrides, {"use_dynamic_shifting": False, "shift": 1.0, "shift_terminal": None})
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            qwen21_sampling_profile(QWEN_TURBO_PROFILE, 40)

    def test_scheduler_runs_the_nodes_unchanged_then_reaches_zero(self):
        from diffusers import FlowMatchEulerDiscreteScheduler

        # WanGP's shared Qwen 2.1 scheduler config enables dynamic shifting and
        # a terminal stretch; the profile overrides must neutralize both.
        config = json.loads((ROOT / "app/models/qwen21/configs/scheduler_scheduler_config.json")
                            .read_text(encoding="utf-8"))
        overrides, sigmas = qwen21_sampling_profile(QWEN_TURBO_PROFILE, 8)
        scheduler = FlowMatchEulerDiscreteScheduler.from_config(config, **overrides)
        self.assertFalse(scheduler.config.use_dynamic_shifting)
        scheduler.set_timesteps(sigmas=list(sigmas), device="cpu")
        torch.testing.assert_close(scheduler.sigmas, torch.tensor(PUBLISHED_TURBO_SIGMAS + [0.0]),
                                   atol=1e-6, rtol=0)
        torch.testing.assert_close(scheduler.timesteps, torch.tensor(PUBLISHED_TURBO_SIGMAS) * 1000,
                                   atol=1e-3, rtol=0)

    def test_fixed_profile_forces_eight_steps_cfg_one_and_drops_viggle_adapters(self):
        model_def = _definition(TURBO_DEF_PATH)["model"]
        self.assertEqual(fixed_sampling_profile(model_def),
                         {"solver": QWEN_TURBO_PROFILE, "label": "Qwen Turbo (8 steps, checkpoint schedule)",
                          "steps": 8, "guidance": 1.0})
        self.assertIsNone(fixed_sampling_profile(_definition(BASE_DEF_PATH)["model"]))
        user_lora = "https://example.com/style.safetensors"
        viggle = ("https://huggingface.co/Viggle/Qwen-Image-2.1-viggle-turbo/resolve/main/"
                  "Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors")
        settings = apply_acceleration_profile(
            {"sample_solver": "viggle_v01", "num_inference_steps": 4, "guidance_scale": 4,
             "activated_loras": [user_lora, viggle], "loras_multipliers": "0.7 1"},
            selected_profile=QWEN_TURBO_PROFILE)
        self.assertEqual(settings["sample_solver"], QWEN_TURBO_PROFILE)
        self.assertEqual(settings["num_inference_steps"], 8)
        self.assertEqual(settings["guidance_scale"], 1.0)
        self.assertEqual(settings["activated_loras"], [user_lora])
        self.assertEqual(settings["loras_multipliers"], "0.7")

    def test_validation_normalizes_turbo_settings_and_rejects_turbo_on_base(self):
        turbo = _resolved_model_def(TURBO_DEF_PATH)
        inputs = {"sample_solver": "default", "num_inference_steps": 40, "guidance_scale": 4.0,
                  "video_prompt_type": "I", "image_refs": [Image.new("RGB", (8, 8))] * 4}
        self.assertIsNone(family_handler.validate_generative_settings(MODEL_TYPE, turbo, inputs))
        self.assertEqual((inputs["sample_solver"], inputs["num_inference_steps"], inputs["guidance_scale"]),
                         (QWEN_TURBO_PROFILE, 8, 1.0))
        base = _resolved_model_def(BASE_DEF_PATH)
        error = family_handler.validate_generative_settings(
            MODEL_TYPE, base, {"sample_solver": QWEN_TURBO_PROFILE, "video_prompt_type": ""})
        self.assertIn("Turbo", error)

    def test_turbo_model_def_offers_only_its_schedule_and_enables_the_kv_cache(self):
        turbo = _resolved_model_def(TURBO_DEF_PATH)
        self.assertEqual([value for _, value in turbo["sample_solvers"]], [QWEN_TURBO_PROFILE])
        self.assertEqual(turbo["qwen21_acceleration_profiles"], {})
        kv_setting = next(item for item in turbo["custom_settings"] if item["id"] == "qwen21_kv_cache")
        self.assertEqual(kv_setting["default"], "Enabled")
        self.assertEqual(turbo["max_image_refs"], 10)
        base = _resolved_model_def(BASE_DEF_PATH)
        self.assertEqual([value for _, value in base["sample_solvers"]],
                         ["default", "viggle_v01", "viggle_v02", "viggle_v021"])
        self.assertEqual(next(item for item in base["custom_settings"]
                              if item["id"] == "qwen21_kv_cache")["default"], "Disabled")
        defaults = {}
        family_handler.update_default_settings(MODEL_TYPE, turbo, defaults)
        self.assertEqual((defaults["num_inference_steps"], defaults["guidance_scale"], defaults["sample_solver"]),
                         (8, 1.0, QWEN_TURBO_PROFILE))


class TurboDefinitionTests(unittest.TestCase):
    def test_official_shards_download_verified_under_unique_local_names(self):
        definition = _definition(TURBO_DEF_PATH)
        model = definition["model"]
        self.assertEqual(model["architecture"], MODEL_TYPE)
        self.assertTrue(model["qwen21_sharded_transformer"])
        shards = model["URLs"] + model["URLs2"]
        self.assertEqual(shards, ["qwen_image_21_7B_turbo_bf16_part1of2.safetensors",
                                  "qwen_image_21_7B_turbo_bf16_part2of2.safetensors"])
        for index, name in enumerate(shards, start=1):
            # mmgp expands "-of-" names itself; these must stay plain names.
            self.assertNotIn("-of-", name)
            source = model["download_sources"][name]
            self.assertEqual(
                source["url"],
                f"https://huggingface.co/Qwen/Qwen-Image-2.1-Turbo/resolve/{TURBO_REVISION}/transformer/"
                f"diffusion_pytorch_model-0000{index}-of-00002.safetensors")
            self.assertRegex(source["sha256"], r"^[0-9a-f]{64}$")
            self.assertGreater(source["size_bytes"], 0)
        self.assertEqual(definition["num_inference_steps"], 8)
        self.assertEqual(definition["guidance_scale"], 1.0)
        self.assertEqual(definition["sample_solver"], QWEN_TURBO_PROFILE)
        self.assertEqual(model["qwen21_min_render_pixels"], TURBO_PIXELS)
        self.assertEqual(model["qwen21_reference_max_side"], 768)

    def test_named_sources_pass_the_checkpoint_download_validation(self):
        from shared.checkpoint_downloads import _validate_source, find_named_checkpoint_source

        model = _definition(TURBO_DEF_PATH)["model"]
        for name in model["URLs"] + model["URLs2"]:
            source = find_named_checkpoint_source(model, name)
            self.assertIsNotNone(source)
            url, _, size, host = _validate_source(source)
            self.assertEqual(host, "huggingface.co")
            self.assertEqual(size, source["size_bytes"])


class RenderSizeTests(unittest.TestCase):
    def test_small_requests_render_near_the_turbo_canvas_with_the_same_shape(self):
        self.assertEqual(render_dimensions(480, 848, TURBO_PIXELS), (864, 1536))
        self.assertEqual(render_dimensions(848, 480, TURBO_PIXELS), (1536, 864))
        self.assertEqual(render_dimensions(720, 1280, TURBO_PIXELS), (864, 1536))
        self.assertEqual(render_dimensions(512, 512, TURBO_PIXELS), (1152, 1152))
        for width, height in ((480, 848), (640, 480), (1000, 333), (256, 1024)):
            with self.subTest(size=(width, height)):
                render_width, render_height = render_dimensions(width, height, TURBO_PIXELS)
                self.assertEqual((render_width % 32, render_height % 32), (0, 0))
                self.assertAlmostEqual(render_width * render_height / TURBO_PIXELS, 1, delta=0.08)
                self.assertAlmostEqual(math.log((render_width / render_height) / (width / height)), 0, delta=0.05)

    def test_large_requests_and_models_without_a_minimum_keep_the_historic_floor(self):
        self.assertEqual(render_dimensions(1536, 2752, TURBO_PIXELS), (1536, 2752))
        self.assertEqual(render_dimensions(1000, 1400, TURBO_PIXELS), (992, 1376))
        self.assertEqual(render_dimensions(480, 848, None), (480, 832))

    def test_output_downscale_keeps_channels_dtype_and_flat_colour(self):
        images = torch.full((2, 4, 1536, 864), 200, dtype=torch.uint8)
        images[:, 3] = 255
        resized = resize_uint8_images(images, 480, 848)
        self.assertEqual(tuple(resized.shape), (2, 4, 848, 480))
        self.assertEqual(resized.dtype, torch.uint8)
        self.assertTrue(torch.all(resized[:, :3] == 200))
        self.assertTrue(torch.all(resized[:, 3] == 255))
        self.assertIs(resize_uint8_images(resized, 480, 848), resized)

    def test_references_are_capped_at_768_on_their_long_side(self):
        portrait = downscale_reference(Image.new("RGBA", (1536, 2048)), 768)
        self.assertEqual(portrait.size, (576, 768))
        self.assertEqual(portrait.mode, "RGBA")
        landscape = downscale_reference(Image.new("RGB", (4000, 1000)), 768)
        self.assertEqual(landscape.size, (768, 192))
        small = Image.new("RGB", (640, 480))
        self.assertIs(downscale_reference(small, 768), small)
        self.assertIs(downscale_reference(portrait, None), portrait)


def _runtime_available():
    try:
        import models.qwen21.runtime  # noqa: F401
    except Exception:  # The engine needs the app's diffusers/transformers/mmgp.
        return False
    return True


@unittest.skipUnless(_runtime_available(), "Qwen Image 2.1 runtime dependencies are not installed")
class TurboRuntimeTests(unittest.TestCase):
    def runtime(self, model_def):
        from models.qwen21.runtime import model_factory

        instance = model_factory.__new__(model_factory)
        instance.model_def = model_def
        instance.device = torch.device("cpu")
        instance._abort = False
        instance.transformer = torch.nn.Linear(2, 2)
        instance.text_encoder = SimpleNamespace(
            visual=SimpleNamespace(blocks=[torch.nn.Linear(2, 2)]),
            language_model=SimpleNamespace(layers=[torch.nn.Linear(2, 2)]))
        instance.vae = SimpleNamespace(enable_tiling=Mock(), enable_slicing=Mock(), clear_cache=Mock(),
                                      decoder=torch.nn.Linear(2, 2))

        def produce(**kw):
            frame = torch.full((kw["num_images_per_prompt"], 4, kw["height"], kw["width"]), 90,
                               dtype=torch.uint8)
            return SimpleNamespace(images=frame)

        instance.pipeline = Mock(side_effect=produce)
        return instance

    def test_director_request_renders_large_with_the_checkpoint_recipe_then_downscales(self):
        instance = self.runtime(_resolved_model_def(TURBO_DEF_PATH))
        refs = [Image.new("RGB", (1536, 2048)), Image.new("RGB", (512, 512)), Image.new("RGBA", (3000, 1500))]
        output = instance.generate(input_prompt="Picture the scene", width=480, height=848, sampling_steps=40,
                                   guide_scale=4.0, sample_solver="default", video_prompt_type="KI",
                                   input_ref_images=refs)
        call = instance.pipeline.call_args.kwargs
        self.assertEqual((call["width"], call["height"]), (864, 1536))
        self.assertEqual(call["sample_solver"], QWEN_TURBO_PROFILE)
        self.assertEqual(call["num_inference_steps"], 8)
        self.assertEqual(call["true_cfg_scale"], 1.0)
        self.assertIsNone(call["negative_prompt"])
        self.assertTrue(call["use_kv_cache"])
        self.assertEqual(call["output_resolution"], 1024)
        self.assertEqual([image.size for image in call["image"]], [(576, 768), (512, 512), (768, 384)])
        self.assertEqual(tuple(output.shape), (4, 1, 848, 480))

    def test_explicit_cache_choice_and_base_model_keep_their_behaviour(self):
        instance = self.runtime(_resolved_model_def(TURBO_DEF_PATH))
        instance.generate(input_prompt="Cat", width=480, height=848,
                          custom_settings={"qwen21_kv_cache": "Disabled"})
        self.assertFalse(instance.pipeline.call_args.kwargs["use_kv_cache"])

        base = self.runtime(_resolved_model_def(BASE_DEF_PATH))
        reference = Image.new("RGB", (1536, 2048))
        output = base.generate(input_prompt="Cat", width=480, height=848, video_prompt_type="I",
                               input_ref_images=[reference])
        call = base.pipeline.call_args.kwargs
        self.assertEqual((call["width"], call["height"]), (480, 832))
        self.assertEqual(call["num_inference_steps"], 40)
        self.assertEqual(call["sample_solver"], "default")
        self.assertFalse(call["use_kv_cache"])
        self.assertIs(call["image"][0], reference)
        self.assertEqual(tuple(output.shape), (4, 1, 832, 480))

    def test_sharded_transformer_loads_every_shard_into_one_module(self):
        from models.qwen21 import runtime

        model_def = _resolved_model_def(TURBO_DEF_PATH)
        shards = ["/ckpts/part1.safetensors", "/ckpts/part2.safetensors"]
        loads = []
        with patch.object(runtime.offload, "load_model_data", side_effect=lambda module, files, **_:
                          loads.append(files)), \
                patch.object(runtime.Qwen3VLProcessor, "from_pretrained",
                             return_value=SimpleNamespace(tokenizer=None)), \
                patch.object(runtime, "MaestroQwenImage21Pipeline", return_value=Mock()), \
                patch.object(runtime.fl, "locate_file", side_effect=lambda name: f"/ckpts/{name}"):
            runtime.model_factory(shards, "/ckpts/encoder.safetensors", model_def=model_def)
        self.assertEqual(loads[0], shards)


if __name__ == "__main__":
    unittest.main()
