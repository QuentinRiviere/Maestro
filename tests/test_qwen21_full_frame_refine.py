"""A full-frame (all-white) mask turns Qwen Image 2.1's masked edit into img2img.

Director's texture refine relies on this: denoising starts from the source
image's latents at ``denoising_strength`` instead of from pure noise, and the
mask blend never pins any region back to the source.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "tests"))

try:
    import torch
    from PIL import Image
    from test_qwen21_upstream import rgba_source, tiny_pipeline
    from models.qwen21.runtime import model_factory
except Exception as exc:  # pragma: no cover - depends on the local diffusers version
    _SKIP_REASON = f"Qwen Image 2.1 runtime unavailable: {exc}"
else:
    _SKIP_REASON = None

SEED = 5
STEPS = 20
STRENGTH = 0.35


def white_mask(size=(64, 64)):
    return Image.new("L", size, 255)


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class FullFrameRefinePipelineTests(unittest.TestCase):
    def run_refine(self, strength=STRENGTH):
        pipe = tiny_pipeline()
        calls = []

        def record(*, hidden_states, timestep, **kwargs):
            calls.append((float(timestep[0]) * 1000, hidden_states.clone()))
            return (torch.zeros_like(hidden_states),)

        pipe.transformer.forward = record
        image = rgba_source()
        result = pipe(
            prompt="Refine the texture.", image=[image], mask_image=white_mask(), inpaint_image=image,
            height=64, width=64, output_resolution=64, num_inference_steps=STEPS,
            denoising_strength=strength, generator=torch.Generator("cpu").manual_seed(SEED),
            output_type="latent", use_kv_cache=False,
        ).images
        return pipe, image, calls, result

    @torch.inference_mode()
    def test_denoising_starts_from_the_source_latents_at_the_strength_step(self):
        pipe, image, calls, result = self.run_refine()
        first_step = int(STEPS * (1 - STRENGTH))  # 13
        self.assertEqual(len(calls), STEPS - first_step)  # 7 effective steps

        from diffusers import FlowMatchEulerDiscreteScheduler
        from models.qwen21.pipeline_qwenimage21 import calculate_shift, retrieve_timesteps
        import numpy as np

        scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        mu = calculate_shift(16, scheduler.config.get("base_image_seq_len", 256),
                             scheduler.config.get("max_image_seq_len", 4096),
                             scheduler.config.get("base_shift", 0.5), scheduler.config.get("max_shift", 1.15))
        timesteps, _ = retrieve_timesteps(
            scheduler, STEPS, torch.device("cpu"), sigmas=np.linspace(1.0, 1 / STEPS, STEPS),
            **({} if not scheduler.config.use_dynamic_shifting else {"mu": mu}))
        self.assertAlmostEqual(calls[0][0], float(timesteps[first_step]), places=3)
        sigma = scheduler.sigmas[first_step]

        noise, _ = pipe.prepare_latents(None, 1, 4, 64, 64, torch.float32, torch.device("cpu"),
                                        torch.Generator("cpu").manual_seed(SEED))
        source_pixels = pipe.image_processor.preprocess(image.convert("RGBA"), width=64, height=64).unsqueeze(2)
        source = pipe._encode_vae_image(source_pixels, None)[:, :, 0].flatten(2).transpose(1, 2)
        expected = source * (1 - sigma) + noise * sigma
        target = calls[0][1][:, -expected.shape[1]:]
        torch.testing.assert_close(target, expected, atol=1e-5, rtol=1e-5)
        # The source image is also the edit's reference 1 (the prefix tokens).
        self.assertGreater(calls[0][1].shape[1], expected.shape[1])
        # A zero velocity keeps the latents: the white mask never re-pins them.
        torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    @torch.inference_mode()
    def test_full_strength_still_starts_from_noise(self):
        _, _, calls, _ = self.run_refine(strength=1.0)
        self.assertEqual(len(calls), STEPS)


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class FullFrameRefineRuntimeTests(unittest.TestCase):
    def runtime(self):
        instance = model_factory.__new__(model_factory)
        instance.device = torch.device("cpu")
        instance._abort = False
        instance.model_def = {}
        instance.transformer = torch.nn.Linear(2, 2)
        instance.text_encoder = SimpleNamespace(
            visual=SimpleNamespace(blocks=[torch.nn.Linear(2, 2)]),
            language_model=SimpleNamespace(layers=[torch.nn.Linear(2, 2)]))
        instance.vae = SimpleNamespace(enable_tiling=Mock(), enable_slicing=Mock(), clear_cache=Mock(),
                                      decoder=torch.nn.Linear(2, 2))
        instance.pipeline = Mock(return_value=SimpleNamespace(images=torch.zeros((1, 3, 64, 64))))
        return instance

    def test_refine_job_reaches_the_pipeline_as_a_masked_edit_of_the_source(self):
        instance = self.runtime()
        source, sheet = Image.new("RGB", (64, 64), (10, 20, 30)), Image.new("RGB", (32, 32))
        mask = white_mask()
        instance.generate(input_prompt="Refine", seed=3, video_prompt_type="VAGI", image_mode=2,
                          input_frames=source, input_masks=mask, original_input_ref_images=[sheet],
                          denoising_strength=0.35, masking_strength=1.0, model_mode=0,
                          sampling_steps=30, guide_scale=4.0, width=64, height=64)
        call = instance.pipeline.call_args.kwargs
        self.assertIs(call["image"][0], source)
        self.assertIs(call["image"][1], sheet)
        self.assertIs(call["inpaint_image"], source)
        self.assertIs(call["mask_image"], mask)
        self.assertEqual(call["denoising_strength"], 0.35)
        self.assertEqual(call["num_inference_steps"], 30)
        self.assertEqual(call["true_cfg_scale"], 4.0)


if __name__ == "__main__":
    unittest.main()
