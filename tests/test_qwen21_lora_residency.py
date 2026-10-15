"""LoRAs must change a Qwen Image 2.1 transformer that is already resident in VRAM.

Remote workers run with ``MAESTRO_KEEP_MODELS_IN_VRAM=1``: the transformer is
left on the GPU between generations. MMGP only builds the ``<adapter>_GPU``
tensors its LoRA hooks read when it moves a model's blocks to the GPU, so the
adapters loaded for the next generation were silently ignored and a LoRA job
rendered pixel-identical to the same job without the LoRA.
"""

import ast
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

try:
    import torch
    from safetensors.torch import save_file
    from mmgp import offload
    from models.qwen21.transformer_qwenimage21 import QwenImage21Transformer2DModel
except Exception as exc:  # pragma: no cover - depends on the local environment
    _SKIP_REASON = f"torch/mmgp/diffusers runtime unavailable: {exc}"
else:
    _SKIP_REASON = None

from services.lora_residency import is_model_resident, sync_resident_loras

LORA_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                "img_mlp.gate_layer", "img_mlp.proj", "img_mlp.out")
RANK = 2
ALPHA = 4.0
STRENGTH = 0.8


def tiny_transformer():
    return QwenImage21Transformer2DModel(in_channels=4, out_channels=4, num_layers=2,
        attention_head_dim=16, num_attention_heads=2, context_in_dim=8,
        mlp_ratio=2, axes_dims_rope=(4, 6, 6)).eval()


def forward_inputs():
    generator = torch.Generator().manual_seed(3)
    return dict(
        hidden_states=torch.randn(1, 8, 4, generator=generator),
        encoder_hidden_states=torch.randn(1, 5, 8, generator=generator),
        encoder_hidden_states_mask=None,
        img_shapes=[[(1, 2, 2), (1, 2, 2)]],
        img_mask=torch.tensor([[False, True, False, False, False, True]]),
        timestep=torch.tensor([0.6]),
    )


def profile_like_mmgp(model):
    """Install MMGP's LoRA hooks the way ``offload.profile(loras=["transformer"])`` does."""
    manager = offload.offload.__new__(offload.offload)
    manager.models = {"transformer": model}
    manager.active_models_ids = []
    model._loras_model_data = {}
    model._loras_model_shortcuts = {}
    for name, module in model.named_modules():
        if name and hasattr(module, "weight"):
            module._mm_manager = manager
            module.forward = manager.hook_lora(module, model, "transformer", model._loras_model_data,
                                               model._loras_model_shortcuts, name)
    return manager


def synthetic_lora(model):
    """A LoRA in the trainer's ``diffusion_model.`` naming over every attention/MLP linear."""
    generator = torch.Generator().manual_seed(7)
    state = {}
    modules = dict(model.named_modules())
    for block in range(len(model.transformer_blocks)):
        for target in LORA_TARGETS:
            weight = modules[f"transformer_blocks.{block}.{target}"].weight
            prefix = f"diffusion_model.transformer_blocks.{block}.{target}"
            state[f"{prefix}.lora_A.weight"] = torch.randn(RANK, weight.shape[1], generator=generator) * 0.2
            state[f"{prefix}.lora_B.weight"] = torch.randn(weight.shape[0], RANK, generator=generator) * 0.2
            state[f"{prefix}.alpha"] = torch.tensor(ALPHA)
    return state


def load_like_wgp(model, path):
    # Same call and options as wgp.generate_video. Qwen Image 2.1 has no
    # ``preprocess_loras``, so get_loras_preprocessor() yields None.
    assert getattr(model, "preprocess_loras", None) is None
    offload.load_loras_into_model(
        model, [path], [STRENGTH], activate_all_loras=True, preprocess_sd=None,
        pinnedLora=False, maxReservedLoras=-1, split_linear_modules_map=model.split_linear_modules_map,
    )
    assert model._loras_errors == [], model._loras_errors
    # MMGP memory-maps adapter tensors; keep them valid after the file is removed.
    for module_loras in model._loras_model_data.values():
        for adapter in module_loras.values():
            for index, tensor in enumerate(adapter):
                if torch.is_tensor(tensor):
                    adapter[index] = tensor.clone()


def cpu_cuda(self, *args, **kwargs):
    return self


@unittest.skipIf(_SKIP_REASON is not None, _SKIP_REASON or "")
class ResidentTransformerLoraTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        self.model = tiny_transformer()
        self.reference = tiny_transformer()
        self.reference.load_state_dict(self.model.state_dict())
        self.manager = profile_like_mmgp(self.model)
        self.lora = synthetic_lora(self.model)
        with torch.inference_mode():
            self.baseline = self.model(**forward_inputs()).sample.clone()
        directory = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, directory, True)
        self.path = os.path.join(directory, "qwen21_synthetic.safetensors")
        save_file({k: v.contiguous() for k, v in self.lora.items()}, self.path)

    def run_model(self):
        with torch.inference_mode():
            return self.model(**forward_inputs()).sample

    def merged_reference_output(self):
        """The base transformer with every LoRA delta merged into its weights."""
        modules = dict(self.reference.named_modules())
        scale = STRENGTH * ALPHA / RANK
        with torch.no_grad():
            for block in range(len(self.reference.transformer_blocks)):
                for target in LORA_TARGETS:
                    prefix = f"diffusion_model.transformer_blocks.{block}.{target}"
                    delta = self.lora[f"{prefix}.lora_B.weight"] @ self.lora[f"{prefix}.lora_A.weight"]
                    modules[f"transformer_blocks.{block}.{target}"].weight.add_(delta, alpha=scale)
        with torch.inference_mode():
            return self.reference(**forward_inputs()).sample

    def test_every_lora_target_module_exists_in_the_transformer(self):
        modules = dict(self.model.named_modules())
        prefixes = {key.rsplit(".lora_", 1)[0].removeprefix("diffusion_model.")
                    for key in self.lora if ".lora_" in key}
        self.assertTrue(prefixes)
        self.assertEqual({p for p in prefixes if p not in modules}, set())

    def test_lora_loaded_into_a_resident_transformer_is_ignored_without_sync(self):
        # The model is already on the GPU from the previous generation.
        self.manager.active_models_ids = ["transformer"]
        load_like_wgp(self.model, self.path)
        torch.testing.assert_close(self.run_model(), self.baseline, rtol=0, atol=0)

    def test_sync_applies_the_lora_to_a_resident_transformer(self):
        self.manager.active_models_ids = ["transformer"]
        load_like_wgp(self.model, self.path)
        with mock.patch.object(torch.Tensor, "cuda", cpu_cuda):
            synced = sync_resident_loras(self.manager, self.model)
        self.assertEqual(synced, len(LORA_TARGETS) * len(self.model.transformer_blocks))
        output = self.run_model()
        self.assertGreater((output - self.baseline).abs().mean().item(), 1e-4)
        torch.testing.assert_close(output, self.merged_reference_output(), atol=1e-4, rtol=1e-4)

    def test_unloading_restores_the_base_output(self):
        self.manager.active_models_ids = ["transformer"]
        load_like_wgp(self.model, self.path)
        with mock.patch.object(torch.Tensor, "cuda", cpu_cuda):
            sync_resident_loras(self.manager, self.model)
        offload.unload_loras_from_model(self.model)
        torch.testing.assert_close(self.run_model(), self.baseline, rtol=0, atol=0)

    def test_sync_is_a_no_op_for_a_model_mmgp_will_load(self):
        self.assertFalse(is_model_resident(self.manager, self.model))
        load_like_wgp(self.model, self.path)
        move = mock.Mock()
        self.manager._move_loras = move
        self.assertEqual(sync_resident_loras(self.manager, self.model), 0)
        move.assert_not_called()


class SyncHelperContractTests(unittest.TestCase):
    def test_missing_manager_or_model_is_ignored(self):
        self.assertEqual(sync_resident_loras(None, object()), 0)
        self.assertEqual(sync_resident_loras(mock.Mock(models={}, active_models_ids=[]), None), 0)

    def test_only_modules_holding_an_active_adapter_are_moved(self):
        model = mock.Mock()
        with_adapter, without = object(), object()
        model._loras_active_adapters = ["0"]
        model._loras_model_data = {with_adapter: {"0": [None] * 6}, without: {}}
        manager = mock.Mock(models={"transformer": model}, active_models_ids=["transformer"])
        self.assertEqual(sync_resident_loras(manager, model), 1)
        manager._move_loras.assert_called_once_with(["0"], {with_adapter: {"0": [None] * 6}}, True, model)

    def test_wgp_syncs_after_loading_loras(self):
        tree = ast.parse((ROOT / "app/wgp.py").read_text(encoding="utf-8"))
        generate = next(node for node in tree.body
                        if isinstance(node, ast.FunctionDef) and node.name == "generate_video")
        source = ast.get_source_segment((ROOT / "app/wgp.py").read_text(encoding="utf-8"), generate)
        load = source.index("offload.load_loras_into_model(\n                trans_lora")
        finalize = source.index("wan_model.finalize_loras()")
        sync = source.index("sync_resident_loras(offloadobj, lora_target)")
        self.assertLess(load, finalize)
        self.assertLess(finalize, sync)
        self.assertIn("for lora_target in (trans_lora, trans2_lora):", source)


if __name__ == "__main__":
    unittest.main()
