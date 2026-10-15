"""Generation provenance: building, merging and storing the per-job record."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
for _path in (_APP_DIR, _HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from services import generation_provenance as gp  # noqa: E402
from services.remote_worker.job_forwarder import RemoteJobForwarder  # noqa: E402
from services.remote_worker.manager import offer_details  # noqa: E402

from test_remote_job_forwarder import FakeManager, FakeWorker  # noqa: E402


H3_RUNTIME = {
    "model_type": "minimax_h3_fl2va_full",
    "model_name": "MiniMax H3 First / Last - Full",
    "architecture": "minimax_h3_fl2va",
    "files": {
        "transformer": "/ckpts/minimax_h3_full_int8_convrot.safetensors",
        "video_vae": "h3_video_vae_fp16.safetensors",
        "text_encoder": "h3_te_nvfp4_awq.safetensors",
    },
    "precision_overrides": {"transformer_quantization": "int8"},
    "configured_quantization": "int8",
    "transformer_dtype": None,
    "text_encoder_variant": "nvfp4_awq",
    "vae_precision": None,
    "attention": "sage2",
    "compile": False,
    "profile": 3,
    "num_inference_steps": 8,
    "model_reloaded": True,
    "model_load_seconds": 41.26,
}


class PhaseClockTests(unittest.TestCase):
    def test_wall_time_is_split_by_phase(self):
        clock = gp.PhaseClock()
        clock.observe("Downloading model H3 (first use)", now=0)
        clock.observe("Loading model H3...", now=30)
        clock.observe("Model loaded", now=50)
        clock.observe("Something unrelated", now=51)
        clock.observe("Encoding Prompt", now=52)
        clock.observe("Denoising First Pass | 3s", now=54)
        clock.observe("Denoising Second Pass", now=60)  # same phase, keeps running
        clock.observe("VAE Decoding", now=94)
        clock.enter("postprocess", now=100)
        clock.stop(now=112)
        self.assertEqual(clock.seconds(), {
            "download": 30.0, "model_load": 20.0, "text_encode": 2.0,
            "denoise": 40.0, "vae_decode": 6.0, "postprocess": 12.0,
        })

    def test_upsampling_is_postprocess_not_denoise(self):
        self.assertEqual(gp.classify_phase("Applying upsampling"), "postprocess")
        self.assertEqual(gp.classify_phase("Upscaling clip.mp4 (FlashVSR)..."), "postprocess")
        self.assertIsNone(gp.classify_phase("Preparing..."))

    def test_running_phase_counts_until_now(self):
        ticks = iter([10.0, 25.0])
        clock = gp.PhaseClock(clock=lambda: next(ticks))
        clock.observe("Denoising")
        self.assertEqual(clock.seconds(), {"denoise": 15.0})


class QuantizationTests(unittest.TestCase):
    def test_reads_the_loaded_file_name(self):
        self.assertEqual(gp.quantization_from_filename("/x/h3_full_int8_convrot.safetensors"), "INT8")
        self.assertEqual(gp.quantization_from_filename("wan_fp8_e4m3fn.safetensors"), "FP8")
        self.assertEqual(gp.quantization_from_filename("te_nvfp4_awq.safetensors"), "NVFP4")
        self.assertEqual(gp.quantization_from_filename("model-Q4_K_M.gguf"), "GGUF Q4_K_M")
        self.assertEqual(gp.quantization_from_filename("ltx_bf16.safetensors"), "BF16")

    def test_unknown_when_the_name_says_nothing(self):
        self.assertEqual(gp.quantization_from_filename("model.safetensors"), "")
        self.assertEqual(gp.quantization_from_filename(""), "")


class BuildTests(unittest.TestCase):
    def test_local_job_captures_model_runtime_loras_turbo_and_upscale(self):
        params = {
            "model_type": "minimax_h3_fl2va_full",
            "num_inference_steps": 40,  # requested; the engine ran 8
            "activated_loras": ["/loras/h3/pdd_8step.safetensors", "style.safetensors"],
            "loras_multipliers": "1.0 0.8",
            "minimax_h3_turbo_mode": True,
            "minimax_h3_turbo_preset": "alibaba-pai-fl2va-pdd-8step",
            "skip_steps_cache_type": "first_block",
            "skip_steps_multiplier": 0.15,
            "spatial_upsampling": "flashvsr1.5",
            "film_grain_intensity": 0.1,
            "film_grain_saturation": 0.5,
        }
        record = gp.build_provenance(
            params,
            runtime_metadata=H3_RUNTIME,
            hardware=gp.hardware_from_probe({"gpu_name": "NVIDIA A100 80GB", "gpu_vram_gb": 79.15}),
            phases={"text_encode": 2.0, "denoise": 40.0},
            total_seconds=86.04,
            generation_seconds=60,
            turbo_label=lambda preset: "PDD 8-step",
            captured_at=1.0,
        )
        self.assertEqual(record["model"], {
            "type": "minimax_h3_fl2va_full", "name": "MiniMax H3 First / Last - Full",
            "architecture": "minimax_h3_fl2va",
            "files": {"transformer": "minimax_h3_full_int8_convrot.safetensors",
                      "video_vae": "h3_video_vae_fp16.safetensors",
                      "text_encoder": "h3_te_nvfp4_awq.safetensors"},
        })
        runtime = record["runtime"]
        self.assertEqual(runtime["transformer_quantization"], "INT8")
        self.assertEqual(runtime["text_encoder"], "nvfp4_awq")
        self.assertEqual(runtime["text_encoder_quantization"], "NVFP4")
        self.assertEqual(runtime["attention"], "sage2")
        self.assertIs(runtime["compile"], False)
        self.assertEqual(runtime["precision_overrides"], {"transformer_quantization": "int8"})
        self.assertNotIn("transformer_dtype", runtime)
        self.assertNotIn("vae_precision", runtime)
        self.assertEqual(record["loras"], [
            {"name": "pdd_8step.safetensors", "multiplier": 1.0},
            {"name": "style.safetensors", "multiplier": 0.8},
        ])
        self.assertEqual(record["acceleration"], {
            "steps": 8, "turbo": True, "turbo_preset": "alibaba-pai-fl2va-pdd-8step",
            "turbo_preset_label": "PDD 8-step", "step_cache": "first_block",
            "step_cache_multiplier": 0.15,
        })
        self.assertEqual(record["postprocessing"], {
            "spatial_upsampling": "flashvsr1.5", "film_grain_intensity": 0.1, "film_grain_saturation": 0.5,
        })
        self.assertEqual(record["hardware"], {"location": "local", "gpu": "NVIDIA A100 80GB", "vram_gb": 79.2})
        self.assertEqual(record["timing"], {
            "total_seconds": 86.0, "generation_seconds": 60.0,
            "phases": {"text_encode": 2.0, "denoise": 40.0, "model_load": 41.3},
        })
        self.assertNotIn("cost", record)

    def test_missing_facts_are_omitted_not_guessed(self):
        record = gp.build_provenance(
            {"model_type": "flux", "minimax_h3_turbo_mode": False,
             "minimax_h3_turbo_preset": "default-preset", "loras_multipliers": ""},
            hardware=gp.hardware_from_probe({"gpu_name": "", "gpu_vram_gb": 0.0}),
            captured_at=1.0,
        )
        # No engine report, no GPU, Turbo off (its stored preset is not
        # reported), no LoRAs, no timing: only what is known remains.
        self.assertEqual(record, {
            "version": gp.PROVENANCE_VERSION, "captured_at": 1.0,
            "model": {"type": "flux"}, "hardware": {"location": "local"},
        })

    def test_phased_lora_multipliers_stay_verbatim(self):
        loras = gp.parse_loras({"activated_loras": ["a.safetensors", "b.safetensors"],
                                "loras_multipliers": "1;0.5 0.8"})
        self.assertEqual(loras, [{"name": "a.safetensors", "multiplier": "1;0.5"},
                                 {"name": "b.safetensors", "multiplier": 0.8}])

    def test_request_attention_is_a_fallback_only(self):
        record = gp.build_provenance({"model_type": "x", "override_attention": "sol"})
        self.assertEqual(record["runtime"]["attention"], "sol")
        self.assertEqual(record["acceleration"], {"sparse_attention": "sol"})


class RemoteMergeTests(unittest.TestCase):
    WORKER = {
        "version": 1,
        "model": {"type": "minimax_h3_fl2va_full"},
        "hardware": {"location": "remote", "gpu": "NVIDIA A100-SXM4-80GB", "vram_gb": 79.2},
        "timing": {"total_seconds": 80.0, "phases": {"denoise": 40.0}},
    }
    BILLING = {
        "pod": "maestro-worker", "price_per_hour": 1.8, "wait_seconds": 120.4, "work_seconds": 86.0,
        "gpu_name": "A100", "gpu_count": 1, "gpu_memory_gb": 80.0, "gpu_backend": "runpod",
        "gpu_region": "EU-RO-1", "instance_type": "NVIDIA A100 80GB PCIe", "spot": False,
    }

    def test_worker_gpu_wins_and_pod_offer_and_cost_are_added(self):
        record = gp.merge_remote(self.WORKER, self.BILLING, total_seconds=210.0)
        self.assertEqual(record["hardware"], {
            "location": "remote", "gpu": "NVIDIA A100-SXM4-80GB", "vram_gb": 79.2,
            "gpu_count": 1, "backend": "runpod", "region": "EU-RO-1",
            "instance_type": "NVIDIA A100 80GB PCIe", "spot": False, "price_per_hour": 1.8,
            "pod": "maestro-worker",
        })
        self.assertEqual(record["timing"], {
            "worker_seconds": 80.0, "phases": {"denoise": 40.0}, "total_seconds": 210.0,
            "remote_wait_seconds": 120.4, "remote_work_seconds": 86.0,
        })
        self.assertEqual(record["cost"]["estimated_usd"], 0.043)
        self.assertEqual(record["cost"]["price_per_hour"], 1.8)

    def test_older_worker_without_provenance_still_gets_the_pod(self):
        record = gp.merge_remote({}, self.BILLING, total_seconds=None)
        self.assertEqual(record["hardware"]["gpu"], "A100")
        self.assertEqual(record["hardware"]["vram_gb"], 80.0)

    def test_unknown_price_means_no_cost(self):
        record = gp.merge_remote(self.WORKER, {"work_seconds": 86.0, "price_per_hour": None})
        self.assertNotIn("cost", record)
        self.assertNotIn("price_per_hour", record["hardware"])

    def test_dstack_offer_details(self):
        provisioning = {
            "backend": "runpod", "region": "EU-RO-1", "price": 1.8,
            "instance_type": {"name": "NVIDIA A100 80GB PCIe", "resources": {
                "spot": False, "gpus": [{"name": "A100", "memory_mib": 81920, "vendor": "nvidia"}]}},
        }
        self.assertEqual(offer_details(provisioning), {
            "gpu_backend": "runpod", "instance_type": "NVIDIA A100 80GB PCIe", "spot": False,
            "gpu_name": "A100", "gpu_count": 1, "gpu_memory_gb": 80.0,
        })
        self.assertEqual(offer_details(None), {})
        self.assertEqual(offer_details({"instance_type": "weird"}), {})


class SidecarTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_stamp_keeps_the_sidecar_and_narrows_timing_to_the_output(self):
        meta = os.path.join(self.tmp, "clip.meta.json")
        with open(meta, "w") as handle:
            json.dump({"params": {"seed": 1}, "job_id": "worker-1", "generation_time": 55,
                       "generation_time_basis": "active"}, handle)
        record = {"model": {"type": "m"}, "timing": {"generation_seconds": 120.0}}
        self.assertTrue(gp.stamp_sidecar(meta, record, job_id="local-9", output_filename="clip.mp4"))
        with open(meta) as handle:
            stamped = json.load(handle)
        self.assertEqual(stamped["params"], {"seed": 1})
        self.assertEqual(stamped["job_id"], "local-9")
        self.assertEqual(stamped["provenance"]["timing"]["generation_seconds"], 55.0)
        self.assertEqual(gp.read_output_provenance(self.tmp, "clip.mp4"), stamped["provenance"])
        self.assertIsNone(gp.read_output_provenance(self.tmp, "missing.mp4"))
        self.assertIsNone(gp.read_output_provenance(self.tmp, None))

    def test_stamp_creates_a_minimal_sidecar_and_skips_empty_records(self):
        meta = os.path.join(self.tmp, "image.meta.json")
        self.assertFalse(gp.stamp_sidecar(meta, {}))
        self.assertFalse(os.path.exists(meta))
        self.assertTrue(gp.stamp_sidecar(meta, {"model": {"type": "flux"}}))
        with open(meta) as handle:
            stamped = json.load(handle)
        self.assertNotIn("params", stamped)
        self.assertEqual(stamped["provenance"], {"model": {"type": "flux"}})


class ForwarderProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.out_dir = os.path.join(self.tmp, "outputs")
        self.manager = FakeManager()
        self.manager.status = lambda: {"price_per_hour": 1.8, "gpu_name": "A100", "gpu_backend": "runpod"}
        self.worker = FakeWorker()
        self.worker.final = {**self.worker.final, "sidecars": ["clip.json", "clip.meta.json"],
                             "provenance": {"model": {"type": "flux"}, "hardware": {"gpu": "NVIDIA A100"}}}
        self.worker.contents["clip.meta.json"] = b'{"job_id": "r1"}'
        self.forwarder = RemoteJobForwarder(
            self.manager, client_factory=lambda url, token: self.worker,
            store_factory=lambda settings: None, poll_interval=0, sleep=lambda s: None,
        )

    def _run(self, params):
        return self.forwarder.run(params, out_dir=self.out_dir, resolve_lora=lambda m, n: None,
                                  update=lambda **u: None, is_cancelled=lambda: False)

    def test_worker_record_gallery_sidecar_and_pod_offer_come_back(self):
        result = self._run({"model_type": "flux", "prompt": "x"})
        self.assertEqual(result.provenance["hardware"], {"gpu": "NVIDIA A100"})
        self.assertEqual(result.billing["gpu_name"], "A100")
        self.assertEqual(result.billing["gpu_backend"], "runpod")
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, "clip.meta.json")))
        merged = gp.merge_remote(result.provenance, result.billing)
        self.assertEqual(merged["hardware"]["gpu"], "NVIDIA A100")
        self.assertEqual(merged["hardware"]["backend"], "runpod")

    def test_internal_outputs_get_no_gallery_sidecar(self):
        self._run({"model_type": "flux", "prompt": "x", "_defer_output_publication": True})
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, "clip.json")))
        self.assertFalse(os.path.isfile(os.path.join(self.out_dir, "clip.meta.json")))


if __name__ == "__main__":
    unittest.main()
