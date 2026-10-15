"""Per-job precision overrides (transformer quantization, H3 video VAE, H3 text encoder).

CPU-only: the engine functions are executed from ``wgp.py`` / ``launch.py``
source with fakes, as in the other engine regression tests.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

from shared import precision_overrides as po  # noqa: E402
from models.minimax_h3.minimax_h3_handler import family_handler  # noqa: E402
from services.remote_worker.job_forwarder import RemoteJobForwarder  # noqa: E402

WGP = APP / "wgp.py"
LAUNCH = APP / "launch.py"
WGP_TREE = ast.parse(WGP.read_text(encoding="utf-8"))
LAUNCH_TREE = ast.parse(LAUNCH.read_text(encoding="utf-8"))

INT8_VAE = "minimax_h3_video_vae_int8_convrot.safetensors"
FP16_VAE = "minimax_h3_video_vae_fp16.safetensors"


def _functions(tree, *names):
    return [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]


def _exec(nodes, namespace, filename):
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), filename, "exec"), namespace)
    return namespace


def _h3_def(model_type="minimax_h3"):
    definition = json.loads((APP / "defaults" / f"{model_type}.json").read_text(encoding="utf-8"))["model"]
    base = "minimax_h3_full" if model_type.endswith("_full") else "minimax_h3"
    return family_handler.query_model_def(base, definition) | {
        "architecture": definition["architecture"],
        "URLs": definition["URLs"],
    }


def _model_filename_selector(models_def):
    class QuantRouter:
        @staticmethod
        def get_quantization_tokens(quantization):
            return [str(quantization).lower()]

    namespace = {
        "os": os,
        "models_def": models_def,
        "quant_router": QuantRouter(),
        "torch": SimpleNamespace(float16=object(), bfloat16=object()),
        "get_transformer_dtype": lambda *_args, **_kwargs: object(),
        "get_base_model_type": lambda model_type: model_type,
    }
    return _exec(_functions(WGP_TREE, "get_model_filename"), namespace, str(WGP))["get_model_filename"]


class NormalizeTests(unittest.TestCase):
    def test_unset_is_empty(self):
        self.assertEqual(po.normalize(_h3_def()), {})
        self.assertEqual(po.from_job_params({"override_text_encoder": "  ", "override_video_vae_precision": None}), {})

    def test_canonical_values(self):
        overrides = po.normalize(
            _h3_def(),
            transformer_quantization=" BF16 ",
            video_vae_precision="int8_convrot",
            text_encoder="bf16",
        )
        self.assertEqual(overrides, {
            "transformer_quantization": "bf16",
            "video_vae_precision": "int8",
            "text_encoder": "bf16",
        })

    def test_invalid_values_are_rejected(self):
        h3 = _h3_def()
        for kwargs in (
            {"transformer_quantization": "fp4"},
            {"video_vae_precision": "bf16"},
            {"text_encoder": "gguf_q8"},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                po.normalize(h3, **kwargs)

    def test_h3_only_overrides_are_rejected_for_other_models(self):
        other = {"architecture": "wan", "URLs": ["a_bf16.safetensors", "a_int8.safetensors"]}
        self.assertEqual(po.normalize(other, transformer_quantization="int8"), {"transformer_quantization": "int8"})
        for kwargs in ({"video_vae_precision": "fp16"}, {"text_encoder": "bf16"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                po.normalize(other, **kwargs)

    def test_cache_key_is_stable_and_default_free(self):
        self.assertEqual(po.cache_key("minimax_h3", {}), "minimax_h3")
        self.assertEqual(
            po.cache_key("minimax_h3", {"text_encoder": "bf16", "transformer_quantization": "bf16", "video_vae_precision": "fp16"}),
            "minimax_h3@te-bf16+tq-bf16+vae-fp16",
        )

    def test_scope_is_restored(self):
        with po.scope({"transformer_quantization": "bf16"}):
            self.assertEqual(po.effective_transformer_quantization("int8"), "bf16")
        self.assertEqual(po.effective_transformer_quantization("int8"), "int8")
        self.assertEqual(po.current(), {})


class TransformerSelectionTests(unittest.TestCase):
    def test_pruned_h3_variants_select_distinct_files(self):
        definition = _h3_def()
        select = _model_filename_selector({"minimax_h3": definition})
        expected = {
            "int8": "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors",
            "bf16": "MiniMax-H3-FL2VA-pruned_rank8_bf16.safetensors",
            "fp8": "minimax_h3_fl2va_pruned_fp8_scaled.safetensors",
        }
        for quantization, filename in expected.items():
            with self.subTest(quantization=quantization):
                selected = select("minimax_h3", quantization)
                self.assertEqual(os.path.basename(selected), filename)
                self.assertTrue(po.transformer_file_matches(selected, quantization))

    def test_full_h3_has_no_fp8_file(self):
        select = _model_filename_selector({"minimax_h3_full": _h3_def("minimax_h3_full")})
        fallback = select("minimax_h3_full", "fp8")
        self.assertIn("int8", fallback)
        self.assertFalse(po.transformer_file_matches(fallback, "fp8"))
        self.assertTrue(po.transformer_file_matches(select("minimax_h3_full", "bf16"), "bf16"))


class VideoVaeSelectionTests(unittest.TestCase):
    @staticmethod
    def _downloads(definition):
        return [name for item in family_handler.query_model_files([], "minimax_h3", definition)
                for group in item["fileList"] for name in group]

    def test_override_beats_transformer_derived_choice(self):
        definition = _h3_def()
        cases = (
            ({"transformer_quantization": "int8"}, INT8_VAE),
            ({"transformer_quantization": "int8", "video_vae_precision": "fp16"}, FP16_VAE),
            ({"transformer_quantization": "bf16"}, FP16_VAE),
            ({"transformer_quantization": "bf16", "video_vae_precision": "int8"}, INT8_VAE),
        )
        for context, filename in cases:
            with self.subTest(context=context):
                resolved = family_handler.resolve_runtime_model_def(dict(definition), context)
                self.assertEqual(resolved["minimax_h3_video_vae_filename"], filename)
                self.assertIn(filename, self._downloads(resolved))

    def test_override_applies_to_definitions_with_an_explicit_decoder(self):
        fused = family_handler.query_model_def("minimax_h3", {"minimax_h3_fused_turbo": True})
        self.assertEqual(
            family_handler.resolve_runtime_model_def(dict(fused), {"transformer_quantization": "bf16"})["minimax_h3_video_vae_filename"],
            INT8_VAE,
        )
        self.assertEqual(
            family_handler.resolve_runtime_model_def(dict(fused), {"video_vae_precision": "fp16"})["minimax_h3_video_vae_filename"],
            FP16_VAE,
        )


class CompatibleAliasTests(unittest.TestCase):
    def _resolver(self, present):
        locator = SimpleNamespace(locate_file=lambda name, error_if_none=False: name if os.path.basename(name) in present else None)
        definition = {"compatible_model_paths": {"fp8.safetensors": ["int8.safetensors"]}}
        namespace = {"os": os, "fl": locator, "get_model_def": lambda _m: definition, "precision_overrides_lib": po}
        names = ("get_local_model_filename", "get_compatible_local_model_filename")
        return _exec(_functions(WGP_TREE, *names), namespace, str(WGP))["get_compatible_local_model_filename"]

    def test_alias_is_used_only_without_a_transformer_override(self):
        resolve = self._resolver({"int8.safetensors"})
        self.assertEqual(resolve("fp8.safetensors", "m", file_type=0), "int8.safetensors")
        with po.scope({"transformer_quantization": "fp8"}):
            self.assertIsNone(resolve("fp8.safetensors", "m", file_type=0))
        with po.scope({"video_vae_precision": "fp16"}):
            self.assertEqual(resolve("fp8.safetensors", "m", file_type=0), "int8.safetensors")


class EngineWiringTests(unittest.TestCase):
    def _generate_video(self):
        return _functions(WGP_TREE, "generate_video")[0]

    def test_generate_video_accepts_the_job_params(self):
        params = [arg.arg for arg in self._generate_video().args.args]
        for name in po.JOB_PARAMS:
            self.assertIn(name, params)
        defaults = dict(zip(params[-len(self._generate_video().args.defaults):], self._generate_video().args.defaults))
        for name in po.JOB_PARAMS:
            self.assertEqual(ast.literal_eval(defaults[name]), "")

    def _reload_namespace(self, job_files, loaded_files, job_overrides, loaded_overrides, model_type="m", transformer_type="m"):
        block = next(
            node for node in self._generate_video().body
            if isinstance(node, ast.If) and "job_model_files != loaded_model_files" in ast.unparse(node.test)
        )
        namespace = {
            "wan_model": object(), "model_type": model_type, "transformer_type": transformer_type,
            "job_precision_overrides": job_overrides, "loaded_precision_overrides": loaded_overrides,
            "job_model_files": job_files, "loaded_model_files": loaded_files, "reload_needed": False,
            "print": lambda *a, **k: None,
        }
        return _exec([block], namespace, str(WGP))

    def test_reload_only_when_overridden_files_differ(self):
        default = {"transformer": "int8.safetensors", "video_vae": INT8_VAE, "text_encoder": "q2.gguf"}
        bf16 = dict(default, transformer="bf16.safetensors")
        cases = (
            (default, default, {}, {}, False),           # unset -> unset: today's behavior
            (bf16, default, {"transformer_quantization": "bf16"}, {}, True),
            (bf16, bf16, {"transformer_quantization": "bf16"}, {"transformer_quantization": "bf16"}, False),
            (default, bf16, {}, {"transformer_quantization": "bf16"}, True),  # back to the default
            (default, default, {"transformer_quantization": "int8"}, {}, False),  # same files: no reload
        )
        for job_files, loaded_files, job_overrides, loaded_overrides, expected in cases:
            with self.subTest(job=job_overrides, loaded=loaded_overrides):
                namespace = self._reload_namespace(job_files, loaded_files, job_overrides, loaded_overrides)
                self.assertIs(namespace["reload_needed"], expected)

    def test_resolve_job_model_files_follows_the_scope(self):
        definition = _h3_def()
        select = _model_filename_selector({"minimax_h3": definition})
        namespace = {
            "os": os, "args": SimpleNamespace(save_quantized=False), "precision_overrides_lib": po,
            "transformer_quantization": "int8", "transformer_dtype_policy": "", "text_encoder_quantization": "int8",
            "get_model_filename": select,
            "get_runtime_model_def": lambda m: family_handler.resolve_runtime_model_def(dict(definition), {
                "transformer_quantization": po.effective_transformer_quantization("int8"),
                **({"video_vae_precision": po.current()["video_vae_precision"]} if po.current().get("video_vae_precision") else {}),
            }),
            "get_model_recursive_prop": lambda *a, **k: [],
        }
        resolve = _exec(_functions(WGP_TREE, "resolve_job_model_files"), namespace, str(WGP))["resolve_job_model_files"]
        self.assertEqual(resolve("minimax_h3", "gguf_q2_k"), {
            "transformer": "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors",
            "video_vae": INT8_VAE,
            "text_encoder": "qwen3vl-32B-MiniMax-H3-Q2_K.gguf",
        })
        with po.scope({"transformer_quantization": "bf16", "text_encoder": "bf16"}):
            self.assertEqual(resolve("minimax_h3", "gguf_q2_k"), {
                "transformer": "MiniMax-H3-FL2VA-pruned_rank8_bf16.safetensors",
                "video_vae": FP16_VAE,
                "text_encoder": "Qwen3-VL-32B-Instruct-layer50_bf16.safetensors",
            })

    def test_job_load_goes_through_the_override_scope(self):
        source = ast.unparse(self._generate_video())
        self.assertIn("load_models_for_job(", source)
        self.assertIn("precision_overrides=job_precision_overrides", source)
        loader = ast.unparse(_functions(WGP_TREE, "load_models")[0])
        self.assertIn("effective_transformer_quantization(transformer_quantization)", loader)
        self.assertNotIn("else transformer_quantization", loader)

    def test_describe_files_is_one_line(self):
        line = po.describe_files("minimax_h3", {"transformer": "/x/a.safetensors", "video_vae": "", "text_encoder": "b.gguf"},
                                 {"transformer_quantization": "bf16"})
        self.assertEqual(line, "[Model Files] minimax_h3: transformer=a.safetensors video_vae=- text_encoder=b.gguf "
                               "(job overrides: transformer_quantization=bf16)")


class WorkerDownloadTests(unittest.TestCase):
    def _download_namespace(self, definition):
        downloads = []
        wgp = SimpleNamespace(
            get_model_def=lambda m: definition,
            transformer_quantization="int8", transformer_dtype_policy="", text_encoder_quantization="int8",
            get_transformer_dtype=lambda *a: None,
            get_model_filename=lambda model_type=None, quantization="", dtype_policy="", submodel_no=1, URLs=None, **k: (
                os.path.basename(URLs[0]) if URLs else f"transformer_{quantization}.safetensors"
            ),
            get_model_recursive_prop=lambda m, prop, **k: definition.get(prop, []) if prop != "modules" else [],
            download_models=lambda filename, *a, **k: downloads.append((filename, po.current())),
            get_compatible_local_model_filename=lambda *a, **k: "found",
        )
        namespace = {"wgp": wgp, "os": os, "check_download_cancelled": lambda: None,
                     "_check_model_downloaded": lambda m: True}
        names = ("_download_model_files", "_download_model_files_in_scope")
        _exec(_functions(LAUNCH_TREE, *names), namespace, str(LAUNCH))
        return namespace["_download_model_files"], downloads

    def test_worker_download_resolves_the_job_variant(self):
        definition = _h3_def()
        download, downloads = self._download_namespace(definition)
        download("minimax_h3")
        self.assertEqual([name for name, _ in downloads], [
            "transformer_int8.safetensors", "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        ])
        downloads.clear()
        overrides = {"transformer_quantization": "bf16", "text_encoder": "gguf_q4_k_m", "video_vae_precision": "fp16"}
        download("minimax_h3", overrides)
        self.assertEqual([name for name, _ in downloads], [
            "transformer_bf16.safetensors", "qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf",
        ])
        # download_models (and so the handler's VAE list) runs inside the scope.
        self.assertTrue(all(active == overrides for _, active in downloads))
        self.assertEqual(po.current(), {})


class ForwarderPreloadTests(unittest.TestCase):
    class Client:
        def __init__(self):
            self.status = {}
            self.posts = []

        def get_json(self, path, **_kwargs):
            assert path == "api/v1/models/downloads/status"
            return {"downloads": self.status}

        def post_json(self, path, body=None, **_kwargs):
            self.posts.append((path, body))
            key = "minimax_h3@tq-bf16+vae-fp16" if body else "minimax_h3"
            self.status[key] = {"status": "completed"}
            return {"status": "downloading", "download_key": key}

    def _preload(self, client, params):
        forwarder = RemoteJobForwarder(SimpleNamespace(), poll_interval=0, sleep=lambda s: None)
        forwarder._preload(client, "minimax_h3", update=lambda **u: None, is_cancelled=lambda: False, params=params)

    def test_default_job_keeps_the_plain_model_download(self):
        client = self.Client()
        self._preload(client, {"model_type": "minimax_h3", "override_text_encoder": ""})
        self.assertEqual(client.posts, [("api/v1/models/minimax_h3/download", None)])

    def test_override_job_downloads_its_variant(self):
        client = self.Client()
        client.status["minimax_h3"] = {"status": "completed"}  # default files are not enough
        params = {"model_type": "minimax_h3", "override_transformer_quantization": "bf16",
                  "override_video_vae_precision": "fp16"}
        self._preload(client, params)
        self.assertEqual(client.posts, [(
            "api/v1/models/minimax_h3/download",
            {"precision_overrides": {"override_transformer_quantization": "bf16", "override_video_vae_precision": "fp16"}},
        )])
        client.posts.clear()
        self._preload(client, params)
        self.assertEqual(client.posts, [])

    def test_override_values_are_forwarded_unchanged(self):
        forwarder = RemoteJobForwarder(SimpleNamespace(), poll_interval=0, sleep=lambda s: None)
        params = {"model_type": "minimax_h3", "override_transformer_quantization": "bf16",
                  "override_video_vae_precision": "fp16", "override_text_encoder": "bf16"}
        remote = forwarder._transfer_inputs(None, None, params, "minimax_h3",
                                            resolve_lora=lambda *a: None, update=lambda **u: None)
        self.assertEqual(remote, params)


if __name__ == "__main__":
    unittest.main()
