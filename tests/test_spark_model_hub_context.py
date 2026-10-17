"""docker/spark-stack/model-hub: GGUF header parsing, exact KV cache, max context, budgets.

No network: remote headers go through a fake opener, GGUF bytes are built here.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import struct
import sys
import tempfile
import unittest
import urllib.error

import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_HUB_DIR = os.path.join(_ROOT, "docker", "spark-stack", "model-hub")
_STANDALONE = os.path.join(_ROOT, "docker", "spark-stack", "llama-swap-standalone")
if _HUB_DIR not in sys.path:
    sys.path.insert(0, _HUB_DIR)

from model_hub import gguf_meta, ggufs, registry, render  # noqa: E402
from model_hub.gguf_meta import ArchInfo  # noqa: E402

TOKEN = "test-token-0123456789abcdef"
GB = ggufs.GB

# -- synthetic GGUF ---------------------------------------------------------

U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 = range(13)
_FMT = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i", F32: "<f", BOOL: "<?", U64: "<Q", I64: "<q",
        F64: "<d"}


def _s(text: str) -> bytes:
    data = text.encode()
    return struct.pack("<Q", len(data)) + data


def _val(type_id: int, value) -> bytes:
    if type_id == STR:
        return _s(value)
    if type_id == ARR:
        elem_type, items = value
        out = struct.pack("<IQ", elem_type, len(items))
        return out + b"".join(_val(elem_type, v) for v in items)
    return struct.pack(_FMT[type_id], value)


def gguf_bytes(kv: list[tuple[str, int, object]], version: int = 3, tensors: int = 7) -> bytes:
    """A GGUF header: kv is [(key, type, value)], arrays as (elem_type, [items])."""
    out = b"GGUF" + struct.pack("<IQQ", version, tensors, len(kv))
    for key, type_id, value in kv:
        out += _s(key) + struct.pack("<I", type_id) + _val(type_id, value)
    return out + b"\0" * 64  # start of the tensor infos: never parsed


def model_kv(arch="llama", layers=40, heads=32, heads_kv=8, head_dim=128, ctx=131072, extra=()) -> list:
    return [
        ("general.architecture", STR, arch),
        (f"{arch}.block_count", U32, layers),
        (f"{arch}.context_length", U32, ctx),
        (f"{arch}.embedding_length", U32, heads * head_dim),
        (f"{arch}.attention.head_count", U32, heads),
        (f"{arch}.attention.head_count_kv", U32, heads_kv),
        (f"{arch}.attention.key_length", U32, head_dim),
        (f"{arch}.attention.value_length", U32, head_dim),
        *extra,
    ]


DENSE = ArchInfo("llama", 40, 32, 8, 128, 128, 131072, 4096)
DENSE_KV = 40 * 8 * (128 + 128) * 2   # 163840 bytes per token, f16


class _FakeOpener:
    """Serves Range requests over `data`, recording the ranges asked."""

    def __init__(self, data: bytes, honour_range: bool = True):
        self.data, self.honour_range, self.ranges = data, honour_range, []

    def open(self, req, timeout=None):
        self.ranges.append(req.get_header("Range"))
        self.last_headers = dict(req.header_items())
        start, end = (int(x) for x in req.get_header("Range").split("=")[1].split("-"))
        if start >= len(self.data):
            raise urllib.error.HTTPError(req.full_url, 416, "range", {}, None)
        body = self.data[start:end + 1] if self.honour_range else self.data
        resp = io.BytesIO(body)
        resp.status = 206 if self.honour_range else 200
        resp.__enter__ = lambda *a: resp
        resp.__exit__ = lambda *a: None
        return resp


class HeaderParsingTests(unittest.TestCase):
    def test_scalars_strings_arrays_v2_and_v3(self):
        kv = [
            ("u8", U8, 7), ("i8", I8, -3), ("u16", U16, 65000), ("i16", I16, -2), ("u32", U32, 4000000000),
            ("i32", I32, -5), ("f32", F32, 1.5), ("bool", BOOL, True), ("str", STR, "héllo"),
            ("u64", U64, 1 << 40), ("i64", I64, -(1 << 40)), ("f64", F64, 2.25),
            ("arr_i32", ARR, (I32, [1, 0, 2])), ("arr_str", ARR, (STR, ["a", "bc"])),
            ("nested", ARR, (ARR, [(U8, [1, 2]), (STR, ["x"])])),
        ]
        for version in (2, 3):
            with self.subTest(version=version):
                head = gguf_meta.parse(gguf_bytes(kv, version=version, tensors=3))
                self.assertEqual((head["version"], head["tensor_count"], head["kv_count"]), (version, 3, len(kv)))
                got = head["kv"]
                self.assertEqual(got["u8"], 7)
                self.assertEqual(got["i8"], -3)
                self.assertEqual(got["u32"], 4000000000)
                self.assertEqual(got["f32"], 1.5)
                self.assertIs(got["bool"], True)
                self.assertEqual(got["str"], "héllo")
                self.assertEqual(got["u64"], 1 << 40)
                self.assertEqual(got["i64"], -(1 << 40))
                self.assertEqual(got["f64"], 2.25)
                self.assertEqual(got["arr_i32"], [1, 0, 2])
                self.assertEqual(got["arr_str"], ["a", "bc"])
                self.assertEqual(got["nested"], [[1, 2], ["x"]])
                self.assertLess(head["header_bytes"], len(gguf_bytes(kv, version=version)))

    def test_large_arrays_are_summarized_not_kept(self):
        tokens = [f"tok{i}" for i in range(gguf_meta.KEEP_ARRAY + 1)]
        scores = [float(i) for i in range(gguf_meta.KEEP_ARRAY + 1)]
        kv = [("tokenizer.ggml.tokens", ARR, (STR, tokens)), ("tokenizer.ggml.scores", ARR, (F32, scores)),
              ("after", U32, 42), ("small", ARR, (I32, [0, 2, 0]))]
        head = gguf_meta.parse(gguf_bytes(kv))
        self.assertEqual(head["kv"]["tokenizer.ggml.tokens"], {"array": "string", "count": len(tokens)})
        self.assertEqual(head["kv"]["tokenizer.ggml.scores"], {"array": "float32", "count": len(scores)})
        self.assertEqual(head["kv"]["after"], 42)       # the skip landed on the right offset
        self.assertEqual(head["kv"]["small"], [0, 2, 0])
        self.assertEqual(ArchInfo.from_kv(head["kv"]).vocab_size, len(tokens))

    def test_bad_magic_version_and_truncation(self):
        with self.assertRaises(gguf_meta.GgufError):
            gguf_meta.parse(b"GGML" + b"\0" * 40)
        with self.assertRaises(gguf_meta.GgufError):
            gguf_meta.parse(gguf_bytes([("a", U32, 1)], version=1))
        full = gguf_bytes(model_kv())
        for cut in (3, 10, 30, len(full) - 70):
            with self.subTest(cut=cut), self.assertRaises(gguf_meta.Truncated):
                gguf_meta.parse(full[:cut])
        self.assertTrue(issubclass(gguf_meta.Truncated, gguf_meta.GgufError))

    def test_local_read_grows_until_complete(self):
        data = gguf_bytes(model_kv(extra=[("pad", ARR, (STR, ["x" * 100] * 50))]))
        tmp = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp, "m.gguf")
            with open(path, "wb") as h:
                h.write(data)
            head = gguf_meta.read_local(path, chunks=(64, 1024, 1 << 20))
            self.assertEqual(head["kv"]["llama.block_count"], 40)
            with open(path, "wb") as h:
                h.write(data[:200])
            with self.assertRaises(gguf_meta.GgufError):
                gguf_meta.read_local(path, chunks=(64, 1024, 1 << 20))
        finally:
            shutil.rmtree(tmp)

    def test_remote_range_chunks_grow_and_append(self):
        data = gguf_bytes(model_kv(extra=[("pad", ARR, (STR, ["y" * 100] * 30))]))
        opener = _FakeOpener(data)
        url = gguf_meta.remote_url("owner/Repo-GGUF", "a" * 40, "m-Q4_K_M.gguf")
        self.assertEqual(url, f"https://huggingface.co/owner/Repo-GGUF/resolve/{'a' * 40}/m-Q4_K_M.gguf")
        head = gguf_meta.read_remote(url, chunks=(256, 1024, 4096, 1 << 20), opener=opener)
        self.assertEqual(head["kv"]["llama.attention.head_count_kv"], 8)
        self.assertEqual(opener.ranges, ["bytes=0-255", "bytes=256-1023", "bytes=1024-4095"])
        self.assertEqual(opener.last_headers.get("User-agent"), "spark-model-hub/1")
        # A header that outgrows the last chunk, and a file shorter than the header, are errors.
        with self.assertRaises(gguf_meta.GgufError):
            gguf_meta.read_remote(url, chunks=(256, 512), opener=_FakeOpener(data))
        with self.assertRaises(gguf_meta.GgufError):
            gguf_meta.read_remote(url, chunks=(256, 1 << 20), opener=_FakeOpener(data[:300]))
        # A server ignoring the Range (200) still works for the first chunk.
        self.assertEqual(gguf_meta.read_remote(url, chunks=(1 << 20,), opener=_FakeOpener(data, honour_range=False))
                         ["kv"]["llama.block_count"], 40)

    def test_remote_url_is_validated(self):
        for repo, rev, name in (("../x", "main", "a.gguf"), ("o/r", "../x", "a.gguf"), ("o/r", "main", "../a.gguf")):
            with self.subTest(repo=repo, rev=rev, name=name), self.assertRaises(ValueError):
                gguf_meta.remote_url(repo, rev, name)

    def test_meta_store_caches_as_json_and_prefers_local(self):
        tmp = tempfile.mkdtemp()
        try:
            data = gguf_bytes(model_kv(arch="qwen3moe", layers=48, heads_kv=4, extra=[("qwen3moe.expert_count", U32, 128)]))
            opener = _FakeOpener(data)
            store = gguf_meta.MetaStore(tmp, remote=True, opener=opener)
            info = store.info("o/r", "main", "m.gguf")
            self.assertEqual((info.architecture, info.expert_count, info.head_count_kv), ("qwen3moe", 128, 4))
            cached = os.path.join(tmp, "gguf-meta", gguf_meta.MetaStore.key("o/r", "main", "m.gguf") + ".json")
            with open(cached, encoding="utf-8") as h:
                self.assertEqual(json.load(h)["kv"]["qwen3moe.expert_count"], 128)
            # Second call: the memory cache; a fresh store: the JSON file; neither hits the network.
            n = len(opener.ranges)
            store.info("o/r", "main", "m.gguf")
            gguf_meta.MetaStore(tmp, remote=True, opener=opener).info("o/r", "main", "m.gguf")
            self.assertEqual(len(opener.ranges), n)
            # An installed file is read locally, and remote=False never opens a connection.
            local = os.path.join(tmp, "local.gguf")
            with open(local, "wb") as h:
                h.write(gguf_bytes(model_kv(arch="gemma3", layers=10)))
            offline = gguf_meta.MetaStore(tmp, remote=False, opener=opener)
            self.assertEqual(offline.info("o/r", "main", "other.gguf", local).architecture, "gemma3")
            self.assertIsNone(offline.info("o/r", "main", "missing.gguf"))
            self.assertEqual(len(opener.ranges), n)
            # Unreadable remote: None, with the error kept for the UI.
            broken = gguf_meta.MetaStore(tmp, remote=True, opener=_FakeOpener(b"not a gguf" * 10))
            self.assertIsNone(broken.info("o/r", "main", "bad.gguf"))
            self.assertIn("not a GGUF", broken.last_error("o/r", "main", "bad.gguf"))
        finally:
            shutil.rmtree(tmp)


class ArchInfoTests(unittest.TestCase):
    def test_dense_defaults_and_unknown_keys(self):
        info = ArchInfo.from_kv(gguf_meta.parse(gguf_bytes(model_kv()))["kv"])
        self.assertEqual(info, DENSE)
        self.assertTrue(info.complete)
        # key_length falls back to embedding / heads; head_count_kv to head_count; nothing crashes on junk.
        kv = {"general.architecture": "llama", "llama.block_count": 2, "llama.embedding_length": 256,
              "llama.attention.head_count": 4, "llama.attention.head_count_kv": "weird"}
        info = ArchInfo.from_kv(kv)
        self.assertEqual((info.key_length, info.value_length, info.head_count_kv), (64, 64, 4))
        empty = ArchInfo.from_kv({"general.architecture": 5})
        self.assertFalse(empty.complete)
        self.assertEqual(empty.layers(), [])
        self.assertIsNone(ggufs.kv_profile(empty))

    def test_hybrid_and_sliding_window_layers(self):
        hybrid = ArchInfo("nemotron_h", 6, 32, [8, 0, 0, 8, 0, 8], 128, 128, 8192)
        self.assertEqual([h for h, _ in hybrid.layers()], [8, 0, 0, 8, 0, 8])
        linear = ArchInfo("qwen3next", 8, 16, 2, 256, 256, 262144, full_attention_interval=4)
        self.assertEqual([h for h, _ in linear.layers()], [0, 0, 0, 2, 0, 0, 0, 2])
        gpt_oss = ArchInfo.from_kv(gguf_meta.parse(gguf_bytes(model_kv(
            "gpt-oss", layers=4, heads=64, heads_kv=8, head_dim=64,
            extra=[("gpt-oss.attention.sliding_window", U32, 128)])))["kv"])
        self.assertEqual(gpt_oss.sliding_window_pattern, 2)   # llama.cpp's built-in pattern
        self.assertEqual([swa for _, swa in gpt_oss.layers()], [True, False, True, False])
        explicit = ArchInfo("x", 6, 8, 8, 64, 64, 4096, sliding_window=512, sliding_window_pattern=3)
        self.assertEqual([swa for _, swa in explicit.layers()], [True, True, False, True, True, False])
        every = ArchInfo("x", 3, 8, 8, 64, 64, 4096, sliding_window=512)
        self.assertEqual([swa for _, swa in every.layers()], [True, True, True])
        mla = ArchInfo("deepseek2", 2, 128, 128, 192, 128, 4096, kv_lora_rank=512, rope_dimension_count=64)
        self.assertEqual(mla.kv_dims(), (576, 512))


class KvCacheTests(unittest.TestCase):
    def test_dense_moe_hybrid_swa_and_cache_types(self):
        self.assertEqual(ggufs.kv_profile(DENSE).per_token, DENSE_KV)
        self.assertEqual(ggufs.kv_profile(DENSE).as_dict()["attention_layers"], 40)
        moe = ArchInfo("qwen3moe", 48, 32, 4, 128, 128, 262144, expert_count=128)   # experts do not change the KV
        self.assertEqual(ggufs.kv_profile(moe).per_token, 48 * 4 * 256 * 2)
        hybrid = ArchInfo("nemotron_h", 6, 32, [8, 0, 0, 8, 0, 8], 128, 128, 8192)
        p = ggufs.kv_profile(hybrid)
        self.assertEqual((p.per_token, p.attention_layers, p.linear_layers), (3 * 8 * 256 * 2, 3, 3))
        linear = ArchInfo("qwen3next", 48, 16, 2, 256, 256, 262144, full_attention_interval=4)
        self.assertEqual(ggufs.kv_profile(linear).per_token, 12 * 2 * 512 * 2)
        swa = ArchInfo("gpt-oss", 4, 64, 8, 64, 64, 131072, sliding_window=128, sliding_window_pattern=2)
        p = ggufs.kv_profile(swa)
        per_layer = 8 * 128 * 2
        self.assertEqual((p.full_per_token, p.swa_per_token, p.window, p.swa_layers), (2 * per_layer, 2 * per_layer, 128, 2))
        # Sliding layers hold min(ctx, window * (parallel + 1) + 512) tokens.
        self.assertEqual(p.bytes_at(65536), 2 * per_layer * 65536 + 2 * per_layer * (128 * 2 + 512))
        self.assertEqual(p.bytes_at(512), 4 * per_layer * 512)
        self.assertEqual(p.bytes_at(65536, parallel=4), 2 * per_layer * 65536 + 2 * per_layer * (128 * 5 + 512))
        # q8_0: 34 bytes per 32 elements, q4_0: 18.
        self.assertEqual(ggufs.kv_profile(DENSE, "q8_0", "q8_0").per_token, int(40 * 8 * 256 * 34 / 32))
        self.assertEqual(ggufs.kv_profile(DENSE, "q8_0", "f16").per_token, int(40 * 8 * (128 * 34 / 32 + 128 * 2)))
        self.assertEqual(ggufs.kv_profile(DENSE, "q4_0", "q4_0").per_token, int(40 * 8 * 256 * 18 / 32))
        self.assertEqual(ggufs.kv_profile(DENSE, "f32", "f32").per_token, 2 * DENSE_KV)
        with self.assertRaises(ValueError):
            ggufs.kv_profile(DENSE, "q3_k", "f16")

    def test_estimate_uses_metadata_or_falls_back(self):
        est = ggufs.estimate_memory(int(20e9), 65536, info=DENSE)
        self.assertEqual((est["kv"], est["kv_per_token"], est["kv_source"]), (DENSE_KV * 65536, DENSE_KV, "metadata"))
        self.assertEqual(est["overhead"], int(0.5 * GB + 0.03 * 20e9))
        self.assertEqual(est["total"], est["weights"] + est["kv"] + est["overhead"])
        fallback = ggufs.estimate_memory(int(20e9), 65536)
        self.assertEqual((fallback["kv_per_token"], fallback["kv_source"]), (20 * 6 * 1024, "heuristic"))
        incomplete = ggufs.estimate_memory(int(20e9), 65536, info=ArchInfo("x", 40))
        self.assertEqual(incomplete["kv_source"], "heuristic")
        quant = ggufs.estimate_memory(int(20e9), 65536, info=DENSE, cache_type_k="q8_0", cache_type_v="q8_0")
        self.assertLess(quant["kv"], est["kv"] * 0.54)


class MaxContextTests(unittest.TestCase):
    def test_budget_limit_rounding_and_monotonicity(self):
        weights, budget = int(20e9), 40 * GB
        best = ggufs.max_context(DENSE, weights, 0, budget)
        self.assertEqual(best["ctx"] % 256, 0)
        self.assertEqual((best["limited_by"], best["fits"], best["kv_source"]), ("budget", True, "metadata"))
        self.assertLessEqual(ggufs.estimate_memory(weights, best["ctx"], info=DENSE)["total"], budget)
        self.assertGreater(ggufs.estimate_memory(weights, best["ctx"] + 256, info=DENSE)["total"], budget)
        self.assertAlmostEqual(best["kv_gb"], DENSE_KV * best["ctx"] / GB, places=2)
        previous = 0
        for gb in (25, 30, 40, 60, 100):
            ctx = ggufs.max_context(DENSE, weights, 0, gb * GB)["ctx"]
            self.assertGreaterEqual(ctx, previous)
            previous = ctx
        # q8_0 cache: more context in the same budget; an mmproj: less.
        self.assertGreater(ggufs.max_context(DENSE, weights, 0, budget, "q8_0", "q8_0")["ctx"], best["ctx"])
        self.assertLess(ggufs.max_context(DENSE, weights, int(2e9), budget)["ctx"], best["ctx"])

    def test_trained_cap_minimum_and_parallel(self):
        best = ggufs.max_context(DENSE, int(20e9), 0, 120 * GB)
        self.assertEqual((best["ctx"], best["limited_by"], best["trained"]), (131072, "trained", 131072))
        tiny = ggufs.max_context(DENSE, int(20e9), 0, 20.5 * GB)
        self.assertEqual((tiny["ctx"], tiny["fits"]), (2048, False))
        four = ggufs.max_context(DENSE, int(20e9), 0, 40 * GB, parallel=4)
        self.assertEqual(four["per_slot"], four["ctx"] // 4)
        # Without metadata: heuristic KV, no trained cap, still a multiple of 256 that fits.
        guess = ggufs.max_context(None, int(20e9), 0, 40 * GB)
        self.assertEqual((guess["kv_source"], guess["trained"], guess["ctx"] % 256), ("heuristic", None, 0))
        self.assertLessEqual(ggufs.estimate_memory(int(20e9), guess["ctx"])["total"], 40 * GB)

    def test_context_report(self):
        report = ggufs.context_report(DENSE, int(20e9), int(1e9), 40 * GB, ctx=8192, parallel=2)
        self.assertEqual(report["arch"]["architecture"], "llama")
        self.assertEqual(report["kv_per_token"], DENSE_KV)
        self.assertEqual(report["estimate"]["ctx"], 8192)
        self.assertEqual(report["estimate"]["kv"], round(DENSE_KV * 8192 / GB, 2))
        self.assertEqual(report["max"]["per_slot"], report["max"]["ctx"] // 2)
        self.assertEqual(report["budget_gb"], 40.0)
        self.assertIsNone(ggufs.context_report(None, int(1e9), 0, 10 * GB)["arch"])


class BudgetTests(unittest.TestCase):
    def test_resident_vs_on_demand_with_swap_reserve(self):
        budget = ggufs.Budget(128, 10, 25, ["small"], swap_reserve_gb=70)
        self.assertEqual(budget.available_gb, 93)              # on-demand: total - reserved - residents
        self.assertEqual(budget.resident_available_gb, 23)     # resident: minus the swap reserve too
        self.assertEqual(budget.available_for(True), 23)
        self.assertEqual(budget.available_for(False), 93)
        d = budget.as_dict()
        self.assertEqual((d["swap_reserve_gb"], d["resident_available_gb"], d["available_gb"]), (70, 23, 93))
        self.assertEqual(ggufs.fit_badge(int(70e9), budget), "fits")
        self.assertEqual(ggufs.fit_badge(int(70e9), budget, resident=True), "too-big")
        self.assertEqual(ggufs.fit_badge(int(21e9), budget, resident=True), "tight")
        groups = {"resident": {"persistent": True}, "on-demand": {"swap": True}}
        self.assertTrue(ggufs.is_resident("resident", groups))
        self.assertFalse(ggufs.is_resident("on-demand", groups))
        self.assertTrue(ggufs.is_resident("on-demand", groups, preload=True))
        self.assertFalse(ggufs.is_resident("unknown", groups))
        self.assertFalse(ggufs.is_resident(None, {}))


class RenderCacheFlagTests(unittest.TestCase):
    def test_cache_types_in_serve_and_cmd(self):
        serve = registry.normalize_serve({"id": "m", "model": "m.gguf", "cache_type_k": "q8_0", "cache_type_v": "q4_0"})
        self.assertEqual((serve["cache_type_k"], serve["cache_type_v"]), ("q8_0", "q4_0"))
        cmd = render.build_cmd({"repo": "o/r"}, serve, "${models}")
        self.assertIn("--ctx-size 32768 --parallel 1 --cache-type-k q8_0 --cache-type-v q4_0", cmd)
        plain = registry.normalize_serve({"id": "m", "model": "m.gguf", "cache_type_k": "f16"})
        self.assertNotIn("cache_type_k", plain)   # f16 is the default: nothing to render
        self.assertNotIn("--cache-type", render.build_cmd({"repo": "o/r"}, plain, "${models}"))
        for bad in ("q3_k", "F16", 8, "q8_0; rm"):
            with self.subTest(bad=bad), self.assertRaises(registry.ConfigError):
                registry.normalize_serve({"id": "m", "model": "m.gguf", "cache_type_v": bad})
        with open(os.path.join(_STANDALONE, "config", "llama-swap.base.example.yaml"), encoding="utf-8") as h:
            base = yaml.safe_load(h)
        config, _ = render.render(base, [{"repo": "o/r", "revision": "a" * 40, "include": ["m.gguf"], "serve": serve}])
        self.assertEqual(render.validate(config), [])
        self.assertIn("--cache-type-k q8_0", config["models"]["m"]["cmd"])


class AppContextTests(unittest.TestCase):
    """The /api/context route and the badges, with installed files carrying a synthetic header."""

    def setUp(self):
        from fastapi.testclient import TestClient
        from model_hub import app as app_module

        self.tmp = tempfile.mkdtemp()
        self.models = os.path.join(self.tmp, "models")
        self.config = os.path.join(self.tmp, "config")
        os.makedirs(self.config)
        shutil.copy(os.path.join(_STANDALONE, "config", "llama-swap.base.example.yaml"), self.config)
        with open(os.path.join(self.config, "models.example.yml"), "w", encoding="utf-8") as h:
            h.write("[]\n")
        settings = app_module.Settings(token=TOKEN, models_dir=self.models, config_dir=self.config,
                                       catalog_path=os.path.join(_HUB_DIR, "catalog.yml"), remote_metadata=False,
                                       swap_reserve_gb=70)
        self.app = app_module.create_app(settings)
        self.hub = self.app.state.hub
        self.client = TestClient(self.app)
        r = self.client.post("/api/login", json={"token": TOKEN})
        self.h = {"X-CSRF-Token": r.json()["csrf"]}

    def tearDown(self):
        self.client.close()
        shutil.rmtree(self.tmp)

    def install(self, repo: str, name: str, data: bytes, size: int) -> None:
        """A verified file in the store, with the header bytes on disk and the Hub size in the manifest."""
        path = self.hub.store.target(repo, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as h:
            h.write(data)
        self.hub.store.update_manifest({f"{repo}/{name}": {"commit": "a" * 40, "expected": "x", "algo": "sha256",
                                                            "sha256": "x", "size": size}})

    def test_context_route_exact_and_fallback(self):
        repo = "owner/Dense-GGUF"
        self.install(repo, "dense-Q4_K_M.gguf", gguf_bytes(model_kv()), int(30e9))
        self.install(repo, "mmproj-F16.gguf", gguf_bytes(model_kv()), int(1e9))
        r = self.client.get(f"/api/context/{repo}", params={"file": "dense-Q4_K_M.gguf", "revision": "a" * 40})
        self.assertEqual(r.status_code, 200, r.text)
        d = r.json()
        self.assertEqual((d["kv_source"], d["kv_per_token"], d["arch"]["architecture"]), ("metadata", DENSE_KV, "llama"))
        self.assertEqual((d["resident"], d["budget_gb"]), (False, 118.0))
        self.assertEqual((d["max"]["ctx"], d["max"]["limited_by"]), (131072, "trained"))
        self.assertEqual(d["weights"], int(30e9))
        # Options: mmproj, parallel, a resident group with the 70 GB swap reserve (f16 KV: 21.5 GB at 131072).
        params = {"file": "dense-Q4_K_M.gguf", "mmproj": "mmproj-F16.gguf", "parallel": 2, "group": "resident", "ctx": 4096}
        d = self.client.get(f"/api/context/{repo}", params=params).json()
        self.assertEqual((d["resident"], d["budget_gb"]), (True, 48.0))
        self.assertEqual(d["estimate"]["mmproj"], 1.0)
        self.assertEqual(d["estimate"]["ctx"], 4096)
        self.assertEqual(d["max"]["limited_by"], "budget")
        self.assertEqual(d["max"]["per_slot"], d["max"]["ctx"] // 2)
        self.assertLessEqual(d["max"]["total_gb"], 48.0)
        # q8_0 on both halves the KV cache: the trained length fits again.
        q8 = self.client.get(f"/api/context/{repo}", params={**params, "cache_type_k": "q8_0", "cache_type_v": "q8_0"}).json()
        self.assertEqual(q8["kv_per_token"], int(DENSE_KV * 34 / 64))
        self.assertEqual((q8["max"]["ctx"], q8["max"]["limited_by"]), (131072, "trained"))
        # Unknown file, bad cache type, unsafe names.
        self.assertEqual(self.client.get(f"/api/context/{repo}", params={"file": "nope.gguf"}).status_code, 404)
        self.assertEqual(self.client.get(f"/api/context/{repo}", params={"file": "dense-Q4_K_M.gguf", "cache_type_k": "q3"}).status_code, 400)
        self.assertEqual(self.client.get("/api/context/o/r", params={"file": "../x.gguf"}).status_code, 400)
        # A file without a readable header: the heuristic, flagged as such.
        self.install("owner/Opaque-GGUF", "opaque-Q4_K_M.gguf", b"\0" * 4096, int(10e9))
        d = self.client.get("/api/context/owner/Opaque-GGUF", params={"file": "opaque-Q4_K_M.gguf"}).json()
        self.assertEqual((d["kv_source"], d["arch"], d["kv_per_token"]), ("heuristic", None, 10 * 6 * 1024))
        self.assertIn("not a GGUF", d["metadata_error"])

    def test_residents_use_metadata_and_shrink_the_budget(self):
        repo = "owner/Small-GGUF"
        self.install(repo, "small-Q8_0.gguf", gguf_bytes(model_kv(layers=28, heads_kv=8, head_dim=128)), int(8e9))
        serve = {"id": "small", "model": "small-Q8_0.gguf", "ctx_size": 32768, "group": "resident", "ttl": 0}
        registry.save(self.hub.s.models_yml, [{"repo": repo, "revision": "a" * 40, "include": ["small-Q8_0.gguf"],
                                               "serve": serve}])
        budget = self.hub.budget()
        kv = 28 * 8 * 256 * 2 * 32768
        expected = (8e9 + kv + 0.5e9 + 0.03 * 8e9) / 1e9
        self.assertEqual(budget.resident, ["small"])
        self.assertAlmostEqual(budget.resident_gb, expected, places=3)
        self.assertEqual(self.client.get("/api/status").json()["budget"]["swap_reserve_gb"], 70)
        # The resident's own budget excludes itself but keeps the swap reserve.
        d = self.client.get(f"/api/context/{repo}", params={"file": "small-Q8_0.gguf", "group": "resident"}).json()
        self.assertEqual((d["budget_gb"], d["budget"]["resident"]), (48.0, []))
        d = self.client.get("/api/context/owner/Other-GGUF", params={"file": "x.gguf"})
        self.assertIn(d.status_code, (404, 502))   # not installed, and no network here

    def test_catalog_badges_still_work_without_metadata(self):
        items = self.client.get("/api/catalog").json()["items"]
        self.assertTrue(all(i["kv_source"] == "heuristic" and i["fit"] in ("fits", "tight", "too-big") for i in items))
        self.assertTrue(any(i["resident"] for i in items))

    def test_cli_context_on_an_installed_model(self):
        from model_hub import cli
        repo = "owner/Dense-GGUF"
        self.install(repo, "dense-Q4_K_M.gguf", gguf_bytes(model_kv()), int(20e9))
        import contextlib
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["context", repo, "--parallel", "2", "--cache", "q8_0", "--models", self.models,
                             "--config", self.hub.s.models_yml, "--base", self.hub.s.base_yaml])
        self.assertEqual(code, 0, out.getvalue())
        text = out.getvalue()
        self.assertIn("llama: 40 layers (40 attention), KV heads 8 x 128/128, trained context 131072", text)
        self.assertIn("max --ctx-size 131072 (65536 per slot x 2, trained limit)", text)
        self.assertIn("KB/token (q8_0/q8_0)", text)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main(["context", repo, "--file", "nope.gguf", "--models", self.models,
                                       "--config", self.hub.s.models_yml]), 1)


if __name__ == "__main__":
    unittest.main()
