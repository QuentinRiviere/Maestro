"""docker/spark-stack/model-hub: rendering, file grouping, path safety, auth, fetcher, memory fit.

Network tests (tiny real downloads from huggingface.co and the catalog check)
are skipped when MODEL_HUB_OFFLINE=1 or when the Hub is unreachable.
"""
from __future__ import annotations

import copy
import os
import shutil
import sys
import tempfile
import time
import unittest
import urllib.request

import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_HUB_DIR = os.path.join(_ROOT, "docker", "spark-stack", "model-hub")
_STANDALONE = os.path.join(_ROOT, "docker", "spark-stack", "llama-swap-standalone")
_MAESTRO_BASE = os.path.join(_ROOT, "docker", "spark-stack", "llama-swap.yaml")
if _HUB_DIR not in sys.path:
    sys.path.insert(0, _HUB_DIR)

from model_hub import auth, catalog, fetcher, ggufs, paths, registry, render  # noqa: E402

TOKEN = "test-token-0123456789abcdef"
TINY_REPO = "ggml-org/models"
TINY_COMMIT = "499bc8821c6b12b4e53c5bffcb21ec206f212d81"
TINY_FILE = "tinyllamas/stories260K.gguf"   # 1185376 bytes, LFS (sha256)
TINY_TEXT = "README.md"                      # 81 bytes, plain git blob (sha1)


def _online() -> bool:
    if os.environ.get("MODEL_HUB_OFFLINE") == "1":
        return False
    try:
        urllib.request.urlopen("https://huggingface.co/api/models/ggml-org/models", timeout=5).close()
        return True
    except OSError:
        return False


ONLINE = _online()
network = unittest.skipUnless(ONLINE, "Hugging Face unreachable or MODEL_HUB_OFFLINE=1")


def _base_example() -> dict:
    with open(os.path.join(_STANDALONE, "config", "llama-swap.base.example.yaml"), encoding="utf-8") as h:
        return yaml.safe_load(h)


def _entry(repo="owner/Some-GGUF", **serve) -> dict:
    s = {"id": "some-model", "model": "some-Q4_K_M.gguf", "ctx_size": 8192}
    s.update(serve)
    return {"repo": repo, "revision": "a" * 40, "include": ["*-Q4_K_M.gguf"], "serve": s}


class RenderTests(unittest.TestCase):
    def test_standalone_base_renders_managed_models(self):
        entries = [
            _entry("huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF", id="qwen36", model="m-Q4_K.gguf",
                   mmproj="mmproj-model-f16.gguf", ctx_size=65536, mtp=True, mtp_draft_n_max=6, ttl=0,
                   group="resident", preload=True, aliases=["small-llm"], extra_args=["--temp", "0.6"]),
            _entry("ggml-org/gpt-oss-120b-GGUF", id="gpt-oss-120b", model="gpt-oss-120b-MXFP4.gguf", ttl=600),
        ]
        config, warnings = render.render(_base_example(), entries)
        self.assertEqual(render.validate(config), [])
        cmd = config["models"]["qwen36"]["cmd"]
        self.assertIn("${llama-server}", cmd)
        self.assertIn("--model ${models}/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF/m-Q4_K.gguf", cmd)
        self.assertIn("--mmproj ${models}/huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF/mmproj-model-f16.gguf", cmd)
        self.assertIn("--ctx-size 65536 --parallel 1", cmd)
        self.assertIn("--spec-type draft-mtp --spec-draft-n-max 6", cmd)
        self.assertIn("--temp 0.6", cmd)
        self.assertNotIn("--spec-type", config["models"]["gpt-oss-120b"]["cmd"])
        self.assertEqual(config["models"]["qwen36"]["ttl"], 0)
        self.assertEqual(config["models"]["qwen36"]["aliases"], ["small-llm"])
        groups = config["routing"]["router"]["settings"]["groups"]
        self.assertEqual(groups["resident"]["members"], ["qwen36"])
        self.assertEqual(groups["on-demand"]["members"], ["gpt-oss-120b"])
        self.assertEqual(config["hooks"]["on_startup"]["preload"], ["qwen36"])
        self.assertTrue(any("MTP" in w for w in warnings))  # mmproj + MTP warning
        # The dumped YAML round-trips and keeps cmd as a block.
        text = render.dump(config)
        self.assertTrue(text.startswith("# yaml-language-server"))
        self.assertIn("cmd: |", text)
        self.assertEqual(yaml.safe_load(text), config)

    def test_maestro_lease_and_peer_survive(self):
        with open(_MAESTRO_BASE, encoding="utf-8") as h:
            base = yaml.safe_load(h)
        original = copy.deepcopy(base)
        entries = [_entry("ggml-org/gpt-oss-120b-GGUF", id="gpt-oss-120b", model="gpt-oss-120b-MXFP4.gguf",
                          group="gpu-heavy", ttl=600)]
        config, _ = render.render(base, entries)
        self.assertEqual(base, original, "render must not mutate the base")
        self.assertEqual(config["models"]["maestro-worker"], original["models"]["maestro-worker"])
        self.assertEqual(config["peers"], original["peers"])
        self.assertEqual(config["apiKeys"], original["apiKeys"])
        self.assertEqual(config["hooks"]["on_startup"]["preload"], ["small-llm", "maestro-worker"])
        members = config["routing"]["router"]["settings"]["groups"]["gpu-heavy"]["members"]
        self.assertEqual(members, ["maestro-worker", "big-llm", "gpt-oss-120b"])
        self.assertEqual(render.validate(config), [])

    def test_name_collisions_are_refused(self):
        with open(_MAESTRO_BASE, encoding="utf-8") as h:
            base = yaml.safe_load(h)
        for serve in ({"id": "maestro-worker"}, {"id": "x", "aliases": ["small-llm"]}, {"id": "maestro-api"}):
            with self.subTest(serve=serve), self.assertRaises(render.RenderError):
                render.render(base, [_entry(**serve)])
        with self.assertRaises(render.RenderError):
            render.render(_base_example(), [_entry(id="dup"), _entry(repo="other/Repo", id="dup")])

    def test_schema_errors_block_the_render(self):
        base = _base_example()
        base["healthCheckTimeout"] = "soon"
        with self.assertRaises(render.RenderError) as ctx:
            render.render(base, [])
        self.assertIn("healthCheckTimeout", str(ctx.exception))
        base = _base_example()
        del base["macros"]["llama-server"]
        with self.assertRaises(render.RenderError):
            render.render(base, [_entry()])

    def test_missing_files_and_disabled_entries_are_not_mapped(self):
        entries = [_entry(id="present"), _entry(repo="o/Missing", id="missing"), _entry(repo="o/Off", id="off", enabled=False)]
        config, warnings = render.render(_base_example(), entries, available=lambda repo, name: repo != "o/Missing")
        self.assertEqual(sorted(config["models"]), ["present"])
        self.assertTrue(any("missing" in w for w in warnings))

    def test_unknown_group_is_created(self):
        config, warnings = render.render(_base_example(), [_entry(group="experiments")])
        group = config["routing"]["router"]["settings"]["groups"]["experiments"]
        self.assertEqual(group, {"swap": True, "exclusive": False, "members": ["some-model"]})
        self.assertTrue(any("experiments" in w for w in warnings))

    def test_legacy_groups_are_used_when_present(self):
        base = _base_example()
        del base["routing"]
        base["groups"] = {"on-demand": {"swap": True, "exclusive": False, "members": []}}
        config, _ = render.render(base, [_entry()])
        self.assertEqual(config["groups"]["on-demand"]["members"], ["some-model"])
        self.assertNotIn("routing", config)
        self.assertEqual(render.validate(config), [])

    def test_unsafe_serve_values_are_refused(self):
        bad = [
            {"extra_args": ["${env.LLAMA_SWAP_API_KEY}"]},
            {"extra_args": ["--x; rm -rf /"]},
            {"model": "../../etc/passwd.gguf"},
            {"model": "with space.gguf"},
            {"model": "weights.bin"},
            {"id": "bad id"},
            {"name": "line\nbreak"},
            {"ctx_size": -1},
            {"surprise": 1},
        ]
        for serve in bad:
            with self.subTest(serve=serve), self.assertRaises(registry.ConfigError):
                registry.normalize_serve({"id": "ok", "model": "a.gguf", **serve})

    def test_atomic_write_keeps_a_backup(self):
        tmp = tempfile.mkdtemp()
        try:
            out = os.path.join(tmp, "llama-swap.yaml")
            self.assertTrue(render.write(out, "a: 1\n"))
            self.assertFalse(render.write(out, "a: 1\n"))
            self.assertTrue(render.write(out, "a: 2\n"))
            with open(out + ".bak", encoding="utf-8") as h:
                self.assertEqual(h.read(), "a: 1\n")
            self.assertEqual(sorted(os.listdir(tmp)), ["llama-swap.yaml", "llama-swap.yaml.bak"])
        finally:
            shutil.rmtree(tmp)

    def test_examples_and_catalog_render_and_validate(self):
        with open(os.path.join(_STANDALONE, "config", "models.example.yml"), encoding="utf-8") as h:
            entries = yaml.safe_load(h)
        config, _ = render.render(_base_example(), entries)
        self.assertIn("qwen36-35b-abliterated", config["models"])
        items = catalog.load(os.path.join(_HUB_DIR, "catalog.yml"))
        self.assertGreaterEqual(len(items), 8)
        self.assertIn("huihui-ai/Huihui-Qwen3.6-35B-A3B-abliterated-MTP-GGUF", [i["repo"] for i in items])
        all_entries = [{"repo": i["repo"], "revision": i["revision"], "include": i["include"], "serve": i["serve"]}
                       for i in items]
        config, _ = render.render(_base_example(), all_entries)
        self.assertEqual(len(config["models"]), len(items))


class GroupingTests(unittest.TestCase):
    def test_split_files_merge_and_kinds(self):
        files = [
            {"name": "Q/gpt-oss-120b-Q4_K_M-00001-of-00003.gguf", "size": 30},
            {"name": "Q/gpt-oss-120b-Q4_K_M-00003-of-00003.gguf", "size": 10},
            {"name": "Q/gpt-oss-120b-Q4_K_M-00002-of-00003.gguf", "size": 20},
            {"name": "big-Q8_0-00001-of-00002.gguf", "size": 5},
            {"name": "model-UD-Q4_K_XL.gguf", "size": 7},
            {"name": "mmproj-F16.gguf", "size": 1},
            {"name": "MTP/mtp-model-Q8_0.gguf", "size": 1},
            {"name": "eagle3-model-Q8_0.gguf", "size": 1},
            {"name": "README.md", "size": 1},
        ]
        groups = {g.key: g for g in ggufs.group_files(files)}
        split = groups["Q/gpt-oss-120b-Q4_K_M-*-of-00003.gguf"]
        self.assertEqual(split.files, [f"Q/gpt-oss-120b-Q4_K_M-0000{i}-of-00003.gguf" for i in (1, 2, 3)])
        self.assertEqual((split.size, split.parts, split.complete, split.quant, split.kind), (60, 3, True, "Q4_K_M", "model"))
        self.assertEqual(split.first, "Q/gpt-oss-120b-Q4_K_M-00001-of-00003.gguf")
        self.assertFalse(groups["big-Q8_0-*-of-00002.gguf"].complete)
        self.assertEqual(groups["model-UD-Q4_K_XL.gguf"].quant, "UD-Q4_K_XL")
        self.assertEqual(groups["mmproj-F16.gguf"].kind, "mmproj")
        self.assertEqual(groups["MTP/mtp-model-Q8_0.gguf"].kind, "mtp")
        self.assertEqual(groups["eagle3-model-Q8_0.gguf"].kind, "draft")
        self.assertNotIn("README.md", groups)

    def test_quant_names(self):
        cases = {
            "Huihui-Qwen3.6-35B-A3B-abliterated-ggml-model-Q4_K.gguf": "Q4_K",
            "gpt-oss-20b-MXFP4.gguf": "MXFP4",
            "Qwen3.6-35B-A3B-MXFP4_MOE.gguf": "MXFP4_MOE",
            "Qwen3-Coder-30B-A3B-Instruct-IQ4_XS.gguf": "IQ4_XS",
            "Qwen3.8-27B-BF16.gguf": "BF16",
            "Qwen3-1.7B-f16.gguf": "F16",
            "stories260K.gguf": "?",
        }
        for name, quant in cases.items():
            self.assertEqual(ggufs.quant_of(name), quant, name)

    def test_mtp_detection(self):
        self.assertTrue(ggufs.detect_mtp("x/Model-MTP-GGUF", [], "")["mtp"])
        found = ggufs.detect_mtp("x/Model-GGUF", [], "run with --spec-type draft-mtp --spec-draft-n-max 2")
        self.assertEqual((found["mtp"], found["draft_n_max"]), (True, 2))
        self.assertTrue(ggufs.detect_mtp("x/m", ["MTP"], "")["mtp"])
        self.assertFalse(ggufs.detect_mtp("x/Smtpd-GGUF", ["gguf"], "nothing")["mtp"])


class PathSafetyTests(unittest.TestCase):
    def test_repo_ids(self):
        for ok in ("ggml-org/gpt-oss-20b-GGUF", "Qwen/Qwen3-VL-8B-Instruct-GGUF", "a/b.c_d"):
            self.assertEqual(paths.check_repo_id(ok), ok)
        for bad in ("../etc", "a/../b", "a/b/c", "/abs/x", "a", "a/..", "a/.hidden", "a/b\0", "a/b c", "", None):
            with self.subTest(bad=bad), self.assertRaises(paths.UnsafePathError):
                paths.check_repo_id(bad)

    def test_file_names(self):
        for ok in ("model.gguf", "tinyllamas/stories260K.gguf", "MTP/mtp-x-Q8_0.gguf"):
            self.assertEqual(paths.check_filename(ok), ok)
        for bad in ("../x.gguf", "/etc/passwd", "a/../../b.gguf", "a/./b.gguf", ".env", "a\\b.gguf", "a//b.gguf", ""):
            with self.subTest(bad=bad), self.assertRaises(paths.UnsafePathError):
                paths.check_filename(bad)

    def test_paths_stay_inside_the_models_dir(self):
        tmp = tempfile.mkdtemp()
        try:
            inside = paths.model_file(tmp, "o/r", "sub/f.gguf")
            self.assertTrue(inside.startswith(os.path.realpath(tmp) + os.sep))
            with self.assertRaises(paths.UnsafePathError):
                paths.inside(tmp, "..", "x")
            os.symlink("/", os.path.join(tmp, "o"))  # a symlink escape is caught too
            with self.assertRaises(paths.UnsafePathError):
                paths.model_file(tmp, "o/r", "f.gguf")
        finally:
            shutil.rmtree(tmp)

    def test_only_hugging_face_hosts(self):
        self.assertTrue(paths.allowed_host("https://huggingface.co/api/models"))
        self.assertTrue(paths.allowed_host("https://cas-bridge.xethub.hf.co/x"))
        self.assertTrue(paths.allowed_host("https://cdn-lfs-us-1.hf.co/x"))
        for bad in ("http://huggingface.co/x", "https://evilhuggingface.co/x", "https://huggingface.co.evil.com/x",
                    "https://example.com/x", "file:///etc/passwd"):
            self.assertFalse(paths.allowed_host(bad), bad)
        with self.assertRaises(paths.UnsafePathError):
            paths.resolve_url("o/r", "main", "f.gguf")  # downloads need a commit

    def test_refused_formats(self):
        snap = fetcher.Snapshot("o/r", "a" * 40, [
            fetcher.RemoteFile("model.gguf", 1, "x", "sha256"),
            fetcher.RemoteFile("pytorch_model.bin", 1, "x", "sha256"),
            fetcher.RemoteFile("model.pt", 1, "x", "sha256"),
            fetcher.RemoteFile("setup.py", 1, "x", "git-sha1"),
            fetcher.RemoteFile("config.json", 1, "x", "git-sha1"),
        ])
        chosen, refused = fetcher.select(snap, None, None)
        self.assertEqual([f.name for f in chosen], ["model.gguf", "config.json"])
        self.assertEqual(sorted(refused), ["model.pt", "pytorch_model.bin", "setup.py"])
        self.assertEqual(fetcher.glob_escape("a[1]*.gguf"), "a[[]1][*].gguf")


class MemoryTests(unittest.TestCase):
    def test_estimate_and_badges(self):
        est = ggufs.estimate_memory(int(20e9), 65536, int(1e9))
        self.assertEqual(est["kv_per_token"], int(20 * 6 * 1024))
        self.assertEqual(est["total"], est["weights"] + est["mmproj"] + est["kv"] + est["overhead"])
        self.assertGreater(est["total"], 21e9)
        self.assertLess(est["total"], 32e9)
        budget = ggufs.Budget(128, 10, 25)
        self.assertEqual(budget.available_gb, 93)
        self.assertEqual(ggufs.fit_badge(int(70e9), budget), "fits")
        self.assertEqual(ggufs.fit_badge(int(85e9), budget), "tight")
        self.assertEqual(ggufs.fit_badge(int(100e9), budget), "too-big")
        self.assertEqual(ggufs.kv_bytes_per_token(int(0.5e9)), 16 * 1024)
        self.assertEqual(ggufs.kv_bytes_per_token(int(500e9)), 320 * 1024)


class _AppCase(unittest.TestCase):
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
                                       catalog_path=os.path.join(_HUB_DIR, "catalog.yml"))
        self.app = app_module.create_app(settings)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.client.close()
        shutil.rmtree(self.tmp)

    def login(self) -> str:
        r = self.client.post("/api/login", json={"token": TOKEN})
        self.assertEqual(r.status_code, 200)
        return r.json()["csrf"]


class AuthTests(_AppCase):
    def test_refuses_to_start_without_token(self):
        from model_hub import app as app_module
        for token in ("", "short"):
            with self.assertRaises(auth.TokenMissing):
                app_module.create_app(app_module.Settings(token=token, models_dir=self.models, config_dir=self.config))

    def test_seeds_config_and_renders_at_startup(self):
        self.assertTrue(os.path.exists(os.path.join(self.config, "llama-swap.base.yaml")))
        self.assertTrue(os.path.exists(os.path.join(self.config, "models.yml")))
        with open(os.path.join(self.config, "llama-swap.yaml"), encoding="utf-8") as h:
            self.assertEqual(render.validate(yaml.safe_load(h)), [])

    def test_requires_login(self):
        for path in ("/api/status", "/api/catalog", "/api/installed", "/api/downloads", "/api/config"):
            self.assertEqual(self.client.get(path).status_code, 401, path)
        self.assertEqual(self.client.post("/api/render").status_code, 401)
        self.assertEqual(self.client.get("/api/status", headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.assertEqual(self.client.get("/api/session").status_code, 401)
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn("script-src 'self'", page.headers["content-security-policy"])
        self.assertNotIn("access-control-allow-origin", page.headers)

    def test_cookie_session_and_csrf(self):
        r = self.client.post("/api/login", json={"token": TOKEN})
        cookie = r.headers["set-cookie"].lower()
        self.assertIn("httponly", cookie)
        self.assertIn("samesite=strict", cookie)
        csrf = r.json()["csrf"]
        self.assertEqual(self.client.get("/api/status").status_code, 200)
        self.assertEqual(self.client.get("/api/session").json()["csrf"], csrf)
        self.assertEqual(self.client.post("/api/render").status_code, 403)
        self.assertEqual(self.client.post("/api/render", headers={"X-CSRF-Token": "nope"}).status_code, 403)
        self.assertEqual(self.client.post("/api/render", headers={"X-CSRF-Token": csrf}).status_code, 200)
        self.assertEqual(self.client.delete("/api/installed/o/r", params={"confirm": "o/r"}).status_code, 403)
        self.client.post("/api/logout", headers={"X-CSRF-Token": csrf})
        self.assertEqual(self.client.get("/api/status").status_code, 401)

    def test_bearer_token_needs_no_csrf(self):
        headers = {"Authorization": f"Bearer {TOKEN}"}
        self.assertEqual(self.client.post("/api/render", headers=headers).status_code, 200)

    def test_login_rate_limit(self):
        for _ in range(5):
            self.assertEqual(self.client.post("/api/login", json={"token": "wrong"}).status_code, 401)
        self.assertEqual(self.client.post("/api/login", json={"token": "wrong"}).status_code, 429)
        self.assertEqual(self.client.post("/api/login", json={"token": TOKEN}).status_code, 429)

    def test_rejects_unsafe_input(self):
        csrf = self.login()
        h = {"X-CSRF-Token": csrf}
        r = self.client.post("/api/downloads", headers=h,
                             json={"repo": "../etc", "revision": "a" * 40, "files": ["x.gguf"]})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/downloads", headers=h,
                             json={"repo": "o/r", "revision": "main", "files": ["x.gguf"]})
        self.assertEqual(r.status_code, 400)
        r = self.client.post("/api/downloads", headers=h,
                             json={"repo": "o/r", "revision": "a" * 40, "files": ["../../x.gguf"]})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.client.delete("/api/installed/o/r", headers=h, params={"confirm": "x"}).status_code, 400)
        self.assertEqual(self.client.get("/api/repo/o/a..b").status_code, 400)

    def test_catalog_and_status(self):
        self.login()
        status = self.client.get("/api/status").json()
        self.assertEqual(status["groups"], ["on-demand", "resident"])
        self.assertEqual(status["budget"]["available_gb"], 118)
        items = self.client.get("/api/catalog").json()["items"]
        self.assertTrue(all(i["fit"] in ("fits", "tight", "too-big") for i in items))
        big = next(i for i in items if i["repo"] == "ggml-org/gpt-oss-120b-GGUF")
        self.assertIn(big["fit"], ("fits", "tight"))


@network
class FetcherNetworkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = fetcher.Store(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_download_verify_and_tamper_detection(self):
        snap = fetcher.resolve(TINY_REPO, TINY_COMMIT)
        self.assertEqual(snap.commit, TINY_COMMIT)
        files, _ = snap.by_name(), None
        tiny, text = files[TINY_FILE], files[TINY_TEXT]
        self.assertEqual((tiny.algo, tiny.size), ("sha256", 1185376))
        self.assertEqual(text.algo, "git-sha1")
        seen = []
        self.assertEqual(self.store.fetch_file(TINY_REPO, snap.commit, tiny, progress=seen.append), "downloaded")
        self.assertEqual(self.store.fetch_file(TINY_REPO, snap.commit, text), "downloaded")
        self.assertEqual(self.store.fetch_file(TINY_REPO, snap.commit, tiny), "skipped")
        target = self.store.target(TINY_REPO, TINY_FILE)
        self.assertEqual(os.path.getsize(target), 1185376)
        manifest = self.store.load_manifest()
        self.assertEqual(manifest[f"{TINY_REPO}/{TINY_FILE}"]["sha256"], tiny.expected)
        with open(os.path.join(self.tmp, "models.sha256"), encoding="utf-8") as h:
            self.assertIn(f"{tiny.expected}  {TINY_REPO}/{TINY_FILE}", h.read())
        self.assertTrue(self.store.verify(TINY_REPO, TINY_FILE))
        self.assertTrue(self.store.verify(TINY_REPO, TINY_TEXT))
        # Tamper with the installed file: re-verification catches it.
        with open(target, "r+b") as h:
            h.seek(1000)
            h.write(b"\xff\xfe")
        self.assertFalse(self.store.verify(TINY_REPO, TINY_FILE))
        # A forced re-check re-downloads and restores the verified file.
        self.assertEqual(self.store.fetch_file(TINY_REPO, snap.commit, tiny, force_verify=True), "downloaded")
        self.assertTrue(self.store.verify(TINY_REPO, TINY_FILE))
        self.assertEqual(os.listdir(os.path.join(self.tmp, ".staging")), [])

    def test_hash_mismatch_discards_the_download(self):
        snap = fetcher.resolve(TINY_REPO, TINY_COMMIT)
        real = snap.by_name()[TINY_FILE]
        forged = fetcher.RemoteFile(real.name, real.size, "0" * 64, "sha256")
        with self.assertRaises(fetcher.FetchError) as ctx:
            self.store.fetch_file(TINY_REPO, snap.commit, forged)
        self.assertIn("mismatch", str(ctx.exception))
        self.assertFalse(os.path.exists(self.store.target(TINY_REPO, TINY_FILE)))
        self.assertEqual(os.listdir(os.path.join(self.tmp, ".staging")), [])

    def test_resume_from_partial_file(self):
        snap = fetcher.resolve(TINY_REPO, TINY_COMMIT)
        tiny = snap.by_name()[TINY_FILE]
        part = self.store.part_path(TINY_REPO, TINY_FILE)
        os.makedirs(os.path.dirname(part), exist_ok=True)
        # The first 500 kB, fetched separately, as a cancelled download would leave them.
        req = urllib.request.Request(paths.resolve_url(TINY_REPO, TINY_COMMIT, TINY_FILE),
                                     headers={"Range": "bytes=0-499999"})
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
            out.write(resp.read())
        self.assertEqual(self.store.staged_bytes(TINY_REPO, TINY_FILE), 500000)
        progress = []
        self.store.fetch_file(TINY_REPO, snap.commit, tiny, progress=progress.append)
        self.assertTrue(self.store.verify(TINY_REPO, TINY_FILE))

    def test_cli_fetch_and_render(self):
        from model_hub import cli
        config = os.path.join(self.tmp, "config")
        os.makedirs(config)
        shutil.copy(os.path.join(_STANDALONE, "config", "llama-swap.base.example.yaml"),
                    os.path.join(config, "llama-swap.base.yaml"))
        models_yml = os.path.join(config, "models.yml")
        registry.save(models_yml, [{"repo": TINY_REPO, "revision": TINY_COMMIT, "include": [TINY_FILE, "*.bin"],
                                    "serve": {"id": "tiny", "model": TINY_FILE, "ctx_size": 256}}])
        models = os.path.join(self.tmp, "models")
        os.makedirs(models)
        self.assertEqual(cli.main(["--config", models_yml, "--models", models]), 0)  # legacy flags = fetch
        self.assertTrue(os.path.exists(os.path.join(models, TINY_REPO, TINY_FILE)))
        out = os.path.join(config, "llama-swap.yaml")
        self.assertEqual(cli.main(["render", "--base", os.path.join(config, "llama-swap.base.yaml"),
                                   "--config", models_yml, "--out", out, "--models", models]), 0)
        with open(out, encoding="utf-8") as h:
            rendered = yaml.safe_load(h)
        self.assertIn(f"--model ${{models}}/{TINY_REPO}/{TINY_FILE}", rendered["models"]["tiny"]["cmd"])


@network
class AppDownloadNetworkTests(_AppCase):
    def test_download_through_the_api_maps_the_model(self):
        csrf = self.login()
        h = {"X-CSRF-Token": csrf}
        detail = self.client.get(f"/api/repo/{TINY_REPO}", params={"revision": TINY_COMMIT, "ctx": 512}).json()
        self.assertEqual(detail["commit"], TINY_COMMIT)
        keys = {g["key"]: g for g in detail["ggufs"]}
        self.assertIn(TINY_FILE, keys)
        self.assertIn("tinyllamas/split/stories15M-*-of-00003.gguf", keys)
        self.assertEqual(keys["tinyllamas/split/stories15M-*-of-00003.gguf"]["parts"], 3)
        serve = {"id": "tiny", "model": TINY_FILE, "ctx_size": 256, "group": "on-demand", "ttl": 60}
        r = self.client.post("/api/serve/preview", headers=h, json={"repo": TINY_REPO, "serve": serve})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertIn("+  tiny:", r.json()["diff"])
        r = self.client.post("/api/downloads", headers=h,
                             json={"repo": TINY_REPO, "revision": TINY_COMMIT, "files": [TINY_FILE], "serve": serve})
        self.assertEqual(r.status_code, 200, r.text)
        job_id = r.json()["id"]
        for _ in range(240):
            job = next(j for j in self.client.get("/api/downloads").json()["jobs"] if j["id"] == job_id)
            if job["state"] not in ("queued", "running"):
                break
            time.sleep(0.25)
        self.assertEqual(job["state"], "done", job)
        with open(os.path.join(self.config, "llama-swap.yaml"), encoding="utf-8") as fh:
            rendered = yaml.safe_load(fh)
        self.assertEqual(render.validate(rendered), [])
        self.assertIn("tiny", rendered["models"])
        self.assertIn("tiny", rendered["routing"]["router"]["settings"]["groups"]["on-demand"]["members"])
        entries = registry.load(os.path.join(self.config, "models.yml"))
        self.assertEqual(entries[0]["revision"], TINY_COMMIT)
        installed = self.client.get("/api/installed").json()["repos"]
        self.assertTrue(installed[0]["mapped"])
        # Delete needs the repo id repeated; then files and mapping are gone.
        r = self.client.delete(f"/api/installed/{TINY_REPO}", headers=h, params={"confirm": TINY_REPO})
        self.assertEqual(r.status_code, 200, r.text)
        self.assertFalse(os.path.exists(os.path.join(self.models, TINY_REPO)))
        with open(os.path.join(self.config, "llama-swap.yaml"), encoding="utf-8") as fh:
            self.assertNotIn("tiny", yaml.safe_load(fh)["models"])

    def test_search_trusted_only(self):
        self.login()
        data = self.client.get("/api/search", params={"q": "gpt-oss", "limit": 5}).json()
        self.assertTrue(data["trusted_only"])
        self.assertTrue(data["results"])
        self.assertTrue(all(r["trusted"] for r in data["results"]))


@network
class CatalogNetworkTests(unittest.TestCase):
    def test_catalog_entries_match_the_hub(self):
        for item in catalog.load(os.path.join(_HUB_DIR, "catalog.yml")):
            with self.subTest(repo=item["repo"]):
                snap = fetcher.resolve(item["repo"], item["revision"])
                self.assertEqual(snap.commit, item["revision"])
                files, refused = fetcher.select(snap, item["include"])
                self.assertEqual(refused, [])
                self.assertEqual(len(files), len(item["include"]))
                self.assertAlmostEqual(sum(f.size for f in files) / 1e9, item["size_gb"], delta=0.02)
                names = {f.name for f in files}
                serve = item["serve"]
                self.assertIn(serve["model"], names)
                if serve.get("mmproj"):
                    self.assertIn(serve["mmproj"], names)



class BundledExampleTests(unittest.TestCase):
    def test_bundled_examples_match_the_standalone_stack(self):
        for _, example in registry.EXAMPLES:
            with open(os.path.join(registry.BUNDLED_EXAMPLES_DIR, example), encoding="utf-8") as bundled, \
                    open(os.path.join(_STANDALONE, "config", example), encoding="utf-8") as standalone:
                self.assertEqual(bundled.read(), standalone.read(), example)

    def test_seed_falls_back_to_bundled_examples(self):
        with tempfile.TemporaryDirectory() as config_dir:
            with open(os.path.join(config_dir, "models.yml"), "w", encoding="utf-8") as handle:
                handle.write("[]\n")
            created = registry.seed_config(config_dir)
            self.assertEqual(created, ["llama-swap.base.yaml"])
            with open(os.path.join(config_dir, "models.yml"), encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "[]\n")


if __name__ == "__main__":
    unittest.main()
