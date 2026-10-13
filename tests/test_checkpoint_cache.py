"""R2 checkpoint cache tests with a fake s5cmd backed by a local directory."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.remote_worker.checkpoint_cache import (  # noqa: E402
    CheckpointCacheError,
    R2CheckpointCache,
)


class FakeS5:
    """Implements ``cat``, ``cp`` and ``run`` against ``<root>/<bucket>/...``."""

    def __init__(self, root):
        self.root = root
        self.commands = []
        self.timeouts = []
        self.drop_on_run = set()  # remote paths silently not copied (exit 0)

    def _path(self, url):
        return os.path.join(self.root, url[len("s3://"):]) if url.startswith("s3://") else url

    def _cp(self, src, dst):
        if src.startswith("s3://") and src in self.drop_on_run:
            return
        source = self._path(src)
        if not os.path.exists(source):
            raise FileNotFoundError(src)
        destination = self._path(dst)
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copyfile(source, destination)

    def __call__(self, argv, capture_output=True, text=True, env=None, timeout=None):
        self.timeouts.append(timeout)
        assert argv[1] == "--endpoint-url"
        assert env["AWS_ACCESS_KEY_ID"] == "kid"
        command, args = argv[3], argv[4:]
        self.commands.append((command, args))
        try:
            if command == "cat":
                with open(self._path(args[0])) as handle:
                    return subprocess.CompletedProcess(argv, 0, handle.read(), "")
            if command == "cp":
                self._cp(args[0], args[1])
            elif command == "run":
                with open(args[0]) as handle:
                    for line in handle:
                        parts = shlex.split(line)
                        if parts:
                            self._cp(parts[1], parts[2])
        except FileNotFoundError as exc:
            return subprocess.CompletedProcess(argv, 1, "", f"ERROR: {exc}")
        return subprocess.CompletedProcess(argv, 0, "", "")


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


class CheckpointCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.s5 = FakeS5(os.path.join(self.tmp, "r2"))
        os.environ.update({"R2_ACCESS_KEY_ID": "kid", "R2_SECRET_ACCESS_KEY": "secret"})
        self.addCleanup(lambda: [os.environ.pop(k, None) for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")])

    def _pod(self, name):
        base = os.path.join(self.tmp, name)
        roots = {"ckpts": os.path.join(base, "ckpts"), "hf": os.path.join(base, "hf")}
        for root in roots.values():
            os.makedirs(root, exist_ok=True)
        cache = R2CheckpointCache(
            bucket="videogen", prefix="maestro", endpoint_url="https://acct.r2.example",
            roots=roots, runner=self.s5,
        )
        return cache, roots

    def _download(self, roots):
        """Stand-in for Maestro's own download logic."""
        _write(os.path.join(roots["ckpts"], "h3", "transformer.safetensors"), b"T" * 10)
        _write(os.path.join(roots["ckpts"], "vae.safetensors"), b"V" * 4)
        blob = os.path.join(roots["hf"], "hub", "models--x", "blobs", "abc")
        _write(blob, b"B" * 3)
        link = os.path.join(roots["hf"], "hub", "models--x", "snapshots", "main", "config.json")
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if not os.path.lexists(link):
            os.symlink("../../blobs/abc", link)
        _write(os.path.join(roots["ckpts"], "partial.safetensors.part"), b"x")

    def test_first_pod_publishes_second_pod_hydrates(self):
        first, first_roots = self._pod("pod1")
        self.assertEqual(first.hydrate("minimax_h3"), 0)  # no manifest yet
        before = first.snapshot()
        self._download(first_roots)
        self.assertEqual(first.publish("minimax_h3", before), 3)
        # The upload is bounded so a stalled R2 transfer cannot hang the worker.
        self.assertIn(300, self.s5.timeouts)

        second, roots = self._pod("pod2")
        self.assertEqual(second.hydrate("minimax_h3"), 3)
        with open(os.path.join(roots["ckpts"], "h3", "transformer.safetensors"), "rb") as handle:
            self.assertEqual(handle.read(), b"T" * 10)
        link = os.path.join(roots["hf"], "hub", "models--x", "snapshots", "main", "config.json")
        self.assertTrue(os.path.islink(link))
        with open(link, "rb") as handle:
            self.assertEqual(handle.read(), b"B" * 3)
        self.assertFalse(os.path.exists(os.path.join(roots["ckpts"], "partial.safetensors.part")))

        # Warm pod: Maestro finds everything, nothing new to publish.
        before = second.snapshot()
        self.assertEqual(second.publish("minimax_h3", before), 0)
        self.assertEqual(second.hydrate("minimax_h3"), 0)

    def test_manifest_grows_with_newly_needed_files(self):
        first, roots = self._pod("pod1")
        before = first.snapshot()
        self._download(roots)
        first.publish("ltx25", before)

        second, roots2 = self._pod("pod2")
        second.hydrate("ltx25")
        before = second.snapshot()
        _write(os.path.join(roots2["ckpts"], "new_text_encoder.safetensors"), b"E" * 5)
        self.assertEqual(second.publish("ltx25", before), 1)
        files, _links = second.read_manifest("ltx25")
        self.assertIn("new_text_encoder.safetensors", {entry.path for entry in files})
        self.assertIn("vae.safetensors", {entry.path for entry in files})

    def test_silent_s5cmd_refusal_is_detected(self):
        first, roots = self._pod("pod1")
        before = first.snapshot()
        self._download(roots)
        first.publish("m", before)
        self.s5.drop_on_run.add("s3://videogen/maestro/ckpts/vae.safetensors")
        second, _roots = self._pod("pod2")
        with self.assertRaises(CheckpointCacheError):
            second.hydrate("m")

    def test_manifest_paths_cannot_escape_roots(self):
        _write(os.path.join(self.tmp, "r2", "videogen", "maestro", "manifests", "evil.json"),
               b'{"files": [{"root": "ckpts", "path": "../../etc/passwd", "size": 1},'
               b' {"root": "nope", "path": "a", "size": 1}], "links": []}')
        cache, _roots = self._pod("pod1")
        self.assertEqual(cache.read_manifest("evil"), ([], []))

    def test_disabled_without_environment(self):
        for key in ("R2_ENDPOINT_URL", "MAESTRO_R2_BUCKET"):
            os.environ.pop(key, None)
        self.assertIsNone(R2CheckpointCache.from_environment({"ckpts": self.tmp}))


if __name__ == "__main__":
    unittest.main()
