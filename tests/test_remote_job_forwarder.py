"""RemoteJobForwarder tests with an in-memory fake worker (no network, no GPU)."""
from __future__ import annotations

import json
import io
import os
import shutil
import sys
import tempfile
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from services.remote_worker.job_forwarder import (  # noqa: E402
    RemoteJobCancelled,
    RemoteJobError,
    RemoteJobForwarder,
)


class FakeManager:
    def __init__(self):
        self.started = 0
        self.finished = 0
        self.ready_calls = []

        class S:
            token = "tok"
            worker_token = "tok"

        self.settings = S()

    def job_started(self):
        self.started += 1

    def job_finished(self):
        self.finished += 1

    def ensure_ready(self, model_type, *, should_abort, on_update):
        self.ready_calls.append(model_type)
        on_update({"state": "booting"})
        return {"state": "ready", "service_url": "http://worker/"}


class FakeWorker:
    """Scripted worker: download status, file store, one job with a timeline."""

    def __init__(self):
        self.files = {}  # worker path -> sha
        self.uploads = []
        self.submitted = None
        self.cancelled = False
        self.download_status = {}
        self.download_requests = []
        self.timeline = []
        self.final = {"status": "completed", "output_files": ["clip.mp4"], "files": ["clip.mp4"],
                      "sidecars": ["clip.json"], "clip_output_files": {}, "join_output_file": None}
        self.contents = {"clip.mp4": b"video", "clip.json": b"{}"}
        self.submit_body = None
        self.previews = {}  # url -> (bytes, content type)
        self.preview_fetches = []

    # WorkerClient surface
    def get_json(self, path, *, timeout=30.0, params=None):
        if path == "api/v1/models/downloads/status":
            return {"downloads": self.download_status}
        if path == "api/v1/worker/files":
            return None
        if path.startswith("api/v1/worker/jobs/"):
            if self.cancelled:
                return {"status": "cancelled"}
            if self.timeline:
                return self.timeline.pop(0)
            return self.final
        raise AssertionError(path)

    def post_json(self, path, body=None, *, timeout=60.0, retry=True):
        if path.startswith("api/v1/models/"):
            model = path.split("/")[3]
            self.download_requests.append(model)
            self.download_status[model] = {"status": "completed"}
            return {"status": "downloading"}
        if path == "api/v1/worker/jobs":
            self.submit_body = body
            self.submitted = body["params"]
            return {"job_id": "r1"}
        if path.startswith("api/v1/cancel/"):
            self.cancelled = True
            return {}
        raise AssertionError(path)

    def upload_file(self, local_path, *, kind, sha256, model_type=""):
        remote = f"/workspace/data/{kind}/{sha256[:8]}/{os.path.basename(local_path)}"
        self.uploads.append((kind, os.path.basename(local_path), model_type))
        self.files[remote] = sha256
        return remote

    def get_bytes(self, path, *, timeout=30.0):
        self.preview_fetches.append(path)
        return self.previews.get(path)

    def download(self, path, dest):
        name = path.rsplit("/", 1)[1]
        with open(dest, "wb") as handle:
            handle.write(self.contents[name])


class ForwarderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.out_dir = os.path.join(self.tmp, "outputs")
        self.manager = FakeManager()
        self.worker = FakeWorker()
        self.forwarder = RemoteJobForwarder(
            self.manager,
            client_factory=lambda url, token: self.worker,
            poll_interval=0,
            sleep=lambda s: None,
        )
        self.updates = []
        self.cancel = False

    def _file(self, name, data=b"x"):
        path = os.path.join(self.tmp, name)
        with open(path, "wb") as handle:
            handle.write(data)
        return path

    def _run(self, params, resolve_lora=lambda m, n: None):
        return self.forwarder.run(
            params,
            out_dir=self.out_dir,
            resolve_lora=resolve_lora,
            update=lambda **u: self.updates.append(u),
            is_cancelled=lambda: self.cancel,
        )

    def test_r2_only_lora_is_fetched_by_the_worker(self):
        weights = os.path.join(self.tmp, "remote_style.safetensors")
        with open(os.path.join(self.tmp, "remote_style.r2.json"), "w") as handle:
            json.dump({"filename": "remote_style.safetensors", "key": "maestro/assets/abc/remote_style.safetensors",
                       "sha256": "a" * 64}, handle)
        fetched = []
        self.worker.fetch_from_r2 = lambda **kw: fetched.append(kw) or "/workspace/data/lora/remote_style.safetensors"
        self._run({"model_type": "flux", "prompt": "x", "activated_loras": ["remote_style.safetensors"]},
                  resolve_lora=lambda m, n: weights)
        self.assertEqual(fetched, [{"kind": "lora", "name": "remote_style.safetensors",
                                    "sha256": "a" * 64, "model_type": "flux"}])
        self.assertFalse([u for u in self.worker.uploads if u[0] == "lora"])

    def test_previews_are_relayed_once_per_revision(self):
        def preview(revision, notice=None):
            return {"status": "running", "step": revision, "total_steps": 3,
                    "preview": {"url": f"/api/v1/jobs/r1/preview/{revision}", "kind": "video",
                                "revision": revision, "mode": "tiny_vae_video", "window": 1, "total_windows": 1},
                    "preview_notice": notice}
        self.worker.previews = {"/api/v1/jobs/r1/preview/1": (b"v1", "video/mp4"),
                                "/api/v1/jobs/r1/preview/2": (b"v2", "video/mp4")}
        self.worker.timeline = [preview(1), preview(1), preview(2), preview(2, "Previews paused")]
        relayed, notices = [], []
        self.forwarder.run(
            {"model_type": "ltx2_22B_distilled_1_1", "prompt": "a cat"},
            out_dir=self.out_dir,
            resolve_lora=lambda m, n: None,
            update=lambda **u: None,
            is_cancelled=lambda: False,
            preview_mode="tiny_vae_video",
            on_preview=lambda data, ctype, info: relayed.append((data, ctype, info["revision"])),
            on_preview_notice=notices.append,
        )
        self.assertEqual(self.worker.submit_body["preview_mode"], "tiny_vae_video")
        self.assertEqual(relayed, [(b"v1", "video/mp4", 1), (b"v2", "video/mp4", 2)])
        self.assertEqual(len(self.worker.preview_fetches), 2)
        self.assertEqual(notices, ["Previews paused"])

    def test_preview_failures_do_not_fail_the_job(self):
        self.worker.timeline = [{"status": "running", "preview": {"url": "/p/1", "revision": 1}}]

        def boom(path, *, timeout=30.0):
            raise RuntimeError("proxy hiccup")
        self.worker.get_bytes = boom
        result = self.forwarder.run(
            {"model_type": "ltx2_22B_distilled_1_1", "prompt": "a cat"},
            out_dir=self.out_dir, resolve_lora=lambda m, n: None,
            update=lambda **u: None, is_cancelled=lambda: False,
            on_preview=lambda *a: None,
        )
        self.assertEqual(len(result.output_files), 1)

    def test_a_multi_repeat_job_returns_every_output_in_order_with_its_sidecar(self):
        self.worker.final = {"status": "completed", "output_files": ["s1.png", "s2.png", "s3.png"],
                             "files": ["s1.png", "s2.png", "s3.png"], "sidecars": ["s1.json", "s3.json"],
                             "clip_output_files": {}, "join_output_file": None}
        self.worker.contents = {"s1.png": b"1", "s2.png": b"2", "s3.png": b"3",
                                "s1.json": b'{"seed": 10}', "s3.json": b'{"seed": 12}'}
        result = self._run({"model_type": "qwen_image_21_7B", "prompt": "a shot", "repeat_generation": 3})
        self.assertEqual(self.worker.submitted["repeat_generation"], 3)
        self.assertEqual(result.output_files, ["s1.png", "s2.png", "s3.png"])
        self.assertEqual(sorted(os.listdir(self.out_dir)), ["s1.json", "s1.png", "s2.png", "s3.json", "s3.png"])
        with open(os.path.join(self.out_dir, "s3.json"), "rb") as handle:
            self.assertEqual(json.load(handle), {"seed": 12})

    def test_no_preview_mode_keeps_the_worker_default(self):
        self._run({"model_type": "ltx2_22B_distilled_1_1", "prompt": "a cat"})
        self.assertNotIn("preview_mode", self.worker.submit_body)

    def test_full_round_trip(self):
        image = self._file("start.png")
        ref = self._file("ref.wav")
        lora = self._file("style.safetensors")
        self.worker.timeline = [{"status": "running", "progress": 40, "step": 4, "total_steps": 10,
                                 "message": "Denoising", "phase": "Denoising"}]
        params = {
            "model_type": "minimax_h3_frames",
            "prompt": start_prompt(image),
            "image_start": image,
            "minimax_h3_references": [{"path": ref, "role": "voice"}],
            "activated_loras": ["style.safetensors", "https://civitai.com/x.safetensors"],
            "_defer_output_publication": True,
            "video_length": 97,
        }
        result = self._run(params, resolve_lora=lambda m, n: lora if n == "style.safetensors" else None)

        sent = self.worker.submitted
        self.assertTrue(sent["image_start"].startswith("/workspace/data/input/"))
        self.assertTrue(sent["minimax_h3_references"][0]["path"].endswith("/ref.wav"))
        self.assertEqual(sent["prompt"], params["prompt"])  # free text is never rewritten
        self.assertNotIn("_defer_output_publication", sent)
        self.assertEqual(sent["video_length"], 97)
        self.assertIn(("lora", "style.safetensors", "minimax_h3_frames"), self.worker.uploads)
        self.assertEqual(sum(1 for u in self.worker.uploads if u[0] == "lora"), 1)
        self.assertEqual(self.worker.download_requests, ["minimax_h3_frames"])

        self.assertEqual(result.output_files, ["clip.mp4"])
        with open(os.path.join(self.out_dir, "clip.mp4"), "rb") as handle:
            self.assertEqual(handle.read(), b"video")
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, "clip.json")))
        self.assertIn({"progress": 40, "step": 4, "total_steps": 10, "message": "Denoising", "phase": "Denoising"},
                      self.updates)
        self.assertEqual((self.manager.started, self.manager.finished), (1, 1))

    def test_same_file_is_uploaded_once(self):
        image = self._file("start.png")
        self._run({"model_type": "m", "image_start": image, "image_end": [image]})
        self.assertEqual([u for u in self.worker.uploads if u[0] == "input"], [("input", "start.png", "")])

    def test_preloaded_model_is_not_requested_again(self):
        self.worker.download_status = {"m": {"status": "completed"}}
        self._run({"model_type": "m"})
        self.assertEqual(self.worker.download_requests, [])

    def test_output_name_collision_is_avoided(self):
        os.makedirs(self.out_dir)
        with open(os.path.join(self.out_dir, "clip.mp4"), "wb") as handle:
            handle.write(b"older")
        result = self._run({"model_type": "m"})
        self.assertEqual(result.output_files, ["clip_1.mp4"])
        with open(os.path.join(self.out_dir, "clip.mp4"), "rb") as handle:
            self.assertEqual(handle.read(), b"older")
        self.assertTrue(os.path.isfile(os.path.join(self.out_dir, "clip_1.json")))

    def test_remote_failure_raises(self):
        self.worker.final = {"status": "failed", "error": "CUDA out of memory"}
        with self.assertRaises(RemoteJobError) as ctx:
            self._run({"model_type": "m"})
        self.assertIn("CUDA out of memory", str(ctx.exception))
        self.assertEqual(self.manager.finished, 1)

    def test_local_cancel_cancels_remote_job(self):
        self.worker.timeline = [{"status": "running"}]

        original = self.worker.get_json

        def get_json(path, **kwargs):
            if path.startswith("api/v1/worker/jobs/"):
                self.cancel = True
            return original(path, **kwargs)

        self.worker.get_json = get_json
        with self.assertRaises(RemoteJobCancelled):
            self._run({"model_type": "m"})
        self.assertTrue(self.worker.cancelled)

    def test_unknown_remote_job_is_an_error(self):
        self.worker.get_json = lambda path, **kw: (
            {"downloads": {"m": {"status": "completed"}}} if "downloads" in path else None
        )
        with self.assertRaises(RemoteJobError):
            self._run({"model_type": "m"})


def start_prompt(path):
    # A prompt that happens to contain a real path must stay untouched.
    return f"A cat, see {path}"


if __name__ == "__main__":
    unittest.main()


class LiveClipRelayTests(unittest.TestCase):
    """Finished clips of a running multi-clip job reach the controller mid-job."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.out_dir = os.path.join(self.tmp, "outputs")
        self.worker = FakeWorker()
        self.forwarder = RemoteJobForwarder(
            FakeManager(),
            client_factory=lambda url, token: self.worker,
            poll_interval=0,
            sleep=lambda s: None,
        )
        self.worker.contents = {"c0.mp4": b"clip0", "c1.mp4": b"clip1", "c1.json": b"{}"}
        self.worker.timeline = [
            {"status": "running", "live_clip_output_files": {"0": "c0.mp4"}},
            {"status": "running", "live_clip_output_files": {"0": "c0.mp4", "1": "c1.mp4"}},
        ]
        self.worker.final = {
            "status": "completed", "output_files": ["c0.mp4", "c1.mp4"],
            "files": ["c0.mp4", "c1.mp4"], "sidecars": ["c1.json"],
            "clip_output_files": {"0": "c0.mp4", "1": "c1.mp4"}, "join_output_file": None,
        }
        self.seen = []

    def _run(self, on_clip_output):
        return self.forwarder.run(
            {"model_type": "minimax_h3"},
            out_dir=self.out_dir,
            resolve_lora=lambda m, n: None,
            update=lambda **u: None,
            is_cancelled=lambda: False,
            on_clip_output=on_clip_output,
        )

    def test_each_finished_clip_is_downloaded_and_announced_once(self):
        def on_clip(index, name):
            # The file is on disk when the controller is told about it.
            with open(os.path.join(self.out_dir, name), "rb") as handle:
                self.seen.append((index, name, handle.read()))

        result = self._run(on_clip)
        self.assertEqual(self.seen, [("0", "c0.mp4", b"clip0"), ("1", "c1.mp4", b"clip1")])
        # The final download reuses the live names instead of adding c0_1.mp4.
        self.assertEqual(result.clip_output_files, {"0": "c0.mp4", "1": "c1.mp4"})
        self.assertEqual(sorted(os.listdir(self.out_dir)), ["c0.mp4", "c1.json", "c1.mp4"])

    def test_the_final_copy_replaces_a_clip_changed_after_its_live_fetch(self):
        self.worker.timeline = [{"status": "running", "live_clip_output_files": {"0": "c0.mp4"}}]
        original_download = self.worker.download

        def download(path, dest):
            original_download(path, dest)
            self.worker.contents["c0.mp4"] = b"final"

        self.worker.download = download
        self._run(lambda index, name: None)
        with open(os.path.join(self.out_dir, "c0.mp4"), "rb") as handle:
            self.assertEqual(handle.read(), b"final")

    def test_a_failed_live_fetch_never_fails_the_job(self):
        attempts = []
        original_download = self.worker.download

        def download(path, dest):
            attempts.append(path)
            if len(attempts) == 1:
                raise RemoteJobError("proxy hiccup")
            original_download(path, dest)

        self.worker.download = download
        result = self._run(lambda index, name: self.seen.append((index, name)))
        self.assertEqual(result.clip_output_files, {"0": "c0.mp4", "1": "c1.mp4"})
        # The clip that failed on its first poll is retried on the next one.
        self.assertEqual(self.seen, [("0", "c0.mp4"), ("1", "c1.mp4")])

    def test_without_a_callback_nothing_is_fetched_mid_job(self):
        downloads = []
        original_download = self.worker.download
        self.worker.download = lambda path, dest: (downloads.append(path), original_download(path, dest))
        self._run(None)
        # Only the final download of the two clips and the sidecar.
        self.assertEqual(len(downloads), 3)


class WorkerClientRetryTests(unittest.TestCase):
    def _client(self, statuses):
        import requests

        from services.remote_worker.job_forwarder import WorkerClient

        calls = []

        class Session:
            def request(self, method, url, **kwargs):
                calls.append((method, url))
                response = requests.Response()
                response.status_code = statuses[min(len(calls), len(statuses)) - 1]
                response._content = b'{"path": "/inputs/x.png", "job_id": "j1"}'
                response.raw = io.BytesIO()
                return response

        sleeps = []
        return WorkerClient("https://w/", "t", session=Session(), sleep=sleeps.append), calls, sleeps

    def test_a_proxy_hiccup_is_sent_again(self):
        client, calls, sleeps = self._client([502, 502, 200])
        self.assertEqual(client.fetch_from_r2(kind="input", name="x.png", sha256="a" * 64), "/inputs/x.png")
        self.assertEqual((len(calls), sleeps), (3, [2.0, 5.0]))

    def test_a_job_submission_is_never_sent_twice(self):
        client, calls, _ = self._client([502, 200])
        with self.assertRaises(Exception):
            client.post_json("api/v1/worker/jobs", {}, retry=False)
        self.assertEqual(len(calls), 1)
