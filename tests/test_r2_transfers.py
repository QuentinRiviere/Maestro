"""R2 transfer path: R2Store keys/transfers and the forwarder's R2 routing."""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile
import unittest


_HERE = os.path.dirname(os.path.abspath(__file__))
_APP_DIR = os.path.abspath(os.path.join(_HERE, "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)
sys.path.insert(0, _HERE)

from services.remote_worker.job_forwarder import RemoteJobForwarder  # noqa: E402
from services.remote_worker.r2_store import R2Store, R2StoreError  # noqa: E402
from test_remote_job_forwarder import FakeManager, FakeWorker  # noqa: E402


class _NotFound(Exception):
    response = {"Error": {"Code": "404"}}


class FakeS3:
    """In-memory bucket implementing the boto3 calls R2Store makes."""

    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}
        self.uploads = []

    def head_object(self, Bucket, Key):
        if (Bucket, Key) not in self.objects:
            raise _NotFound()
        return {}

    def upload_file(self, Filename, Bucket, Key):
        with open(Filename, "rb") as handle:
            self.objects[(Bucket, Key)] = handle.read()
        self.uploads.append(Key)

    def download_file(self, Bucket, Key, Filename):
        if (Bucket, Key) not in self.objects:
            raise _NotFound()
        with open(Filename, "wb") as handle:
            handle.write(self.objects[(Bucket, Key)])


def _store(s3):
    return R2Store(account_id="a", access_key_id="k", secret_access_key="s",
                   bucket="videogen", prefix="maestro", client=s3)


class R2StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_keys_are_content_addressed_and_sanitized(self):
        store = _store(FakeS3())
        sha = "a" * 64
        self.assertEqual(store.asset_key(sha, "../my ref.png"), f"maestro/assets/{sha}/my_ref.png")
        self.assertEqual(store.output_key("job1", "clip.mp4"), "maestro/outputs/job1/clip.mp4")
        with self.assertRaises(R2StoreError):
            store.asset_key("nothex", "x.png")

    def test_upload_skips_existing_and_download_roundtrips(self):
        s3 = FakeS3()
        store = _store(s3)
        path = os.path.join(self.tmp, "a.bin")
        with open(path, "wb") as handle:
            handle.write(b"data")
        store.upload(path, "maestro/assets/x/a.bin")
        store.upload(path, "maestro/assets/x/a.bin")
        self.assertEqual(s3.uploads, ["maestro/assets/x/a.bin"])
        dest = os.path.join(self.tmp, "out", "a.bin")
        store.download("maestro/assets/x/a.bin", dest)
        with open(dest, "rb") as handle:
            self.assertEqual(handle.read(), b"data")
        with self.assertRaises(R2StoreError):
            store.download("maestro/assets/missing", os.path.join(self.tmp, "m"))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "m.part")))

    def test_from_environment_requires_all_credentials(self):
        self.assertIsNone(R2Store.from_environment({"R2_ACCOUNT_ID": "a"}))


class R2Worker(FakeWorker):
    """Fake worker that pulls inputs from and publishes outputs to the bucket."""

    def __init__(self, store, s3, *, has_r2=True):
        super().__init__()
        self.store, self.s3, self.has_r2 = store, s3, has_r2
        self.fetched = []
        self.published = None
        self.proxy_downloads = []

    def fetch_from_r2(self, *, kind, name, sha256, model_type=""):
        if not self.has_r2:
            return None
        key = self.store.asset_key(sha256, name)
        assert (self.store.bucket, key) in self.s3.objects, key
        self.fetched.append(key)
        return f"/workspace/data/{kind}/{sha256[:16]}/{name}"

    def post_json(self, path, body=None, *, timeout=60.0):
        if path.endswith("/publish"):
            self.published = body
            objects = {}
            for name in self.final["files"] + self.final["sidecars"]:
                key = self.store.output_key(body["job_id"], name)
                self.s3.objects[(self.store.bucket, key)] = self.contents[name]
                objects[name] = key
            return {"objects": objects}
        return super().post_json(path, body, timeout=timeout)

    def download(self, path, dest):
        self.proxy_downloads.append(path)
        super().download(path, dest)


class ForwarderR2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.s3 = FakeS3()
        self.store = _store(self.s3)
        self.manager = FakeManager()
        self.out_dir = os.path.join(self.tmp, "outputs")
        self.image = os.path.join(self.tmp, "ref.png")
        with open(self.image, "wb") as handle:
            handle.write(b"png-bytes")

    def _forwarder(self, worker):
        return RemoteJobForwarder(
            self.manager,
            client_factory=lambda url, token: worker,
            store_factory=lambda settings: self.store,
            poll_interval=0,
            sleep=lambda s: None,
        )

    def _run(self, forwarder, job_id="local1"):
        return forwarder.run(
            {"model_type": "m", "image_start": self.image},
            job_id=job_id,
            out_dir=self.out_dir,
            resolve_lora=lambda m, n: None,
            update=lambda **u: None,
            is_cancelled=lambda: False,
        )

    def test_inputs_go_through_r2_once_and_outputs_stay_in_r2(self):
        worker = R2Worker(self.store, self.s3)
        result = self._run(self._forwarder(worker))

        sha = hashlib.sha256(b"png-bytes").hexdigest()
        asset_key = f"maestro/assets/{sha}/ref.png"
        self.assertEqual(worker.fetched, [asset_key])
        self.assertTrue(worker.submitted["image_start"].startswith("/workspace/data/input/"))
        self.assertEqual(worker.uploads, [])  # nothing through the proxy
        self.assertEqual(worker.published, {"job_id": "local1"})
        self.assertEqual(worker.proxy_downloads, [])
        self.assertIn(("videogen", "maestro/outputs/local1/clip.mp4"), self.s3.objects)
        self.assertEqual(result.r2_objects["clip.mp4"], "maestro/outputs/local1/clip.mp4")
        with open(os.path.join(self.out_dir, "clip.mp4"), "rb") as handle:
            self.assertEqual(handle.read(), b"video")

        # A second job (new pod or not) does not upload the same reference again.
        uploads_before = list(self.s3.uploads)
        self._run(self._forwarder(R2Worker(self.store, self.s3)), job_id="local2")
        self.assertEqual([k for k in self.s3.uploads if k not in uploads_before
                          and k.startswith("maestro/assets/")], [])

    def test_worker_without_r2_falls_back_to_the_proxy(self):
        worker = R2Worker(self.store, self.s3, has_r2=False)
        worker.post_json = lambda path, body=None, timeout=60.0: (
            (_ for _ in ()).throw(__import__("services.remote_worker.job_forwarder", fromlist=["x"]).RemoteJobError("409"))
            if path.endswith("/publish") else FakeWorker.post_json(worker, path, body, timeout=timeout)
        )
        self._run(self._forwarder(worker))
        self.assertEqual([u[0] for u in worker.uploads], ["input"])
        self.assertTrue(worker.proxy_downloads)


if __name__ == "__main__":
    unittest.main()
