import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from services.remote_worker import kernel_cache  # noqa: E402


class FakeR2:
    """Object URLs map to files of a local folder."""

    def __init__(self, folder, readonly=False):
        self.folder = folder
        self.readonly = readonly
        self.uploads = 0

    def object_url(self, *parts):
        return "s3://bucket/maestro/" + "/".join(parts)

    def _path(self, url):
        return os.path.join(self.folder, url.split("s3://bucket/maestro/", 1)[1])

    def copy(self, source, destination, *, timeout=None):
        if source.startswith("s3://"):
            if not os.path.isfile(self._path(source)):
                return False
            shutil.copyfile(self._path(source), destination)
            return True
        os.makedirs(os.path.dirname(self._path(destination)), exist_ok=True)
        shutil.copyfile(source, self._path(destination))
        self.uploads += 1
        return True


def _write(path, data=b"k"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(data)


class KernelCacheTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.r2 = FakeR2(os.path.join(self.tmp, "r2"))

    def _pod(self, name, r2=None):
        return kernel_cache.KernelCache(r2 or self.r2, "sm_89-torch2.10-triton3.6-py310",
                                        os.path.join(self.tmp, name))

    def test_kernels_compiled_on_one_pod_are_pulled_by_the_next(self):
        first = self._pod("pod1")
        self.assertEqual(first.pull(), 0)
        self.assertFalse(first.publish_if_grown())
        _write(os.path.join(self.tmp, "pod1", "triton", "abc", "int8_matmul.cubin"))
        self.assertTrue(first.publish_if_grown())
        self.assertFalse(first.publish_if_grown())  # nothing new since
        second = self._pod("pod2")
        self.assertEqual(second.pull(), 1)
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "pod2", "triton", "abc", "int8_matmul.cubin")))

    def test_a_read_only_host_never_publishes(self):
        pod = self._pod("pod", FakeR2(os.path.join(self.tmp, "r2"), readonly=True))
        _write(os.path.join(self.tmp, "pod", "triton", "abc", "k.cubin"))
        self.assertFalse(pod.publish_if_grown())

    def test_each_card_and_version_has_its_own_archive(self):
        self.assertEqual(kernel_cache.cache_key(""), "")
        self.assertTrue(kernel_cache.cache_key("sm_80").startswith("sm_80-torch"))
        self.assertNotEqual(kernel_cache.cache_key("sm_80"), kernel_cache.cache_key("sm_89"))


if __name__ == "__main__":
    unittest.main()
