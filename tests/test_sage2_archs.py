"""Which GPU architectures may use Maestro's SageAttention 2 path."""
from __future__ import annotations

import importlib.util
import os
import unittest

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))


def _load_sage2_archs():
    """shared/sage2_archs.py without importing torch through the package."""
    path = os.path.join(_APP_DIR, "shared", "sage2_archs.py")
    spec = importlib.util.spec_from_file_location("maestro_sage2_archs", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sage2_archs = _load_sage2_archs()


class Sage2ArchTests(unittest.TestCase):
    def test_builtin_archs_without_declaration(self):
        self.assertEqual(sage2_archs.supported_archs({}), {(8, 0), (8, 6), (8, 9)})
        self.assertEqual(sage2_archs.extra_archs({}), frozenset())

    def test_the_spark_image_declares_blackwell(self):
        env = {sage2_archs.EXTRA_ARCHS_ENV: "12.0,12.1"}
        self.assertEqual(sage2_archs.extra_archs(env), {(12, 0), (12, 1)})
        self.assertIn((12, 1), sage2_archs.supported_archs(env))

    def test_lenient_parsing(self):
        env = {sage2_archs.EXTRA_ARCHS_ENV: "sm_120; 121, 8.9, junk, , 9"}
        self.assertEqual(sage2_archs.extra_archs(env), {(12, 0), (12, 1)})

    def test_sm121_runs_the_sm120_kernels(self):
        self.assertEqual(sage2_archs.dispatch_arch("sm121"), "sm120")
        self.assertEqual(sage2_archs.dispatch_arch("sm89"), "sm89")


if __name__ == "__main__":
    unittest.main()
