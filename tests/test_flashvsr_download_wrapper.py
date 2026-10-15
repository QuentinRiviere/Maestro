"""FlashVSR weight download can be wrapped (R2 checkpoint cache on remote workers)."""
import os
import sys

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from postprocessing.flashvsr.wgp_bridge import FlashVSRBridge  # noqa: E402


class _MissingFiles:
    def locate_file(self, _path, error_if_none=True):
        return None


def _bridge():
    return FlashVSRBridge({"flashvsr_mode": FlashVSRBridge.MODE_TINY}, _MissingFiles())


def test_download_runs_through_the_wrapper():
    bridge = _bridge()
    fetched, wrapped = [], []
    bridge.download_wrapper = lambda fetch: (wrapped.append(True), fetch())
    assert bridge.download(lambda **kwargs: fetched.append(kwargs["repoId"])) is True
    assert wrapped == [True]
    assert fetched == ["DeepBeepMeep/Wan2.1"]


def test_download_without_wrapper_fetches_directly():
    fetched = []
    assert _bridge().download(lambda **kwargs: fetched.append(kwargs)) is True
    assert len(fetched) == 1
