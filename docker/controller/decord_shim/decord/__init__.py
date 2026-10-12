"""Minimal decord-compatible reader backed by OpenCV.

decord has no linux/arm64 build, so the CPU controller image installs this
package in its place on Apple Silicon. It covers the API Maestro uses:
``VideoReader(path)`` with ``len()``, indexing, ``get_batch``,
``get_avg_fps`` and ``seek``; ``bridge.set_bridge('native' | 'torch')``;
``NDArray.asnumpy()`` and ``cpu()``. Frames are RGB uint8 ``(H, W, 3)``.
The controller only probes and samples a few frames; heavy decoding happens
on the GPU worker, which ships the real decord.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np

__version__ = "0.0.0+maestro-shim"
_bridge = "native"


class NDArray:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def asnumpy(self) -> np.ndarray:
        return self._array

    @property
    def shape(self) -> tuple[int, ...]:
        return self._array.shape

    def __len__(self) -> int:
        return len(self._array)

    def __array__(self, dtype=None):
        return self._array if dtype is None else self._array.astype(dtype)


def _set_bridge(name: str) -> None:
    global _bridge
    if name not in ("native", "torch"):
        raise ValueError(f"Unsupported decord bridge in the Maestro shim: {name}")
    _bridge = name


bridge = SimpleNamespace(set_bridge=_set_bridge)


def cpu(device_id: int = 0) -> None:
    return None


def gpu(device_id: int = 0) -> None:
    raise RuntimeError("decord GPU decoding is not available in the Maestro shim")


def _wrap(array: np.ndarray) -> Any:
    if _bridge == "torch":
        import torch

        return torch.from_numpy(np.ascontiguousarray(array))
    return NDArray(array)


class VideoReader:
    def __init__(self, uri: Any, ctx: Any = None, width: int = -1, height: int = -1, **_: Any) -> None:
        self._path = os.fspath(uri)
        self._capture = cv2.VideoCapture(self._path)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open video {self._path}")
        self._width = int(width)
        self._height = int(height)
        self._fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        self._count = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._position = 0

    def __len__(self) -> int:
        return self._count

    def __del__(self) -> None:
        capture = getattr(self, "_capture", None)
        if capture is not None:
            capture.release()

    def get_avg_fps(self) -> float:
        return self._fps

    def seek(self, position: int) -> None:
        self._position = max(0, int(position))

    def _read(self, index: int) -> np.ndarray:
        if index < 0:
            index += self._count
        if not 0 <= index < max(self._count, index + 1):
            raise IndexError(index)
        if index != self._position:
            self._capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise IndexError(f"frame {index} out of range for {self._path}")
        self._position = index + 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self._width > 0 and self._height > 0:
            frame = cv2.resize(frame, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return frame

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, slice):
            return self.get_batch(list(range(*item.indices(self._count))))
        return _wrap(self._read(int(item)))

    def get_batch(self, indices: Any) -> Any:
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        frames = [self._read(int(index)) for index in indices]
        if not frames:
            return _wrap(np.zeros((0, 0, 0, 3), dtype=np.uint8))
        return _wrap(np.stack(frames))
