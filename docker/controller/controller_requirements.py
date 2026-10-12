"""Rewrite app/requirements.txt for a CPU-only controller host.

The controller (Maestro UI + API driving a remote GPU worker) never runs a
model, but it still imports the engine, so it needs the same packages minus
the CUDA-only builds. Platform markers cannot express "no CUDA" (a DGX Spark
is linux/aarch64 *with* CUDA), so the controller image applies this mapping
explicitly at build time.


Usage: python controller_requirements.py app/requirements.txt [--arch arm64] > out.txt

On arm64 (Apple Silicon, native image) decord, torchcodec and taichi have no
wheels: they are dropped and the image installs docker/controller/decord_shim, an
OpenCV-backed decord replacement covering the API Maestro uses.
"""
from __future__ import annotations

import re
import sys

# Package name (as written in requirements.txt) -> replacement requirement,
# or None to drop it. Markers on the original line are discarded.
REPLACEMENTS: dict[str, str | None] = {
    "onnxruntime-gpu": "onnxruntime==1.22.1",
    "onnxruntime": "onnxruntime==1.22.1",
    "rembg[gpu]": "rembg==2.0.65",
    "rembg": "rembg==2.0.65",
    # Needs triton kernels and is never imported by a controller.
    "flash-linear-attention": None,
}
# Extra drops for linux/arm64: no wheels exist for these. None is imported at
# engine startup (decord is replaced by the shim; torchcodec and taichi are
# only imported lazily by the Kiwi and SCAIL models).
ARM64_DROPS = {"decord", "eva-decord", "torchcodec", "taichi"}

_NAME = re.compile(r"^\s*([A-Za-z0-9_.\-]+(?:\[[^\]]+\])?)")


def rewrite(lines: list[str], arch: str = "amd64") -> list[str]:
    replacements = dict(REPLACEMENTS)
    if arch in ("arm64", "aarch64"):
        replacements.update({name: None for name in ARM64_DROPS})
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("--"):
            out.append(line.rstrip("\n"))
            continue
        match = _NAME.match(stripped)
        name = match.group(1).lower() if match else ""
        if name not in replacements:
            out.append(line.rstrip("\n"))
            continue
        replacement = replacements[name]
        if replacement is not None and replacement not in seen:
            out.append(replacement)
            seen.add(replacement)
    return out


if __name__ == "__main__":
    arch = sys.argv[sys.argv.index("--arch") + 1] if "--arch" in sys.argv else "amd64"
    with open(sys.argv[1], encoding="utf-8") as handle:
        print("\n".join(rewrite(handle.readlines(), arch)))
