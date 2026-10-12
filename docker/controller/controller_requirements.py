"""Rewrite app/requirements.txt for a CPU-only controller host.

The controller (Maestro UI + API driving a remote GPU worker) never runs a
model, but it still imports the engine, so it needs the same packages minus
the CUDA-only builds. Platform markers cannot express "no CUDA" (a DGX Spark
is linux/aarch64 *with* CUDA), so the controller image applies this mapping
explicitly at build time.

The image is built for linux/amd64 (Rosetta on Apple Silicon): decord, which
the engine imports at startup, has no linux/aarch64 build at all.

Usage: python controller_requirements.py app/requirements.txt > out.txt
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
_NAME = re.compile(r"^\s*([A-Za-z0-9_.\-]+(?:\[[^\]]+\])?)")


def rewrite(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("--"):
            out.append(line.rstrip("\n"))
            continue
        match = _NAME.match(stripped)
        name = match.group(1).lower() if match else ""
        if name not in REPLACEMENTS:
            out.append(line.rstrip("\n"))
            continue
        replacement = REPLACEMENTS[name]
        if replacement is not None and replacement not in seen:
            out.append(replacement)
            seen.add(replacement)
    return out


if __name__ == "__main__":
    with open(sys.argv[1], encoding="utf-8") as handle:
        print("\n".join(rewrite(handle.readlines())))
