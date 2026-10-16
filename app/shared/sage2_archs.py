"""GPU architectures Maestro's SageAttention 2 path may use.

Kept free of torch and sageattention imports so the policy can be tested
anywhere. The amd64 worker image compiles SageAttention for sm_80 and sm_89
(sm_86 runs the sm_80 cubins plus the Triton path). An image that compiles
more architectures declares them in ``MAESTRO_SAGE2_EXTRA_ARCHS`` (e.g. the
DGX Spark image: sm_120 cubins, which also load on the GB10's sm_121). Those
extra architectures are trusted only once a one-off kernel probe succeeds.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

BUILTIN_ARCHS = frozenset({(8, 0), (8, 6), (8, 9)})
EXTRA_ARCHS_ENV = "MAESTRO_SAGE2_EXTRA_ARCHS"
# sage2_core dispatch name of each architecture whose kernels it shares.
DISPATCH_ALIASES = {"sm121": "sm120"}


def _parse_arch(text: str) -> tuple[int, int] | None:
    text = text.strip().lower().replace("sm_", "").replace("sm", "")
    if not text:
        return None
    if "." in text:
        major, _, minor = text.partition(".")
    elif text.isdigit() and len(text) >= 2:
        major, minor = text[:-1], text[-1]
    else:
        return None
    if not (major.isdigit() and minor.isdigit()):
        return None
    return int(major), int(minor)


def extra_archs(environ: Mapping[str, str] | None = None) -> frozenset[tuple[int, int]]:
    """Architectures the image declares on top of the built-in ones."""
    env = os.environ if environ is None else environ
    raw = env.get(EXTRA_ARCHS_ENV, "")
    parsed = (_parse_arch(item) for item in raw.replace(";", ",").split(","))
    return frozenset(arch for arch in parsed if arch is not None) - BUILTIN_ARCHS


def supported_archs(environ: Mapping[str, str] | None = None) -> frozenset[tuple[int, int]]:
    return BUILTIN_ARCHS | extra_archs(environ)


def dispatch_arch(arch: str) -> str:
    """The ``smXY`` name sage2_core dispatches on for this device."""
    return DISPATCH_ALIASES.get(arch, arch)
