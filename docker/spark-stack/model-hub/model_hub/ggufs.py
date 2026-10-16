"""GGUF file grouping (quants, split files, mmproj, drafts) and memory estimates."""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field

SPLIT_RE = re.compile(r"^(?P<stem>.+)-(?P<idx>\d{5})-of-(?P<total>\d{5})\.gguf$", re.I)
QUANT_RE = re.compile(
    r"(?:UD-)?(?:I?Q\d+_[A-Z0-9]+(?:_[A-Z0-9]+)*|TQ\d_\d|MXFP4(?:_MOE)?|NVFP4|BF16|F16|F32|FP16|FP8)",
    re.I,
)
DRAFT_N_RE = re.compile(r"--spec-draft-n-max\s+(\d+)")

GB = 1e9


@dataclass
class GgufEntry:
    """One loadable GGUF: a single file, or every part of a split model."""
    key: str                 # display name: file name, or "<stem>-*-of-N.gguf"
    kind: str                # model | mmproj | mtp | draft
    quant: str
    files: list[str]
    size: int
    complete: bool = True
    parts: int = 1

    @property
    def first(self) -> str:
        return self.files[0]

    def as_dict(self) -> dict:
        return {"key": self.key, "kind": self.kind, "quant": self.quant, "files": self.files, "size": self.size,
                "complete": self.complete, "parts": self.parts, "first": self.first}


def quant_of(name: str) -> str:
    base = posixpath.basename(name)
    m = SPLIT_RE.match(base)
    stem = m.group("stem") if m else re.sub(r"\.gguf$", "", base, flags=re.I)
    found = QUANT_RE.findall(stem)
    return found[-1].upper() if found else "?"


def kind_of(name: str) -> str:
    base = posixpath.basename(name).lower()
    parts = name.lower().split("/")
    if "mmproj" in base:
        return "mmproj"
    if base.startswith("mtp-") or "mtp" in parts[:-1]:
        return "mtp"
    if base.startswith(("dflash-", "eagle3-", "draft-")):
        return "draft"
    return "model"


def group_files(files: list[dict]) -> list[GgufEntry]:
    """Group {'name', 'size'} dicts: split parts merged, non-GGUF files ignored."""
    singles: list[GgufEntry] = []
    splits: dict[str, dict] = {}
    for f in files:
        name, size = f["name"], int(f.get("size") or 0)
        if not name.lower().endswith(".gguf"):
            continue
        m = SPLIT_RE.match(posixpath.basename(name))
        if m:
            directory = posixpath.dirname(name)
            stem = posixpath.join(directory, m.group("stem")) if directory else m.group("stem")
            key = f"{stem}-*-of-{m.group('total')}.gguf"
            group = splits.setdefault(key, {"total": int(m.group("total")), "parts": {}})
            group["parts"][int(m.group("idx"))] = (name, size)
        else:
            singles.append(GgufEntry(name, kind_of(name), quant_of(name), [name], size))
    for key, group in splits.items():
        idx = sorted(group["parts"])
        names = [group["parts"][i][0] for i in idx]
        complete = idx == list(range(1, group["total"] + 1))
        singles.append(GgufEntry(key, kind_of(names[0]), quant_of(names[0]), names,
                                 sum(group["parts"][i][1] for i in idx), complete, group["total"]))
    order = {"model": 0, "mmproj": 1, "mtp": 2, "draft": 3}
    singles.sort(key=lambda e: (order.get(e.kind, 9), e.size))
    return singles


def detect_mtp(repo: str, tags: list[str], readme: str) -> dict:
    """Whether the model carries MTP heads usable with llama-server's draft-mtp."""
    tagset = {t.lower() for t in tags}
    name_hit = bool(re.search(r"(^|[-_/])mtp([-_/]|$)", repo, re.I))
    readme_hit = "draft-mtp" in readme
    m = DRAFT_N_RE.search(readme) if readme_hit else None
    return {
        "mtp": "mtp" in tagset or name_hit or readme_hit,
        "draft_n_max": int(m.group(1)) if m else None,
        "source": "tags" if "mtp" in tagset else "name" if name_hit else "readme" if readme_hit else None,
    }


def kv_bytes_per_token(weights_bytes: int) -> int:
    """Rough f16 KV cache per token of context, from the weights size.

    Real values depend on layers x KV heads x head size (and hybrid / sliding
    attention). ~6 KB per GB of weights matches common 20-35B MoE models
    within a factor of ~2, clamped to 16-320 KB.
    """
    return int(min(max(weights_bytes / GB * 6 * 1024, 16 * 1024), 320 * 1024))


def estimate_memory(weights_bytes: int, ctx: int, mmproj_bytes: int = 0,
                    kv_per_token: int | None = None) -> dict:
    """Estimated resident memory of one llama-server, in bytes.

    llama-server splits --ctx-size across its --parallel slots, so the KV
    cache depends on ctx-size only. Overhead covers compute buffers.
    """
    per_token = kv_per_token or kv_bytes_per_token(weights_bytes)
    kv = per_token * max(int(ctx), 0)
    overhead = int(0.5 * GB + 0.03 * weights_bytes)
    total = weights_bytes + mmproj_bytes + kv + overhead
    return {"weights": weights_bytes, "mmproj": mmproj_bytes, "kv": kv, "overhead": overhead, "total": total,
            "kv_per_token": per_token}


@dataclass
class Budget:
    total_gb: float = 128.0
    reserved_gb: float = 10.0       # OS, Docker, llama-swap itself
    resident_gb: float = 0.0        # models that are always loaded
    resident: list[str] = field(default_factory=list)

    @property
    def available_gb(self) -> float:
        return self.total_gb - self.reserved_gb - self.resident_gb

    def as_dict(self) -> dict:
        return {"total_gb": self.total_gb, "reserved_gb": self.reserved_gb, "resident_gb": round(self.resident_gb, 2),
                "resident": self.resident, "available_gb": round(self.available_gb, 2)}


def fit_badge(total_bytes: int, budget: Budget) -> str:
    """fits (<= 85% of what is left), tight (<= 100%), too-big."""
    need = total_bytes / GB
    avail = budget.available_gb
    if need <= avail * 0.85:
        return "fits"
    if need <= avail:
        return "tight"
    return "too-big"
