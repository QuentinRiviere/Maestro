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
    installed: bool = False

    @property
    def first(self) -> str:
        return self.files[0]

    def as_dict(self) -> dict:
        return {"key": self.key, "kind": self.kind, "quant": self.quant, "files": self.files, "size": self.size,
                "complete": self.complete, "parts": self.parts, "first": self.first, "installed": self.installed}


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

    The fallback when the GGUF metadata is unavailable (see kv_profile for
    the exact figure). ~6 KB per GB of weights matches common 20-35B MoE
    models within a factor of ~2, clamped to 16-320 KB.
    """
    return int(min(max(weights_bytes / GB * 6 * 1024, 16 * 1024), 320 * 1024))


# Bytes per cached element: f16/bf16 2, f32 4, quantized types per 32-element
# block (q8_0: 32 bytes + f16 scale, q4_0: 16 + scale, q4_1: 16 + scale + min...).
CACHE_TYPE_BYTES = {"f16": 2.0, "bf16": 2.0, "f32": 4.0, "q8_0": 34 / 32, "q4_0": 18 / 32, "q4_1": 20 / 32,
                    "q5_0": 22 / 32, "q5_1": 24 / 32}
CACHE_TYPES = tuple(CACHE_TYPE_BYTES)
UBATCH = 512          # llama-server's default --ubatch-size, part of the SWA cache size
MIN_CTX = 2048
CTX_STEP = 256
MAX_CTX = 1 << 21     # the serve.ctx_size limit (registry)


def cache_type_bytes(cache_type: str) -> float:
    try:
        return CACHE_TYPE_BYTES[cache_type]
    except KeyError:
        raise ValueError(f"unknown cache type {cache_type!r}; one of {', '.join(CACHE_TYPES)}") from None


@dataclass
class KvProfile:
    """KV cache bytes per token, split by layer kind so the sliding window can be applied."""
    full_per_token: int             # full-attention layers
    swa_per_token: int = 0          # sliding-window layers (count min(ctx, window) tokens)
    window: int | None = None
    attention_layers: int = 0
    swa_layers: int = 0
    linear_layers: int = 0          # layers without a KV cache (recurrent / linear attention)
    source: str = "metadata"        # metadata | heuristic

    @property
    def per_token(self) -> int:
        return self.full_per_token + self.swa_per_token

    def swa_tokens(self, ctx: int, parallel: int = 1) -> int:
        """Tokens a sliding-window layer keeps: llama.cpp sizes its SWA cache at
        n_swa * (n_seq_max + 1) + n_ubatch, never more than the full context."""
        if not self.window:
            return ctx
        return min(ctx, self.window * (max(parallel, 1) + 1) + UBATCH)

    def bytes_at(self, ctx: int, parallel: int = 1) -> int:
        ctx = max(int(ctx), 0)
        return self.full_per_token * ctx + self.swa_per_token * self.swa_tokens(ctx, parallel)

    def as_dict(self) -> dict:
        return {"per_token": self.per_token, "full_per_token": self.full_per_token,
                "swa_per_token": self.swa_per_token, "window": self.window, "attention_layers": self.attention_layers,
                "swa_layers": self.swa_layers, "linear_layers": self.linear_layers, "source": self.source}


def heuristic_profile(weights_bytes: int) -> KvProfile:
    return KvProfile(kv_bytes_per_token(weights_bytes), source="heuristic")


def kv_profile(info, cache_type_k: str = "f16", cache_type_v: str = "f16") -> KvProfile | None:
    """Exact KV cache per token from the GGUF metadata: sum over attention layers of
    kv_heads * (key_length * bytes(type_k) + value_length * bytes(type_v)).
    None when the metadata lacks what is needed (caller falls back to the heuristic)."""
    if info is None or not info.complete:
        return None
    dims = info.kv_dims()
    if dims is None:
        return None
    k_bytes, v_bytes = cache_type_bytes(cache_type_k), cache_type_bytes(cache_type_v)
    per_head = dims[0] * k_bytes + dims[1] * v_bytes
    full = swa = 0.0
    attention = swa_layers = linear = 0
    for heads, is_swa in info.layers():
        if not heads:
            linear += 1
            continue
        attention += 1
        if is_swa:
            swa_layers += 1
            swa += heads * per_head
        else:
            full += heads * per_head
    return KvProfile(int(full), int(swa), info.sliding_window if swa_layers else None, attention, swa_layers, linear)


def estimate_memory(weights_bytes: int, ctx: int, mmproj_bytes: int = 0, kv_per_token: int | None = None,
                    info=None, cache_type_k: str = "f16", cache_type_v: str = "f16", parallel: int = 1) -> dict:
    """Estimated resident memory of one llama-server, in bytes.

    llama-server splits --ctx-size across its --parallel slots, so the KV
    cache depends on ctx-size only (plus the SWA cache's per-slot window).
    Overhead stays a heuristic for the compute buffers: ~0.5 GB + 3 % of the
    weights; with --flash-attn on, llama-server needs no large extra buffer.
    """
    if kv_per_token:
        profile = KvProfile(kv_per_token, source="heuristic")
    else:
        profile = kv_profile(info, cache_type_k, cache_type_v) or heuristic_profile(weights_bytes)
    ctx = max(int(ctx), 0)
    kv = profile.bytes_at(ctx, parallel)
    overhead = int(0.5 * GB + 0.03 * weights_bytes)
    total = weights_bytes + mmproj_bytes + kv + overhead
    return {"weights": weights_bytes, "mmproj": mmproj_bytes, "kv": kv, "overhead": overhead, "total": total,
            "kv_per_token": profile.per_token, "kv_source": profile.source, "ctx": ctx}


def max_context(info, weights_bytes: int, mmproj_bytes: int, budget_bytes: float, cache_type_k: str = "f16",
                cache_type_v: str = "f16", parallel: int = 1) -> dict:
    """The largest --ctx-size (multiple of 256, >= 2048, at most the trained length) whose estimate fits."""
    trained = info.context_length if info is not None else None
    cap = trained if trained and trained >= MIN_CTX else MAX_CTX

    def fits(ctx: int) -> bool:
        return estimate_memory(weights_bytes, ctx, mmproj_bytes, info=info, cache_type_k=cache_type_k,
                               cache_type_v=cache_type_v, parallel=parallel)["total"] <= budget_bytes

    if fits(cap):
        ctx, limited_by, ok = cap, "trained" if cap == trained else "budget", True
    elif not fits(MIN_CTX):
        ctx, limited_by, ok = MIN_CTX, "budget", False
    else:
        lo, hi = MIN_CTX // CTX_STEP, cap // CTX_STEP   # lo fits, hi does not (monotonic in ctx)
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fits(mid * CTX_STEP):
                lo = mid
            else:
                hi = mid
        ctx, limited_by, ok = lo * CTX_STEP, "budget", True
    est = estimate_memory(weights_bytes, ctx, mmproj_bytes, info=info, cache_type_k=cache_type_k,
                          cache_type_v=cache_type_v, parallel=parallel)
    return {"ctx": ctx, "per_slot": ctx // max(parallel, 1), "fits": ok, "limited_by": limited_by,
            "trained": trained, "kv_gb": round(est["kv"] / GB, 2), "total_gb": round(est["total"] / GB, 2),
            "kv_source": est["kv_source"]}


def context_report(info, weights_bytes: int, mmproj_bytes: int, budget_bytes: float, ctx: int = 0,
                   cache_type_k: str = "f16", cache_type_v: str = "f16", parallel: int = 1) -> dict:
    """Everything the UI and the CLI show: architecture, KV per token, max context, estimate at ctx."""
    profile = kv_profile(info, cache_type_k, cache_type_v) or heuristic_profile(weights_bytes)
    best = max_context(info, weights_bytes, mmproj_bytes, budget_bytes, cache_type_k, cache_type_v, parallel)
    est = estimate_memory(weights_bytes, ctx or best["ctx"], mmproj_bytes, info=info, cache_type_k=cache_type_k,
                          cache_type_v=cache_type_v, parallel=parallel)
    return {
        "arch": info.as_dict() if info is not None else None,
        "kv": profile.as_dict(),
        "kv_per_token": profile.per_token,
        "kv_source": profile.source,
        "cache_type_k": cache_type_k, "cache_type_v": cache_type_v, "parallel": parallel,
        "max": best,
        "estimate": {k: round(v / GB, 2) if k in ("weights", "mmproj", "kv", "overhead", "total") else v
                     for k, v in est.items()},
        "budget_gb": round(budget_bytes / GB, 2),
    }


@dataclass
class Budget:
    total_gb: float = 128.0
    reserved_gb: float = 10.0       # OS, Docker, llama-swap itself
    resident_gb: float = 0.0        # models that are always loaded
    resident: list[str] = field(default_factory=list)
    swap_reserve_gb: float = 0.0    # kept free for on-demand models while the residents are loaded

    @property
    def available_gb(self) -> float:
        """For an on-demand model: total - reserved - residents."""
        return self.total_gb - self.reserved_gb - self.resident_gb

    @property
    def resident_available_gb(self) -> float:
        """For a resident model: the on-demand swap reserve must stay free too."""
        return self.available_gb - self.swap_reserve_gb

    def available_for(self, resident: bool) -> float:
        return self.resident_available_gb if resident else self.available_gb

    def as_dict(self) -> dict:
        return {"total_gb": self.total_gb, "reserved_gb": self.reserved_gb, "resident_gb": round(self.resident_gb, 2),
                "resident": self.resident, "available_gb": round(self.available_gb, 2),
                "swap_reserve_gb": self.swap_reserve_gb, "resident_available_gb": round(self.resident_available_gb, 2)}


def is_resident(group: str | None, groups: dict, preload: bool = False) -> bool:
    """A model stays loaded when its llama-swap group is persistent or it is preloaded."""
    return bool(((groups or {}).get(group or "") or {}).get("persistent") or preload)


def fit_badge(total_bytes: int, budget: Budget, resident: bool = False) -> str:
    """fits (<= 85% of what is left), tight (<= 100%), too-big."""
    need = total_bytes / GB
    avail = budget.available_for(resident)
    if need <= avail * 0.85:
        return "fits"
    if need <= avail:
        return "tight"
    return "too-big"
