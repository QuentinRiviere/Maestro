"""GGUF metadata: the header's key/value section, read locally or over HTTP Range.

Only the key/value section is parsed (the tensor infos that follow are never
read), so the first few MB of a file are enough: the tokenizer vocabulary is
the bulk of it. A remote file is read through growing Range requests to the
Hub (2, 8, 32 then 64 MB) with the same opener as the downloads, so the
HF token never reaches the CDN. Parsed metadata is cached as JSON per
(repo, revision, file) under the hub's config directory.

GGUF layout (v2 and v3, little endian): magic "GGUF", u32 version, u64
tensor count, u64 kv count, then kv pairs: string key, u32 type, value.
Arrays carry their element type and count; strings a u64 length.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import time
import urllib.error
import urllib.parse
from dataclasses import asdict, dataclass

from . import hf
from .paths import HUB, check_filename, check_repo_id, check_revision

MAGIC = b"GGUF"
VERSIONS = (2, 3)
STRING, ARRAY = 8, 9
# type id -> (struct format, size); strings and arrays are handled apart.
SCALARS = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4),
           7: ("<?", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}
TYPE_NAMES = {0: "uint8", 1: "int8", 2: "uint16", 3: "int16", 4: "uint32", 5: "int32", 6: "float32", 7: "bool",
              8: "string", 9: "array", 10: "uint64", 11: "int64", 12: "float64"}
KEEP_ARRAY = 4096          # longer arrays (vocabularies, merges) are kept as a count only
MAX_KV = 1 << 20           # a sane upper bound on kv_count: anything more is garbage
CHUNKS = (2 << 20, 8 << 20, 32 << 20, 64 << 20)
NEGATIVE_TTL = 300         # seconds before a failed read is attempted again


class GgufError(ValueError):
    """Not a GGUF file, an unsupported version, or a header that cannot be read."""


class Truncated(GgufError):
    """The buffer ends inside the header: read more bytes and retry."""


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def take(self, n: int) -> bytes:
        end = self.pos + n
        if end > len(self.buf):
            raise Truncated(f"need {end} bytes, have {len(self.buf)}")
        chunk = self.buf[self.pos:end]
        self.pos = end
        return chunk

    def skip(self, n: int) -> None:
        self.take(n)

    def scalar(self, type_id: int):
        fmt, size = SCALARS[type_id]
        return struct.unpack(fmt, self.take(size))[0]

    def u32(self) -> int:
        return self.scalar(4)

    def u64(self) -> int:
        return self.scalar(10)

    def string(self) -> str:
        return self.take(self.u64()).decode("utf-8", "replace")

    def value(self, type_id: int):
        if type_id in SCALARS:
            return self.scalar(type_id)
        if type_id == STRING:
            return self.string()
        if type_id == ARRAY:
            return self.array()
        raise GgufError(f"unknown GGUF value type {type_id}")

    def array(self):
        """A list, or {"array": type name, "count": n} for arrays longer than KEEP_ARRAY."""
        type_id, count = self.u32(), self.u64()
        if type_id in SCALARS:
            fmt, size = SCALARS[type_id]
            if count > KEEP_ARRAY:
                self.skip(count * size)
                return {"array": TYPE_NAMES[type_id], "count": count}
            return [v[0] for v in struct.iter_unpack(fmt, self.take(count * size))]
        if type_id == STRING:
            if count > KEEP_ARRAY:
                for _ in range(count):
                    self.skip(self.u64())
                return {"array": "string", "count": count}
            return [self.string() for _ in range(count)]
        if type_id == ARRAY:
            return [self.array() for _ in range(count)]
        raise GgufError(f"unknown GGUF array type {type_id}")


def parse(buf: bytes) -> dict:
    """{"version", "tensor_count", "kv_count", "header_bytes", "kv": {...}}; raises Truncated when buf is short."""
    r = _Reader(buf)
    if r.take(4) != MAGIC:
        raise GgufError("not a GGUF file")
    version = r.u32()
    if version not in VERSIONS:
        raise GgufError(f"unsupported GGUF version {version}")
    tensor_count, kv_count = r.u64(), r.u64()
    if kv_count > MAX_KV:
        raise GgufError(f"implausible kv_count {kv_count}")
    kv = {}
    for _ in range(kv_count):
        key = r.string()
        kv[key] = r.value(r.u32())
    return {"version": version, "tensor_count": tensor_count, "kv_count": kv_count, "header_bytes": r.pos, "kv": kv}


def read_local(path: str, chunks: tuple[int, ...] = CHUNKS) -> dict:
    with open(path, "rb") as handle:
        buf = b""
        for size in chunks:
            buf += handle.read(size - len(buf))
            try:
                return parse(buf)
            except Truncated:
                if len(buf) < size:
                    raise GgufError("file ends inside the GGUF header")
    raise GgufError(f"GGUF header larger than {chunks[-1] >> 20} MB")


def remote_url(repo: str, revision: str, name: str) -> str:
    check_repo_id(repo)
    check_revision(revision)
    check_filename(name)
    return f"{HUB}/{repo}/resolve/{urllib.parse.quote(revision, safe='')}/{urllib.parse.quote(name)}"


def _range(url: str, start: int, end: int, opener, timeout: float) -> bytes:
    """bytes [start, end] of url (fewer when the file is shorter)."""
    req = hf.request(url, {"Range": f"bytes={start}-{end}"})
    try:
        resp = opener.open(req, timeout=timeout)
    except urllib.error.HTTPError as exc:
        if exc.code == 416:
            return b""
        raise GgufError(f"Hub answered {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise GgufError(f"Hub unreachable: {exc.reason}") from exc
    with resp:
        if resp.status == 200 and start:
            raise GgufError("server ignored the Range request")
        want = end - start + 1
        parts, got = [], 0
        while got < want:
            block = resp.read(min(1 << 20, want - got))
            if not block:
                break
            parts.append(block)
            got += len(block)
    return b"".join(parts)


def read_remote(url: str, chunks: tuple[int, ...] = CHUNKS, opener=None, timeout: float = 30) -> dict:
    """Parse the header of a remote GGUF, fetching growing byte ranges until it is complete."""
    opener = opener or hf.OPENER
    buf = b""
    for size in chunks:
        if len(buf) < size:
            buf += _range(url, len(buf), size - 1, opener, timeout)
        try:
            return parse(buf)
        except Truncated:
            if len(buf) < size:
                raise GgufError("file ends inside the GGUF header")
    raise GgufError(f"GGUF header larger than {chunks[-1] >> 20} MB")


# -- architecture -----------------------------------------------------------

# Sliding-window layer patterns llama.cpp hard-codes when the GGUF has no
# `<arch>.attention.sliding_window_pattern`: every n-th layer is full attention.
SWA_PATTERNS = {"gemma2": 2, "gemma3": 6, "gemma3n": 5, "gpt-oss": 2, "cohere2": 4, "llama4": 4}


@dataclass
class ArchInfo:
    """What the KV cache size depends on. Unknown keys stay None."""
    architecture: str | None = None
    block_count: int | None = None
    head_count: int | None = None
    head_count_kv: int | list[int] | None = None    # per layer when a list; 0 = no KV cache (linear/recurrent)
    key_length: int | None = None
    value_length: int | None = None
    context_length: int | None = None               # trained maximum
    embedding_length: int | None = None
    vocab_size: int | None = None
    expert_count: int | None = None
    sliding_window: int | None = None
    sliding_window_pattern: int | None = None       # every n-th layer is full attention
    full_attention_interval: int | None = None      # qwen3next / qwen3.5: 1 attention layer every n (rest linear)
    kv_lora_rank: int | None = None                 # MLA (deepseek2): compressed KV
    rope_dimension_count: int | None = None

    @classmethod
    def from_kv(cls, kv: dict) -> "ArchInfo":
        arch = kv.get("general.architecture")
        arch = arch if isinstance(arch, str) else None

        def num(key: str):
            value = kv.get(f"{arch}.{key}") if arch else None
            return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

        heads_kv = kv.get(f"{arch}.attention.head_count_kv") if arch else None
        if isinstance(heads_kv, list) and all(isinstance(v, int) for v in heads_kv):
            heads_kv = [int(v) for v in heads_kv]
        elif isinstance(heads_kv, (int, float)) and not isinstance(heads_kv, bool):
            heads_kv = int(heads_kv)
        else:
            heads_kv = None
        tokens = kv.get("tokenizer.ggml.tokens")
        vocab = num("vocab_size")
        if vocab is None:
            if isinstance(tokens, dict):
                vocab = int(tokens.get("count") or 0) or None
            elif isinstance(tokens, list):
                vocab = len(tokens)
        info = cls(
            architecture=arch, block_count=num("block_count"), head_count=num("attention.head_count"),
            head_count_kv=heads_kv, key_length=num("attention.key_length"), value_length=num("attention.value_length"),
            context_length=num("context_length"), embedding_length=num("embedding_length"), vocab_size=vocab,
            expert_count=num("expert_count"), sliding_window=num("attention.sliding_window"),
            sliding_window_pattern=num("attention.sliding_window_pattern"),
            full_attention_interval=num("full_attention_interval"), kv_lora_rank=num("attention.kv_lora_rank"),
            rope_dimension_count=num("rope.dimension_count"),
        )
        if info.head_count_kv is None and info.head_count is not None:
            info.head_count_kv = info.head_count  # multi-head attention: no GQA key written
        if info.key_length is None and info.embedding_length and info.head_count:
            info.key_length = info.embedding_length // info.head_count
        if info.value_length is None:
            info.value_length = info.key_length
        if info.sliding_window and info.sliding_window_pattern is None:
            info.sliding_window_pattern = SWA_PATTERNS.get(arch or "")
        return info

    @property
    def complete(self) -> bool:
        """Enough to size the KV cache exactly."""
        return bool(self.block_count and self.head_count_kv is not None and self.key_length and self.value_length)

    def layers(self) -> list[tuple[int, bool]]:
        """Per layer: (KV heads, sliding window?). 0 heads = no KV cache for that layer."""
        n = self.block_count or 0
        out = []
        for i in range(n):
            if isinstance(self.head_count_kv, list):
                heads = self.head_count_kv[i] if i < len(self.head_count_kv) else 0
            else:
                heads = self.head_count_kv or 0
            if self.full_attention_interval and (i + 1) % self.full_attention_interval:
                heads = 0
            swa = False
            if self.sliding_window and heads:
                pattern = self.sliding_window_pattern
                swa = not pattern or pattern < 2 or i % pattern != pattern - 1
            out.append((heads, swa))
        return out

    def kv_dims(self) -> tuple[int, int] | None:
        """(K elements, V elements) per KV head and token; MLA stores one compressed head."""
        if self.kv_lora_rank and self.rope_dimension_count:
            return self.kv_lora_rank + self.rope_dimension_count, self.kv_lora_rank
        if self.key_length and self.value_length:
            return self.key_length, self.value_length
        return None

    def as_dict(self) -> dict:
        return asdict(self)


# -- cache ------------------------------------------------------------------

class MetaStore:
    """Parsed headers cached as JSON under <config_dir>/gguf-meta/, one file per (repo, revision, file)."""

    def __init__(self, config_dir: str, remote: bool = True, opener=None):
        self.dir = os.path.join(config_dir, "gguf-meta")
        self.remote = remote
        self.opener = opener
        self._mem: dict[str, dict] = {}
        self._failed: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def key(repo: str, revision: str, name: str) -> str:
        return hashlib.sha1(f"{repo}@{revision}/{name}".encode()).hexdigest()

    def path(self, repo: str, revision: str, name: str) -> str:
        return os.path.join(self.dir, self.key(repo, revision, name) + ".json")

    def header(self, repo: str, revision: str, name: str, local_path: str | None = None) -> dict | None:
        """The cached header, read from local_path or the Hub on a miss. None when unreadable."""
        key = self.key(repo, revision, name)
        with self._lock:
            hit = self._mem.get(key)
        if hit is not None:
            return hit
        path = self.path(repo, revision, name)
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict) and isinstance(data.get("kv"), dict):
                    with self._lock:
                        self._mem[key] = data
                    return data
            except (OSError, ValueError):
                pass
        failed = self._failed.get(key)
        if failed and time.time() - failed[0] < NEGATIVE_TTL:
            return None
        try:
            if local_path and os.path.exists(local_path):
                data, source = read_local(local_path), "local"
            elif self.remote:
                data, source = read_remote(remote_url(repo, revision, name), opener=self.opener), "remote"
            else:
                return None
        except (GgufError, OSError, ValueError, hf.HubError) as exc:
            self._failed[key] = (time.time(), str(exc))
            return None
        data = {"repo": repo, "revision": revision, "file": name, "source": source, "version": data["version"],
                "tensor_count": data["tensor_count"], "header_bytes": data["header_bytes"], "kv": data["kv"]}
        with self._lock:
            self._mem[key] = data
        try:
            os.makedirs(self.dir, exist_ok=True)
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle)
            os.replace(tmp, path)
        except OSError:
            pass  # read-only config dir: memory cache only
        return data

    def last_error(self, repo: str, revision: str, name: str) -> str | None:
        failed = self._failed.get(self.key(repo, revision, name))
        return failed[1] if failed else None

    def info(self, repo: str, revision: str, name: str, local_path: str | None = None) -> ArchInfo | None:
        data = self.header(repo, revision, name, local_path)
        return ArchInfo.from_kv(data["kv"]) if data else None
