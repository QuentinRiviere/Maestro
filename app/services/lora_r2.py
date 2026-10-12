"""LoRAs stored in the R2 bucket instead of (or as well as) the local disk.

A LoRA kept only in R2 leaves a small ``<stem>.r2.json`` pointer where its
weights would be, beside the usual sidecars (``.civitai.json``, guide,
previews). The pointer names the content-addressed R2 asset the remote
worker already uses, so listings show the LoRA, the GPU worker fetches it
from R2, cloud providers read it through a presigned URL, and a local run
downloads it on demand.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

STORAGE_CHOICES = ("local", "r2", "both")
POINTER_SUFFIX = ".r2.json"
WEIGHT_EXTENSIONS = (".safetensors", ".sft")


def normalize_storage(value: Any) -> str:
    value = str(value or "local").strip().lower()
    if value not in STORAGE_CHOICES:
        raise ValueError(f"storage must be one of {', '.join(STORAGE_CHOICES)}")
    return value


def pointer_path(weights_path: str) -> str:
    return os.path.splitext(weights_path)[0] + POINTER_SUFFIX


def read_pointer(weights_path: str) -> dict | None:
    try:
        with open(pointer_path(weights_path), encoding="utf-8") as handle:
            pointer = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(pointer, dict) or not pointer.get("key") or not pointer.get("sha256"):
        return None
    return pointer


def pointer_weight_names(directory: str) -> list[str]:
    """Weight filenames that exist only as R2 pointers in ``directory``."""
    try:
        entries = os.listdir(directory)
    except OSError:
        return []
    names = []
    for entry in entries:
        if not entry.endswith(POINTER_SUFFIX):
            continue
        pointer = read_pointer(os.path.join(directory, entry[: -len(POINTER_SUFFIX)] + ".safetensors"))
        name = os.path.basename(str((pointer or {}).get("filename") or ""))
        if name.endswith(WEIGHT_EXTENSIONS) and not os.path.isfile(os.path.join(directory, name)):
            names.append(name)
    return names


def storage_of(weights_path: str) -> str:
    """local, r2 (pointer only) or both (local file and R2 copy)."""
    has_pointer = read_pointer(weights_path) is not None
    if os.path.isfile(weights_path):
        return "both" if has_pointer else "local"
    return "r2" if has_pointer else "local"


def store_in_r2(
    store: Any,
    weights_path: str,
    *,
    keep_local: bool,
    sha256: Callable[[str], str],
    update: Callable[[str], None] = lambda message: None,
) -> dict:
    """Upload a local LoRA to R2, write its pointer, optionally drop the file."""
    name = os.path.basename(weights_path)
    update(f"Uploading {name} to R2…")
    digest = sha256(weights_path)
    key = store.asset_key(digest, name)
    store.upload(weights_path, key)
    pointer = {
        "filename": name,
        "key": key,
        "sha256": digest,
        "size": os.path.getsize(weights_path),
        "bucket": getattr(store, "bucket", ""),
        "uploaded_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    target = pointer_path(weights_path)
    temporary = f"{target}.tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(pointer, handle, indent=2)
    os.replace(temporary, target)
    if not keep_local:
        os.remove(weights_path)
    return pointer


def materialize(store: Any, weights_path: str) -> bool:
    """Download a pointer-only LoRA to its local path. True when fetched."""
    if os.path.isfile(weights_path):
        return False
    pointer = read_pointer(weights_path)
    if pointer is None:
        return False
    store.download(pointer["key"], weights_path)
    return True
