"""Cloud image generation through hosted APIs (fal.ai, Replicate).

A cloud model is an ordinary model definition whose ``model`` block carries a
``cloud`` spec. The spec maps Maestro's generation parameters onto one
provider endpoint, so a new hosted model needs only a JSON file:

    "cloud": {
        "provider": "fal" | "replicate",
        "endpoint": "fal-ai/nano-banana",          # model route
        "edit_endpoint": "fal-ai/nano-banana/edit", # optional, used with images
        "image_field": "image_urls",               # optional, input images
        "image_list": true,                        # list or single URL
        "max_images_in": 10,
        "requires_image": false,
        "size": "aspect_ratio" | "image_size" | "width_height" | "none",
        "aspect_ratios": ["1:1", "16:9", ...],
        "count_field": "num_images",               # optional
        "max_count": 4,
        "seed_field": "seed",                      # optional
        "steps_field": "num_inference_steps",      # optional
        "guidance_field": "guidance_scale",        # optional
        "lora": {"style": "fal" | "replicate", "max": 2},  # optional
        "defaults": {"num_inference_steps": 28},   # Studio defaults
        "input": {"output_format": "png"}          # static extra input
    }

LoRAs are local files: fal receives them through its storage upload (cached),
Replicate through the CivitAI model ID when known, else its Files API.
"""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import httpx

PROVIDERS = ("fal", "replicate")
PROVIDER_LABELS = {"fal": "fal.ai", "replicate": "Replicate"}
PROVIDER_API_KEY_SETTING = {"fal": "fal_api_key", "replicate": "replicate_api_key"}
PROVIDER_API_KEY_ENV = {
    "fal": ("MAESTRO_FAL_KEY", "FAL_KEY"),
    "replicate": ("MAESTRO_REPLICATE_API_TOKEN", "REPLICATE_API_TOKEN"),
}
FAL_QUEUE_URL = "https://queue.fal.run"
FAL_STORAGE_URL = "https://rest.fal.ai/storage/upload/initiate?storage_type=gcs"
REPLICATE_API_URL = "https://api.replicate.com/v1"
POLL_SECONDS = 1.5
TIMEOUT_SECONDS = 15 * 60
REPLICATE_FILE_LIMIT = 100 * 1024 * 1024
# Uploaded LoRAs are reused while the provider keeps the file around.
UPLOAD_TTL_SECONDS = {"fal": 7 * 24 * 3600, "replicate": 20 * 3600}
_CONTENT_EXTENSIONS = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp"}


class CloudImageError(RuntimeError):
    pass


class CloudImageCancelled(RuntimeError):
    pass


@dataclass
class CloudImage:
    data: bytes
    content_type: str

    @property
    def extension(self) -> str:
        return _CONTENT_EXTENSIONS.get(self.content_type, ".png")


@dataclass
class CloudLora:
    path: str
    scale: float = 1.0
    civitai_model_id: int | None = None
    civitai_version_id: int | None = None

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


class UploadCache:
    """Provider URLs of uploaded LoRAs, keyed by file identity."""

    def __init__(self, path: str | None):
        self._path = path
        self._lock = threading.Lock()

    @staticmethod
    def key(provider: str, path: str) -> str:
        stat = os.stat(path)
        return f"{provider}:{os.path.realpath(path)}:{stat.st_size}:{int(stat.st_mtime)}"

    def _load(self) -> dict:
        try:
            with open(self._path, encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def get(self, provider: str, path: str, now: float) -> str | None:
        if not self._path:
            return None
        with self._lock:
            entry = self._load().get(self.key(provider, path)) or {}
        fresh = now - float(entry.get("at") or 0) < UPLOAD_TTL_SECONDS[provider]
        return entry.get("url") if fresh else None

    def put(self, provider: str, path: str, url: str, now: float) -> None:
        if not self._path:
            return
        with self._lock:
            data = self._load()
            data[self.key(provider, path)] = {"url": url, "at": now}
            os.makedirs(os.path.dirname(self._path) or ".", exist_ok=True)
            tmp = f"{self._path}.tmp"
            with open(tmp, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=1)
            os.replace(tmp, self._path)


def parse_lora_scales(value: Any, count: int) -> list[float]:
    """First number of each multiplier (phase-specific values are local-only)."""
    tokens = [str(v) for v in value] if isinstance(value, list) else str(value or "").split()
    scales = []
    for token in tokens:
        try:
            scales.append(float(token.replace(",", ";").split(";")[0]))
        except ValueError:
            scales.append(1.0)
    return (scales + [1.0] * count)[:count]


def cloud_spec(model_def: Mapping[str, Any] | None) -> dict | None:
    spec = (model_def or {}).get("cloud")
    return spec if isinstance(spec, dict) and spec.get("provider") in PROVIDERS else None


def provider_api_key(provider: str, services: Mapping[str, Any] | None = None,
                     environ: Mapping[str, str] | None = None) -> str:
    """Stored key first, then the provider's environment variables."""
    stored = str((services or {}).get(PROVIDER_API_KEY_SETTING.get(provider, ""), "") or "").strip()
    if stored:
        return stored
    environ = os.environ if environ is None else environ
    for name in PROVIDER_API_KEY_ENV.get(provider, ()):
        value = str(environ.get(name, "") or "").strip()
        if value:
            return value
    return ""


def parse_resolution(value: Any) -> tuple[int, int]:
    try:
        width, height = (int(part) for part in str(value or "").lower().split("x", 1))
    except ValueError:
        return 1024, 1024
    return (width, height) if width > 0 and height > 0 else (1024, 1024)


def nearest_aspect_ratio(width: int, height: int, allowed: list[str]) -> str:
    target = width / height
    choices = []
    for ratio in allowed:
        try:
            a, b = (float(part) for part in ratio.split(":", 1))
        except ValueError:
            continue
        choices.append((abs(target - a / b), ratio))
    if not choices:
        raise CloudImageError("Cloud model has no usable aspect ratio")
    return min(choices)[1]


def input_images(params: Mapping[str, Any]) -> list[str]:
    images = params.get("image_refs") or []
    if isinstance(images, str):
        images = [images]
    return [str(path) for path in images if path]


def build_input(spec: Mapping[str, Any], params: Mapping[str, Any], image_urls: list[str]) -> dict:
    """Translate Maestro params into the provider endpoint's input object."""
    payload: dict[str, Any] = {"prompt": str(params.get("prompt") or "")}
    width, height = parse_resolution(params.get("resolution"))
    size_mode = spec.get("size", "aspect_ratio")
    if size_mode == "aspect_ratio":
        payload["aspect_ratio"] = nearest_aspect_ratio(width, height, list(spec.get("aspect_ratios") or []))
    elif size_mode == "image_size":
        payload["image_size"] = {"width": width, "height": height}
    elif size_mode == "width_height":
        payload["width"], payload["height"] = width, height
    count_field = spec.get("count_field")
    if count_field:
        count = int(params.get("batch_size") or 1)
        payload[count_field] = max(1, min(count, int(spec.get("max_count") or 1)))
    seed_field = spec.get("seed_field")
    seed = params.get("seed")
    if seed_field and seed is not None and int(seed) >= 0:
        payload[seed_field] = int(seed)
    for field, param, cast in (("steps_field", "num_inference_steps", int),
                               ("guidance_field", "guidance_scale", float)):
        if spec.get(field) and params.get(param) is not None:
            payload[spec[field]] = cast(params[param])
    image_field = spec.get("image_field")
    if image_urls and image_field:
        if spec.get("image_list", True):
            payload[image_field] = image_urls[: int(spec.get("max_images_in") or len(image_urls))]
        else:
            payload[image_field] = image_urls[0]
    payload.update(spec.get("input") or {})
    return payload


def _data_uri(path: str) -> str:
    content_type = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as handle:
        return f"data:{content_type};base64,{base64.b64encode(handle.read()).decode()}"


def _error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:300] or f"HTTP {response.status_code}"
    detail = body.get("detail") if isinstance(body, dict) else body
    if isinstance(detail, list):
        detail = "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
    return str(detail or body)[:300]


def _check(response: httpx.Response, provider: str) -> httpx.Response:
    if response.status_code >= 400:
        label = PROVIDER_LABELS[provider]
        if response.status_code in (401, 403):
            raise CloudImageError(f"{label} rejected the API key ({response.status_code})")
        raise CloudImageError(f"{label} error {response.status_code}: {_error_detail(response)}")
    return response


class FalClient:
    """fal.ai queue API: submit, poll status_url, read response_url."""

    provider = "fal"

    def __init__(self, api_key: str, http: httpx.Client):
        self._http = http
        self._headers = {"Authorization": f"Key {api_key}"}

    def upload(self, path: str) -> str:
        return _data_uri(path)

    def upload_lora(self, lora: CloudLora) -> str:
        initiate = self._http.post(
            FAL_STORAGE_URL,
            json={"file_name": lora.name, "content_type": "application/octet-stream"},
            headers=self._headers,
        )
        target = _check(initiate, self.provider).json()
        with open(lora.path, "rb") as handle:
            put = self._http.put(target["upload_url"], content=handle,
                                 headers={"Content-Type": "application/octet-stream"},
                                 timeout=httpx.Timeout(1800.0, connect=15.0))
        _check(put, self.provider)
        return target["file_url"]

    @staticmethod
    def lora_reference(lora: CloudLora) -> str | None:
        return None

    @staticmethod
    def apply_loras(payload: dict, refs: list[tuple[str, float]], civitai_token: str) -> None:
        payload["loras"] = [{"path": url, "scale": scale} for url, scale in refs]

    def submit(self, endpoint: str, payload: dict) -> dict:
        response = self._http.post(f"{FAL_QUEUE_URL}/{endpoint}", json=payload, headers=self._headers)
        return _check(response, self.provider).json()

    def poll(self, handle: dict) -> tuple[str, Any]:
        status = _check(self._http.get(handle["status_url"], headers=self._headers), self.provider).json()
        state = status.get("status")
        if state == "COMPLETED":
            result = _check(self._http.get(handle["response_url"], headers=self._headers), self.provider).json()
            return "done", result
        if state == "IN_QUEUE":
            position = status.get("queue_position")
            return "queued", position
        return "running", None

    def cancel(self, handle: dict) -> None:
        if handle.get("cancel_url"):
            self._http.put(handle["cancel_url"], headers=self._headers)

    @staticmethod
    def output_urls(result: Any) -> list[str]:
        images = (result or {}).get("images") or ([result["image"]] if (result or {}).get("image") else [])
        return [image["url"] for image in images if isinstance(image, dict) and image.get("url")]


class ReplicateClient:
    """Replicate predictions API on official model routes."""

    provider = "replicate"

    def __init__(self, api_key: str, http: httpx.Client):
        self._http = http
        self._headers = {"Authorization": f"Bearer {api_key}"}

    def upload(self, path: str) -> str:
        content_type = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as handle:
            response = self._http.post(
                f"{REPLICATE_API_URL}/files",
                files={"content": (os.path.basename(path), handle, content_type)},
                headers=self._headers,
            )
        return _check(response, self.provider).json()["urls"]["get"]

    def upload_lora(self, lora: CloudLora) -> str:
        if os.path.getsize(lora.path) > REPLICATE_FILE_LIMIT:
            raise CloudImageError(
                f"LoRA {lora.name} is over 100 MB, Replicate's upload limit, and has no CivitAI source"
            )
        with open(lora.path, "rb") as handle:
            response = self._http.post(
                f"{REPLICATE_API_URL}/files",
                files={"content": (lora.name, handle, "application/octet-stream")},
                headers=self._headers,
                timeout=httpx.Timeout(1800.0, connect=15.0),
            )
        return _check(response, self.provider).json()["urls"]["get"]

    @staticmethod
    def lora_reference(lora: CloudLora) -> str | None:
        # Replicate downloads CivitAI models itself; no upload needed.
        if lora.civitai_model_id is None:
            return None
        return f"https://civitai.com/models/{lora.civitai_model_id}"

    @staticmethod
    def apply_loras(payload: dict, refs: list[tuple[str, float]], civitai_token: str) -> None:
        for (url, scale), (weights, scale_field) in zip(
            refs, (("lora_weights", "lora_scale"), ("extra_lora", "extra_lora_scale"))
        ):
            payload[weights], payload[scale_field] = url, scale
        if civitai_token and any("civitai.com/" in url for url, _ in refs):
            payload["civitai_api_token"] = civitai_token

    def submit(self, endpoint: str, payload: dict) -> dict:
        response = self._http.post(
            f"{REPLICATE_API_URL}/models/{endpoint}/predictions",
            json={"input": payload},
            headers=self._headers,
        )
        return _check(response, self.provider).json()

    def poll(self, handle: dict) -> tuple[str, Any]:
        prediction = _check(self._http.get(handle["urls"]["get"], headers=self._headers), self.provider).json()
        state = prediction.get("status")
        if state == "succeeded":
            return "done", prediction
        if state in ("failed", "canceled"):
            raise CloudImageError(f"Replicate prediction {state}: {prediction.get('error') or 'no detail'}")
        return ("queued", None) if state == "starting" else ("running", None)

    def cancel(self, handle: dict) -> None:
        cancel_url = (handle.get("urls") or {}).get("cancel")
        if cancel_url:
            self._http.post(cancel_url, headers=self._headers)

    @staticmethod
    def output_urls(result: Any) -> list[str]:
        output = (result or {}).get("output")
        if isinstance(output, str):
            return [output]
        return [str(item) for item in output or [] if item]


CLIENTS = {"fal": FalClient, "replicate": ReplicateClient}


def generate(
    spec: Mapping[str, Any],
    params: Mapping[str, Any],
    *,
    api_key: str,
    update: Callable[..., Any] = lambda **_: None,
    is_cancelled: Callable[[], bool] = lambda: False,
    http: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    loras: list[CloudLora] | None = None,
    civitai_token: str = "",
    upload_cache: UploadCache | None = None,
    wall_clock: Callable[[], float] = time.time,
) -> list[CloudImage]:
    """Run one cloud request and return the downloaded images."""
    provider = spec["provider"]
    label = PROVIDER_LABELS[provider]
    if not api_key:
        raise CloudImageError(f"No {label} API key: add it in Settings → Services")
    images = input_images(params)
    if spec.get("requires_image") and not images:
        raise CloudImageError("This cloud model needs at least one reference image")
    for path in images:
        if not os.path.isfile(path):
            raise CloudImageError(f"Reference image not found: {os.path.basename(path)}")
    loras = list(loras or [])
    lora_spec = spec.get("lora") or {}
    if loras and not lora_spec:
        raise CloudImageError("This cloud model does not accept LoRAs")
    if loras and len(loras) > int(lora_spec.get("max") or 1):
        raise CloudImageError(f"This cloud model accepts at most {lora_spec.get('max')} LoRAs")
    for lora in loras:
        if not os.path.isfile(lora.path):
            raise CloudImageError(f"LoRA not found: {lora.name}")
    owns_http = http is None
    http = http or httpx.Client(timeout=httpx.Timeout(120.0, connect=15.0), follow_redirects=True)
    try:
        client = CLIENTS[provider](api_key, http)
        update(phase="Cloud", message=f"Sending to {label}…", progress=5)
        urls = [client.upload(path) for path in images] if spec.get("image_field") else []
        endpoint = spec.get("edit_endpoint") if urls and spec.get("edit_endpoint") else spec["endpoint"]
        payload = build_input(spec, params, urls)
        if loras:
            refs = []
            for lora in loras:
                url = client.lora_reference(lora)
                if url is None and upload_cache is not None:
                    url = upload_cache.get(provider, lora.path, wall_clock())
                if url is None:
                    update(phase="Cloud", message=f"Uploading LoRA {lora.name} to {label}…", progress=5)
                    url = client.upload_lora(lora)
                    if upload_cache is not None:
                        upload_cache.put(provider, lora.path, url, wall_clock())
                refs.append((url, lora.scale))
            client.apply_loras(payload, refs, civitai_token)
        handle = client.submit(endpoint, payload)
        deadline = clock() + TIMEOUT_SECONDS
        while True:
            if is_cancelled():
                try:
                    client.cancel(handle)
                except httpx.HTTPError:
                    pass
                raise CloudImageCancelled()
            state, detail = client.poll(handle)
            if state == "done":
                result = detail
                break
            if state == "queued":
                where = f" (position {detail})" if isinstance(detail, int) else ""
                update(phase="Cloud", message=f"Queued on {label}{where}…", progress=10)
            else:
                update(phase="Cloud", message=f"Generating on {label}…", progress=50)
            if clock() > deadline:
                client.cancel(handle)
                raise CloudImageError(f"{label} did not finish within {TIMEOUT_SECONDS // 60} min")
            sleep(POLL_SECONDS)
        output_urls = client.output_urls(result)
        if not output_urls:
            raise CloudImageError(f"{label} returned no image")
        update(phase="Cloud", message="Downloading result…", progress=90)
        outputs = []
        for url in output_urls:
            if url.startswith("data:"):
                header, _, encoded = url.partition(",")
                outputs.append(CloudImage(base64.b64decode(encoded), header[5:].split(";")[0]))
                continue
            response = _check(http.get(url), provider)
            content_type = response.headers.get("content-type", "image/png").split(";")[0].strip()
            outputs.append(CloudImage(response.content, content_type))
        return outputs
    except httpx.HTTPError as error:
        raise CloudImageError(f"{label} request failed: {error}") from error
    finally:
        if owns_http:
            http.close()
