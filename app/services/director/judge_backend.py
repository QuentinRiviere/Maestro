"""Where the Director's vision judges run: fal.ai or a local OpenAI-compatible server.

The storyboard judge (``storyboard_judge``) and the video judge
(``video_judge``) both ask a vision LLM through one ``ask`` callable:
``ask(http, key, model, system, prompt, media, sleep, *, max_tokens, timeout,
router=None) -> (text, cost_usd)``. Two backends implement it:

- ``fal`` (default): fal's ``openrouter/router/video`` queue, which only takes
  videos, so stills are sent as one-frame mp4 data URIs (see
  ``storyboard_judge.still_as_video``). Billed per call.
- ``openai``: any ``/v1/chat/completions`` server with vision, e.g.
  llama-server behind llama-swap on a DGX Spark (model alias ``judge``,
  Qwen3.6-35B-A3B with its mmproj). Images go as ``image_url`` data URIs
  (downscaled to 768 px), videos as ``VIDEO_FRAMES`` evenly spaced frames
  with their timestamps in the prompt. Costs nothing per call.

Services settings (``Settings -> Services``), filled from the environment when
unset: ``director_judge_provider`` (``MAESTRO_JUDGE_PROVIDER``),
``director_judge_url`` (``MAESTRO_JUDGE_URL``), ``director_judge_api_key``
(``MAESTRO_JUDGE_API_KEY``) and ``director_judge_model``
(``MAESTRO_JUDGE_MODEL``).
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence

import httpx

from .video_judge import JudgeError, _ask as _fal_ask

PROVIDERS = ("fal", "openai")
DEFAULT_PROVIDER = "fal"
DEFAULT_OPENAI_MODEL = "judge"
PROVIDER_SETTING = "director_judge_provider"
URL_SETTING = "director_judge_url"
API_KEY_SETTING = "director_judge_api_key"
MODEL_SETTING = "director_judge_model"
ENV = {
    PROVIDER_SETTING: "MAESTRO_JUDGE_PROVIDER",
    URL_SETTING: "MAESTRO_JUDGE_URL",
    API_KEY_SETTING: "MAESTRO_JUDGE_API_KEY",
    MODEL_SETTING: "MAESTRO_JUDGE_MODEL",
}
# Media sent to the OpenAI-compatible server: long side of every picture, and
# how many frames stand for a video.
IMAGE_LONG_SIDE = 768
VIDEO_FRAMES = 8
MAX_VIDEO_FRAMES = 12
# A frame-based judge hears nothing: without this it reported "audio is
# completely missing" as a critical defect on every clip (run 3087cc42).
SILENT_FRAMES_NOTE = ("These videos are given to you as silent stills: you cannot hear their audio, dialogue, "
                      "music or sound. Do not judge or mention sound at all, never report audio or dialogue as "
                      "missing, and leave any audio field null.")
JPEG_QUALITY = 85
TEMPERATURE = 0.2
VIDEO_SUFFIXES = (".mp4", ".mov", ".webm", ".mkv", ".m4v")
_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.S)


def _setting(services: Mapping[str, Any], key: str, environ: Mapping[str, str]) -> str:
    """The stored value, else the environment default (stored settings win)."""
    stored = str(services.get(key, "") or "").strip()
    return stored or str(environ.get(ENV[key], "") or "").strip()


@dataclass(frozen=True)
class JudgeBackend:
    """One way of asking a vision LLM; shared by both judges."""

    provider: str = DEFAULT_PROVIDER
    url: str = ""
    key: str = ""
    model: str = ""

    @property
    def available(self) -> bool:
        """Whether a vision call can be made at all."""
        if self.provider == "openai":
            return bool(self.url)
        return bool(self.key)

    @property
    def media_noun(self) -> str:
        """How prompts should name a still: fal only shows videos."""
        return "image" if self.provider == "openai" else "video"

    def resolve_model(self, fal_model: str) -> str:
        """The model a call uses: the local alias, or the fal model asked for."""
        return (self.model or DEFAULT_OPENAI_MODEL) if self.provider == "openai" else fal_model

    def media_for_images(self, paths: Sequence[str], tmp: str) -> list[str]:
        """The media items a judge sends for stills (``tmp`` holds any conversions)."""
        if self.provider == "openai":
            return [str(path) for path in paths]
        from . import storyboard_judge

        return [storyboard_judge.data_uri(storyboard_judge.still_as_video(path, os.path.join(tmp, f"still_{n}.mp4")))
                for n, path in enumerate(paths)]

    def ask(self, http: httpx.Client, key: str, model: str, system: str, prompt: str, media: Sequence[str],
            sleep: Callable[[float], None], *, max_tokens: int = 12000, timeout: float = 300,
            router: str | None = None) -> tuple[str, float | None]:
        """One vision call; same signature for both providers, returns (text, cost)."""
        if self.provider == "openai":
            return ask_openai(http, key or self.key, model, system, prompt, media, sleep,
                              max_tokens=max_tokens, timeout=timeout, url=self.url)
        extra = {"router": router} if router else {}
        return _fal_ask(http, key, model, system, prompt, list(media), sleep, max_tokens=max_tokens,
                        timeout=timeout, **extra)


def settings(services: Mapping[str, Any] | None, environ: Mapping[str, str] | None = None) -> JudgeBackend:
    """The configured backend; the fal one carries the fal key of ``services``."""
    services = services or {}
    environ = os.environ if environ is None else environ
    provider = _setting(services, PROVIDER_SETTING, environ).lower() or DEFAULT_PROVIDER
    if provider not in PROVIDERS:
        provider = DEFAULT_PROVIDER
    if provider == "openai":
        return JudgeBackend(provider="openai", url=_setting(services, URL_SETTING, environ).rstrip("/"),
                            key=_setting(services, API_KEY_SETTING, environ),
                            model=_setting(services, MODEL_SETTING, environ) or DEFAULT_OPENAI_MODEL)
    from services import cloud_image

    return JudgeBackend(provider="fal", key=cloud_image.provider_api_key("fal", services, environ))


def with_key(backend: JudgeBackend, key: str) -> JudgeBackend:
    return replace(backend, key=key) if key else backend


# ── OpenAI-compatible chat completions ────────────────────────────────


def chat_completions_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


def image_data_uri(path: str, long_side: int = IMAGE_LONG_SIDE) -> str:
    """A JPEG data URI of an image, downscaled so its long side is ``long_side``."""
    from PIL import Image

    with Image.open(path) as image:
        picture = image.convert("RGB")
        picture.thumbnail((long_side, long_side), Image.LANCZOS)
        buffer = io.BytesIO()
        picture.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def probe_duration(path: str) -> float | None:
    try:
        result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                 "default=nw=1:nk=1", path], check=True, capture_output=True, text=True, timeout=30)
        return float(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def frame_times(duration: float | None, count: int = VIDEO_FRAMES) -> list[float]:
    """``count`` evenly spaced timestamps from 0, capped at ``MAX_VIDEO_FRAMES``."""
    count = max(1, min(int(count), MAX_VIDEO_FRAMES))
    if not duration or duration <= 0:
        return [float(index) for index in range(count)]
    step = duration / count
    return [round(index * step, 2) for index in range(count)]


def extract_frames(path: str, dest_dir: str, count: int = VIDEO_FRAMES,
                   long_side: int = IMAGE_LONG_SIDE) -> list[tuple[float, str]]:
    """(timestamp, jpeg path) of ``count`` evenly spaced frames of a video."""
    times = frame_times(probe_duration(path), count)
    step = times[1] - times[0] if len(times) > 1 else 1.0
    pattern = os.path.join(dest_dir, "frame_%03d.jpg")
    scale = (f"scale=w='if(gt(iw,ih),min({long_side},iw),-2)':h='if(gt(iw,ih),-2,min({long_side},ih))'")
    command = ["ffmpeg", "-v", "error", "-y", "-i", path, "-vf", f"fps=1/{max(step, 0.01):.4f},{scale}",
               "-frames:v", str(len(times)), "-q:v", "4", pattern]
    subprocess.run(command, check=True, capture_output=True, timeout=300)
    frames = sorted(name for name in os.listdir(dest_dir) if name.startswith("frame_") and name.endswith(".jpg"))
    if not frames:
        raise JudgeError(f"No frame could be extracted from {os.path.basename(path)}")
    return [(times[index], os.path.join(dest_dir, name)) for index, name in enumerate(frames[:len(times)])]


def _materialize(item: str, tmp: str, index: int, http: httpx.Client) -> str:
    """A local file for a media item: a path, a data URI or a URL."""
    if item.startswith("data:"):
        header, _, payload = item.partition(",")
        mime = header[5:].split(";", 1)[0]
        suffix = ".mp4" if mime.startswith("video/") else ".png" if "png" in mime else ".jpg"
        path = os.path.join(tmp, f"media_{index}{suffix}")
        with open(path, "wb") as handle:
            handle.write(base64.b64decode(payload))
        return path
    if item.startswith(("http://", "https://")):
        response = http.get(item)
        if response.status_code >= 400:
            raise JudgeError(f"Could not fetch media {index + 1}: HTTP {response.status_code}")
        kind = response.headers.get("content-type", "")
        name = os.path.basename(item.split("?", 1)[0])
        suffix = os.path.splitext(name)[1].lower() or (".mp4" if kind.startswith("video/") else ".jpg")
        path = os.path.join(tmp, f"media_{index}{suffix}")
        with open(path, "wb") as handle:
            handle.write(response.content)
        return path
    return item


def _is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_SUFFIXES


def build_messages(system: str, prompt: str, media: Sequence[str], tmp: str, http: httpx.Client,
                   frames: int = VIDEO_FRAMES) -> list[dict]:
    """OpenAI chat messages: the prompt, then every media item as image parts.

    Media keep the judges' numbering (video 1, video 2, ...; image 1, ...). A
    video becomes ``frames`` stills, announced in the prompt with their
    timestamps so the model can place what it sees in time.
    """
    parts: list[dict] = []
    notes: list[str] = []
    for index, item in enumerate(media):
        path = _materialize(str(item), tmp, index, http)
        number = index + 1
        if _is_video(path):
            frame_dir = os.path.join(tmp, f"frames_{index}")
            os.makedirs(frame_dir, exist_ok=True)
            extracted = extract_frames(path, frame_dir, frames)
            stamps = ", ".join(f"{seconds:.1f}s" for seconds, _ in extracted)
            notes.append(f"Video {number} is shown as {len(extracted)} frames at t={stamps}.")
            parts.append({"type": "text", "text": f"Video {number}, frames in order:"})
            for seconds, frame in extracted:
                parts.append({"type": "text", "text": f"t={seconds:.1f}s"})
                parts.append({"type": "image_url", "image_url": {"url": image_data_uri(frame)}})
        else:
            parts.append({"type": "text", "text": f"Image {number}:"})
            parts.append({"type": "image_url", "image_url": {"url": image_data_uri(path)}})
    if notes:
        notes.append(SILENT_FRAMES_NOTE)
    text = prompt if not notes else prompt + "\n" + "\n".join(notes)
    return [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": text}] + parts}]


_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _json_depth_scan(text: str) -> tuple[list[str], list[tuple[int, int]], int]:
    """Open closers, (index, depth) of every comma outside strings, and the index of the root's end (-1 if open)."""
    stack: list[str] = []
    commas: list[tuple[int, int]] = []
    in_string = escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                raise ValueError(f"mismatched bracket at {index}")
            stack.pop()
            if not stack:
                return stack, commas, index
        elif char == ",":
            commas.append((index, len(stack)))
    return stack, commas, -1


def _close_open_json(text: str) -> str:
    """``text`` cut back to its last complete element, with its open brackets closed.

    The cut is at the last comma outside the innermost unfinished container,
    so a half-written element goes with its comma instead of surviving as
    an object missing half its fields.
    """
    stack, commas, end = _json_depth_scan(text)
    if end >= 0:
        return text[:end + 1]
    depth = len(stack)
    cuts = [index for index, comma_depth in commas if comma_depth < depth] or [index for index, _ in commas]
    if not cuts:
        return text
    cut = text[:cuts[-1]]
    stack, _, _ = _json_depth_scan(cut)
    return cut + "".join(reversed(stack))


def load_json_object(text: str, who: str = "The judge") -> dict:
    """The first JSON object in ``text``, repaired when a local model cut or dirtied it.

    Tried in order: a plain decode from the first brace (ignores any text
    after the object), the same without trailing commas, then the object
    cut back to its last complete element with its brackets closed (an
    answer stopped by the token limit). Raises JudgeError otherwise.
    """
    start = (text or "").find("{")
    if start < 0:
        raise JudgeError(f"{who} returned no JSON")
    body = text[start:]
    decoder = json.JSONDecoder()
    candidates = [body, _TRAILING_COMMA_RE.sub(r"\1", body)]
    error: Exception | None = None
    try:
        candidates.append(_TRAILING_COMMA_RE.sub(r"\1", _close_open_json(candidates[1])))
    except ValueError as failure:  # a closer that matches no opener: not a cut answer, junk
        error = failure
    for candidate in candidates:
        try:
            value, _ = decoder.raw_decode(candidate)
        except ValueError as failure:
            error = failure
            continue
        if isinstance(value, dict):
            return value
        error = ValueError("the JSON is not an object")
    where = getattr(error, "pos", None)
    excerpt = ""
    if isinstance(where, int):
        excerpt = " near: " + repr(body[max(0, where - 120):where + 120])
    raise JudgeError(f"{who} returned invalid JSON: {error}{excerpt}") from error


def _content_text(body: Mapping[str, Any]) -> str:
    choices = body.get("choices") or []
    message = (choices[0].get("message") or {}) if choices and isinstance(choices[0], dict) else {}
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(str(part.get("text") or "") for part in content if isinstance(part, dict))
    return _THINK_BLOCK.sub("", str(content or "")).strip()


def ask_openai(http: httpx.Client, key: str, model: str, system: str, prompt: str, media: Sequence[str],
               sleep: Callable[[float], None], *, max_tokens: int = 12000, timeout: float = 300,
               url: str, frames: int = VIDEO_FRAMES) -> tuple[str, float]:
    """One ``/v1/chat/completions`` vision call; retried once on a 5xx or a timeout. Free."""
    if not url:
        raise JudgeError("The local judge has no server URL (Settings -> Services)")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with tempfile.TemporaryDirectory(prefix="director-judge-") as tmp:
        payload = {
            "model": model or DEFAULT_OPENAI_MODEL,
            "messages": build_messages(system, prompt, media, tmp, http, frames),
            "temperature": TEMPERATURE, "max_tokens": max_tokens, "stream": False,
            # Qwen3-class models think by default and spend the whole token
            # budget on it with five images in the prompt (empty content,
            # finish_reason "length"): the judge answers straight away. The
            # verdicts were the same with and without thinking (2 s vs 9 s).
            "chat_template_kwargs": {"enable_thinking": False},
        }
    last_error: str = ""
    for attempt in range(2):
        try:
            response = http.post(chat_completions_url(url), headers=headers, json=payload, timeout=timeout)
        except (httpx.TimeoutException, httpx.TransportError) as error:
            last_error = f"{type(error).__name__}: {error}"[:300]
            if attempt == 0:
                sleep(2.0)
                continue
            raise JudgeError(f"The local judge did not answer: {last_error}") from error
        if response.status_code >= 500 and attempt == 0:
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            sleep(2.0)
            continue
        if response.status_code >= 400:
            raise JudgeError(f"The local judge failed: HTTP {response.status_code}: {response.text[:300]}")
        try:
            body = response.json()
        except ValueError as error:
            raise JudgeError("The local judge returned no JSON body") from error
        text = _content_text(body)
        if not text:
            choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
            raise JudgeError("The local judge returned an empty answer"
                             f" (finish_reason {choice.get('finish_reason')!r})")
        return text, 0.0
    raise JudgeError(f"The local judge failed: {last_error}")
