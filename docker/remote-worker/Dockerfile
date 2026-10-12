# Remote Maestro worker image: CUDA runtime + Maestro's Python environment.
#
# The application code is NOT baked in: the pod clones the requested branch at
# boot (seconds), so code changes never need an image rebuild. Only the
# dependency set lives here; scripts/remote_worker_boot.sh reinstalls on top
# of it when app/requirements.txt no longer matches the baked hash.
#
# Built by .github/workflows/remote-worker-image.yml, which copies
# app/requirements.txt into this directory as the build context.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential \
      git curl ca-certificates ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ARG S5CMD_VERSION=2.2.2
RUN curl -sL --fail --connect-timeout 30 --max-time 300 \
      "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_Linux-64bit.tar.gz" \
    | tar xz -C /usr/local/bin s5cmd \
 && s5cmd version

ENV VIRTUAL_ENV=/opt/maestro-venv \
    PATH=/opt/maestro-venv/bin:$PATH

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128
RUN python3 -m venv "$VIRTUAL_ENV" \
 && pip install --upgrade pip uv \
 && uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url "$TORCH_INDEX"

# Same resolution strategy as install.js: requirements.txt mixes an extra
# index with PyPI pins (e.g. onnxruntime-gpu).
COPY requirements.txt /opt/maestro/requirements.txt
RUN uv pip install -r /opt/maestro/requirements.txt --index-strategy unsafe-best-match \
 && sha256sum /opt/maestro/requirements.txt | cut -d' ' -f1 > /opt/maestro/requirements.sha256

WORKDIR /workspace
