# CPU image for the R2 cache warmer (services/remote_worker/cache_warmer.py).
#
# The warmer only resolves model files with Maestro's own download logic and
# copies them to R2: no GPU, no CUDA. The GPU worker image is ~20 GB and takes
# a CPU pod a quarter of an hour to pull; this one keeps the same venv path
# and boot script but installs the controller's CPU-only dependency set.
#
# Published as the `cpu` tag of the worker package (same GHCR visibility), by
# .github/workflows/remote-worker-image.yml. Build context: repository root.
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/maestro-venv \
    PATH=/opt/maestro-venv/bin:$PATH \
    MAESTRO_IMAGE_FLAVOR=cpu

# build-essential: insightface only ships an sdist on linux.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 git curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

ARG S5CMD_VERSION=2.2.2
RUN curl -sL --fail --connect-timeout 30 --max-time 300 \
      "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_Linux-64bit.tar.gz" \
    | tar xz -C /usr/local/bin s5cmd \
 && s5cmd version

RUN python -m venv "$VIRTUAL_ENV" \
 && pip install --upgrade pip uv \
 && uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
      --index-url https://download.pytorch.org/whl/cpu

COPY app/requirements.txt /opt/maestro/requirements.txt
COPY docker/controller/controller_requirements.py /tmp/controller_requirements.py
RUN python /tmp/controller_requirements.py /opt/maestro/requirements.txt --arch amd64 > /tmp/requirements-cpu.txt \
 && uv pip install -r /tmp/requirements-cpu.txt --index-strategy unsafe-best-match \
 && sha256sum /opt/maestro/requirements.txt | cut -d' ' -f1 > /opt/maestro/requirements.sha256

WORKDIR /opt/maestro
