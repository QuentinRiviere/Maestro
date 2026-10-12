# Maestro controller image: UI + API without a local GPU, for a Mac (or any
# CPU host) that sends generations to the remote GPU worker.
#
#   docker compose -f docker/controller/compose.yaml up --build
#
# Build context is the repository root. The image targets linux/amd64 (run
# through Rosetta on Apple Silicon) because decord, imported by the engine at
# startup, has no linux/aarch64 wheels; the controller does no heavy compute.

# ── UI ─────────────────────────────────────────────────────────────────────
FROM --platform=$BUILDPLATFORM node:22-bookworm-slim AS ui
WORKDIR /src/ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY ui/ ./
RUN npm run build

# ── Python runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/maestro-venv \
    PATH=/opt/maestro-venv/bin:$PATH

# build-essential: insightface only ships an sdist on linux.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ffmpeg libgl1 libglib2.0-0 git curl ca-certificates build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN python -m venv "$VIRTUAL_ENV" \
 && pip install --upgrade pip uv \
 && uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
      --index-url https://download.pytorch.org/whl/cpu

COPY app/requirements.txt /tmp/requirements.txt
COPY docker/controller/controller_requirements.py /tmp/controller_requirements.py
RUN python /tmp/controller_requirements.py /tmp/requirements.txt > /tmp/requirements-controller.txt \
 && uv pip install -r /tmp/requirements-controller.txt --index-strategy unsafe-best-match

COPY app/ /opt/maestro/app/
COPY --from=ui /src/ui/dist /opt/maestro/ui/dist
COPY docker/controller/entrypoint.sh /usr/local/bin/maestro-controller

WORKDIR /opt/maestro/app
ENV SERVER_NAME=0.0.0.0 \
    SERVER_PORT=7860 \
    MAESTRO_DATA=/data \
    MAESTRO_LOCALHOST_ALIAS=host.docker.internal
EXPOSE 7860
VOLUME ["/data"]
ENTRYPOINT ["maestro-controller"]
