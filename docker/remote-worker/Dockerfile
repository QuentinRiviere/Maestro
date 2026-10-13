# Remote Maestro worker image: CUDA runtime + Maestro's Python environment.
#
# The application code is NOT baked in: the pod clones the requested branch at
# boot (seconds), so code changes never need an image rebuild. Only the
# dependency set lives here; scripts/remote_worker_boot.sh reinstalls on top
# of it when app/requirements.txt no longer matches the baked hash.
#
# Built by .github/workflows/remote-worker-image.yml, which copies
# app/requirements.txt into this directory as the build context.

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# Stage 1: compile SageAttention 2.2 against the exact torch the worker runs.
#
# No prebuilt Linux wheel exists for torch 2.10 / CUDA 12.8 / CPython 3.10
# (PyPI only carries the Triton-only 1.x line, which Maestro cannot use for
# "sage2"). The runtime image has no nvcc, so the wheel is built here in the
# matching CUDA devel image and only the wheel is copied into the final stage.
# This stage does not depend on requirements.txt, so the registry build cache
# reuses it when only Maestro's Python dependencies change.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS sageattention-builder

ARG TORCH_INDEX
# thu-ml/SageAttention tag v2.2.0 (the version string must start with "2.2"
# for shared/sage2_core.py to enable the SageAttention2++ kernels).
ARG SAGEATTENTION_COMMIT=eb615cf6cf4d221338033340ee2de1c37fbdba4a
# A100 (8.0) and L40S/RTX 4090/RTX 6000 Ada (8.9). v2.2.0's setup.py applies
# one gencode list to every extension, so its Hopper-only sm90 kernels cannot
# be built together with 8.x (ptxas rejects TMA/wgmma) nor with 12.0; H100
# and Blackwell stay on SageAttention's Triton path or SDPA. sm_86 GPUs (A40, RTX A6000)
# run the sm_80 cubins plus SageAttention's Triton path. Each extra arch adds
# compile time to every kernel file, so keep the list to GPUs the fleet uses.
ARG SAGEATTENTION_CUDA_ARCHS="8.0;8.9"
# GitHub's hosted runner has 4 vCPUs and 16 GB RAM; nvcc already fans out
# over the arch list (--threads=8 in setup.py), so keep ninja jobs low to
# stay clear of the OOM killer.
ARG SAGEATTENTION_MAX_JOBS=2

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/build-venv \
 && /opt/build-venv/bin/pip install --upgrade pip uv \
 && /opt/build-venv/bin/uv pip install --python /opt/build-venv/bin/python \
      torch==2.10.0 --index-url "$TORCH_INDEX" \
 && /opt/build-venv/bin/uv pip install --python /opt/build-venv/bin/python \
      ninja packaging setuptools wheel

RUN mkdir -p /src/sageattention /wheels \
 && curl -sL --fail --connect-timeout 30 --max-time 300 \
      "https://github.com/thu-ml/SageAttention/archive/${SAGEATTENTION_COMMIT}.tar.gz" \
    | tar xz --strip-components=1 -C /src/sageattention \
 && cd /src/sageattention \
 && PATH=/opt/build-venv/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="${SAGEATTENTION_CUDA_ARCHS}" \
    MAX_JOBS="${SAGEATTENTION_MAX_JOBS}" \
    EXT_PARALLEL=1 \
    /opt/build-venv/bin/pip wheel . --no-build-isolation --no-deps -w /wheels \
 && ls -l /wheels \
 && rm -rf /src/sageattention

# ---------------------------------------------------------------------------
# Stage 2: the worker image.
# ---------------------------------------------------------------------------
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

ARG TORCH_INDEX
RUN python3 -m venv "$VIRTUAL_ENV" \
 && pip install --upgrade pip uv \
 && uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url "$TORCH_INDEX"

# Same resolution strategy as install.js: requirements.txt mixes an extra
# index with PyPI pins (e.g. onnxruntime-gpu).
COPY requirements.txt /opt/maestro/requirements.txt
RUN uv pip install -r /opt/maestro/requirements.txt --index-strategy unsafe-best-match \
 && sha256sum /opt/maestro/requirements.txt | cut -d' ' -f1 > /opt/maestro/requirements.sha256

# Attention kernels. Both are ABI-bound to torch 2.10 + CUDA 12.8 + CPython
# 3.10, so they are installed last with --no-deps and never move torch.
# Maestro's "auto" attention mode picks sage2 > sage > sdpa from what imports
# and passes a capability check at startup, so a GPU these kernels do not
# cover falls back to SDPA instead of failing.
COPY --from=sageattention-builder /wheels/ /tmp/wheels/
# Prebuilt FlashAttention 2 (no source build: it would take hours). Used by the
# models that call flash_attn directly (TTS, LTX-2, MultiTalk...); Maestro
# probes it once per process and drops it when the GPU has no kernel for it.
ARG FLASH_ATTN_WHEEL=flash_attn-2.8.3+cu128torch2.10-cp310-cp310-linux_x86_64.whl
ARG FLASH_ATTN_URL=https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.0/flash_attn-2.8.3%2Bcu128torch2.10-cp310-cp310-linux_x86_64.whl
ARG FLASH_ATTN_SHA256=8d4b9f6ec17b999315a302f7ea96afeaae7037ccf212fbdca008dbcfe2893edb
RUN curl -sL --fail --connect-timeout 30 --max-time 900 -o "/tmp/wheels/${FLASH_ATTN_WHEEL}" "$FLASH_ATTN_URL" \
 && echo "${FLASH_ATTN_SHA256}  /tmp/wheels/${FLASH_ATTN_WHEEL}" | sha256sum -c - \
 && uv pip install --no-deps /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels \
 && python -c "import importlib.metadata as m; print('sageattention', m.version('sageattention'), '| flash_attn', m.version('flash_attn'))"

# uv installs without bytecode, so every pod used to compile torch, diffusers
# and the rest on its first imports. Compile once here instead.
RUN python -m compileall -q -j 0 "$VIRTUAL_ENV/lib" >/dev/null || true

WORKDIR /workspace
