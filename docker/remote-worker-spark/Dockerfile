# Remote Maestro worker image for NVIDIA DGX Spark: GB10 Grace Blackwell,
# linux/arm64, GPU compute capability 12.1 (sm_121), 128 GB unified memory.
#
# Same contract as docker/remote-worker/Dockerfile (the amd64 cloud image):
# the application code is NOT baked in, the pod clones the requested branch
# at boot and app/scripts/remote_worker_boot.sh reinstalls on top of the baked
# environment when app/requirements.txt no longer matches the baked hash.
# Only what differs on the Spark lives here:
#
# * torch 2.10.0 from the cu130 index, the CUDA line NVIDIA documents for
#   DGX Spark. Its aarch64 wheel carries sm_120 SASS plus compute_120 PTX;
#   CUDA runs sm_120 cubins on sm_121 (same major, newer minor).
# * SageAttention 2.2 compiled for 12.0 (v2.2.0's setup.py has no 12.1 entry,
#   and the sm_120 cubins load on GB10 for the reason above). The image sets
#   MAESTRO_SAGE2_EXTRA_ARCHS so Maestro's sage2 path accepts 12.0/12.1; a
#   one-off kernel probe at startup still drops sage2 for "auto" attention
#   (then Triton "sage", then SDPA) if the kernel does not run.
# * decord built from source (no aarch64 wheel exists on PyPI).
# * torchcodec 0.10.0 from the PyTorch index (PyPI has no aarch64 wheel).
# * A CUDA 13 ptxas for Triton, which DGX Spark guides require for sm_121.
# * No FlashAttention: no prebuilt aarch64 wheel exists for torch 2.10, and
#   Maestro drops "flash" when the module is missing.
#
# app/requirements.txt carries `platform_machine` markers for the packages
# with no aarch64 build (onnxruntime-gpu -> onnxruntime CPU, rembg[gpu] ->
# rembg, decord, taichi), so the boot-time delta install resolves on the
# Spark too. The amd64 resolution is unchanged by those markers.
#
# Build context: repository root.
#   docker buildx build --platform linux/arm64 -f docker/remote-worker-spark/Dockerfile \
#     -t ghcr.io/quentinriviere/maestro-worker:spark .
# Built and pushed by the `spark` job of .github/workflows/remote-worker-image.yml.

ARG TORCH_INDEX=https://download.pytorch.org/whl/cu130
ARG CUDA_DEVEL_IMAGE=nvidia/cuda:13.0.3-devel-ubuntu22.04

# The CUDA toolkit image, as is: the runtime stage takes ptxas from here
# without waiting for the SageAttention compile below.
FROM ${CUDA_DEVEL_IMAGE} AS cuda-devel

# ---------------------------------------------------------------------------
# Stage 1: SageAttention 2.2 against the exact torch the worker runs.
# ---------------------------------------------------------------------------
FROM cuda-devel AS sageattention-builder

ARG TORCH_INDEX
# thu-ml/SageAttention tag v2.2.0 (the version string must start with "2.2"
# for shared/sage2_core.py to enable the SageAttention2++ kernels).
ARG SAGEATTENTION_COMMIT=eb615cf6cf4d221338033340ee2de1c37fbdba4a
ARG SAGEATTENTION_CUDA_ARCHS="12.0"
# The ubuntu-24.04-arm hosted runner has 4 vCPUs and 16 GB RAM.
ARG SAGEATTENTION_MAX_JOBS=2

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# The PyTorch and NVIDIA indexes can stall on large wheels.
ENV UV_HTTP_TIMEOUT=300

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
# Stage 2: decord 0.6.0 (CPU decoding, linked against Ubuntu 22.04's FFmpeg
# 4.4, which the runtime stage installs through the ffmpeg package).
# ---------------------------------------------------------------------------
FROM ubuntu:22.04 AS decord-builder

ARG DECORD_REF=v0.6.0

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates python3 python3-venv python3-dev \
      libavcodec-dev libavfilter-dev libavformat-dev libavutil-dev libavdevice-dev \
      libswresample-dev libswscale-dev \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --quiet --depth 1 --branch "${DECORD_REF}" --recursive \
      https://github.com/dmlc/decord /src/decord \
 && cmake -S /src/decord -B /src/decord/build -DUSE_CUDA=0 -DCMAKE_BUILD_TYPE=Release \
 && cmake --build /src/decord/build -j"$(nproc)" \
 && python3 -m venv /opt/build-venv \
 && /opt/build-venv/bin/pip install --upgrade pip setuptools wheel \
 && cd /src/decord/python \
 && /opt/build-venv/bin/python setup.py -q bdist_wheel -d /wheels \
 && ls -l /wheels

# ---------------------------------------------------------------------------
# Stage 3: the worker image.
# ---------------------------------------------------------------------------
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-dev build-essential \
      git curl ca-certificates ffmpeg libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

ARG S5CMD_VERSION=2.2.2
RUN curl -sL --fail --connect-timeout 30 --max-time 300 \
      "https://github.com/peak/s5cmd/releases/download/v${S5CMD_VERSION}/s5cmd_${S5CMD_VERSION}_Linux-arm64.tar.gz" \
    | tar xz -C /usr/local/bin s5cmd \
 && s5cmd version

# Triton compiles with this ptxas instead of the one it bundles: a CUDA 13
# ptxas knows sm_121 (DGX Spark guides set TRITON_PTXAS_PATH the same way).
COPY --from=cuda-devel /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas

ENV VIRTUAL_ENV=/opt/maestro-venv \
    PATH=/opt/maestro-venv/bin:$PATH \
    TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
    UV_HTTP_TIMEOUT=300 \
    MAESTRO_SAGE2_EXTRA_ARCHS=12.0,12.1

ARG TORCH_INDEX
# torchcodec only has aarch64 wheels on the PyTorch index; installing it here
# satisfies requirements.txt's torchcodec==0.10.0 (local version +cu130).
# That CUDA build links NPP (libnppicc.so.13), which torch does not pull in.
ARG NVIDIA_NPP_VERSION=13.0.3.3
RUN python3 -m venv "$VIRTUAL_ENV" \
 && pip install --upgrade pip uv \
 && uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 torchcodec==0.10.0 \
      --index-url "$TORCH_INDEX" \
 && uv pip install "nvidia-npp==${NVIDIA_NPP_VERSION}"

# Same resolution strategy as the amd64 image; the aarch64 markers in
# requirements.txt swap the packages that have no aarch64 build.
COPY app/requirements.txt /opt/maestro/requirements.txt
RUN uv pip install -r /opt/maestro/requirements.txt --index-strategy unsafe-best-match \
 && sha256sum /opt/maestro/requirements.txt | cut -d' ' -f1 > /opt/maestro/requirements.sha256

# Libraries that load CUDA without going through torch (compiled extensions)
# find the pip-installed CUDA libraries system-wide. CUDA 13 wheels keep the
# toolkit libraries under nvidia/cu13/lib.
RUN find "$VIRTUAL_ENV" -path '*/site-packages/nvidia/*/lib' -type d > /etc/ld.so.conf.d/maestro-cuda.conf \
 && ldconfig \
 && ldconfig -p | grep -q libcudnn.so.9 \
 && ldconfig -p | grep -q libcublasLt.so.13 \
 && ldconfig -p | grep -q libnppicc.so.13

# SageAttention and decord are ABI-bound to this torch / CPython 3.10 /
# FFmpeg 4.4: installed last with --no-deps so they never move torch.
COPY --from=sageattention-builder /wheels/ /tmp/wheels/
COPY --from=decord-builder /wheels/ /tmp/wheels/
RUN uv pip install --no-deps /tmp/wheels/*.whl \
 && rm -rf /tmp/wheels \
 && python -c "import importlib.metadata as m, decord, torch, torchcodec; print('sageattention', m.version('sageattention'), '| decord', decord.__version__, '| torch', torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())"

# uv installs without bytecode, so every pod used to compile torch, diffusers
# and the rest on its first imports. Compile once here instead.
RUN python -m compileall -q -j 0 "$VIRTUAL_ENV/lib" >/dev/null || true

WORKDIR /workspace
