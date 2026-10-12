#!/usr/bin/env bash
# Boot a headless Maestro worker inside a dstack service (RunPod).
#
# The service clones this repository, then runs this script. Expensive state
# lives on the network volume mounted at $MAESTRO_VOLUME so later boots are
# fast: the Python venv (keyed by a hash of the dependency pins), model
# checkpoints, LoRAs and the Hugging Face cache.
#
# Inputs (environment, set by the run spec built in
# services/remote_worker/worker_spec.py):
#   MAESTRO_VOLUME         network volume mount point (default /workspace/vol)
#   MAESTRO_TORCH_INDEX    PyTorch wheel index (default cu128)
#   SERVER_PORT            Maestro API port (default 7860)
#
# Model weights are not fetched here: once the API is up, the controlling
# Maestro asks the worker to download the selected model through
# POST /api/v1/models/{model_type}/download, reusing Maestro's own logic.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOLUME="${MAESTRO_VOLUME:-/workspace/vol}"
TORCH_INDEX="${MAESTRO_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
TORCH_PINS="torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0"
export SERVER_PORT="${SERVER_PORT:-7860}"

log() { echo "[maestro-worker] $(date -u +%H:%M:%S) $*"; }

log "app dir ${APP_DIR}, volume ${VOLUME}, $(nproc) vCPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

mkdir -p "${VOLUME}"/{ckpts,loras,hf,venvs} /workspace/outputs

if ! command -v ffmpeg >/dev/null 2>&1; then
  log "installing system packages"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends ffmpeg libgl1 libglib2.0-0 git >/dev/null
fi

PYTHON_BIN="$(command -v python3 || command -v python)"
PY_VERSION="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
DEPS_HASH="$( { echo "${PY_VERSION} ${TORCH_PINS} ${TORCH_INDEX}"; cat "${APP_DIR}/requirements.txt"; } | sha256sum | cut -c1-16)"
VENV="${VOLUME}/venvs/py${PY_VERSION}-${DEPS_HASH}"

if [ ! -f "${VENV}/.ready" ]; then
  log "building venv ${VENV} (first boot for these pins)"
  rm -rf "${VENV}"
  "${PYTHON_BIN}" -m venv "${VENV}"
  "${VENV}/bin/python" -m pip install -q --upgrade pip uv
  "${VENV}/bin/uv" pip install -q --python "${VENV}/bin/python" ${TORCH_PINS} --index-url "${TORCH_INDEX}"
  "${VENV}/bin/uv" pip install -q --python "${VENV}/bin/python" -r "${APP_DIR}/requirements.txt"
  touch "${VENV}/.ready"
  log "venv ready"
else
  log "reusing venv ${VENV}"
fi

export HF_HOME="${VOLUME}/hf"
export PYTHONUNBUFFERED=1

# Worker-local engine config: weights and LoRAs on the volume, outputs on the
# pod disk (they are downloaded by the controlling Maestro, then discarded).
"${VENV}/bin/python" - "${APP_DIR}/wgp_config.json" "${VOLUME}" <<'PY'
import json, os, sys
path, volume = sys.argv[1], sys.argv[2]
config = {}
if os.path.exists(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
config.update({
    "checkpoints_paths": [os.path.join(volume, "ckpts")],
    "loras_root": os.path.join(volume, "loras"),
    "save_path": "/workspace/outputs",
    "image_save_path": "/workspace/outputs",
    "audio_save_path": "/workspace/outputs",
})
# A worker never routes to another worker.
config.pop("inference", None)
config.pop("remote_worker", None)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
PY

cd "${APP_DIR}"
log "starting Maestro API on port ${SERVER_PORT}"
exec "${VENV}/bin/python" launch.py --worker
