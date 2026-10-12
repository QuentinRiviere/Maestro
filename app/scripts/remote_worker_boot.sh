#!/usr/bin/env bash
# Boot a headless Maestro worker inside a dstack service (RunPod).
#
# Runs inside the maestro-worker image (docker/remote-worker/Dockerfile), which
# already holds the Python environment at /opt/maestro-venv. The service has
# cloned this repository at the requested branch and runs this script.
#
# Inputs (environment, set by the run spec built in
# services/remote_worker/worker_spec.py; secrets come from dstack secrets):
#   SERVER_PORT              Maestro API port (default 7860)
#   MAESTRO_DATA             pod-local data root (default /workspace/data)
#   R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, MAESTRO_R2_BUCKET,
#   MAESTRO_R2_PREFIX        optional checkpoint cache in Cloudflare R2
#
# Model weights are not fetched here: once the API is up, the controlling
# Maestro asks the worker to preload the selected model, which pulls cached
# files from R2 and lets Maestro's own download logic fetch anything missing.
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${MAESTRO_DATA:-/workspace/data}"
VENV="${VIRTUAL_ENV:-/opt/maestro-venv}"
export SERVER_PORT="${SERVER_PORT:-7860}"

log() { echo "[maestro-worker] $(date -u +%H:%M:%S) $*"; }

log "app dir ${APP_DIR}, data ${DATA}, $(nproc) vCPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

if [ ! -x "${VENV}/bin/python" ]; then
  echo "FATAL: ${VENV} is missing; run this worker from the maestro-worker image" >&2
  exit 1
fi

# The image bakes one dependency set; a branch may have moved on since.
BAKED_HASH="$(cat /opt/maestro/requirements.sha256 2>/dev/null || true)"
CURRENT_HASH="$(sha256sum "${APP_DIR}/requirements.txt" | cut -d' ' -f1)"
if [ "${BAKED_HASH}" != "${CURRENT_HASH}" ]; then
  log "requirements.txt differs from the image; installing the delta"
  "${VENV}/bin/uv" pip install --python "${VENV}/bin/python" -r "${APP_DIR}/requirements.txt"
else
  log "dependencies match the image"
fi

mkdir -p "${DATA}"/{ckpts,loras,hf,outputs}
export HF_HOME="${DATA}/hf"
export PYTHONUNBUFFERED=1
if [ -n "${R2_ACCOUNT_ID:-}" ]; then
  export R2_ENDPOINT_URL="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"
  log "R2 checkpoint cache: ${MAESTRO_R2_BUCKET:-?}/${MAESTRO_R2_PREFIX:-maestro}"
else
  log "no R2 cache configured; weights come from their original sources"
fi

# Worker-local engine config: everything on the pod disk.
"${VENV}/bin/python" - "${APP_DIR}/wgp_config.json" "${DATA}" <<'PY'
import json, os, sys
path, data = sys.argv[1], sys.argv[2]
config = {}
if os.path.exists(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
config.update({
    "checkpoints_paths": [os.path.join(data, "ckpts")],
    "loras_root": os.path.join(data, "loras"),
    "save_path": os.path.join(data, "outputs"),
    "image_save_path": os.path.join(data, "outputs"),
    "audio_save_path": os.path.join(data, "outputs"),
})
# A worker never routes to another worker.
config.pop("inference", None)
with open(path, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
PY

cd "${APP_DIR}"
log "starting Maestro API on port ${SERVER_PORT}"
exec "${VENV}/bin/python" launch.py --worker
