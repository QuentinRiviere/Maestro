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
# Read by launch.py (worker file uploads land under $MAESTRO_DATA/inputs).
export MAESTRO_DATA="${DATA}"
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
  "${VENV}/bin/uv" pip install --python "${VENV}/bin/python" -r "${APP_DIR}/requirements.txt" \
    --index-strategy unsafe-best-match
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

# Worker-local paths, applied by wgp on top of its own (auto-tuned on first
# boot) config. Writing a partial wgp_config.json instead would skip wgp's
# defaults entirely.
# "auto" attention picks the fastest kernel the image ships that this GPU
# supports (sage2 > sage > sdpa, see wgp.get_auto_attention), so a stored
# config can never pin the worker to plain SDPA, and a GPU without a matching
# SageAttention kernel still falls back to SDPA instead of failing.
export MAESTRO_CONFIG_OVERRIDES="$(cat <<JSON
{"attention_mode": "auto",
 "checkpoints_paths": ["${DATA}/ckpts"],
 "loras_root": "${DATA}/loras",
 "save_path": "${DATA}/outputs",
 "image_save_path": "${DATA}/outputs",
 "audio_save_path": "${DATA}/outputs"}
JSON
)"

cd "${APP_DIR}"
if [ "${1:-}" = "--cache-warm" ]; then
  log "caching ${2:-?} in R2, then exiting"
else
  log "starting Maestro API on port ${SERVER_PORT}"
fi
exec "${VENV}/bin/python" launch.py --worker "$@"
