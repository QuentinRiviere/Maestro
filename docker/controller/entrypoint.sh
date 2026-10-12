#!/usr/bin/env bash
# Keep every piece of user state under /data so the container is disposable:
# settings (incl. the dstack/R2 credentials), engine config, outputs,
# uploads, LoRAs and imported model definitions.
set -euo pipefail

APP=/opt/maestro/app
DATA="${MAESTRO_DATA:-/data}"
mkdir -p "${DATA}"/{settings,outputs,uploads,loras,finetunes}

for dir in settings outputs uploads loras finetunes; do
  if [ -d "${APP}/${dir}" ] && [ ! -L "${APP}/${dir}" ]; then
    # Seed the volume with files shipped in the image (e.g. finetunes/README).
    cp -rn "${APP}/${dir}/." "${DATA}/${dir}/" 2>/dev/null || true
    rm -rf "${APP:?}/${dir}"
  fi
  ln -sfn "${DATA}/${dir}" "${APP}/${dir}"
done
[ -e "${DATA}/wgp_config.json" ] || [ ! -f "${APP}/wgp_config.json" ] || mv "${APP}/wgp_config.json" "${DATA}/"
ln -sfn "${DATA}/wgp_config.json" "${APP}/wgp_config.json"

cd "${APP}"
exec python launch.py "$@"
