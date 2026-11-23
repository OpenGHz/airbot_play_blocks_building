#!/usr/bin/env bash
# `pixi run vision ...`: Best_Contour.py with configs/vision/<env_type>.yaml.
cd "$(dirname "$0")/.." || exit 1
ENV_TYPE=$(bash scripts/env_type.sh)
. scripts/sim_camera_args.sh
map_camera_args -vd "$@"
exec python Best_Contour.py -cfg "./configs/vision/${ENV_TYPE}.yaml" "${ARGS[@]}"
