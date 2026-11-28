#!/usr/bin/env bash
# `pixi run hsv-finder ...`
cd "$(dirname "$0")/.." || exit 1
ENV_TYPE=$(bash scripts/env_type.sh)
. scripts/sim_camera_args.sh
map_camera_args -d "$@"
exec python demo/hsv_finder.py "${ARGS[@]}"
