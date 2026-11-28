#!/usr/bin/env bash
# `pixi run bringup` (manual terminal 1): find the arm (real or simulated), then start
# the bridge node + MoveIt/RViz with the matching driver and /env_type.
cd "$(dirname "$0")/.." || exit 1
out=$(python scripts/detect_arm.py) || exit 1
read -r driver env_type <<<"$out"
if [ "$env_type" = "sim" ]; then
    printf '\033[1;35m%s\033[0m\n' "==== 仿真模式（MuJoCo）：驱动 sim，配置 configs/*/sim.yaml ===="
else
    printf '\033[1;32m%s\033[0m\n' "==== 真机：驱动 ${driver}（${ARM_URL}:${ARM_PORT}），配置 configs/*/real.yaml ===="
fi
exec roslaunch launch/bringup.launch driver:="$driver" env_type:="$env_type" url:="$ARM_URL" port:="$ARM_PORT" "$@"
