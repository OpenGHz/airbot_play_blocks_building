#!/usr/bin/env bash
# Is an arm (real or simulated) reachable? Prints what was found; used by `pixi run doctor`.
cd "$(dirname "$0")/.." || exit 1
out=$(python scripts/detect_arm.py) || exit 1
read -r driver env_type <<<"$out"
if [ "$env_type" = "sim" ]; then
    echo "仿真进程在运行（pixi run sim）"
else
    echo "机械臂主服务可以连接（驱动 ${driver}，${ARM_URL:-localhost}:${ARM_PORT:-50051}）"
fi
