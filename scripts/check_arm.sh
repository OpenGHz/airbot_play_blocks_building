#!/usr/bin/env bash
# Fail fast when the arm service is not reachable. Without this, the gRPC
# bridge node waits silently forever and ignores Ctrl+C.
ARM_URL="${ARM_URL:-localhost}"
ARM_PORT="${ARM_PORT:-50051}"
if timeout 3 bash -c "</dev/tcp/${ARM_URL}/${ARM_PORT}" 2>/dev/null; then
    echo "arm service reachable at ${ARM_URL}:${ARM_PORT}"
    exit 0
fi
cat >&2 <<MSG
无法连接机械臂主服务 ${ARM_URL}:${ARM_PORT}。
请先在机械臂容器中启动：  xhost + && airbot_server -i can0 -p ${ARM_PORT}
如果服务在其他地址，设置环境变量后重试，例如：  ARM_URL=192.168.1.10 pixi run bringup
MSG
exit 1
