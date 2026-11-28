#!/usr/bin/env bash
# `pixi run keyboard`: keyboard teleoperation with configs/control/<env_type>.yaml.
cd "$(dirname "$0")/.." || exit 1
ENV_TYPE=$(bash scripts/env_type.sh)
exec rosrun easy_arm_control airbot_play_keyboard.py --init_pose "configs/control/${ENV_TYPE}.yaml" "$@"
