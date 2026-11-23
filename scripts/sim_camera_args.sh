#!/usr/bin/env bash
# Sourced by the camera-using wrappers. In the simulation without a v4l2loopback device,
# the wrist camera is the ROS topic /camera/color/image_raw: map a device number given
# with OPT (e.g. -vd 2, -d 2) to that topic so the manual's commands work unchanged.
# Usage: map_camera_args OPT "$@"; result in the array ARGS.
map_camera_args() {
    local opt=$1
    shift
    ARGS=()
    while [ $# -gt 0 ]; do
        if [ "$1" = "$opt" ] && [ $# -ge 2 ] && [[ "$2" =~ ^[0-9]+$ ]] && [ "$ENV_TYPE" = "sim" ] \
            && ! grep -qs AIRBOT-Sim "/sys/class/video4linux/video$2/name"; then
            echo "仿真：没有虚拟 USB 相机 /dev/video$2，改用 ROS 话题 /camera/color/image_raw" >&2
            ARGS+=("$1" /camera/color/image_raw)
            shift 2
            continue
        fi
        ARGS+=("$1")
        shift
    done
}
