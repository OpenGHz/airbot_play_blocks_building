#!/usr/bin/env bash
# Environment self-check. Run through `pixi run doctor`.
cd "$(dirname "$0")/.." || exit 1

fails=0
ok() { printf '  \033[32m✔\033[0m %s\n' "$*"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$*"; }
fail() {
    printf '  \033[31m✘\033[0m %s\n' "$*"
    fails=$((fails + 1))
}

echo "源码"
for sub in ros_ws/src/easy_arm_control/pkgs ros_ws/src/airbot_play_moveit1_base/src ros_ws/src/trac_ik/trac_ik_lib; do
    if [ -d "$sub" ]; then ok "$sub"; else fail "$sub 不存在：执行 git submodule update --init --recursive"; fi
done

echo "ROS 工作空间"
if [ -f ros_ws/devel/setup.sh ]; then
    ok "已编译（ros_ws/devel）"
    for pkg in easy_arm_control airbot_play_launch airbot_play_v3_gripper_gazebo_all_config trac_ik_kinematics_plugin; do
        if rospack find "$pkg" >/dev/null 2>&1; then ok "rospack find $pkg"; else fail "找不到 ROS 包 $pkg：执行 pixi run build"; fi
    done
else
    fail "尚未编译：执行 pixi run build"
fi

echo "Python 模块"
python - <<'PY' 2>/dev/null || fails=$((fails + 1))
import importlib, pathlib, sys
bad = 0
for name in ["rospy", "moveit_commander", "cv_bridge", "easy_arm_control.control",
             "airbot_py.arm", "robot_tools", "std_vision.video", "cv2"]:
    try:
        importlib.import_module(name)
        print(f"  \033[32m✔\033[0m {name}")
    except Exception as e:  # noqa: BLE001
        bad += 1
        print(f"  \033[31m✘\033[0m {name}: {type(e).__name__}: {e}")
copies = {str(p.resolve()) for d in sys.path if d for p in pathlib.Path(d).glob("cv2/__init__.py")}
if len(copies) == 1:
    print("  \033[32m✔\033[0m 只有一份 OpenCV")
else:
    bad += 1
    print(f"  \033[31m✘\033[0m 发现 {len(copies)} 份 OpenCV：{sorted(copies)}")
sys.exit(1 if bad else 0)
PY

echo "图形界面（RViz / OpenCV 窗口）"
if [ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]; then
    ok "DISPLAY=${DISPLAY:-} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-}"
else
    warn "没有图形会话（DISPLAY 未设置），RViz 和视觉窗口无法显示"
fi

echo "机械臂主服务"
if out=$(bash scripts/check_arm.sh 2>&1); then
    ok "$out"
else
    warn "${ARM_URL:-localhost}:${ARM_PORT:-50051} 无法连接（运行 bringup 前需先在机械臂容器中启动 airbot_server）"
fi

echo "相机"
shopt -s nullglob
devices=(/dev/video*)
if [ ${#devices[@]} -eq 0 ]; then
    warn "没有找到 /dev/video* 设备：检查机械臂相机的 USB 连接"
fi
for dev in "${devices[@]}"; do
    n=${dev#/dev/video}
    name=$(cat "/sys/class/video4linux/video$n/name" 2>/dev/null)
    if [ -r "$dev" ] && [ -w "$dev" ]; then
        ok "$dev（设备号 $n）${name:+ - $name}"
    else
        warn "$dev 无读写权限：执行 sudo usermod -aG video \$USER 后重新登录"
    fi
done

echo
if [ "$fails" -eq 0 ]; then
    echo "检查通过。"
else
    echo "有 $fails 项检查未通过，见上方 ✘。"
    exit 1
fi
