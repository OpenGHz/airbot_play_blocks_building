# Sourced by pixi on activation (`pixi run`, `pixi shell`).
# Do not enable `set -u` here: catkin's setup scripts reference unset variables.

# Address of the arm service (`airbot_server -i can0 -p 50051` in the arm container).
export ARM_URL="${ARM_URL:-localhost}"
export ARM_PORT="${ARM_PORT:-50051}"

# Overlay the built ROS workspace, if it has been built (`pixi run build`).
if [ -f "$PIXI_PROJECT_ROOT/ros_ws/devel/setup.sh" ]; then
    if [ -n "${BASH_VERSION:-}" ]; then
        . "$PIXI_PROJECT_ROOT/ros_ws/devel/setup.bash"
    elif [ -n "${ZSH_VERSION:-}" ]; then
        . "$PIXI_PROJECT_ROOT/ros_ws/devel/setup.zsh"
    else
        . "$PIXI_PROJECT_ROOT/ros_ws/devel/setup.sh"
    fi
fi
