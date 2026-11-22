#!/usr/bin/env bash
# Initialise and configure the catkin workspace in ros_ws/. Idempotent.
# Run through `pixi run configure` (also run automatically by `pixi run build`).
set -eo pipefail
cd "$(dirname "$0")/../ros_ws"

[ -d .catkin_tools ] || catkin init >/dev/null

# Always extend the pixi environment itself, never a previously built
# ros_ws/devel that the activation script may have sourced.
catkin config --extend "$CONDA_PREFIX" \
    --skiplist \
        trac_ik trac_ik_python trac_ik_examples \
        airbot_play_inspire_eg2_4c_v2_arm_group_config \
        airbot_play_v3_inspire_eg2_4c_rm2_v1_all_config \
        airbot_play_v3_inspire_eg2_4c_v2_all_config \
    --cmake-args \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DCMAKE_CXX_STANDARD=17 \
        -Wno-dev \
    >/dev/null
# Why:
# - CMAKE_POLICY_VERSION_MINIMUM: CMake 4 rejects cmake_minimum_required(<3.5),
#   which several upstream packages still declare.
# - CMAKE_CXX_STANDARD=17: Eigen 5 needs C++14+, trac_ik 1.6.6 hard-codes C++11.
# - skiplist: trac_ik's Python bindings/examples, and MoveIt configs of other
#   AIRBOT models shipped in moveit1_interface.
echo "catkin workspace configured: $(pwd)"
