#!/bin/bash

set -e

python3 ../DISCOVERSE/discoverse/examples/ros1/airbot_play_cam_ros1.py &

roslaunch ./launch/moveit.launch env_type:=mujoco