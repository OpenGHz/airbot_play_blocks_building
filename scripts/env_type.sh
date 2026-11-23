#!/usr/bin/env bash
# Prints the /env_type that bringup set (real or sim); "real" when ROS is not up.
v=$(rosparam get /env_type 2>/dev/null) || v=real
echo "${v:-real}"
