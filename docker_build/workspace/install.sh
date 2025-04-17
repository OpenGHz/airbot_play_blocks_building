#!/bin/bash

set -ex

SRC_DIR=$1

# Prepare workspace and packages
apt install -y python3-pip python3-rosdep python3-vcstool python3-catkin-pkg git
mkdir -p $SRC_DIR 
vcs import $SRC_DIR < block_building_ros1.repos

# Configure packages
cd $SRC_DIR
pushd airbot_play_moveit1_base
chmod +x *.sh && ./config_basic.sh
apt-get install ros-noetic-rviz ros-noetic-moveit -y
popd

pushd airbot_play_control_upper
chmod +x *.sh && ./update.sh
popd

pushd airbot_play_gazebo/src/airbot_play_gazebo && chmod +x *.sh
apt-get install ros-noetic-gazebo-plugins -y
./gazebo_config.sh
./gazebo_grasp_config.sh
./download_models.sh
popd

cd ..

pip install rosdepc -i https://pypi.mirrors.ustc.edu.cn/simple
rosdepc init && rosdepc update
rosdep install --from-path src --ignore-src -r -y

catkin build

echo "source $(pwd)/devel/setup."${SHELL##*/}"" >> ~/."${SHELL##*/}"rc