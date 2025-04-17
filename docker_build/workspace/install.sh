#!/bin/bash

SRC_DIR=$1

# Prepare workspace and packages
mkdir -p $SRC_DIR && cd $SRC_DIR
pip install vcstool -i https://pypi.mirrors.ustc.edu.cn/simple
cd AIRBOT_PLAY/src
vcs import < airbot_app_build/airbot_app.repos

# Configure packages
pushd airbot_play_moveit1_base
chmod +x *.sh && ./config_basic.sh
apt-get install ros-noetic-rviz ros-noetic-moveit -y
popd

pushd airbot_play_control_upper
chmod +x *.sh && ./update.sh
popd

pushd airbot_play_gazebo/src/airbot_play_gazebo && chmod +x *.sh
apt-get install ros-noetic-gazebo-plugins
./gazebo_config.sh
./gazebo_grasp_config.sh
./download_models.sh
popd

cd ..

pip install rosdepc -i https://pypi.mirrors.ustc.edu.cn/simple
rosdepc init && rosdepc update
rosdep install --from-path src --ignore-src -r -y

apt-get install python3-catkin-tools || pip3 install catkin-tools catkin-tools-python
catkin build

echo "source $(pwd)/devel/setup."${SHELL##*/}"" >> ~/."${SHELL##*/}"rc