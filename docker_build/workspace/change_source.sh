#!/bin/bash

# >=24.04
# sed -i "s http://.*archive.ubuntu.com http://repo.huaweicloud.com g" /etc/apt/sources.list.d/ubuntu.sources
# sed -i "s http://.*security.ubuntu.com http://repo.huaweicloud.com g" /etc/apt/sources.list.d/ubuntu.sources

# <24.04
sed -i "s@http://.*archive.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list
sed -i "s@http://.*security.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list

rm /etc/apt/sources.list.d/ros1-latest.list
sh -c 'echo "deb https://mirrors.huaweicloud.com/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
apt-get update

apt install -y python3-pip
pip config set global.index-url https://mirrors.huaweicloud.com/repository/pypi/simple