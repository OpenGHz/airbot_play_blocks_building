#!/bin/bash

# >=24.04
# sed -i "s http://.*archive.ubuntu.com http://repo.huaweicloud.com g" /etc/apt/sources.list.d/ubuntu.sources
# sed -i "s http://.*security.ubuntu.com http://repo.huaweicloud.com g" /etc/apt/sources.list.d/ubuntu.sources

# <24.04
sudo sed -i "s@http://.*archive.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list
sudo sed -i "s@http://.*security.ubuntu.com@http://mirrors.huaweicloud.com@g" /etc/apt/sources.list
apt-get update