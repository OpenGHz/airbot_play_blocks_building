ARG BASE_IMAGE=osrf/ros:noetic-desktop-full
ARG ARCH=$(uname -m)

FROM --platform=${ARCH} ${BASE_IMAGE}

COPY workspace /root/workspace
WORKDIR /root/workspace

ARG ROS_WS_SRC=AIRBOT_PLAY_ROS1/src

# Install dependencies
RUN bash change_source.sh && mkdir -p ${ROS_WS_SRC} && bash install.sh ${ROS_WS_SRC}
# Configure the demo
RUN git clone https://github.com/OpenGHz/airbot_play_blocks_building.git && \
    pip install -r airbot_play_blocks_building/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    ln -sf $(pwd)/${ROS_WS_SRC}/airbot_play_control_upper/demo /workspace/airbot_play_blocks_building/demo