import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import numpy as np
import argparse

from robot_tools.recorder import ImageRecorderRos
from robot_tools.datar import get_values_by_names
from robot_tools.performancer import Memorier
from robot_tools.pather import get_current_dir

# import sys
# sys.path.insert(0, get_current_dir(__file__))
# from data_driven.hdf5er import save_one_episode
from hdf5er import save_one_episode
from tqdm import tqdm


parser = argparse.ArgumentParser("Record data from the environment")
parser.add_argument("--camera_names", nargs="+", default=("0", "1"))
parser.add_argument("--image_shape", nargs=3, type=int, default=(480, 640, 3))
parser.add_argument("--states_num", type=int, default=7)
parser.add_argument("--name_space", default="/airbot_play")
parser.add_argument("--frequency", type=int, default=25)
parser.add_argument("-mts", "--max_time_steps", type=int, default=25)
parser.add_argument("--output_dir", default=f"{get_current_dir(__file__)}/IL/data/hdf5/blocks_building")
parser.add_argument("-on", "--output_name", type=str, default="episode_0")
args = parser.parse_args()

"""
25*60的时间步（2个480*640相机时）会占用大约2GB的内存
"""

camera_names = args.camera_names
image_shape = tuple(args.image_shape)
states_num = args.states_num
name_space = args.name_space
frequency = args.frequency
max_time_steps = args.max_time_steps
output_name = args.output_name
output_dir = args.output_dir

states_topic = f"{name_space}/joint_states"
arm_action_topic = f"{name_space}/joint_cmd"
gripper_action_topic = f"{name_space}/end_effector/cmd_1dof"

all_data = {
    "/observations/qpos": np.random.rand(max_time_steps, states_num),
    "/observations/qvel": np.random.rand(max_time_steps, states_num),
    "/observations/effort": np.random.rand(max_time_steps, states_num),
    "/action": np.random.rand(max_time_steps, states_num),
}
for name in camera_names:
    all_data[f"/observations/images/{name}"] = [
        np.random.randint(0, 255, image_shape, dtype="uint8")
    ] * max_time_steps

# TODO: Load the current data defination from outside
body_current_data = {
    "/observations/qpos": np.random.rand(states_num),
    "/observations/qvel": np.random.rand(states_num),
    "/observations/effort": np.random.rand(states_num),
    "/action": np.random.rand(states_num),
}

# TODO: Get the keys from the current data defination(observations, observations/images and action are fixed, but the others are not, and the images can not be set)
arm_joint_names = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
gripper_joint_names = ("endleft", "endright")
symmetry = 0.04


def joint_states_callback(data: JointState):
    arm_joints_pos = get_values_by_names(arm_joint_names, data.name, data.position)
    gripper_joints_pos = get_values_by_names(
        gripper_joint_names, data.name, data.position
    )
    gripper_joints_pos = [gripper_joints_pos[0] / symmetry]
    body_current_data["/observations/qpos"] = list(arm_joints_pos) + list(
        gripper_joints_pos
    )
    arm_joints_vel = get_values_by_names(arm_joint_names, data.name, data.velocity)
    gripper_joints_vel = get_values_by_names(
        gripper_joint_names, data.name, data.velocity
    )
    gripper_joints_vel = [gripper_joints_vel[0]]
    body_current_data["/observations/qvel"] = list(arm_joints_vel) + list(
        gripper_joints_vel
    )
    arm_joints_effort = get_values_by_names(arm_joint_names, data.name, data.effort)
    gripper_joints_effort = get_values_by_names(
        gripper_joint_names, data.name, data.effort
    )
    gripper_joints_effort = [gripper_joints_effort[0]]
    body_current_data["/observations/effort"] = list(arm_joints_effort) + list(
        gripper_joints_effort
    )


current_action = {"arm": None, "gripper": None}


def arm_action_callback(data: JointState):
    current_action["arm"] = get_values_by_names(
        arm_joint_names, data.name, data.position
    )

def gripper_action_callback(data: Float64):
    current_action["gripper"] = data.data


rospy.init_node("blocks_building_data_recorder")

print("camera_names:", camera_names)
image_recorder = ImageRecorderRos(camera_names, topic_names=None, show_images=True, image_shape=image_shape)
# # check images shape
# for shape in image_recorder.get_images_shape():
#     assert tuple(shape) == image_shape

states_suber = rospy.Subscriber(states_topic, JointState, joint_states_callback)
arm_action_suber = rospy.Subscriber(arm_action_topic, JointState, arm_action_callback)
gripper_action_suber = rospy.Subscriber(
    gripper_action_topic, Float64, gripper_action_callback
)

print("Waiting for the first arm and gripper message...")
current_body_state: JointState = rospy.wait_for_message(states_topic, JointState)

# 这种两次等待仅适用于持续发送的消息，对于只发送一次的消息，需要使用其他方法
# 否则容易因为不同电脑的性能差异导致错过短期消息而判定为消息未发送
assert len(rospy.wait_for_message(arm_action_topic, JointState).name) == 6
# 所以改成了这种方式
while current_action["gripper"] is None:
    rospy.sleep(0.5)

print("Waiting for the best contour node started...")
while not rospy.get_param("/best_contour_started", False):
    rospy.sleep(0.1)
print("Start recording...")
# TODO: 分次（如每250步）保存数据，避免内存溢出
rate = rospy.Rate(frequency)
for step in tqdm(range(max_time_steps)):
    images = image_recorder.get_images()
    for name in camera_names:
        all_data[f"/observations/images/{name}"][step] = images[name]
    all_data["/observations/qpos"][step] = body_current_data["/observations/qpos"]
    all_data["/observations/qvel"][step] = body_current_data["/observations/qvel"]
    all_data["/observations/effort"][step] = body_current_data["/observations/effort"]
    all_data["/action"][step] = list(current_action["arm"]) + [
        current_action["gripper"]
    ]
    rate.sleep()
print(f"Start saving data to {output_dir}/{output_name}...")
save_one_episode(all_data, camera_names, output_dir, output_name, overwrite=True)
print("Data saved.")
