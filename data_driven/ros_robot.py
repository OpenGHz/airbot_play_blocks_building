import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
from threading import Thread

from robot_tools.datar import get_values_by_names


class RosRobot(object):

    def __init__(
        self,
        states_topic,
        arm_action_topic,
        gripper_action_topic,
        states_num,
        default_joints,
    ) -> None:
        if rospy.get_name() == "/unnamed":
            rospy.init_node("ros_robot_node")

        self.arm_joint_names = (
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        )
        self.gripper_joint_names = ("endleft", "endright")
        self.arm_joints_num = len(self.arm_joint_names)
        self.all_joints_num = self.arm_joints_num + 1
        self.symmetry = 0.04

        # subscribe to the states topics
        assert len(default_joints) == self.all_joints_num
        self.action_cmd = {
            "arm": default_joints[:-1],
            "gripper": self._eef_cmd_convert(default_joints[-1]),
        }
        self.body_current_data = {
            "/observations/qpos": np.random.rand(states_num),
            "/observations/qvel": np.random.rand(states_num),
            "/observations/effort": np.random.rand(states_num),
            "/action": np.random.rand(states_num),
        }
        self.states_suber = rospy.Subscriber(
            states_topic, JointState, self.joint_states_callback
        )
        self.arm_cmd_pub = rospy.Publisher(
            arm_action_topic, Float64MultiArray, queue_size=10
        )
        self.gripper_cmd_pub = rospy.Publisher(
            gripper_action_topic, Float64MultiArray, queue_size=10
        )
        Thread(target=self.publish_action, daemon=True).start()

    def _eef_cmd_convert(self, cmd):
        value = cmd * self.symmetry
        return [value, -value]

    def joint_states_callback(self, data: JointState):
        arm_joints_pos = get_values_by_names(
            self.arm_joint_names, data.name, data.position
        )
        gripper_joints_pos = get_values_by_names(
            self.gripper_joint_names, data.name, data.position
        )
        gripper_joints_pos = [gripper_joints_pos[0] / self.symmetry]
        self.body_current_data["/observations/qpos"] = list(arm_joints_pos) + list(
            gripper_joints_pos
        )
        arm_joints_vel = get_values_by_names(
            self.arm_joint_names, data.name, data.velocity
        )
        gripper_joints_vel = get_values_by_names(
            self.gripper_joint_names, data.name, data.velocity
        )
        gripper_joints_vel = [gripper_joints_vel[0]]
        self.body_current_data["/observations/qvel"] = list(arm_joints_vel) + list(
            gripper_joints_vel
        )
        arm_joints_effort = get_values_by_names(
            self.arm_joint_names, data.name, data.effort
        )
        gripper_joints_effort = get_values_by_names(
            self.gripper_joint_names, data.name, data.effort
        )
        gripper_joints_effort = [gripper_joints_effort[0]]
        self.body_current_data["/observations/effort"] = list(arm_joints_effort) + list(
            gripper_joints_effort
        )

    def publish_action(self):
        rate = rospy.Rate(200)
        while not rospy.is_shutdown():
            self.arm_cmd_pub.publish(Float64MultiArray(data=self.action_cmd["arm"]))
            self.gripper_cmd_pub.publish(
                Float64MultiArray(data=self.action_cmd["gripper"])
            )
            rate.sleep()

    def get_current_joint_positions(self):
        return self.body_current_data["/observations/qpos"]

    def get_current_joint_velocities(self):
        return self.body_current_data["/observations/qvel"]

    def get_current_joint_efforts(self):
        return self.body_current_data["/observations/effort"]

    def set_joint_position_target(self, qpos, qvel=None, blocking=False):  # TODO: add blocking
        self.action_cmd["arm"] = qpos[: self.arm_joints_num]
        if len(qpos) == self.all_joints_num:
            self.action_cmd["gripper"] = self._eef_cmd_convert(
                qpos[self.arm_joints_num]
            )

    def set_target_joint_q(self, qpos, qvel=None, blocking=False):
        self.set_joint_position_target(qpos, qvel, blocking)

    def set_target_end(self, cmd):
        self.action_cmd["gripper"] = self._eef_cmd_convert(cmd)

    def set_joint_velocity_target(self, qvel, blocking=False):
        print("Not implemented yet")

    def set_joint_effort_target(self, qeffort, blocking=False):
        print("Not implemented yet")
