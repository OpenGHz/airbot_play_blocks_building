"""The bridge node's gripper topics use one scale.

/airbot_play/gripper/set_position is the opening width / 0.07 (1 open, 0 closed), and easy_arm_control's
gripper_control ramps from /airbot_play/gripper/position to the target. The position used to be published
as (0.07 - width) / 2 / 0.07, so an open gripper read 0: every "open" started its ramp at "closed" and the
gripper snapped shut before opening (seen at the start of pick-place), and every "close" jumped at once.
"""

from __future__ import annotations

import airbot_play_ros_interface as bridge
import pytest
import rospy


class FakeDriver:
    def __init__(self, width: float):
        self.width = width
        self.eef_cmd = []

    def connect(self, timeout=3.0):
        pass

    def info(self):
        return {"name": "fake", "is_sim": True}

    def enter_servo_mode(self):
        pass

    def ok(self):
        return True

    def servo_joint_pos(self, q):
        pass

    def servo_eef_pos(self, width):
        self.eef_cmd.append(width)

    def get_joint_state(self):
        return [0.0] * 6, [0.0] * 6, [0.0] * 6

    def get_eef_state(self):
        return [self.width], [0.0], [0.0]


class FakePublisher:
    def __init__(self, topic, *args, **kwargs):
        self.topic, self.sent = topic, []

    def publish(self, msg):
        self.sent.append(msg)


@pytest.fixture
def make_bridge(monkeypatch):
    monkeypatch.setattr(rospy, "Publisher", FakePublisher)
    monkeypatch.setattr(rospy, "Subscriber", lambda *a, **k: None)
    monkeypatch.setattr(rospy, "Timer", lambda *a, **k: None)
    monkeypatch.setattr(rospy.Time, "now", staticmethod(lambda: rospy.Time(0)))

    def make(width):
        driver = FakeDriver(width)
        return bridge.AirbotPlayRosInterface(driver), driver

    return make


@pytest.mark.parametrize("width, expected", [(0.07, 1.0), (0.0, 0.0), (0.025, 0.025 / 0.07)])
def test_gripper_position_is_on_the_command_scale(make_bridge, width, expected):
    node, _ = make_bridge(width)
    node.pub_joint_states(None)
    assert node.gripper_position_puber.sent[-1].data == pytest.approx(expected)


def test_feedback_of_a_command_reads_back_the_command(make_bridge):
    node, driver = make_bridge(0.07)
    for cmd in (1.0, 0.36, 0.0):
        node.gripper_float_cmd_cb(bridge.Float64(data=cmd))
        driver.width = driver.eef_cmd[-1]  # the gripper got there
        node.pub_joint_states(None)
        assert node.gripper_position_puber.sent[-1].data == pytest.approx(cmd)
