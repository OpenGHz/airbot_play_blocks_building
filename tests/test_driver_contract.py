"""Contract of the bridge node's arm drivers (moveit1_interface arm_driver.ArmDriver).

The same cases run against every implementation:
  sim        MujocoDriver and a headless `python -m sim.world` on a private socket
  airbot_py  AirbotPyDriver and tests/fake_airbot_server.py on a free port
so the airbot_py / gRPC path the simulation does not exercise still has tests.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time

import pytest
from conftest import REPO

JOINT_TARGET = [0.2, -0.4, 0.4, 0.1, -0.1, 0.3]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def wait_until(cond, timeout: float, period: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(period)
    return cond()


class SimBackend:
    name, is_sim, settle = "sim", True, 5.0

    def __init__(self, tmp_path):
        self.sock = tmp_path / "airbot_sim.sock"
        self.proc = None

    def start(self):
        env = dict(os.environ, AIRBOT_SIM_SOCKET=str(self.sock))
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "sim.world", "--headless", "--no-camera", "--seed", "0"],
            cwd=REPO,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert wait_until(self.sock.exists, 30), "the simulation did not open its socket"

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)  # it homes the arm (3 s) and exits
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def driver(self):
        from sim.driver import MujocoDriver

        return MujocoDriver(path=self.sock, rate=200)

    def absent_driver(self, tmp_path):
        from sim.driver import MujocoDriver

        return MujocoDriver(path=tmp_path / "nobody.sock")


class AirbotPyBackend:
    name, is_sim, settle = "airbot_py", False, 2.0

    def __init__(self, tmp_path):
        self.port = free_port()
        self.server = self.state = None

    def start(self):
        import fake_airbot_server

        self.server, self.state = fake_airbot_server.serve(self.port)

    def stop(self):
        import fake_airbot_server

        if self.server is not None:
            fake_airbot_server.stop(self.server, self.state)
            self.server = None

    def driver(self):
        from arm_driver import AirbotPyDriver

        return AirbotPyDriver(url="localhost", port=self.port)

    def absent_driver(self, tmp_path):
        from arm_driver import AirbotPyDriver

        return AirbotPyDriver(url="localhost", port=free_port())


@pytest.fixture(params=[SimBackend, AirbotPyBackend], ids=["sim", "airbot_py"])
def backend(request, tmp_path):
    b = request.param(tmp_path)
    b.start()
    yield b
    b.stop()


@pytest.fixture
def driver(backend):
    d = backend.driver()
    d.connect(timeout=10.0)
    yield d
    d.disconnect()


def test_info_and_ok(backend, driver):
    info = driver.info()
    assert isinstance(info["name"], str) and info["name"]
    assert info["is_sim"] is backend.is_sim
    assert driver.ok()


def test_state_has_six_joints_and_a_gripper(driver):
    q, dq, tau = driver.get_joint_state()
    assert len(q) == len(dq) == len(tau) == 6
    w, dw, f = driver.get_eef_state()
    assert len(w) >= 1 and len(dw) >= 1 and len(f) >= 1


def test_joint_servo_reaches_the_target(backend, driver):
    driver.enter_servo_mode()
    target = JOINT_TARGET

    def reached():
        driver.servo_joint_pos(target)  # the bridge node streams targets at 200 Hz
        return max(abs(a - b) for a, b in zip(driver.get_joint_state()[0], target)) < 0.01

    assert wait_until(reached, backend.settle, period=0.005), f"joints at {driver.get_joint_state()[0]}"


def test_gripper_servo_reaches_the_width(backend, driver):
    driver.enter_servo_mode()

    def reached(width):
        driver.servo_eef_pos(width)
        return abs(driver.get_eef_state()[0][0] - width) < 0.004

    for width in (0.06, 0.02):
        assert wait_until(lambda width=width: reached(width), backend.settle, period=0.005), (
            f"gripper at {driver.get_eef_state()[0]}"
        )


def test_connect_fails_fast_with_a_hint_when_nothing_is_running(backend, tmp_path):
    d = backend.absent_driver(tmp_path)
    t0 = time.monotonic()
    with pytest.raises(Exception) as err:
        d.connect(timeout=1.0)
    assert time.monotonic() - t0 < 5.0
    assert ("pixi run sim" if backend.is_sim else "airbot_server") in str(err.value)


def test_ok_turns_false_when_the_service_goes_away(backend, driver):
    backend.stop()
    assert wait_until(lambda: not driver.ok(), 15.0), "the bridge node would never notice the lost connection"
