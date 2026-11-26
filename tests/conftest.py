"""Make the repository's top-level modules, the bridge node's arm_driver and the test helpers importable;
the `sim` fixture (a headless simulation on a private socket)."""

import os
import pathlib
import shutil
import signal
import subprocess
import sys
import time

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
BRIDGE_SCRIPTS = REPO / "ros_ws/src/airbot_play_moveit1_base/src/airbot_play_follow_basic/scripts"

for path in (REPO, BRIDGE_SCRIPTS, REPO / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@pytest.fixture
def sim(tmp_path, monkeypatch):
    """A headless `python -m sim.world` on a private socket with a copy of the scene files.
    Yields (ctl, tmp_path): ctl(*args) runs `sim-ctl args` against it and returns its output;
    AIRBOT_SIM_SOCKET is set for the test, so MujocoDriver() connects to it too."""
    for name in ("sim.yaml", "sim_floor.yaml"):
        shutil.copy(REPO / "configs/scene" / name, tmp_path / name)
    env = dict(os.environ, AIRBOT_SIM_SOCKET=str(tmp_path / "s.sock"))
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "sim.world",
            "--scene",
            str(tmp_path / "sim_floor.yaml"),
            "--headless",
            "--no-camera",
            "--seed",
            "0",
        ],
        cwd=REPO,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 30
    while not (tmp_path / "s.sock").exists():
        assert time.monotonic() < deadline and proc.poll() is None, "the simulation did not start"
        time.sleep(0.1)

    def ctl(*args):
        out = subprocess.run(
            [sys.executable, "-m", "sim.ctl", *args], cwd=REPO, env=env, capture_output=True, text=True, timeout=20
        )
        assert out.returncode == 0, out.stderr
        return out.stdout

    monkeypatch.setenv("AIRBOT_SIM_SOCKET", env["AIRBOT_SIM_SOCKET"])
    yield ctl, tmp_path
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()
