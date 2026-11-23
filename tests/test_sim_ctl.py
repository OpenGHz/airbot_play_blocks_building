"""`sim-ctl arm ...` against a headless `python -m sim.world` (private socket, a copy of the scene files)."""

import os
import shutil
import signal
import subprocess
import sys
import time

import pytest

from conftest import REPO
from sim import config_file


@pytest.fixture
def sim(tmp_path):
    for name in ("sim.yaml", "sim_floor.yaml"):
        shutil.copy(REPO / "configs/scene" / name, tmp_path / name)
    env = dict(os.environ, AIRBOT_SIM_SOCKET=str(tmp_path / "s.sock"))
    proc = subprocess.Popen([sys.executable, "-m", "sim.world", "--scene", str(tmp_path / "sim_floor.yaml"),
                             "--headless", "--no-camera", "--seed", "0"], cwd=REPO, env=env,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.monotonic() + 30
    while not (tmp_path / "s.sock").exists():
        assert time.monotonic() < deadline and proc.poll() is None, "the simulation did not start"
        time.sleep(0.1)

    def ctl(*args):
        out = subprocess.run([sys.executable, "-m", "sim.ctl", *args], cwd=REPO, env=env, capture_output=True,
                             text=True, timeout=20)
        assert out.returncode == 0, out.stderr
        return out.stdout

    yield ctl, tmp_path
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        proc.kill()


def test_arm_mount_where_and_save(sim):
    ctl, tmp = sim
    assert "(+0.000, +0.000)" in ctl("arm", "where")
    assert "(-0.100, +0.050)" in ctl("arm", "mount", "--dx", "-0.1", "--dy", "0.05")
    assert "(-0.100, -0.020)" in ctl("arm", "mount", "--y", "-0.02")
    out = ctl("arm", "mount", "--x", "0.5")  # past the front edge of the table
    assert "(+0.000, -0.020)" in out and "桌子边" in out
    before = (tmp / "sim_floor.yaml").read_text()
    ctl("arm", "save")
    after = (tmp / "sim_floor.yaml").read_text()
    assert config_file.load(tmp / "sim_floor.yaml")["MOUNT_OFFSET"] == [0.0, -0.02]
    assert [ln for ln in after.splitlines() if not ln.startswith("MOUNT_OFFSET")] == before.splitlines(), \
        "only the MOUNT_OFFSET line is added; comments and the rest stay"


def test_cube_list_uses_the_moved_base_frame(sim):
    ctl, _ = sim

    def xs():
        return [float(ln.split("(")[1].split(",")[0]) for ln in ctl("cube", "list").splitlines() if "抓取区" in ln]

    x0 = xs()
    ctl("arm", "mount", "--dx", "-0.1")
    assert [round(a - b, 4) for a, b in zip(xs(), x0)] == [0.1] * len(x0), "the cubes are 10 cm further from the base"
