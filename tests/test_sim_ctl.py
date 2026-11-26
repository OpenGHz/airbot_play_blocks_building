"""`sim-ctl arm ...` against a headless `python -m sim.world` (private socket, a copy of the scene
files: the `sim` fixture in conftest.py)."""

from sim import config_file


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
    assert [ln for ln in after.splitlines() if not ln.startswith("MOUNT_OFFSET")] == before.splitlines(), (
        "only the MOUNT_OFFSET line is added; comments and the rest stay"
    )


def test_cube_list_uses_the_moved_base_frame(sim):
    ctl, _ = sim

    def xs():
        return [float(ln.split("(")[1].split(",")[0]) for ln in ctl("cube", "list").splitlines() if "抓取区" in ln]

    x0 = xs()
    ctl("arm", "mount", "--dx", "-0.1")
    assert [round(a - b, 4) for a, b in zip(xs(), x0)] == [0.1] * len(x0), "the cubes are 10 cm further from the base"


def test_arm_turn_and_save(sim):
    ctl, tmp = sim
    out = ctl("arm", "mount", "--x", "-0.1", "--yaw", "10")
    assert "(-0.100, +0.000)" in out and "转角 +10.0°" in out
    assert "转角 +4.5°" in ctl("arm", "mount", "--dyaw", "-5.5")
    out = ctl("arm", "mount", "--x", "0", "--yaw", "30")  # the plate's corner would hang over the front edge
    assert "转角 +30.0°" in out and "底板到桌子边了" in out
    ctl("arm", "mount", "--x", "-0.1", "--yaw", "10")
    ctl("arm", "save")
    cfg = config_file.load(tmp / "sim_floor.yaml")
    assert cfg["MOUNT_OFFSET"] == [-0.1, 0.0] and cfg["MOUNT_YAW"] == 10.0
    lines = (tmp / "sim_floor.yaml").read_text().splitlines()
    assert lines.index("MOUNT_YAW: 10.0") == lines.index("MOUNT_OFFSET: [-0.1, 0.0]") + 1
    ctl("arm", "mount", "--yaw", "0")
    ctl("arm", "save")
    assert config_file.load(tmp / "sim_floor.yaml")["MOUNT_YAW"] == 0.0, "turned back: the saved yaw is reset too"


def test_cube_set(sim):
    ctl, _ = sim
    n = int(ctl("cube", "list").split()[0].removeprefix("cube_"))
    out = ctl("cube", "set", str(n), "--x", "0.34", "--y", "-0.07", "--yaw", "15")
    assert "(+0.340, -0.070," in out and "转角 +15.0°" in out
    line = next(ln for ln in ctl("cube", "list").splitlines() if ln.startswith(f"cube_{n} "))
    assert "(+0.340, -0.070," in line and "+15.0°" in line
    assert "没有移动" in ctl("cube", "set", str(n), "--x", "0", "--y", "0")
