"""The panel (sim/panel.py) against a headless simulation (the `sim` fixture in conftest.py).

Runs on Qt's "minimal" platform: no display needed, and unlike "offscreen" it does not crash in
fontconfig on the Chinese labels. The window is never shown; the widgets work all the same.
"""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")

from sim import config_file  # noqa: E402
from sim.driver import MujocoDriver  # noqa: E402
from sim.panel import Panel  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication(["test"])


@pytest.fixture
def panel(app, sim):
    ctl, tmp = sim
    d = MujocoDriver(rate=1)
    d.connect(5.0)
    p = Panel(d, poll_ms=50)
    yield p, ctl, tmp, d
    p.poll_timer.stop()
    p.send_timer.stop()
    d.disconnect()


def pump(app, seconds=0.3):
    t = time.monotonic()
    while time.monotonic() - t < seconds:
        app.processEvents()
        time.sleep(0.01)


def where(d):
    r = d.request("ctl", cmd="arm_where", args={})
    return r["mount"], r["yaw"]


def test_typed_values_move_the_arm(app, panel):
    p, _, _, d = panel
    p.mount.axes["x"].spin.setValue(-123.4)  # what Enter after typing does
    p.mount.axes["y"].spin.setValue(56.5)
    mount, yaw = where(d)
    assert mount == pytest.approx([-0.1234, 0.0565]) and yaw == pytest.approx(0.0)
    p.mount.axes["yaw"].spin.setValue(12.5)
    assert where(d)[1] == pytest.approx(12.5)
    assert "重新记录" in p.mount.note.text()


def test_slider_moves_the_arm(app, panel):
    p, _, _, d = panel
    ax = p.mount.axes["x"]
    ax.slider.setValue(round(-80.0 / ax.tick))
    assert ax.spin.value() == pytest.approx(-80.0), "the number follows the slider"
    pump(app)
    assert where(d)[0][0] == pytest.approx(-0.08)


def test_turning_at_the_edge_moves_in_and_back_out(app, panel):
    p, _, _, d = panel
    p.mount.axes["yaw"].spin.setValue(20.0)  # at the scene file's mount the plate is flush with the front edge
    (x, _), yaw = where(d)
    assert yaw == pytest.approx(20.0) and x < -0.03
    assert p.mount.axes["x"].spin.value() == pytest.approx(x * 1000, abs=0.05), "the panel shows where the arm went"
    assert "桌子边" in p.mount.status.text()
    p.mount.axes["yaw"].spin.setValue(0.0)
    assert where(d) == (pytest.approx([0.0, 0.0]), pytest.approx(0.0)), "turned back: flush with the edge again"


def test_moves_from_elsewhere_show_up(app, panel):
    p, ctl, _, _ = panel
    ctl("arm", "mount", "--dy", "0.05", "--yaw", "-7")  # like the mouse in the MuJoCo window
    pump(app)
    assert p.mount.axes["y"].spin.value() == pytest.approx(50.0)
    assert p.mount.axes["yaw"].spin.value() == pytest.approx(-7.0)


def test_home_and_save(app, panel):
    p, _, tmp, d = panel
    p.mount.axes["x"].spin.setValue(-100.0)
    p.mount.axes["yaw"].spin.setValue(5.0)
    p.mount.save_btn.click()
    cfg = config_file.load(tmp / "sim_floor.yaml")
    assert cfg["mount_offset"] == [-0.1, 0.0] and cfg["mount_yaw"] == 5.0
    assert "mount_yaw" in p.mount.note.text()
    p.mount.home_btn.click()
    assert where(d) == (pytest.approx([0.0, 0.0]), pytest.approx(0.0))
    assert p.mount.note.text() == ""


def test_closes_when_the_simulation_goes_away(app, panel):
    p, _, _, d = panel
    d.disconnect()
    pump(app, 0.3)
    assert p.closing and not p.isEnabled()


# --- cubes -------------------------------------------------------------------------------------
def cubes(d):
    return {c["cube"]: c for c in d.request("ctl", cmd="cube_list", args={})["cubes"]}


def test_cube_typed_values_move_it(app, panel):
    p, _, _, d = panel
    c = p.cubes
    assert c.cube in cubes(d) and c.combo.count() == len(cubes(d))
    z0 = cubes(d)[c.cube]["base_xyz"][2]  # on the floor
    c.axes["x"].spin.setValue(350.0)  # past the pick area: free floor
    c.axes["y"].spin.setValue(-80.0)
    c.axes["yaw"].spin.setValue(20.0)
    st = cubes(d)[c.cube]
    assert st["base_xyz"][:2] == pytest.approx([0.35, -0.08], abs=1e-4) and st["yaw"] == pytest.approx(20.0)
    assert st["base_xyz"][2] == pytest.approx(z0, abs=1e-3), "still on the floor"
    assert "没有移动" not in c.status.text()


def test_cube_on_another_cube(app, panel):
    p, _, _, d = panel
    c = p.cubes
    a, b = sorted(cubes(d))[:2]
    c.combo.setCurrentIndex(c.combo.findData(a))
    x, y, z = cubes(d)[b]["base_xyz"]
    c.axes["x"].spin.setValue(x * 1000)
    c.axes["y"].spin.setValue(y * 1000)
    za = cubes(d)[a]["base_xyz"][2]
    assert za == pytest.approx(z + 0.025, abs=1e-3), "set down on top of the other cube"
    shown = float(c.status.text().split("中心高度 z ")[1].split()[0])
    assert shown == pytest.approx(za * 1000, abs=0.5), "the panel shows the new height"


def test_cube_refused_inside_the_arm(app, panel):
    p, _, _, d = panel
    c = p.cubes
    before = cubes(d)[c.cube]["base_xyz"]
    c.axes["x"].spin.setValue(0.0)  # the arm's base centre
    c.axes["y"].spin.setValue(0.0)
    assert cubes(d)[c.cube]["base_xyz"] == pytest.approx(before, abs=1e-4), "refused: the cube stays"
    assert "没有移动" in c.status.text() and "机械臂" in c.status.text()


def test_cube_moved_elsewhere_shows_up(app, panel):
    p, ctl, _, _ = panel
    c = p.cubes
    ctl("cube", "set", str(c.cube), "--x", "0.40", "--y", "0", "--yaw", "-10")  # free floor
    pump(app)
    assert c.axes["x"].spin.value() == pytest.approx(400.0)
    assert c.axes["yaw"].spin.value() == pytest.approx(-10.0)


def test_cube_double_clicked_in_mujoco_is_picked(app, panel):
    p, _, _, d = panel
    c = p.cubes
    listing = d.request("ctl", cmd="cube_list", args={})
    other = next(i for i in sorted(cubes(d)) if i != c.cube)
    c._list(dict(listing, selected=other))  # what the poll sees after a double-click on `other`
    assert c.cube == other and c.combo.currentData() == other
    assert c.axes["x"].spin.value() == pytest.approx(cubes(d)[other]["base_xyz"][0] * 1000, abs=0.05)


def test_cube_list_follows_the_scene(app, panel):
    p, ctl, _, d = panel
    c = p.cubes
    before = {i: s["base_xyz"] for i, s in c.states.items()}
    ctl("cube", "take")  # the pick area is emptied and refilled after a moment, from the same pool
    pump(app, 1.5)
    now = cubes(d)
    assert {i: s["base_xyz"] for i, s in now.items()} != before, "a new layout"
    assert set(c.states) == set(now) and c.combo.count() == len(now) and c.cube in now
    assert c.axes["x"].spin.value() == pytest.approx(now[c.cube]["base_xyz"][0] * 1000, abs=0.05)


def test_quit_button_stops_the_simulation(app, panel, monkeypatch):
    p, _, tmp, _ = panel
    monkeypatch.setattr(p, "confirm_quit", lambda: False)
    p.quit_btn.click()
    assert not p.closing and (tmp / "s.sock").exists(), "cancelled: nothing happens"
    monkeypatch.setattr(p, "confirm_quit", lambda: True)
    p.quit_btn.click()
    assert p.closing and not p.isEnabled() and "退出" in p.lost.text()
    deadline = time.monotonic() + 15
    while (tmp / "s.sock").exists():  # the simulation removes its socket when it exits
        assert time.monotonic() < deadline, "the simulation did not exit"
        time.sleep(0.2)
