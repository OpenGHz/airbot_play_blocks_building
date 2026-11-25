"""Moving cubes and the arm's mount in the simulation window (sim/hand.py, World.set_mount / put_down).

The viewer's UI thread fills in its perturbation (selected body, active drag, reference pose)
from the mouse; these tests feed Hand the same values without a window.
"""

import math

import mujoco
import numpy as np
import pytest

from sim import kin, scene
from sim.hand import ROTATE, TRANSLATE, Hand
from sim.world import World


@pytest.fixture
def world():
    w = World(scene.load_config("floor"))
    w.pick.rng = np.random.default_rng(0)
    w.pick.spawn()
    return w


def settle(w, hand=None, seconds=0.5):
    for _ in range(int(seconds / w.model.opt.timestep)):
        if hand is not None:
            hand.before_step()
        mujoco.mj_step(w.model, w.data)


def body(w, name):
    return w.model.body(name).id


def yaw_quat(yaw):
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def drag(hand, w, name, active, start, moves):
    """Ctrl-drag body `name`: start the perturbation at `start` (refpos, refquat), then go through `moves`."""
    sel = body(w, name)
    hand.update(sel, active, *start)
    for refpos, refquat in moves:
        hand.update(sel, active, refpos, refquat)
        settle(w, hand, 0.02)
    return sel


def test_carry_a_cube_and_let_go(world):
    w, hand = world, Hand(world)
    i = w.pick.batch[0]
    p0 = w.cube_pos(i).copy()
    q0 = yaw_quat(w.cube_yaw(i))
    target = p0 + [0.06, -0.04, 0.05]
    sel = drag(hand, w, f"cube_{i}", TRANSLATE, (p0, q0), [(p0 + [0.03, -0.02, 0.03], q0), (target, q0)])
    assert np.allclose(w.cube_pos(i), target, atol=1e-4), "the cube follows the hand"
    hand.update(sel, 0, target, q0)  # let go
    settle(w)
    p = w.cube_pos(i)
    assert np.allclose(p[:2], target[:2], atol=0.003)
    assert p[2] == pytest.approx(w.cfg["CUBE_SIZE"] / 2, abs=0.001), "set down on the floor"
    up = w.data.xmat[body(w, f"cube_{i}")].reshape(3, 3)[:, 2]
    assert up[2] > 0.999, "upright"


def test_let_go_above_another_cube_stacks_it(world):
    w, hand = world, Hand(world)
    a, b = w.pick.batch[:2]
    pa, pb = w.cube_pos(a).copy(), w.cube_pos(b).copy()
    above_b = np.array([pb[0], pb[1], pb[2] + 0.06])
    q = yaw_quat(w.cube_yaw(a))
    sel = drag(hand, w, f"cube_{a}", TRANSLATE, (pa, q), [(pa + [0, 0, 0.06], q), (above_b, q)])
    hand.update(sel, 0, above_b, q)
    settle(w)
    assert w.cube_pos(a)[2] == pytest.approx(pb[2] + w.cfg["CUBE_SIZE"], abs=0.002)
    assert np.allclose(w.cube_pos(a)[:2], pb[:2], atol=0.004)


def test_turn_a_cube(world):
    w, hand = world, Hand(world)
    i = w.pick.batch[0]
    p, yaw0 = w.cube_pos(i).copy(), w.cube_yaw(i)
    sel = drag(hand, w, f"cube_{i}", ROTATE, (p, yaw_quat(yaw0)), [(p, yaw_quat(yaw0 + math.radians(30)))])
    hand.update(sel, 0, p, yaw_quat(yaw0 + math.radians(30)))
    settle(w)
    turned = math.degrees(w.cube_yaw(i) - yaw0)
    assert (turned + 45) % 90 - 45 == pytest.approx(30 - 90 * round(30 / 90), abs=1.0)


def test_drag_the_arm_base_slides_only_the_arm(world):
    w, hand = world, Hand(world)
    table = w.data.geom_xpos[w.model.geom("table_top").id].copy()
    i = w.pick.batch[0]
    cube = w.cube_pos(i).copy()
    base0 = w.data.xpos[w.root].copy()
    drag(
        hand,
        w,
        "base_link",
        TRANSLATE,
        (base0, np.array([1.0, 0, 0, 0])),
        [(base0 + [-0.10, 0.05, 0.02], [1.0, 0, 0, 0])],
    )
    assert np.allclose(w.mount, [-0.10, 0.05], atol=1e-9)
    assert np.allclose(w.data.xpos[w.root][:2], base0[:2] + [-0.10, 0.05], atol=1e-6)
    assert w.data.xpos[w.root][2] == pytest.approx(base0[2]), "the base stays on the table top"
    assert np.allclose(w.data.geom_xpos[w.model.geom("table_top").id], table), "the table does not move"
    assert np.allclose(w.cube_pos(i), cube, atol=1e-3), "the cubes do not move"
    listed = {c["cube"]: c["base_xyz"] for c in w.ctl("cube_list", {})["cubes"]}
    assert np.allclose(listed[i], w.cube_pos(i) - w.base), "positions are reported in the moved base frame"


def test_drag_the_arm_base_prints_nothing(world, capsys):
    w, hand = world, Hand(world)
    base0 = w.data.xpos[w.root].copy()
    q = np.array([1.0, 0, 0, 0])
    for _ in range(3):  # three drags, each let go
        sel = drag(hand, w, "base_link", TRANSLATE, (base0, q), [(base0 + [-0.02, 0.01, 0.0], q)])
        hand.update(sel, 0, base0, q)
        base0 = w.data.xpos[w.root].copy()
    assert capsys.readouterr().out == "", "moving the arm base does not print in the terminal"
    assert np.allclose(w.mount, [-0.06, 0.03], atol=1e-9)
    assert "re-record the positions" in hand.texts()[0][3], "the overlay says the recorded positions no longer match"
    w.set_mount((0.0, 0.0))
    assert "re-record" not in hand.texts()[0][3], "back on the scene file's mount: no reminder"
    assert all(t.isascii() for item in hand.texts() for t in item[2:]), "the viewer font is ASCII only"


def test_mount_stays_on_the_table(world):
    w = world
    where, why = w.set_mount((0.3, 0.9))
    x0, x1, y0, y1 = scene.mount_limits(w.cfg)
    assert why == "edge" and where[0] == pytest.approx(x1) and where[1] == pytest.approx(y1)


def test_mount_refuses_to_put_the_arm_into_the_table(world):
    w = world
    arm = kin.Arm(w.model, mujoco.MjData(w.model), w.base)
    q = arm.ik([0.25, 0.0, -0.46], restarts=60)  # reaching down to the floor in front of the table
    assert q is not None
    scene.set_arm(w.model, w.data, q, scene.FINGER_OPEN)
    mujoco.mj_forward(w.model, w.data)
    where, why = w.set_mount((-0.30, 0.0))  # the forearm would go through the table top
    assert why.startswith("table:")
    assert np.allclose(where, [0.0, 0.0]) and np.allclose(w.mount, [0.0, 0.0])


def test_mount_offset_in_the_scene_file(world):
    cfg = dict(scene.load_config("floor"), MOUNT_OFFSET=[-0.1, 0.05])
    w = World(cfg)
    assert np.allclose(w.data.xpos[w.root][:2], [-0.1, 0.05])
    assert np.allclose(scene.base_offset(cfg)[:2], [-0.1, 0.05])
    assert np.allclose(scene.scene_offset(cfg)[:2], [0.0, 0.0])
    assert "MOUNT_OFFSET" not in scene.without_mount_offset(cfg)
