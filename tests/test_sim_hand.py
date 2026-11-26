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


def plate_corners_world(w):
    """The base plate's corners (world x, y) where the arm stands now."""
    x0, x1, y0, y1 = scene.base_plate()
    return w.frame.to_world(np.array([(x0, y0), (x0, y1), (x1, y0), (x1, y1)]))


def on_table(w, xy, tol=1e-6):
    (x0, x1, y0, y1), _ = scene.table_rect(w.cfg)  # scene frame = world x, y
    return np.all((xy[:, 0] >= x0 - tol) & (xy[:, 0] <= x1 + tol) & (xy[:, 1] >= y0 - tol) & (xy[:, 1] <= y1 + tol))


def test_turn_the_arm_base(world):
    w, hand = world, Hand(world)
    base0 = w.data.xpos[w.root].copy()
    q0 = np.array([1.0, 0, 0, 0])
    sel = drag(hand, w, "base_link", ROTATE, (base0, q0), [(base0, yaw_quat(0.1)), (base0, yaw_quat(0.35))])
    assert w.mount_yaw == pytest.approx(0.35)
    assert np.allclose(w.data.xquat[w.root], yaw_quat(0.35), atol=1e-9)
    # The scene file's mount has the plate's front edge on the table edge: turning there moves the arm in.
    assert w.mount[0] < -0.02 and w.mount[1] == pytest.approx(0.0)
    assert on_table(w, plate_corners_world(w)), "the base plate stays on the table top"
    assert max(plate_corners_world(w)[:, 0]) == pytest.approx(scene.table_rect(w.cfg)[0][1]), "flush with the edge"
    hand.update(sel, 0, base0, yaw_quat(0.35))
    assert "deg" in hand.texts()[0][3] and "re-record" in hand.texts()[0][3]


def test_turned_base_frame(world):
    w = world
    w.set_mount((-0.10, 0.05), math.radians(30))
    assert np.allclose(w.frame.to_world(w.frame.to_base([0.3, -0.2, 0.6])), [0.3, -0.2, 0.6])
    i = w.pick.batch[0]
    p = w.cube_pos(i).copy()
    listed = {c["cube"]: c["base_xyz"] for c in w.ctl("cube_list", {})["cubes"]}
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    d = p - w.base
    assert np.allclose(listed[i], [c * d[0] + s * d[1], -s * d[0] + c * d[1], d[2]]), "cube positions in the base frame"
    # Kinematics in the turned base frame equal those of an arm standing on the scene file's mount.
    q = np.array([0.4, -0.8, 1.2, 0.1, 0.5, -0.3])
    turned = kin.Arm(w.model, mujoco.MjData(w.model), w.frame)
    plain = World(scene.load_config("floor"))
    straight = kin.Arm(plain.model, mujoco.MjData(plain.model), plain.frame)
    (p1, r1), (p2, r2) = turned.fk(q), straight.fk(q)
    assert np.allclose(p1, p2, atol=1e-9) and np.allclose(r1, r2, atol=1e-9)
    assert np.allclose(turned.camera(q)[0], straight.camera(q)[0], atol=1e-9)
    target = [0.25, 0.05, -0.30]
    sol = turned.ik(target, restarts=40)
    assert sol is not None and np.allclose(turned.fk(sol)[0], target, atol=2e-4), "IK in the turned frame"


def test_nudge_moves_along_the_base_frame(world):
    w = world
    w.set_mount((-0.10, 0.0), math.radians(90))
    w.ctl("cube_to_gripper", {"color": "green"})
    i = max(w.active(), key=lambda k: w.cube_pos(k)[2])  # the one in the gripper
    before = w.to_base(w.cube_pos(i)).copy()
    w.ctl("cube_nudge", {"dx": 0.01})
    assert np.allclose(w.to_base(w.cube_pos(i)) - before, [0.01, 0.0, 0.0], atol=1e-9)


def test_mount_limits_keep_the_turned_plate_on_the_table(world):
    w = world
    for deg in (0, 20, 45, 90, 135, -60, 180):
        x0, x1, y0, y1 = scene.mount_limits(w.cfg, math.radians(deg))
        for corner in ((x0, y0), (x1, y1), (x0, y1), (x1, y0)):
            w.set_mount(corner, math.radians(deg))
            assert np.allclose(w.mount, corner, atol=1e-9) or w.mount_yaw != pytest.approx(math.radians(deg))
            assert on_table(w, plate_corners_world(w)), f"plate off the table at {deg} deg, {corner}"


def test_mount_yaw_in_the_scene_file():
    cfg = dict(scene.load_config("floor"), MOUNT_OFFSET=[-0.1, 0.05], MOUNT_YAW=15.0)
    w = World(cfg)
    assert np.allclose(w.data.xquat[w.root], yaw_quat(math.radians(15)), atol=1e-9)
    assert w.ctl("arm_where", {})["yaw"] == pytest.approx(15.0)
    assert not {"MOUNT_OFFSET", "MOUNT_YAW"} & set(scene.without_mount_offset(cfg))
    # A copy of the model (the wrist camera process, sim-record) put where the simulation says.
    model, data, _ = scene.build_physics(scene.load_config("floor"))
    state = w.handle("get_qpos", {})
    data.qpos[:] = state["qpos"]
    scene.place_base(model, state["mount"], state["mount_yaw"])
    mujoco.mj_forward(model, data)
    mujoco.mj_forward(w.model, w.data)
    assert np.allclose(data.cam_xpos[model.camera("eye").id], w.data.cam_xpos[w.model.camera("eye").id], atol=1e-9)


def test_set_a_cube_down(world):
    w = world
    s = w.cfg["CUBE_SIZE"] / 2
    floor_z = w.to_base([0.0, 0.0, 0.0])[2]
    a, b = w.pick.batch[:2]
    assert w.set_cube(a, (0.36, -0.06), math.radians(10)) == ""
    st = w.cube_state(a)
    assert st["base_xyz"] == pytest.approx([0.36, -0.06, floor_z + s + 0.0005], abs=1e-6)
    assert st["yaw"] == pytest.approx(10.0)
    settle(w)
    assert w.cube_state(a)["base_xyz"] == pytest.approx([0.36, -0.06, floor_z + s], abs=1e-3), "it stays"
    bx, by, bz = w.cube_state(b)["base_xyz"]
    assert w.set_cube(a, (bx, by)) == "" and w.cube_state(a)["base_xyz"][2] == pytest.approx(bz + 2 * s, abs=1e-3)
    assert w.cube_state(a)["yaw"] == pytest.approx(10.0), "None keeps the turn"


def test_set_a_cube_refused(world):
    w = world
    a, b = w.pick.batch[:2]
    before = w.cube_pos(a).copy()
    why = w.set_cube(a, (0.0, 0.0))  # the arm's base centre: on the table top, in the base
    assert why.startswith("arm:") and np.allclose(w.cube_pos(a), before)
    bx, by, _ = w.cube_state(b)["base_xyz"]
    before = w.cube_pos(a).copy()
    assert w.set_cube(a, (bx + 0.02, by), 0.0) == f"cube:{b}", "not balanced on the edge of another cube"
    assert np.allclose(w.cube_pos(a), before)
    w.park(b)
    with pytest.raises(RuntimeError):
        w.set_cube(b, (0.3, 0.0))


def test_cube_positions_in_a_turned_base_frame(world):
    w = world
    w.set_mount((-0.10, 0.05), math.radians(30))
    i = w.pick.batch[0]
    assert w.set_cube(i, (0.30, 0.10), math.radians(-20)) == ""
    st = w.cube_state(i)
    assert st["base_xyz"][:2] == pytest.approx([0.30, 0.10], abs=1e-6) and st["yaw"] == pytest.approx(-20.0)
    assert np.allclose(w.cube_pos(i)[:2], w.frame.to_world((0.30, 0.10)))
    assert math.degrees(w.cube_yaw(i)) == pytest.approx(10.0), "-20 deg from the base x axis, which is at +30"
    w.place_cube(i, w.cube_pos(i), math.radians(30 + 80))
    assert w.cube_state(i)["yaw"] == pytest.approx(-10.0), "a cube looks the same every 90 deg"


def test_cube_selection_in_the_viewer(world):
    w = world
    i = w.pick.batch[0]
    assert w.ctl("cube_list", {})["selected"] is None
    w.selected = w.model.body(f"cube_{i}").id
    assert w.ctl("cube_list", {})["selected"] == i
    w.ctl("cube_select", {"cube": w.pick.batch[1]})
    assert w.want_select == w.model.body(f"cube_{w.pick.batch[1]}").id
