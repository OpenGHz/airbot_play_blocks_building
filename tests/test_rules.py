"""The task's pure rules: layer heights, retreat direction, planning-scene heights, tracking limit."""

import math

import pytest

from airbot_play_rules import (
    TrackLimit,
    fingers_beside_top_block,
    first_place_xy,
    floor_and_tower_z,
    place_aligned,
    place_settle_time,
    release_z,
    retreat_sign,
    slot_axis,
    slot_exit_targets,
    tower_footprint,
)

H = 0.025  # cube_height


def test_settle_time_unset_keeps_the_old_waits():
    assert place_settle_time(None, first=True) == 1.0  # before releasing block 1
    assert place_settle_time(None, first=False) == 3.0  # before the vision adjustment: 1 s + 2 s


def test_settle_time_set_applies_to_every_block():
    assert place_settle_time(0.5, first=True) == 0.5
    assert place_settle_time(2, first=False) == 2.0


def test_release_height_with_the_default_place_height_is_unchanged():
    # place_position_z: pick_grasp_z (every shipped config): same heights as the old code, grasp + k * h.
    grasp = 0.015
    for k in range(30):
        assert release_z(grasp, k, H) == pytest.approx(grasp + k * H)


def test_release_height_follows_place_height_on_every_layer():
    # The old code used place_position_z for the first block only and pick_grasp_z after that.
    grasp, place = 0.015, 0.011
    assert release_z(place, 0, H) == pytest.approx(place)
    assert release_z(place, 4, H) == pytest.approx(place + 4 * H)
    assert release_z(place, 4, H) != pytest.approx(grasp + 4 * H)


@pytest.mark.parametrize(
    "pick_y, place_y, sign",
    [
        (0.0, 0.2145, -1.0),  # configs/control/real.yaml: tower to the left, back off to -y like the old fixed shift
        (0.0, -0.2121, 1.0),  # configs/control/sim.yaml: tower to the right
        (0.1, 0.1, 1.0),
    ],
)
def test_retreat_goes_towards_the_pick_area(pick_y, place_y, sign):
    assert retreat_sign(pick_y, place_y) == sign


def test_floor_and_tower_heights_come_from_the_config():
    floor, tower = floor_and_tower_z(-0.460, -0.460, H)  # sim.yaml
    assert floor == pytest.approx(-0.4775)
    assert tower == pytest.approx(-0.4725)
    # The floor goes under the lower of the grasp and release heights.
    floor, tower = floor_and_tower_z(0.015, 0.010, H)
    assert floor == pytest.approx(0.010 - H / 2 - 0.005)
    assert tower == pytest.approx(0.010 - H / 2)


def test_no_track_limit_without_a_half_size():
    t = TrackLimit((0.25, 0.0), None, frames=3)
    assert all(t.check((x, 5.0)) == "ok" for x in (-9.0, 0.0, 9.0))


def test_track_limit_rectangle_includes_its_border():
    t = TrackLimit((0.29, 0.015), (0.065, 0.105))
    assert t.inside((0.29, 0.015))
    assert t.inside((0.29 + 0.065, 0.015 - 0.105))
    assert not t.inside((0.29 + 0.0651, 0.015))
    assert not t.inside((0.29, 0.015 + 0.1051))


def test_track_limit_skips_then_gives_up_after_consecutive_frames():
    t = TrackLimit((0.0, 0.0), (0.02, 0.02), frames=3)
    out = (0.1, 0.0)
    assert [t.check(out) for _ in range(3)] == ["skip", "skip", "out"]
    assert t.check(out) == "skip"  # the counter starts again


def test_track_limit_counts_only_consecutive_frames():
    t = TrackLimit((0.0, 0.0), (0.02, 0.02), frames=3)
    out, inside = (0.1, 0.0), (0.0, 0.0)
    assert [t.check(p) for p in (out, out, inside, out, out)] == ["skip", "skip", "ok", "skip", "skip"]
    t.reset()
    assert [t.check(out) for _ in range(3)] == ["skip", "skip", "out"]


def _quat_down(yaw):
    """custom_end_link pointing down (rpy 0, pi/2, yaw), as x, y, z, w."""
    from robot_tools import transformations

    return list(transformations.quaternion_from_euler(0, math.pi / 2, yaw))


@pytest.mark.parametrize("yaw", [0.0, 0.4, -1.2])
def test_slot_axis_is_perpendicular_to_the_finger_opening(yaw):
    import numpy as np
    from robot_tools import transformations

    q = _quat_down(yaw)
    ux, uy = slot_axis(q)
    r = transformations.quaternion_matrix(q)[:3, :3]
    opening = r[:, 1]  # custom_end_link y: the fingers open along it
    approach = r[:, 0]  # custom_end_link x: down
    assert approach[2] == pytest.approx(-1.0, abs=1e-6)
    assert np.dot([ux, uy, 0.0], opening) == pytest.approx(0.0, abs=1e-6)
    assert math.hypot(ux, uy) == pytest.approx(1.0)


def test_slot_exit_tries_the_end_closer_to_the_base_first():
    first, second = slot_exit_targets([0.2734, 0.2145, 0.178], _quat_down(0.0), 0.06)
    assert math.hypot(*first[:2]) < math.hypot(*second[:2])
    assert first[2] == second[2] == 0.178  # a horizontal slide
    assert math.dist(first, second) == pytest.approx(0.12)


@pytest.mark.parametrize("place", [(0.2734, 0.2145), (0.2121, -0.2121)])  # real.yaml, sim.yaml
def test_first_place_shift_is_the_old_base_x_shift_with_the_shipped_orientation(place):
    # Both configs place with rpy (0, 1.570796, 0): the old code's +0.01 along the base x.
    from robot_tools import transformations

    q = list(transformations.quaternion_from_euler(0, 1.570796, 0))
    assert first_place_xy(place, q) == pytest.approx((place[0] + 0.01, place[1]))


@pytest.mark.parametrize("yaw", [0.4, math.pi / 2, math.pi, -1.2])
def test_first_place_shift_follows_the_gripper_z_axis(yaw):
    from robot_tools import transformations

    place, q = (0.2734, 0.2145), _quat_down(yaw)
    z_axis = transformations.quaternion_matrix(q)[:3, 2]  # custom_end_link z: along the gap between the fingers
    x, y = first_place_xy(place, q)
    assert (x - place[0], y - place[1]) == pytest.approx((0.01 * z_axis[0], 0.01 * z_axis[1]), abs=1e-9)
    assert (x - place[0], y - place[1]) != pytest.approx((0.01, 0.0), abs=1e-4)  # no longer the base x


def test_fingers_beside_top_block():
    tower, top = (0.2734, 0.2145), 0.1775
    assert fingers_beside_top_block((0.282, 0.213), tower, 0.173, top)  # the limit case: tips below the top
    assert not fingers_beside_top_block((0.282, 0.213), tower, 0.185, top)  # risen clear of it
    assert not fingers_beside_top_block((0.35, 0.10), tower, 0.10, top)  # low, but away from the tower


@pytest.mark.parametrize(
    "d, aligned",
    [
        ((1.5, 0.0), False),  # still short along the approach: the old |dx| < 2 px would stop here
        ((0.4, 0.0), True),
        ((0.0, 0.0), True),
        ((-1.5, 0.0), True),  # overshot a little: stop rather than step back and forth
        ((-2.5, 0.0), False),
        ((0.0, 1.5), True),  # across the approach the old 2 px window is kept
        ((0.0, -2.5), False),
    ],
)
def test_place_stop_window_is_asymmetric_along_the_approach(d, aligned):
    assert place_aligned(d, (1.0, 0.0), near=0.5) is aligned


def test_place_stop_window_follows_the_approach_direction():
    # Gripper yawed 90 deg: the approach (slot axis) is base +y, the window turns with it.
    u = slot_axis(_quat_down(math.pi / 2))
    assert u == pytest.approx((0.0, 1.0), abs=1e-9)
    assert not place_aligned((0.0, 1.5), u, near=0.5)
    assert place_aligned((0.0, -1.5), u, near=0.5)
    assert place_aligned((1.5, 0.2), u, near=0.5)


def test_tower_box_is_one_edge_longer_each_way_along_the_finger_gap():
    length, width, yaw = tower_footprint(0.025, 0.01, 0.025, slot_axis(_quat_down(0.0)))
    assert (length, width) == pytest.approx((0.095, 0.045))
    assert yaw == pytest.approx(0.0, abs=1e-9)  # shipped configs: the gap is base x


def test_tower_box_turns_with_the_gripper_yaw():
    _, _, yaw = tower_footprint(0.025, 0.01, 0.025, slot_axis(_quat_down(0.4)))
    assert math.cos(2 * (yaw - 0.4)) == pytest.approx(1.0)  # along the gap, either way round
