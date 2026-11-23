"""Simulation scene, physics model, pick-area placement and the sim protocol (no window, no rendering)."""

import itertools

import numpy as np
import pytest

from sim import protocol, scene
from sim.world import World


@pytest.fixture(scope="module")
def floor_cfg():
    return scene.load_config("floor")


def test_floor_scene_inherits_from_its_base(floor_cfg):
    assert floor_cfg["NAME"] == "floor"
    assert floor_cfg["PICK_SURFACE"] == "floor"
    assert floor_cfg["CAMERA"]["WIDTH"] == 640  # from sim.yaml through BASE
    assert floor_cfg["_path"].name == "sim_floor.yaml"


def test_physics_model_has_the_pool_fingers_and_pick_zone(floor_cfg):
    m, _, _ = scene.build_physics(floor_cfg)
    for name in ("pick_zone", "finger_left", "finger_right", "finger_left_upper", "finger_right_upper"):
        assert m.geom(name).id >= 0
    assert all(m.body(f"cube_{i}").id >= 0 for i in range(floor_cfg["CUBE_POOL"]))
    assert m.opt.noslip_iterations == floor_cfg["NOSLIP_ITERATIONS"]


@pytest.fixture(scope="module")
def world(floor_cfg):
    return World(floor_cfg)


@pytest.mark.parametrize("seed", range(10))
def test_pick_area_cubes_are_inside_and_far_enough_apart(world, seed):
    area = world.pick
    area.rng = np.random.default_rng(seed)
    spots = area._sample(area.count, [])
    assert spots is not None and len(spots) == area.count
    for x, y, _ in spots:
        assert np.all(np.abs(np.array([x, y]) - area.center) <= area.half - area.s + 1e-9)
    for a, b in itertools.permutations(spots, 2):
        assert area._clear_of(a, b), "the open gripper grasping one cube would hit the other"


def test_protocol_round_trip_and_socket_override(monkeypatch, tmp_path):
    msg = {"op": "servo_joint", "q": [0.1, -0.2]}
    assert protocol.decode(protocol.encode(msg)) == msg
    monkeypatch.setenv("AIRBOT_SIM_SOCKET", str(tmp_path / "s.sock"))
    assert protocol.socket_path() == tmp_path / "s.sock"
    assert protocol.lock_path() == tmp_path / "s.lock"
