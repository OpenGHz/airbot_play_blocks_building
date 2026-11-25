"""The YAML configs load, have the keys the programs read, and keep their comments when a tool writes them."""

import pathlib

import pytest
from conftest import REPO

from sim import config_file

CONTROL_KEYS = [
    "PICK_JOINT",
    "PICK_POSITION_X",
    "PICK_POSITION_Y",
    "PICK_POSITION_Z",
    "PICK_ROLL",
    "PICK_PITCH",
    "PICK_YAW",
    "PICK_GRASP_Z",
    "PLACE_POSITION_X",
    "PLACE_POSITION_Y",
    "PLACE_POSITION_Z",
    "PLACE_ROLL",
    "PLACE_PITCH",
    "PLACE_YAW",
    "CUBE_HEIGHT",
]
VISION_KEYS = [
    "HSV1_L",
    "HSV1_H",
    "HSV2_L",
    "HSV2_H",
    "EXPECT_PICK_REF",
    "EXPECT_PLACE_REF",
    "EXPECT_X",
    "USE_PLACE_HSV",
    "AREA_LIMIT_PICK",
    "AREA_LIMIT_PLACE",
]


@pytest.mark.parametrize("name", ["real", "sim"])
def test_control_config(name):
    c = config_file.load(REPO / f"configs/control/{name}.yaml")
    assert not [k for k in CONTROL_KEYS if k not in c]
    # The aliases airbot_play_pick_place.load_configs understands.
    assert c["PLACE_POSITION_Z"] == "PICK_GRASP_Z" or isinstance(c["PLACE_POSITION_Z"], float)
    for k in ("ROLL", "PITCH", "YAW"):
        assert c[f"PLACE_{k}"] == f"PICK_{k}" or isinstance(c[f"PLACE_{k}"], (int, float))


def test_real_robot_config_keeps_the_optional_behaviour_off():
    c = config_file.load(REPO / "configs/control/real.yaml")
    for key in ("TRANSIT_Z_MIN", "PLACE_RETREAT_Y", "PICK_AREA_HALF_SIZE", "PLACE_TRACK_HALF_SIZE"):
        assert key not in c, f"{key} changes the real robot's behaviour; it should stay commented out"


@pytest.mark.parametrize("name", ["real", "sim"])
def test_vision_config(name):
    v = config_file.load(REPO / f"configs/vision/{name}.yaml")
    assert not [k for k in VISION_KEYS if k not in v]
    assert isinstance(v["USE_PLACE_HSV"], bool)
    if v["USE_PLACE_HSV"]:
        assert all(f"HSV{i}_{b}_P" in v for i in (1, 2) for b in "LH")


def test_set_keys_keeps_comments_and_layout(tmp_path: pathlib.Path):
    p = tmp_path / "c.yaml"
    p.write_text("# header\nA: 1.0   # first\nB: [1, 2]  # second\n\nC: x\n")
    config_file.set_keys(p, {"A": 2.5, "B": [3, 4]})
    assert p.read_text() == "# header\nA: 2.5   # first\nB: [3, 4]  # second\n\nC: x\n"


def test_set_keys_adds_missing_keys_after_an_anchor(tmp_path: pathlib.Path):
    p = tmp_path / "c.yaml"
    p.write_text("NAME: floor\nX: 1\n")
    config_file.set_keys(p, {"NEW": [0.1, 0.2]}, after="NAME")
    assert p.read_text() == "NAME: floor\nNEW: [0.1, 0.2]\nX: 1\n"
    config_file.set_keys(p, {"LAST": 3})
    assert config_file.load(p) == {"NAME": "floor", "NEW": [0.1, 0.2], "X": 1, "LAST": 3}
