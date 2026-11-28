"""The YAML configs load, have the keys the programs read, and keep their comments when a tool writes them."""

import pathlib
import re

import pytest
from conftest import REPO

from sim import config_file

CONTROL_KEYS = [
    "pick_joint",
    "pick_position_x",
    "pick_position_y",
    "pick_position_z",
    "pick_roll",
    "pick_pitch",
    "pick_yaw",
    "pick_grasp_z",
    "place_position_x",
    "place_position_y",
    "place_position_z",
    "place_roll",
    "place_pitch",
    "place_yaw",
    "cube_height",
]
VISION_KEYS = [
    "hsv1_l",
    "hsv1_h",
    "hsv2_l",
    "hsv2_h",
    "expect_pick_ref",
    "expect_place_ref",
    "expect_x",
    "use_place_hsv",
    "area_limit_pick",
    "area_limit_place",
]


@pytest.mark.parametrize("name", ["real", "sim"])
def test_control_config(name):
    c = config_file.load(REPO / f"configs/control/{name}.yaml")
    assert not [k for k in CONTROL_KEYS if k not in c]
    # The aliases airbot_play_pick_place.load_configs understands.
    assert c["place_position_z"] == "pick_grasp_z" or isinstance(c["place_position_z"], float)
    for k in ("roll", "pitch", "yaw"):
        assert c[f"place_{k}"] == f"pick_{k}" or isinstance(c[f"place_{k}"], (int, float))


def test_real_robot_config_keeps_the_optional_behaviour_off():
    c = config_file.load(REPO / "configs/control/real.yaml")
    for key in (
        "transit_z_min",
        "place_retreat_y",
        "pick_area_half_size",
        "place_track_half_size",
        "place_settle_time",
    ):
        assert key not in c, f"{key} changes the real robot's behaviour; it should stay commented out"


@pytest.mark.parametrize("name", ["real", "sim"])
def test_vision_config(name):
    v = config_file.load(REPO / f"configs/vision/{name}.yaml")
    assert not [k for k in VISION_KEYS if k not in v]
    assert isinstance(v["use_place_hsv"], bool)
    if v["use_place_hsv"]:
        assert all(f"hsv{i}_{b}_p" in v for i in (1, 2) for b in "lh")


def test_set_keys_keeps_comments_and_layout(tmp_path: pathlib.Path):
    p = tmp_path / "c.yaml"
    p.write_text("# header\na: 1.0   # first\nb: [1, 2]  # second\n\nc: x\n")
    config_file.set_keys(p, {"a": 2.5, "b": [3, 4]})
    assert p.read_text() == "# header\na: 2.5   # first\nb: [3, 4]  # second\n\nc: x\n"


def test_set_keys_adds_missing_keys_after_an_anchor(tmp_path: pathlib.Path):
    p = tmp_path / "c.yaml"
    p.write_text("name: floor\nx: 1\n")
    config_file.set_keys(p, {"new": [0.1, 0.2]}, after="name")
    assert p.read_text() == "name: floor\nnew: [0.1, 0.2]\nx: 1\n"
    config_file.set_keys(p, {"last": 3})
    assert config_file.load(p) == {"name": "floor", "new": [0.1, 0.2], "x": 1, "last": 3}


def test_configs_are_lower_case_and_old_files_are_refused(tmp_path: pathlib.Path):
    from airbot_play_rules import old_key_names

    for path in sorted((REPO / "configs").rglob("*.yaml")):
        text = path.read_text()
        assert not re.findall(r"^\s*#?\s*[A-Z][A-Z0-9_]*\s*:", text, re.M), f"{path}: upper-case key names"
        assert old_key_names(config_file.load(path), path) == ""
    old = tmp_path / "real.yaml"
    old.write_text("PICK_POSITION_X: 0.27\nPLACE_POSITION_Z: PICK_GRASP_Z\n")
    with pytest.raises(ValueError, match="pick_position_x"):
        config_file.load(old)
    assert "PICK_POSITION_X 改为 pick_position_x" in old_key_names({"PICK_POSITION_X": 0.27, "name": "x"}, old)
