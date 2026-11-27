"""Best_Contour.py --held-cube: finding the held cube in the place-stage image and moving the alignment target."""

import numpy as np
import pytest
from conftest import REPO

from Best_Contour import find_held_cube, held_shift
from sim import config_file


def test_the_held_cube_is_the_lowest_blob_in_the_image():
    b = np.zeros((480, 640), np.uint8)
    b[300:330, 300:340] = 255  # a cube of the same colour further down the tower (higher in the image)
    b[415:480, 293:347] = 255  # the held cube, nearest the camera, cut off by the image bottom
    b[470:476, 100:104] = 255  # a speck below it, smaller than the area limit
    held = find_held_cube(b, min_area=300)
    assert held["x"] == pytest.approx(320.0)
    assert held["edge"] == 415.0


def test_no_held_cube():
    assert find_held_cube(np.zeros((480, 640), np.uint8), min_area=300) is None


def test_edge_target_moves_with_the_held_cube_by_the_scale():
    # Only image y (along the finger gap) is corrected; the held cube's x does not matter.
    assert held_shift({"x": 330.0, "edge": 405.0}, 415.0, 0.86) == pytest.approx(-8.6)
    assert held_shift({"x": 300.0, "edge": 415.0}, 415.0, 0.86) == pytest.approx(0.0)


@pytest.mark.parametrize("name", ["real", "sim"])
def test_held_cube_keys_come_together(name):
    v = config_file.load(REPO / f"configs/vision/{name}.yaml")
    keys = [k in v for k in ("expect_held_ref", "held_scale")]
    assert all(keys) or not any(keys)
    if all(keys):
        assert 0 < v["held_scale"] < 2
