"""Pure rules of the block-building task (no ROS, no MoveIt), shared by the task and the tests.

All lengths are in metres, in the robot base frame.
"""

from __future__ import annotations

import math

FLOOR_MARGIN = 0.005  # the floor's top is this much below the lowest cube bottom the config implies


def release_z(place_z0: float, k: int, cube_height: float) -> float:
    """End-effector height when releasing block k+1 onto a tower of k blocks.

    place_z0 is PLACE_POSITION_Z (the release height of the first block); every layer
    adds one CUBE_HEIGHT. (The old code used PICK_GRASP_Z from the second block on.)
    """
    return place_z0 + k * cube_height


def place_settle_time(configured: float | None, first: bool) -> float:
    """Seconds to wait once the gripper is at the tower, before releasing block 1 or, from block 2 on,
    before the vision adjustment starts (the arm has to be still for both).

    configured is PLACE_SETTLE_TIME. Not set: the old program's waits, 1 s before releasing block 1
    and 3 s before adjusting (1 s at the height-check pose plus 2 s at the adjust pose, which is the
    same point without a distance sensor).
    """
    if configured is not None:
        return float(configured)
    return 1.0 if first else 3.0


def retreat_sign(pick_y: float, place_y: float) -> float:
    """Direction along y from the tower towards the pick area (+1 or -1)."""
    return 1.0 if pick_y - place_y >= 0 else -1.0


def floor_and_tower_z(pick_grasp_z: float, place_z0: float, cube_height: float) -> tuple[float, float]:
    """(floor top, tower bottom) for the planning scene, from the control config.

    A grasped cube's centre is about at the end-effector point, so its bottom is half a
    cube below the grasp / release height. The floor goes FLOOR_MARGIN below the lower of
    the two, so neither grasping nor releasing counts as touching it.
    """
    return min(pick_grasp_z, place_z0) - cube_height / 2 - FLOOR_MARGIN, place_z0 - cube_height / 2


class TrackLimit:
    """Keeps vision tracking inside a rectangle (§4.4 of the plan).

    check(xy) for the target of each tracking step returns
      "ok"    inside (or no limit configured): execute the step
      "skip"  outside: do not execute this step (as if nothing was detected)
      "out"   outside for `frames` steps in a row: give up (the counter restarts)
    """

    def __init__(self, center, half, frames: int = 10):
        self.center = tuple(center)
        self.half = None if half is None else tuple(half)
        self.frames = int(frames)
        self.count = 0

    def inside(self, xy) -> bool:
        if self.half is None:
            return True
        return all(abs(v - c) <= h for v, c, h in zip(xy[:2], self.center, self.half))

    def check(self, xy) -> str:
        if self.inside(xy):
            self.count = 0
            return "ok"
        self.count += 1
        if self.count >= self.frames:
            self.count = 0
            return "out"
        return "skip"

    def reset(self) -> None:
        self.count = 0


SLOT_EXIT = 0.06  # how far to slide out along the gap between the open fingers (m)


def slot_axis(xyzw) -> tuple[float, float]:
    """Horizontal direction along the gap between the open fingers, from the end-effector
    orientation (custom_end_link, quaternion x, y, z, w).

    custom_end_link's x axis is the approach (down), its y axis the direction the fingers open,
    so its z axis runs along the gap: a block standing between the open fingers can leave the
    gripper that way without touching them.
    """
    x, y, z, w = xyzw
    ax, ay = 2 * (x * z + w * y), 2 * (y * z - w * x)  # x, y of the rotated z axis
    n = (ax * ax + ay * ay) ** 0.5
    if n < 1e-6:
        raise ValueError("the gripper is not pointing down")
    return ax / n, ay / n


FIRST_PLACE_SHIFT = 0.01  # the real robot releases the first block this much off PLACE_POSITION_X/Y (m)


def first_place_xy(place_xy, xyzw, shift: float = FIRST_PLACE_SHIFT) -> tuple[float, float]:
    """Where the real robot releases the first block: PLACE_POSITION_X/Y moved `shift` along
    custom_end_link's z axis at the place orientation (xyzw), i.e. along the gap between the fingers.

    The fingers hold a block along their opening direction only, so a grasped block can be off
    centre along the gap only. The old code added the shift to the base x, which is the same
    direction only while the gripper points down with yaw 0 (rpy 0, pi/2, 0, every shipped config).
    """
    ux, uy = slot_axis(xyzw)
    return place_xy[0] + shift * ux, place_xy[1] + shift * uy


def place_aligned(d_xy, approach_xy, near: float, far: float = 2.0, side: float = 2.0) -> bool:
    """Place-stage stop test with an asymmetric window along the direction the gripper approaches
    the tower top from (`--place-stop`).

    d_xy: what the vision says the gripper still has to move, base frame (px). approach_xy: unit
    vector of the approach (from block 2 on the gripper starts FIRST_PLACE_SHIFT short of the tower
    along the finger gap, so this is the slot axis). The step gain is below 1, so the remaining move
    along the approach shrinks without changing sign and only the near edge of a symmetric window is
    ever reached; that edge is a bias every layer repeats. Stop when the remaining move along the
    approach is below `near`, or has overshot by less than `far`, and across it is below `side`.
    """
    ux, uy = approach_xy
    along = d_xy[0] * ux + d_xy[1] * uy
    across = -d_xy[0] * uy + d_xy[1] * ux
    return -far < along < near and abs(across) < side


def tower_footprint(cube: float, inflate: float, lean: float, axis_xy) -> tuple[float, float, float]:
    """Horizontal size and yaw of the tower's collision box: `inflate` larger on every side, and
    `lean` more each way along axis_xy (the finger gap at the place orientation, slot_axis).

    The tower leans along the finger gap only: the fingers centre a block along their opening
    direction, and the grasp offset and the closed loop's dead-zone bias both lie along the gap.
    A uniformly leaning tower falls before its top block is one block edge off its bottom one
    (n blocks stepping s each stand while n * s < edge), so lean = one edge covers any standing
    tower of that kind, leaning either way.
    Returns (length along axis_xy, width across it, yaw of axis_xy in rad).
    """
    return cube + 2 * (inflate + lean), cube + 2 * inflate, math.atan2(axis_xy[1], axis_xy[0])


def slot_exit_targets(eef_xyz, xyzw, distance: float = SLOT_EXIT) -> list[list[float]]:
    """Both ends of a horizontal slide along the finger gap, the one closer to the robot base first
    (moving in keeps the arm away from its reach limit)."""
    ux, uy = slot_axis(xyzw)
    x, y, z = eef_xyz
    ends = [[x + s * distance * ux, y + s * distance * uy, z] for s in (1.0, -1.0)]
    return sorted(ends, key=lambda p: p[0] ** 2 + p[1] ** 2)


def fingers_beside_top_block(eef_xy, tower_xy, lowest_z: float, tower_top_z: float, radius: float = 0.03) -> bool:
    """Is the gripper right over the tower with its lowest point (fingertips, or the bottom of the block
    it holds) not above the tower top? Then the top block is between the fingers."""
    dx, dy = eef_xy[0] - tower_xy[0], eef_xy[1] - tower_xy[1]
    return (dx * dx + dy * dy) ** 0.5 < radius and lowest_z < tower_top_z + 0.003
