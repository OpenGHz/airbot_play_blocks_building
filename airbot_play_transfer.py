"""Transfer moves of the block-building task with the tower in the MoveIt planning scene.

The floor and the tower are collision objects in the planning scene, so the moves between
the pick area and the tower are planned around them (OMPL). Only the vertical moves are
hand-made: the lift at the pick area, the last centimetre down onto the tower and the
rise after releasing. They are straight Cartesian lines along the tower axis and are not
collision-checked (they have to get close to the tower top).

Each transfer is planned segment by segment (the next segment starts where the previous
one ends), joined into one trajectory and retimed, so it runs without stopping in between.

All heights come from the control config; nothing has to be measured (see
airbot_play_rules.floor_and_tower_z): the floor top is half a cube plus FLOOR_MARGIN below
the lower of pick_grasp_z and place_position_z, the tower bottom half a cube below
place_position_z, plus one cube_height per block.
The tower box is TOWER_INFLATE larger on every side and on top (a block that was not put down
flat, the arm not following the planned path exactly), and one more block edge each way along
the finger gap, the only direction the tower leans in (airbot_play_rules.tower_footprint).
"""

from __future__ import annotations

import math

from geometry_msgs.msg import Pose, PoseStamped
from moveit_commander import PlanningSceneInterface
from moveit_msgs.msg import RobotState, RobotTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

from airbot_play_rules import tower_footprint

FLOOR_HALF = 1.0  # half size of the floor (m)
FLOOR_HOLE = 0.12  # half size of the square left open around the robot base, so the base never touches the floor box
TOWER_INFLATE = 0.01
HELD_CUBE = "held_cube"
TOWER = "tower"


def clear_scene(ns: str) -> None:
    """Take this program's floor, tower and held cube out of the MoveIt planning scene. move_group keeps
    them after the program exits, so a run without the planning scene would otherwise still plan around
    the boxes a run with it left behind."""
    psi = PlanningSceneInterface(ns=ns, synchronous=True)
    psi.remove_attached_object(name=HELD_CUBE)
    for name in (TOWER, HELD_CUBE, *(f"floor_{i}" for i in range(4))):
        psi.remove_world_object(name)


class TransferError(RuntimeError):
    """A transfer segment could not be planned (the message says which one and where to)."""


class TowerScene:
    """Floor, tower and held cube in the MoveIt planning scene."""

    def __init__(
        self,
        ns: str,
        frame: str,
        floor_z: float,
        tower_bottom_z: float,
        cube: float,
        lean_axis=(1.0, 0.0),
        lean: float = 0.0,
    ):
        """lean_axis: the finger gap at the place orientation (slot_axis); lean: how far the tower may
        lean each way along it (m)."""
        self.psi = PlanningSceneInterface(ns=ns, synchronous=True)
        self.tower_size = tower_footprint(cube, TOWER_INFLATE, lean, lean_axis)  # length, width, yaw
        self.frame = frame
        self.floor_z = floor_z
        self.tower_bottom_z = tower_bottom_z
        self.cube = cube
        self.tower_xy = None
        self.tower_n = 0
        self.holding = False  # the grasped cube is attached to the gripper
        self.psi.remove_attached_object(name=HELD_CUBE)
        self.psi.remove_world_object(TOWER)
        self.psi.remove_world_object(HELD_CUBE)
        self._add_floor()

    def _pose(self, x, y, z, yaw: float = 0.0) -> PoseStamped:
        p = PoseStamped()
        p.header.frame_id = self.frame
        p.pose.position.x, p.pose.position.y, p.pose.position.z = x, y, z
        p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return p

    def _add_floor(self):
        """Four boxes around the base (a floor with a square hole where the robot stands)."""
        t = 0.02
        z = self.floor_z - t / 2
        a, h = FLOOR_HALF, FLOOR_HOLE
        w = a - h
        for i, (x, y, sx, sy) in enumerate(
            ((h + w / 2, 0, w, 2 * a), (-h - w / 2, 0, w, 2 * a), (0, h + w / 2, 2 * h, w), (0, -h - w / 2, 2 * h, w))
        ):
            self.psi.add_box(f"floor_{i}", self._pose(x, y, z), (sx, sy, t))

    def set_tower(self, n: int, xy=None, max_top: float | None = None):
        """Tower of n blocks at xy (kept from the last call when None); n == 0 removes it.

        max_top caps the box top (e.g. just below the fingertips when the gripper is still
        beside the top block), so the start state is not in collision.
        """
        if xy is not None:
            self.tower_xy = tuple(xy)
        self.tower_n = n
        if n <= 0 or self.tower_xy is None:
            self.psi.remove_world_object(TOWER)
            return
        bottom = min(self.floor_z, self.tower_bottom_z)
        top = self.tower_bottom_z + n * self.cube + TOWER_INFLATE
        if max_top is not None:
            top = min(top, max_top)
            if top <= bottom:
                self.psi.remove_world_object(TOWER)
                return
        length, width, yaw = self.tower_size
        self.psi.add_box(TOWER, self._pose(*self.tower_xy, (bottom + top) / 2, yaw), (length, width, top - bottom))

    def hide_tower(self):
        """Take the tower out of collision checking (while working right above it)."""
        self.psi.remove_world_object(TOWER)

    def attach_cube(self, link: str, touch_links):
        """The grasped cube, centred on the end-effector point, moves with the gripper."""
        p = PoseStamped()
        p.header.frame_id = link
        p.pose.orientation.w = 1.0
        self.psi.attach_box(link, HELD_CUBE, p, (self.cube,) * 3, touch_links=list(touch_links))
        self.holding = True

    def detach_cube(self):
        self.psi.remove_attached_object(name=HELD_CUBE)
        self.psi.remove_world_object(HELD_CUBE)
        self.holding = False


class Transfer:
    """Plans and runs the transfer trajectories for a RoboticArmAgent (easy_arm_control)."""

    def __init__(self, arm, scene: TowerScene):
        self.arm = arm
        self.g = arm.basic_c
        self.scene = scene
        self.eef = self.g.get_end_effector_link()

    # ---- poses and states ----
    def pose(self, xyz, xyzw) -> Pose:
        p = Pose()
        p.position.x, p.position.y, p.position.z = (float(v) for v in xyz)
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = (float(v) for v in xyzw)
        return p

    def quat(self, rpy):
        return list(self.arm.euler_to_quaternion(rpy))

    def state(self, joints) -> RobotState:
        s = RobotState()
        s.joint_state.name = list(self.arm.joints_name)
        s.joint_state.position = [float(v) for v in joints]
        s.is_diff = True  # keep the attached cube of the planning scene
        return s

    @staticmethod
    def end(traj: RobotTrajectory) -> list:
        return list(traj.joint_trajectory.points[-1].positions)

    # ---- segments ----
    def line(self, start, xyz, xyzw, what: str, partial=False):
        """Straight Cartesian line from the start joints to the pose (no collision check).

        partial: return as much of the line as can be followed (None when nothing can).
        """
        self.g.set_start_state(self.state(start))
        best, best_f = None, 0.0
        for _ in range(5):
            traj, fraction = self.g.compute_cartesian_path([self.pose(xyz, xyzw)], 0.005, avoid_collisions=False)
            if fraction > 0.999:
                return traj
            if fraction > best_f and len(traj.joint_trajectory.points) > 1:
                best, best_f = traj, fraction
        if partial:
            return best
        raise TransferError(f"{what}：直线路径只能走到 {best_f:.0%}（目标 {_fmt(xyz)}）")

    def plan_to_pose(self, start, xyz, xyzw, what: str) -> RobotTrajectory:
        """OMPL to the pose; the IK is seeded with the start joints so the wrist does not flip."""
        ps = PoseStamped()
        ps.header.frame_id = self.g.get_planning_frame()
        ps.pose = self.pose(xyz, xyzw)
        for _ in range(10):
            self.g.set_joint_value_target(list(start))  # IK seed
            try:
                self.g.set_joint_value_target(ps, self.eef, False)
            except Exception:  # noqa: BLE001
                continue
            traj = self._plan(start)
            if traj is not None:
                return traj
        raise TransferError(f"{what}：规划失败（目标 {_fmt(xyz)}）")

    def plan_to_joints(self, start, joints, what: str) -> RobotTrajectory:
        self.g.set_joint_value_target(list(joints))
        for _ in range(10):
            traj = self._plan(start)
            if traj is not None:
                return traj
        raise TransferError(f"{what}：规划失败（目标关节角 {[round(v, 3) for v in joints]}）")

    def _plan(self, start):
        self.g.set_start_state(self.state(start))
        ok, traj, _, _ = self.g.plan()
        self.g.clear_pose_targets()
        return traj if ok and traj.joint_trajectory.points else None

    # ---- run ----
    def run(self, start, segments, vel: float, acc: float, end_xyz=None, end_rpy=None):
        """Join the segments, retime them as one trajectory, execute, and update the arm's last target."""
        joined = RobotTrajectory()
        joined.joint_trajectory.joint_names = list(segments[0].joint_trajectory.joint_names)
        points = []
        for seg in segments:
            for p in seg.joint_trajectory.points:
                if points and max(abs(a - b) for a, b in zip(p.positions, points[-1].positions)) < 1e-6:
                    continue  # a segment starts where the previous one ended
                points.append(JointTrajectoryPoint(positions=list(p.positions)))
        joined.joint_trajectory.points = points
        timed = self.g.retime_trajectory(
            self.state(start), joined, vel, acc, algorithm="iterative_time_parameterization"
        )
        self.g.stop()
        if not self.g.execute(timed, wait=True):
            raise TransferError("轨迹执行失败")
        self.commit(self.end(timed), end_xyz, end_rpy)

    def commit(self, joints, xyz=None, rpy=None):
        """Make easy_arm_control's 'last target' the pose we ended at (later moves start from it)."""
        if xyz is None:
            xyz, rpy, _ = self.arm.change_joints_to_pose(joints)
        self.arm.set_target_pose(list(xyz), list(rpy), target_base=0)
        self.arm.new_joint_target = list(joints)
        self.arm._RoboticArmAgent__update_last_target()  # noqa: SLF001  (easy_arm_control keeps it private)
        self.arm.wait_flag = True


def _fmt(xyz) -> str:
    return "(" + ", ".join(f"{v:+.3f}" for v in xyz) + ")"
