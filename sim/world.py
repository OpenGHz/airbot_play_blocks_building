"""`pixi run sim`: the simulated "physical world" (arm, gripper, cubes) in real time.

Plays the part of `airbot_server` in manual terminal 0. The bridge node reaches it
through MujocoDriver (sim/driver.py) over a private Unix socket (sim/protocol.py);
everything from the bridge node up is the code the real robot runs.

  * Joints are position servos, limits as in the URDF; the gripper takes a width.
  * Steps at 2 ms in wall-clock time, so ROS/MoveIt need no /use_sim_time.
  * The pick area refills itself: a few cubes at random spots, spaced so the open
    gripper grasping one cannot hit another; when the last one has been taken and
    the arm is out of the way, a new batch appears.
  * Ctrl+C drives the arm back to zero before exiting, like airbot_server.
  * With the window, a second small window (sim/panel.py) sets where the arm stands on the
    table and where the cubes are, with sliders and numbers.
  * One instance at a time (lock file next to the socket).
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import signal
import socketserver
import subprocess
import sys
import threading
import time

import mujoco
import numpy as np

from sim import kin, protocol, scene

OPEN_WIDTH = 0.07  # what the bridge sends for "open" (set_position 1.0 * 0.07)


def width_to_finger_q(width: float) -> float:
    return float(np.clip((scene.FINGER_GAP_AT_ZERO - width) / 2, 0.0, 0.04))


def finger_q_to_width(q: float) -> float:
    return scene.FINGER_GAP_AT_ZERO - 2 * q


class PickArea:
    """Keeps a batch of cubes in the pick area and refills it once it is empty."""

    def __init__(self, world: World):
        self.w = world
        cfg = world.cfg
        area = cfg.get("pick_area", {})
        # Where the cubes go: center (default pick_xy) +- half_size. The wrist camera only sees the
        # floor ahead of the gripper at the pick scan pose, so the area sits in front of the scan point.
        self.center = np.array(area.get("center", cfg["pick_xy"]), dtype=float)
        self.half = np.array(area.get("half_size", [0.04, 0.06]), dtype=float)
        self.count = int(area.get("count", 4))
        self.margin = float(area.get("margin", 0.008))
        self.yaw = math.radians(area.get("yaw_deg", 15.0))
        self.rng = np.random.default_rng(area.get("seed"))
        self.s = cfg["cube_size"] / 2
        self.r = self.s * math.sqrt(2)  # circumradius of a cube's footprint
        self.surface = 0.0 if scene.pick_on_floor(cfg) else cfg["table_height"]  # world z
        self.batch: list[int] = []
        self.waiting_since: float | None = None
        self.zone = world.model.geom("pick_zone").id
        self.footprint = self._footprint()

    # --- geometry ---------------------------------------------------------------
    def _footprint(self) -> list[tuple[float, float, float, float]]:
        """Rectangles (x0, x1, y0, y1), relative to the grasped cube's centre with the
        gripper at yaw 0, covered by the open gripper's collision shapes below the top
        of a neighbouring cube (+5 mm)."""
        m = self.w.model
        d = mujoco.MjData(m)
        frame = self.w.frame
        arm = kin.Arm(m, d, frame)
        z = self.surface + self.s - frame.pos[2]
        down = frame.rot.T @ kin.DOWN  # the gripper at yaw 0 in the world, whichever way the arm is turned
        # The shape only depends on the gripper over a grasped cube. Measure it at the area centre, or,
        # when the arm has been moved on the table so far that it cannot reach that, where the area is
        # from the scene file's mount.
        for gx, gy in (tuple(frame.to_base(self.center)), tuple(self.center)):  # base frame
            q = arm.ik([gx, gy, z], rot=down, restarts=60)
            if q is not None:
                break
        else:
            raise RuntimeError(
                f"cannot reach the pick area centre {list(self.center)} even from the scene file's mount"
            )
        cx, cy = frame.to_world((gx, gy))  # world x, y of the grasped cube
        d.qpos[arm.qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        top = self.surface + 2 * self.s + 0.005
        rects = []
        for g in arm.robot_geoms:
            pts = _geom_points(m, d, g)
            low = pts[pts[:, 2] < top]
            if len(low):
                rel = low[:, :2] - [cx, cy]
                rects.append((rel[:, 0].min(), rel[:, 0].max(), rel[:, 1].min(), rel[:, 1].max()))
        return rects

    def _clear_of(self, a, b) -> bool:
        """Can cube a = (x, y, yaw) be grasped with cube b = (x, y, yaw) standing there?"""
        dx, dy = b[0] - a[0], b[1] - a[1]
        if math.hypot(dx, dy) < 2 * self.r + self.margin:
            return False
        c, s = math.cos(-a[2]), math.sin(-a[2])
        u, v = c * dx - s * dy, s * dx + c * dy
        for x0, x1, y0, y1 in self.footprint:
            ex = max(x0 - u, 0.0, u - x1)
            ey = max(y0 - v, 0.0, v - y1)
            if math.hypot(ex, ey) < self.r + self.margin:
                return False
        return True

    def _sample(self, n: int, others: list) -> list | None:
        lo, hi = self.center - self.half + self.s, self.center + self.half - self.s
        for _ in range(200):
            placed = []
            for _ in range(n):
                for _ in range(300):
                    x, y = self.rng.uniform(lo, hi)
                    c = (x, y, self.rng.uniform(-self.yaw, self.yaw))
                    if all(self._clear_of(c, p) and self._clear_of(p, c) for p in placed + others):
                        placed.append(c)
                        break
                else:
                    break
            if len(placed) == n:
                return placed
        return None

    # --- state ------------------------------------------------------------------
    def inside(self, i: int) -> bool:
        x, y, z = self.w.cube_pos(i)
        out = np.abs(np.array([x, y]) - self.center) > self.half + 0.02
        return not out.any() and z < self.surface + 2 * self.s + 0.01

    def arm_clear(self) -> bool:
        m, d = self.w.model, self.w.data
        return all(mujoco.mj_geomDistance(m, d, g, self.zone, 0.05, None) > 0.01 for g in self.w.robot_geoms)

    def spawn(self) -> None:
        w = self.w
        others = []  # cubes already lying in the area (dropped, nudged ...)
        for i in w.active():
            if self.inside(i):
                x, y, _ = w.cube_pos(i)
                others.append((x, y, w.cube_yaw(i)))
        n = self.count - self.count % len(w.color_names)
        spots = None
        while n >= len(w.color_names):
            spots = self._sample(n, others)
            if spots:
                break
            n -= len(w.color_names)
        if not spots:
            print("[sim] 抓取区放不下新的物块（区域被占满了？），稍后重试", flush=True)
            return
        colors = [w.color_names[k % len(w.color_names)] for k in range(n)]
        self.rng.shuffle(colors)
        batch = []
        for (x, y, yaw), color in zip(spots, colors):
            i = w.take_parked(color)
            if i is None:
                print(f"[sim] 物块池里没有空闲的 {color} 物块了（pixi run sim-ctl cube reset 可以收回）", flush=True)
                break
            w.place_cube(i, (x, y, self.surface + self.s + 0.0005), yaw)
            batch.append(i)
        self.batch = batch
        counts = {c: sum(1 for i in batch if w.cubes[i]["color"] == c) for c in w.color_names}
        print("[sim] 抓取区：放了 " + "、".join(f"{v} 块{c}" for c, v in counts.items()), flush=True)

    def update(self, now: float) -> None:
        self.batch = [i for i in self.batch if self.inside(i)]
        if self.batch:
            self.waiting_since = None
            return
        if not self.arm_clear():
            self.waiting_since = None
            return
        if self.waiting_since is None:
            self.waiting_since = now
        elif now - self.waiting_since > 0.5:
            self.spawn()
            self.waiting_since = None


def _geom_points(m, d, g) -> np.ndarray:
    """World-frame vertices of a mesh geom, or corners of a box geom."""
    rot = d.geom_xmat[g].reshape(3, 3)
    if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
        mid = m.geom_dataid[g]
        v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
    else:
        hx, hy, hz = m.geom_size[g]
        v = np.array([[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return v @ rot.T + d.geom_xpos[g]


class World:
    def __init__(self, cfg: dict):
        self.model, self.data, self.cfg = scene.build_physics(cfg)
        m = self.model
        self.frame = scene.base_frame(cfg)  # the arm base frame in the world; moves with the mount
        self.root = m.body("base_link").id
        self.mount = scene.mount_offset(cfg)  # (dx, dy) and yaw (rad) from the scene file's mount
        self.mount_yaw = scene.mount_yaw(cfg)
        lims = np.array([scene.mount_limits(cfg, math.radians(a)) for a in range(0, 360, 5)])
        self.mount_range = [lims[:, 0].min(), lims[:, 1].max(), lims[:, 2].min(), lims[:, 3].max()]  # any yaw
        self.table_geoms = {g for g in range(m.ngeom) if m.geom(g).name.startswith(("table_top", "column_", "foot_"))}
        self.lock = threading.RLock()
        self.qadr = np.array([m.joint(j).qposadr[0] for j in scene.ARM_JOINTS])
        self.dadr = np.array([m.joint(j).dofadr[0] for j in scene.ARM_JOINTS])
        self.act = np.array([m.actuator(j).id for j in scene.ARM_JOINTS])
        self.grip_act = m.actuator("gripper").id
        self.finger_qadr = m.joint("endleft").qposadr[0]
        self.finger_dadr = m.joint("endleft").dofadr[0]
        self.lo, self.hi = m.jnt_range[[m.joint(j).id for j in scene.ARM_JOINTS]].T
        robot = {m.body(b).id for b in kin._robot_bodies(m) if b != "base_link"}
        self.robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in robot and m.geom_contype[g]]
        self.robot_bodies = robot | {self.root}
        self.color_names = list(cfg["cube_colors"])
        self.cubes = []
        for i, color in enumerate(scene.pool_colors(cfg)):
            j = m.joint(m.body(f"cube_{i}").jntadr[0])
            self.cubes.append({"qadr": j.qposadr[0], "dadr": j.dofadr[0], "color": color, "parked": True})
        self.cmd_q = np.zeros(6)
        self.cmd_width = OPEN_WIDTH
        self.homing = False
        mujoco.mj_forward(m, self.data)
        self.state = self._read_state()
        self.state_cond = threading.Condition()
        self.pick = PickArea(self)
        self.cube_of_body = {m.body(f"cube_{i}").id: i for i in range(len(self.cubes))}
        self.selected = -1  # the body selected in the MuJoCo window (World.run keeps it up to date)
        self.stop_event: threading.Event | None = None  # World.run's stop (sim_quit sets it)
        self.want_select = None  # a body the panel asks the MuJoCo window to select

    @property
    def base(self) -> np.ndarray:
        """World position of the arm base frame's origin (the base centre on the table top)."""
        return self.frame.pos

    def to_base(self, p) -> np.ndarray:
        """World point(s) in the arm base frame, where configs/control/*.yaml and sim-ctl give positions."""
        return self.frame.to_base(p)

    # --- cubes ------------------------------------------------------------------
    def cube_pos(self, i: int) -> np.ndarray:
        a = self.cubes[i]["qadr"]
        return self.data.qpos[a : a + 3]

    def cube_yaw(self, i: int) -> float:
        a = self.cubes[i]["qadr"]
        w, x, y, z = self.data.qpos[a + 3 : a + 7]
        return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    def active(self) -> list[int]:
        return [i for i, c in enumerate(self.cubes) if not c["parked"]]

    def take_parked(self, color: str) -> int | None:
        for i, c in enumerate(self.cubes):
            if c["parked"] and c["color"] == color:
                c["parked"] = False
                return i
        return None

    def place_cube(self, i: int, pos, yaw: float = 0.0) -> None:
        c = self.cubes[i]
        a, v = c["qadr"], c["dadr"]
        self.data.qpos[a : a + 3] = pos
        self.data.qpos[a + 3 : a + 7] = scene.yaw_quat(yaw)
        self.data.qvel[v : v + 6] = 0.0

    def support_below(
        self, i: int, xy=None, start_z: float | None = None, geomgroup=None, centre_only: bool = False
    ) -> float:
        """World z of the highest surface under cube i's footprint (at its current x, y, or at xy; only
        under its centre with centre_only), looking down from just above the cube (or from start_z);
        geomgroup (6 flags) limits what counts."""
        m, d = self.model, self.data
        s = self.cfg["cube_size"] / 2
        pos = self.cube_pos(i)
        x, y = pos[:2] if xy is None else xy
        z = pos[2] + s + 0.002 if start_z is None else start_z
        yaw = self.cube_yaw(i)
        c, sn = math.cos(yaw), math.sin(yaw)
        body = m.body(f"cube_{i}").id
        geomid = np.zeros(1, np.int32)
        top = 0.0  # the floor
        spots = ((0, 0),) if centre_only else ((0, 0), (0.8, 0.8), (0.8, -0.8), (-0.8, 0.8), (-0.8, -0.8))
        for u, v in spots:
            px, py = x + s * (c * u - sn * v), y + s * (sn * u + c * v)
            start = np.array([px, py, z])
            dist = mujoco.mj_ray(m, d, start, np.array([0.0, 0.0, -1.0]), geomgroup, 1, body, geomid)
            if dist >= 0:
                top = max(top, start[2] - dist)
        return top

    def put_down(self, i: int) -> None:
        """Set cube i down upright, at rest, on whatever is under it (like putting it down by hand)."""
        s = self.cfg["cube_size"] / 2
        pos = self.cube_pos(i).copy()
        yaw = self.cube_yaw(i)
        pos[2] = self.support_below(i) + s + 0.0005
        self.place_cube(i, pos, yaw)
        mujoco.mj_forward(self.model, self.data)

    def park(self, i: int) -> None:
        x, y = scene.pool_park_xy(i)
        self.place_cube(i, (x, y, self.cfg["cube_size"] / 2))
        self.cubes[i]["parked"] = True

    # What a cube can be set down on from above: the floor, the table and other cubes (group 0),
    # not the arm (visual geoms in group 1, collision geoms in COLLISION_GROUP).
    SUPPORT_GROUPS = np.array([1, 0, 0, 0, 0, 0], np.uint8)

    def set_cube(self, i: int, xy=None, yaw: float | None = None) -> str:
        """Put cube i down upright at xy (base frame, m), turned yaw (rad, from the base frame's x
        axis), on the highest thing under its centre other than the arm (floor, table top, another
        cube), like setting it down there by hand. None keeps the cube's own value. Refused (the cube
        stays) when it would end up in the arm, the table or another cube; so a cube is not balanced
        on another one's edge: with its centre off that cube it would overlap it.
        Returns "" or why: "arm:<link>", "table:<geom>", "cube:<k>"."""
        if not 0 <= i < len(self.cubes) or self.cubes[i]["parked"]:
            raise RuntimeError(f"cube_{i} 不在场景里（sim-ctl cube list 列出场景里的物块）")
        m, d = self.model, self.data
        s = self.cfg["cube_size"] / 2
        old = self.cube_pos(i).copy(), self.cube_yaw(i)
        x, y = self.cube_pos(i)[:2] if xy is None else self.frame.to_world(np.asarray(xy, dtype=float)[:2])
        yaw_w = old[1] if yaw is None else float(yaw) + self.frame.yaw
        self.place_cube(i, (x, y, old[0][2]), yaw_w)
        mujoco.mj_kinematics(m, d)
        top = self.support_below(i, start_z=10.0, geomgroup=self.SUPPORT_GROUPS, centre_only=True)
        self.place_cube(i, (x, y, top + s + 0.0005), yaw_w)
        mujoco.mj_forward(m, d)
        why = self._cube_overlap(i, 0.002)
        if why:
            self.place_cube(i, *old)
            mujoco.mj_forward(m, d)
        return why

    def _cube_overlap(self, i: int, depth: float) -> str:
        m, d = self.model, self.data
        g = m.geom(f"cube_{i}").id
        for c in d.contact[: d.ncon]:
            if g not in (c.geom1, c.geom2) or c.dist >= -depth:
                continue
            other = c.geom2 if c.geom1 == g else c.geom1
            body = m.geom_bodyid[other]
            if body in self.robot_bodies:
                return f"arm:{m.body(body).name}"
            if body in self.cube_of_body:
                return f"cube:{self.cube_of_body[body]}"
            return f"table:{m.geom(other).name or 'floor'}"
        return ""

    def cube_state(self, i: int) -> dict:
        """Cube i for sim-ctl and the panel: base frame position (m) and yaw (deg; a cube looks the same
        every 90 deg, so in [-45, 45))."""
        yaw = math.degrees(self.cube_yaw(i) - self.frame.yaw)
        return {
            "cube": i,
            "color": self.cubes[i]["color"],
            "base_xyz": self.to_base(self.cube_pos(i)).tolist(),
            "yaw": (yaw + 45.0) % 90.0 - 45.0,
            "where": "抓取区" if i in self.pick.batch else "",
        }

    # --- arm --------------------------------------------------------------------
    def set_mount(self, xy, yaw: float | None = None) -> tuple[np.ndarray, str]:
        """Move the arm on the table: its base centre to xy (scene frame, relative to the scene file's
        mount), turned yaw (rad, from the scene file's mount; None keeps the current one). The base
        plate is kept on the table top (xy is moved in from the edge as far as needed, also when only
        the yaw changes); a move that would put the arm into the table is refused.
        Returns (where the base is now, why it did not get all the way: "" | "edge" | "table:<link>")."""
        m, d = self.model, self.data
        yaw = self.mount_yaw if yaw is None else math.remainder(float(yaw), 2 * math.pi)
        x0, x1, y0, y1 = scene.mount_limits(self.cfg, yaw)
        want = np.array([float(xy[0]), float(xy[1])])
        target = np.array([min(max(want[0], x0), x1), min(max(want[1], y0), y1)])
        why = "" if np.allclose(target, want) else "edge"
        old_pos, old_quat = m.body_pos[self.root].copy(), m.body_quat[self.root].copy()
        scene.place_base(m, target, yaw)
        mujoco.mj_forward(m, d)
        hit = self._arm_in_table(0.002)
        if hit:
            m.body_pos[self.root], m.body_quat[self.root] = old_pos, old_quat
            mujoco.mj_forward(m, d)
            return self.mount.copy(), f"table:{hit}"
        self.mount, self.mount_yaw = target, yaw
        self.frame = scene.BaseFrame([target[0], target[1], self.frame.pos[2]], yaw)
        return target.copy(), why

    def mount_state(self) -> dict:
        """The mount for sim-ctl and the panel: offset (m) and yaw (deg) from the scene file's mount;
        in the same frame, the table top (x0, x1, y0, y1) and where the base centre can go at some yaw."""
        (x0, x1, y0, y1), _ = scene.table_rect(self.cfg)
        return {
            "mount": self.mount.tolist(),
            "yaw": math.degrees(self.mount_yaw),
            "table": [x0, x1, y0, y1],
            "range": [float(v) for v in self.mount_range],
        }

    def mount_text(self) -> str:
        dx, dy = self.mount
        x0, x1, y0, y1 = scene.table_rect(self.cfg)[0]
        return (
            f"机械臂底座相对场景文件里的安装位置 ({dx:+.3f}, {dy:+.3f}) m，转角 {math.degrees(self.mount_yaw):+.1f}°；"
            f"底座中心离桌子四边：前 {x1 - dx:.3f}，后 {dx - x0:.3f}，左 {y1 - dy:.3f}，右 {dy - y0:.3f} m"
        )

    def _arm_in_table(self, depth: float) -> str:
        m, d = self.model, self.data
        for c in d.contact[: d.ncon]:
            for a, b in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if a in self.robot_geoms and b in self.table_geoms and c.dist < -depth:
                    return m.body(m.geom_bodyid[a]).name
        return ""

    def servo_joint(self, q) -> None:
        q = np.asarray(q, dtype=float)
        if q.shape != (6,) or not np.all(np.isfinite(q)):
            return
        with self.lock:
            if not self.homing:
                self.cmd_q = np.clip(q, self.lo, self.hi)

    def servo_eef(self, width: float) -> None:
        with self.lock:
            if not self.homing:
                self.cmd_width = float(width)

    def _read_state(self) -> dict:
        d = self.data
        fq = d.qpos[self.finger_qadr]
        return {
            "type": "state",
            "t": d.time,
            "q": d.qpos[self.qadr].tolist(),
            "dq": d.qvel[self.dadr].tolist(),
            "tau": d.actuator_force[self.act].tolist(),
            "eef": [finger_q_to_width(fq)],
            "eef_v": [-2 * d.qvel[self.finger_dadr]],
            "eef_eff": [float(d.actuator_force[self.grip_act])],
        }

    def gripper_point(self) -> np.ndarray:
        """World position of custom_end_link (the grasp point)."""
        return self.data.xpos[self.model.body(scene.EEF_BODY).id].copy()

    # --- stepping ---------------------------------------------------------------
    def run(self, viewer=None, stop: threading.Event | None = None, home_seconds: float = 3.0) -> None:
        m, d = self.model, self.data
        dt = m.opt.timestep
        stop = stop or threading.Event()
        self.stop_event = stop
        t_wall0, t_sim0 = time.perf_counter(), d.time
        next_sync = next_area = next_text = 0.0
        hand = None
        if viewer is not None:
            from sim.hand import Hand

            hand = Hand(self)
        home_from = None
        lag_warned = 0.0
        while True:
            now = time.perf_counter()
            if stop.is_set() and home_from is None:
                with self.lock:
                    self.homing = True
                    home_from = (now, d.ctrl[self.act].copy())
                print("[sim] 回零点后退出 ...", flush=True)
            if home_from is not None:
                frac = min(1.0, (now - home_from[0]) / home_seconds)
                with self.lock:
                    self.cmd_q = (1 - frac) * home_from[1]
                if frac >= 1.0 and np.max(np.abs(d.qpos[self.qadr])) < 0.02:
                    break
                if now - home_from[0] > home_seconds + 3.0:
                    break
            if viewer is not None and not viewer.is_running():
                stop.set()
            behind = (now - t_wall0) - (d.time - t_sim0)
            steps = 0
            with self.lock:
                d.ctrl[self.act] = self.cmd_q
                d.ctrl[self.grip_act] = width_to_finger_q(self.cmd_width)
                while behind > 0 and steps < 25:
                    if hand is not None:
                        hand.before_step()
                    mujoco.mj_step(m, d)
                    behind -= dt
                    steps += 1
                if behind > 0.1:  # fell behind (heavy contact, debugger): drop the backlog
                    if now - lag_warned > 5.0:
                        print(f"[sim] 仿真跟不上实时，落后 {behind * 1000:.0f} ms", flush=True)
                        lag_warned = now
                    t_wall0, t_sim0 = now, d.time
                state = self._read_state()
                if now >= next_area:
                    self.pick.update(now)
                    next_area = now + 0.05
                if viewer is not None and now >= next_sync:
                    with viewer.lock():  # the mouse drag the viewer's UI thread keeps in its perturbation
                        p = viewer.perturb
                        if self.want_select is not None and not p.active:  # the panel picked a cube
                            p.select, p.skinselect, p.flexselect = self.want_select, -1, -1
                            p.localpos[:] = 0.0
                        self.want_select = None
                        pert = (int(p.select), int(p.active), p.refpos.copy(), p.refquat.copy())
                    self.selected = pert[0]
                    hand.update(*pert)
                    viewer.sync()
                    if now >= next_text:
                        viewer.set_texts(hand.texts())
                        next_text = now + 0.1
                    next_sync = now + 1 / 60
            if steps:
                with self.state_cond:
                    self.state = state
                    self.state_cond.notify_all()
            time.sleep(0.0005)

    # --- requests from clients --------------------------------------------------
    def handle(self, op: str, msg: dict) -> dict:
        if op == "hello":
            return {"name": "mujoco", "is_sim": True, "version": protocol.VERSION}
        if op in ("enter_servo_mode", "subscribe"):
            return {}
        if op == "get_state":
            return dict(self.state)
        if op == "get_qpos":
            with self.lock:
                return {
                    "qpos": self.data.qpos.tolist(),
                    "t": self.data.time,
                    "mount": self.mount.tolist(),
                    "mount_yaw": self.mount_yaw,
                }
        if op == "ctl":
            with self.lock:
                return self.ctl(msg.get("cmd", ""), msg.get("args", {}))
        raise ValueError(f"unknown op {op!r}")

    def ctl(self, cmd: str, args: dict) -> dict:
        """sim-ctl commands: the operator's hands."""
        s = self.cfg["cube_size"] / 2
        if cmd == "cube_reset":
            for i in self.active():
                self.park(i)
            self.pick.batch = []
            self.pick.spawn()
            return {"message": "所有物块已收回，抓取区重新放好"}
        if cmd == "cube_respawn":
            for i in list(self.pick.batch):
                self.park(i)
            self.pick.batch = []
            self.pick.spawn()
            return {"message": "抓取区已重新随机放置"}
        if cmd == "cube_take":
            color = args.get("color")
            taken = [i for i in self.pick.batch if color in (None, self.cubes[i]["color"])]
            for i in taken:
                self.park(i)
            self.pick.batch = [i for i in self.pick.batch if i not in taken]
            return {"message": f"从抓取区拿走了 {len(taken)} 块"}
        if cmd == "cube_to_gripper":
            color = args.get("color") or self.color_names[0]
            i = self.take_parked(color)
            if i is None:
                raise RuntimeError(f"没有空闲的 {color} 物块")
            p = self.gripper_point()
            self.place_cube(i, p, _gripper_yaw(self))
            self.cmd_width = 0.0
            mujoco.mj_forward(self.model, self.data)
            return {"message": f"一块{color}物块已放进夹爪并夹紧"}
        if cmd == "cube_put_under":
            color = args.get("color") or self.color_names[-1]
            i = self.take_parked(color)
            if i is None:
                raise RuntimeError(f"没有空闲的 {color} 物块")
            p = self.gripper_point()
            floor = 0.0
            self.place_cube(i, (p[0], p[1], floor + s + 0.0005), _gripper_yaw(self))
            return {"message": f"一块{color}物块已放到夹爪正下方的地面上"}
        if cmd == "cube_nudge":
            dx, dy = float(args.get("dx", 0.0)), float(args.get("dy", 0.0))  # base frame
            p = self.gripper_point()
            # nearest in 3D: over the tower, the block below is about as close horizontally as the held one
            near = min(self.active(), key=lambda i: np.linalg.norm(self.cube_pos(i) - p), default=None)
            if near is None:
                raise RuntimeError("场景里没有物块")
            pos = self.cube_pos(near).copy()
            pos[:2] += self.frame.rot[:2, :2] @ [dx, dy]
            self.place_cube(near, pos, self.cube_yaw(near) + math.radians(float(args.get("dyaw", 0.0))))
            b = self.to_base(pos)
            return {"message": f"cube_{near} 移到基座坐标 ({b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f})"}
        if cmd == "cube_list":
            selected = self.cube_of_body.get(self.selected)
            return {"cubes": [self.cube_state(i) for i in self.active()], "selected": selected}
        if cmd == "cube_set":
            i = int(args["cube"])
            if "x" in args or "y" in args:
                here = self.to_base(self.cube_pos(i)) if 0 <= i < len(self.cubes) else (0.0, 0.0)
                xy = [float(args.get("x", here[0])), float(args.get("y", here[1]))]
            else:
                xy = None
            yaw = math.radians(float(args["yaw"])) if "yaw" in args else None
            why = self.set_cube(i, xy, yaw)
            st = self.cube_state(i)
            x, y, z = st["base_xyz"]
            msg = f"cube_{i} 在基座坐标 ({x:+.3f}, {y:+.3f}, {z:+.3f})，转角 {st['yaw']:+.1f}°"
            if why:
                kind, name = why.split(":", 1)
                what = {"arm": f"机械臂的 {name}", "cube": f"cube_{name}", "table": f"桌子（{name}）"}[kind]
                msg += f"（没有移动：会和{what}重叠）"
            return {"message": msg, "why": why, **st}
        if cmd == "sim_quit":  # the panel's 退出仿真: like Ctrl+C, home the arm and exit
            if self.stop_event is None:
                raise RuntimeError("仿真没有在运行")
            print("[sim] 面板上点了退出仿真", flush=True)
            self.stop_event.set()
            return {"message": "仿真回零点后退出"}
        if cmd == "cube_select":
            i = int(args["cube"])
            if not 0 <= i < len(self.cubes) or self.cubes[i]["parked"]:
                raise RuntimeError(f"cube_{i} 不在场景里")
            self.want_select = self.model.body(f"cube_{i}").id
            return {}
        if cmd in ("arm_mount", "arm_where"):
            why = ""
            if cmd == "arm_mount":
                xy = [float(args.get("x", self.mount[0])), float(args.get("y", self.mount[1]))]
                xy = np.array(xy) + [float(args.get("dx", 0.0)), float(args.get("dy", 0.0))]
                yaw = math.radians(float(args.get("yaw", math.degrees(self.mount_yaw))) + float(args.get("dyaw", 0.0)))
                _, why = self.set_mount(xy, yaw)
            note = ""
            if why == "edge":
                note = f"（没有移到 ({xy[0]:+.3f}, {xy[1]:+.3f})：底板到桌子边了）"
            elif why:
                note = f"（没有移动：机械臂的 {why.split(':', 1)[1]} 会撞到桌子）"
            return {"message": self.mount_text() + note, "why": why, **self.mount_state()}
        if cmd == "arm_save":
            from sim import config_file

            path = self.cfg["_path"]
            config_file.set_keys(path, {"mount_offset": [round(float(v), 4) for v in self.mount]}, after="name")
            yaw = round(math.degrees(self.mount_yaw), 2)
            keys = "mount_offset"
            if yaw or "mount_yaw" in config_file.load(path):
                config_file.set_keys(path, {"mount_yaw": yaw}, after="mount_offset")
                keys += "、mount_yaw"
            return {"message": f"安装位置 {self.mount_text()} 已写入 {path.name}（{keys}），下次 pixi run sim 沿用"}
        raise ValueError(f"unknown sim-ctl command {cmd!r}")


def _gripper_yaw(world: World) -> float:
    rot = world.data.xmat[world.model.body("link6").id].reshape(3, 3)
    y = rot[:, 1]  # finger opening axis
    return math.atan2(y[1], y[0]) - math.pi / 2


# --- server --------------------------------------------------------------------
class _Handler(socketserver.StreamRequestHandler):
    def handle(self):
        world: World = self.server.world
        wlock = threading.Lock()
        alive = threading.Event()
        alive.set()

        def send(msg):
            with wlock:
                self.wfile.write(protocol.encode(msg))
                self.wfile.flush()

        def stream(rate: float):
            period = 1.0 / max(1.0, min(rate, 1000.0))
            last_t = None
            try:
                while alive.is_set() and not self.server.closing:
                    with world.state_cond:
                        world.state_cond.wait(timeout=period)
                        st = world.state
                    if st["t"] != last_t:
                        send(st)
                        last_t = st["t"]
                    time.sleep(period * 0.9)
            except OSError:
                pass

        try:
            for line in self.rfile:
                if not line.strip():
                    continue
                msg = protocol.decode(line)
                op = msg.get("op")
                if op == "servo_joint":
                    world.servo_joint(msg.get("q"))
                    continue
                if op == "servo_eef":
                    world.servo_eef(msg.get("width", OPEN_WIDTH))
                    continue
                try:
                    reply = {"ok": True, **world.handle(op, msg)}
                except Exception as e:  # noqa: BLE001 - reported to the client
                    reply = {"ok": False, "error": str(e)}
                if op == "subscribe" and reply["ok"]:
                    threading.Thread(target=stream, args=(float(msg.get("rate", 500)),), daemon=True).start()
                if "id" in msg:
                    send({"id": msg["id"], **reply})
        except (OSError, ValueError):
            pass
        finally:
            alive.clear()


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True
    closing = False


def _camera_loop(stop: threading.Event, args) -> None:
    """Keep the virtual camera process running (it exits whenever the ROS master goes away)."""
    cmd = [sys.executable, "-m", "sim.camera", "--scene", args.scene]
    while not stop.is_set():
        proc = subprocess.Popen(cmd)
        while proc.poll() is None and not stop.is_set():
            time.sleep(0.2)
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(3)
            except subprocess.TimeoutExpired:
                proc.kill()
            return
        time.sleep(1.0)


def _start_panel() -> subprocess.Popen | None:
    """The panel (sim/panel.py) in its own process: a Qt window next to the viewer's GLFW one."""
    try:
        return subprocess.Popen([sys.executable, "-m", "sim.panel"])
    except OSError as e:
        print(f"[sim] 仿真面板没有打开：{e}（可以之后运行 pixi run sim-panel）", flush=True)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulated arm, gripper and cubes for the blocks-building demo")
    ap.add_argument("--scene", default=scene.DEFAULT_SCENE, help="scene name or file (default: floor)")
    ap.add_argument("--headless", action="store_true", help="no MuJoCo window")
    ap.add_argument("--no-camera", action="store_true", help="do not run the virtual wrist camera")
    ap.add_argument("--no-panel", action="store_true", help="do not open the panel (pixi run sim-panel opens it)")
    ap.add_argument("--seed", type=int, help="seed for the pick-area cube placement")
    args = ap.parse_args()

    sock = protocol.socket_path()
    lock_file = open(protocol.lock_path(), "w")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        sys.exit(f"已经有一个仿真进程在运行（{sock}）。同一时间只能运行一个。")
    if sock.exists():
        sock.unlink()

    cfg = scene.load_config(args.scene)
    if args.seed is not None:
        cfg.setdefault("pick_area", {})["seed"] = args.seed
    world = World(cfg)
    world.pick.spawn()

    server = _Server(str(sock), _Handler)
    server.world = world
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True).start()

    stop = threading.Event()

    def on_signal(signum, frame):
        if stop.is_set():
            os._exit(1)  # second Ctrl+C: leave now
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    camera = None
    if not args.no_camera:
        camera = threading.Thread(target=_camera_loop, args=(stop, args), daemon=True)
        camera.start()

    print(f"[sim] 仿真已启动（场景 {cfg['name']}，桌高 {cfg['table_height']} m）。接口：{sock}", flush=True)
    print("[sim] 下一步在另一个终端运行：pixi run bringup。按 Ctrl+C 回零点并退出。", flush=True)
    panel = None
    try:
        if args.headless:
            world.run(stop=stop)
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(world.model, world.data, show_left_ui=False, show_right_ui=False) as v:
                v.cam.lookat[:] = [0.25, -0.1, 0.25]
                v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, 200, -25
                if not args.no_panel:
                    panel = _start_panel()
                world.run(viewer=v, stop=stop)
    finally:
        if panel is not None and panel.poll() is None:
            panel.terminate()
        server.closing = True
        server.shutdown()
        server.server_close()
        if sock.exists():
            sock.unlink()
        print("[sim] 已退出", flush=True)
    if camera is not None:
        camera.join(timeout=5.0)  # it stops the wrist camera process
    sys.stdout.flush()
    sys.stderr.flush()
    # Interpreter shutdown sometimes never finished with the window (twice in about 30 exits: after
    # "已退出", every thread waiting on a futex; most likely the viewer's atexit glfw.terminate racing
    # its render thread, not verified). Everything is cleaned up by now, so leave directly, like
    # airbot_play_pick_place.py does.
    os._exit(0)


if __name__ == "__main__":
    main()
