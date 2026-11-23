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


def _yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


class PickArea:
    """Keeps a batch of cubes in the pick area and refills it once it is empty."""

    def __init__(self, world: "World"):
        self.w = world
        cfg = world.cfg
        area = cfg.get("PICK_AREA", {})
        # Where the cubes go: CENTER (default PICK_XY) +- HALF_SIZE. The wrist camera only sees the
        # floor ahead of the gripper at the pick scan pose, so the area sits in front of the scan point.
        self.center = np.array(area.get("CENTER", cfg["PICK_XY"]), dtype=float)
        self.half = np.array(area.get("HALF_SIZE", [0.04, 0.06]), dtype=float)
        self.count = int(area.get("COUNT", 4))
        self.margin = float(area.get("MARGIN", 0.008))
        self.yaw = math.radians(area.get("YAW_DEG", 15.0))
        self.rng = np.random.default_rng(area.get("SEED"))
        self.s = cfg["CUBE_SIZE"] / 2
        self.r = self.s * math.sqrt(2)  # circumradius of a cube's footprint
        self.surface = 0.0 if scene.pick_on_floor(cfg) else cfg["TABLE_HEIGHT"]  # world z
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
        arm = kin.Arm(m, d, self.w.base)
        cx, cy = self.center
        grasp = [cx, cy, self.surface + self.s - self.w.base[2]]
        q = arm.ik(grasp, restarts=60)
        if q is None:
            raise RuntimeError(f"cannot reach the pick area centre {grasp}")
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
        self.base = scene.base_offset(cfg)
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
        self.color_names = list(cfg["CUBE_COLORS"])
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
        self.data.qpos[a + 3 : a + 7] = _yaw_quat(yaw)
        self.data.qvel[v : v + 6] = 0.0

    def park(self, i: int) -> None:
        x, y = scene.pool_park_xy(i)
        self.place_cube(i, (x, y, self.cfg["CUBE_SIZE"] / 2))
        self.cubes[i]["parked"] = True

    # --- arm --------------------------------------------------------------------
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
        t_wall0, t_sim0 = time.perf_counter(), d.time
        next_sync = next_area = 0.0
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
                    viewer.sync()
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
                return {"qpos": self.data.qpos.tolist(), "t": self.data.time}
        if op == "ctl":
            with self.lock:
                return self.ctl(msg.get("cmd", ""), msg.get("args", {}))
        raise ValueError(f"unknown op {op!r}")

    def ctl(self, cmd: str, args: dict) -> dict:
        """sim-ctl commands: the operator's hands."""
        s = self.cfg["CUBE_SIZE"] / 2
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
            dx, dy = float(args.get("dx", 0.0)), float(args.get("dy", 0.0))
            p = self.gripper_point()
            near = min(self.active(), key=lambda i: np.hypot(*(self.cube_pos(i)[:2] - p[:2])), default=None)
            if near is None:
                raise RuntimeError("场景里没有物块")
            pos = self.cube_pos(near).copy()
            pos[0] += dx
            pos[1] += dy
            self.place_cube(near, pos, self.cube_yaw(near) + math.radians(float(args.get("dyaw", 0.0))))
            return {"message": f"cube_{near} 移到 ({pos[0]:.3f}, {pos[1]:.3f})（基座坐标 {pos[0]:.3f}, {pos[1]:.3f}, {pos[2] - self.base[2]:.3f}）"}
        if cmd == "cube_list":
            out = []
            for i in self.active():
                x, y, z = self.cube_pos(i)
                where = "抓取区" if i in self.pick.batch else ""
                out.append({"cube": i, "color": self.cubes[i]["color"], "base_xyz": [x, y, z - self.base[2]], "where": where})
            return {"cubes": out}
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Simulated arm, gripper and cubes for the blocks-building demo")
    ap.add_argument("--scene", default=scene.DEFAULT_SCENE, help="scene name or file (default: floor)")
    ap.add_argument("--headless", action="store_true", help="no MuJoCo window")
    ap.add_argument("--no-camera", action="store_true", help="do not run the virtual wrist camera")
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
        cfg.setdefault("PICK_AREA", {})["SEED"] = args.seed
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

    if not args.no_camera:
        threading.Thread(target=_camera_loop, args=(stop, args), daemon=True).start()

    print(f"[sim] 仿真已启动（场景 {cfg['NAME']}，桌高 {cfg['TABLE_HEIGHT']} m）。接口：{sock}", flush=True)
    print("[sim] 下一步在另一个终端运行：pixi run bringup。按 Ctrl+C 回零点并退出。", flush=True)
    try:
        if args.headless:
            world.run(stop=stop)
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(world.model, world.data, show_left_ui=False, show_right_ui=False) as v:
                v.cam.lookat[:] = [0.25, -0.1, 0.25]
                v.cam.distance, v.cam.azimuth, v.cam.elevation = 1.6, 200, -25
                world.run(viewer=v, stop=stop)
    finally:
        server.closing = True
        server.shutdown()
        server.server_close()
        if sock.exists():
            sock.unlink()
        print("[sim] 已退出", flush=True)


if __name__ == "__main__":
    main()
