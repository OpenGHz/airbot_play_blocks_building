"""End-to-end runs in the simulation (`pixi run test-e2e SCENARIO`).

Starts the whole stack on a private ROS master and simulation socket, so it does not disturb a
stack you have running: `python -m sim.world --headless`, `scripts/bringup.sh use_rviz:=false`,
the vision (Best_Contour.py without windows, or a zero offset on /target_TF) and the pick-place
program. It watches the simulation, answers the program's prompts, and writes the logs and
report.json to runs/e2e/<scenario>_<time>/.

Scenarios
  floor     configs/control/sim.yaml, floor scene; stop after --blocks blocks (default 6) with Ctrl+C;
            the program must exit with status 0.
  open-place  the same with `--open-place`: no vision in the place stage, each block is released
            right over the first one.
  held-cube the same as floor with the vision's `--held-cube`: the place stage also finds the
            held cube and aligns it, not the gripper, with the tower's top block.
  --place-stop PX (floor, held-cube): also run the program with `--place-stop PX`: the place
            stage stops only when the remaining move along the approach is below PX pixels.
  --scene (floor, open-place, held-cube, limit, golden): run the program with `--scene`, the
            planning scene instead of the default fixed waypoints. The transfers are then one
            trajectory each, so floor-like scenarios also require transfers without stops (the
            fixed waypoints stop between segments); golden compares the old program with the new
            one's --scene path.
  limit     configs/control/real.yaml, tabletop scene (the real robot's layout, tower on the
            other side of the pick area): build until the arm cannot go higher, answer the
            prompt with Enter. Expected: LIMIT_BLOCKS (6) blocks, back Home, exit status 0.
  baseline  the program before the changes (git 7a1c2c2) in the same setup. Expected (plan
            §1.1): it freezes after block LIMIT_BLOCKS (6), with a "bad callback" in its log.
  kill-sim  floor scene; kill the simulation while the arm carries block 3. Expected: the
            program says what happened and exits with status 2 (no hang, no crash).
  golden    zero vision offset, tabletop scene, real.yaml: record every motion target of the
            old and the new program and compare them (plan §8). Without the vision the gripper
            closes at the scan point and only pinches pick-area cubes by an edge, then drops
            them high above the tower, so the tower and clearance numbers are not reported.

Every run reports: blocks on the tower; how far each tower block is off the bottom one;
tower blocks that moved by more than 3 mm and whether
the arm touched the tower just before; the smallest distance between the arm (with the block it
holds) and the tower while moving between the pick area and the tower (the straight vertical
moves right at the tower top are excluded, they are meant to be close); and stops in the middle
of those moves (the end effector at rest for 0.2 s or more, away from the start and end).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import signal
import socket
import subprocess
import sys
import threading
import time

import mujoco
import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from sim import config_file, scene  # noqa: E402
from sim.driver import DriverError, MujocoDriver  # noqa: E402

OLD_COMMIT = "7a1c2c2"
PROMPT_FAULT = "按回车回 Home"
PROMPT_REFILL = "补充物块后按回车"
STALL_LIMIT = 10.0  # s without an answer from the simulation (connection still open) before the monitor gives up
LIMIT_BLOCKS = 6  # limit / baseline: blocks before the vertical rise over the tower fails (real robot, old
# model: 7, plan §1.1; the DISCOVERSE model's forearm is 18 mm shorter, so the limit comes a block earlier)
CLOSED_WIDTH = 0.045  # gripper narrower than this: closed (on a cube ~0.025, empty ~0)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------------------
# processes
# --------------------------------------------------------------------------------------
class Stack:
    """The processes of one run, each in its own process group, logs in `out`."""

    def __init__(self, out: pathlib.Path):
        self.out = out
        self.env = dict(
            os.environ,
            ROS_MASTER_URI=f"http://localhost:{free_port()}",
            AIRBOT_SIM_SOCKET=str(out / "sim.sock"),
            PYTHONUNBUFFERED="1",
        )
        self.procs: dict[str, subprocess.Popen] = {}
        self.text: dict[str, list[str]] = {}

    def start(self, name, cmd, cwd=REPO, pipe=False) -> subprocess.Popen:
        log = open(self.out / f"{name}.log", "wb")
        p = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=self.env,
            start_new_session=True,
            stdin=subprocess.PIPE if pipe else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.procs[name] = p
        self.text[name] = []

        def pump():
            while True:
                chunk = p.stdout.read1(4096)
                if not chunk:
                    break
                log.write(chunk)
                log.flush()
                self.text[name].append(chunk.decode(errors="replace"))

        threading.Thread(target=pump, daemon=True).start()
        return p

    def output(self, name) -> str:
        return "".join(self.text.get(name, []))

    def send(self, name, text: str) -> None:
        p = self.procs[name]
        p.stdin.write(text.encode())
        p.stdin.flush()

    def alive(self, name) -> bool:
        p = self.procs.get(name)
        return p is not None and p.poll() is None

    def stop(self, name, sig=signal.SIGINT, wait: float = 15.0) -> int | None:
        p = self.procs.get(name)
        if p is None:
            return None
        if p.poll() is None:
            try:
                os.killpg(p.pid, sig)
                p.wait(wait)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                p.wait()
        return p.returncode

    def rosparam(self, *args) -> None:
        subprocess.run(["rosparam", *map(str, args)], env=self.env, check=True, capture_output=True, timeout=20)

    def wait_for(self, name, text, timeout) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if text in self.output(name):
                return True
            if not self.alive(name):
                return False
            time.sleep(0.2)
        return False


# --------------------------------------------------------------------------------------
# watching the simulation
# --------------------------------------------------------------------------------------
class Monitor(threading.Thread):
    """Replays the simulation's qpos at ~50 Hz in a copy of its model and records what happens."""

    def __init__(self, sock: pathlib.Path, cfg: dict, control: dict):
        super().__init__(daemon=True)
        self.sock, self.cfg, self.control = sock, cfg, control
        self.m, self.d, _ = scene.build_physics(cfg)
        m = self.m
        self.frame = scene.base_frame(cfg)
        self.tower_world = np.array(cfg["TOWER_XY"]) + scene.scene_offset(cfg)[:2]
        self.tower_xy = self.frame.to_base(self.tower_world)  # base frame, like the end-effector samples
        self.h = control["CUBE_HEIGHT"]
        self.cubes = {m.body(f"cube_{i}").id: i for i in range(int(cfg.get("CUBE_POOL", 40)))}
        self.cube_geom = {i: m.geom(f"cube_{i}").id for i in self.cubes.values()}
        root = m.body("base_link").id

        def under_base(b):
            while b > 0:
                if b == root:
                    return True
                b = m.body_parentid[b]
            return False

        self.robot_geoms = [
            g for g in range(m.ngeom) if under_base(m.geom_bodyid[g]) and m.geom_contype[g] | m.geom_conaffinity[g]
        ]
        self.eef = m.body(scene.EEF_BODY).id
        self.fingers = [m.body(n).id for n in scene.FINGER_BODIES]
        self.finger_q = [m.joint(j).qposadr[0] for j in scene.FINGER_JOINTS]
        self.samples = []  # (t, eef xyz, width, moving joints)
        self.held = []  # per sample: held cube centre minus end effector (m) and its tilt (deg); NaN when none
        self.qpos = []  # per sample: the whole simulation state (arm, fingers, cubes), to replay a run
        self.cube_ids = np.array(list(self.cubes))
        self.clearance = []  # (t, distance, robot geom, tower cube) outside the work zone at the tower top
        self.events = []
        self.tower_count = 0
        self.ref = {}  # settled tower cube -> (pos, yaw)
        self.contacts = []  # (t, robot geom name, cube) arm/tower contacts outside the work zone
        self.stop_flag = threading.Event()
        self.error = None
        self.t0 = time.monotonic()

    def log(self, text):
        self.events.append((round(time.monotonic() - self.t0, 2), text))

    def width(self) -> float:
        from sim.world import finger_q_to_width

        return float(finger_q_to_width(float(self.d.qpos[self.finger_q[0]])))

    def run(self):
        drv = MujocoDriver(path=self.sock, rate=1)
        try:
            drv.connect(10.0)
        except DriverError as e:
            self.error = str(e)
            return
        last_q, k = None, 0
        silent_since = None  # the simulation can stall for a few seconds (refilling the pick area under load)
        while not self.stop_flag.is_set():
            try:
                qpos = drv.request("get_qpos", timeout=2.0)["qpos"]
                silent_since = None
            except DriverError as e:
                now = time.monotonic()
                silent_since = silent_since or now
                if drv.ok() and now - silent_since < STALL_LIMIT:
                    continue
                self.error = str(e)
                self.log(f"lost the simulation: {e}")
                break
            t = time.monotonic() - self.t0
            self.d.qpos[:] = qpos
            mujoco.mj_forward(self.m, self.d)
            q = np.array(qpos[:6])
            moving = last_q is not None and float(np.abs(q - last_q).max()) > 1e-4
            last_q = q
            eef = self.frame.to_base(self.d.xpos[self.eef])
            self.samples.append((t, eef.copy(), self.width(), moving))
            self.held.append(self.held_cube())
            self.qpos.append(np.asarray(qpos, dtype=np.float32))
            self.step(t, eef, k % 5 == 0)
            k += 1
            time.sleep(0.02)
        drv.disconnect()

    def held_cube(self) -> list:
        """The cube between the closed fingers: centre minus end effector (m, base axes) and tilt (deg)."""
        if self.width() >= CLOSED_WIDTH:
            return [math.nan] * 4
        rel = (
            self.d.xpos[self.cube_ids] - self.d.xpos[self.eef]
        )  # the finger bodies' origins are 2 cm up, at the finger roots
        dist = np.linalg.norm(rel, axis=1)
        if dist.min() > 0.015:
            return [math.nan] * 4
        b = self.cube_ids[dist.argmin()]
        rel = rel[dist.argmin()]
        tilt = math.degrees(math.acos(min(1.0, abs(float(self.d.xmat[b].reshape(3, 3)[2, 2])))))
        return [*rel, tilt]

    def step(self, t, eef, check_clearance: bool):
        m, d = self.m, self.d
        width = self.width()
        mid = d.xpos[self.fingers].mean(0)
        held = {i for b, i in self.cubes.items() if width < CLOSED_WIDTH and np.linalg.norm(d.xpos[b] - mid) < 0.02}
        tower = {
            i: b
            for b, i in self.cubes.items()
            if i not in held and np.linalg.norm(d.xpos[b][:2] - self.tower_world) < 0.03
        }
        # settled blocks: no robot geom touching them
        touching = set()
        for c in d.contact[: d.ncon]:
            for g_cube, g_other in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                b = m.geom_bodyid[g_cube]
                if b in self.cubes and self.cubes[b] in tower and g_other in self.robot_geoms:
                    touching.add(self.cubes[b])
        for i, b in tower.items():
            if i not in touching and i not in self.ref:
                self.ref[i] = (d.xpos[b].copy(), self._yaw(b))
        n = sum(1 for i in self.ref if i in tower)
        if n != self.tower_count:
            self.log(f"tower: {n} blocks")
            self.tower_count = n
        top = max(tower.values(), key=lambda b: d.xpos[b][2], default=None)
        in_zone = (
            top is not None
            and np.linalg.norm(eef[:2] - self.frame.to_base(d.xpos[top])[:2]) < 0.02
            and eef[2] < self.frame.to_base(d.xpos[top])[2] + self.h / 2 + self.h + 0.02
        )
        if not in_zone and touching:
            names = sorted(
                {
                    m.geom(g).name or m.body(m.geom_bodyid[g]).name
                    for c in d.contact[: d.ncon]
                    for g in (c.geom1, c.geom2)
                    if g in self.robot_geoms
                }
            )
            self.contacts.append((round(t, 2), names, sorted(touching)))
        moved = []
        for i in list(self.ref):
            if i not in tower:
                moved.append(f"cube_{i} left the tower")
                del self.ref[i]
                continue
            p0, y0 = self.ref[i]
            dp = float(np.linalg.norm(d.xpos[tower[i]] - p0)) * 1000
            dyaw = abs((self._yaw(tower[i]) - y0 + 45) % 90 - 45)
            if dp > 3.0 or dyaw > 3.0:
                moved.append(f"cube_{i} moved {dp:.1f} mm, turned {dyaw:.1f} deg")
                self.ref[i] = (d.xpos[tower[i]].copy(), self._yaw(tower[i]))
        if moved:
            recent = [c for c in self.contacts if c[0] > t - 2.0]
            self.log(
                "tower disturbed: "
                + "; ".join(moved)
                + (f" (arm touched the tower: {recent[-1][1]})" if recent else " (no arm contact)")
            )
        if check_clearance and tower and not in_zone:
            arm = list(self.robot_geoms) + [self.cube_geom[i] for i in held]
            tower_geoms = [self.cube_geom[i] for i in tower]
            best = (math.inf, None, None)
            axis = self.tower_world
            for g in arm:
                if math.hypot(*(d.geom_xpos[g][:2] - axis)) - m.geom_rbound[g] > 0.15:
                    continue
                for gt in tower_geoms:
                    dist = mujoco.mj_geomDistance(m, d, g, gt, 0.2, None)
                    if dist < best[0]:
                        best = (dist, g, gt)
            if best[1] is not None:
                self.clearance.append(
                    (
                        t,
                        float(best[0]),
                        m.geom(best[1]).name or m.body(m.geom_bodyid[best[1]]).name,
                        m.geom(best[2]).name,
                    )
                )

    def _yaw(self, b):
        r = self.d.xmat[b].reshape(3, 3)
        return math.degrees(math.atan2(r[1, 0], r[0, 0])) % 90

    def tower_alignment(self) -> dict:
        """Horizontal offsets (mm) of the settled tower blocks, bottom to top, in the last state."""
        bodies = {i: b for b, i in self.cubes.items()}
        pos = sorted((self.d.xpos[bodies[i]].copy() for i in self.ref), key=lambda p: p[2])
        if len(pos) < 2:
            return {}
        xy = np.array([p[:2] for p in pos])
        from_bottom = np.linalg.norm(xy - xy[0], axis=1) * 1000
        steps = np.linalg.norm(np.diff(xy, axis=0), axis=1) * 1000
        return {
            "offset_from_bottom_mm": [round(float(v), 1) for v in from_bottom],
            "max_step_mm": round(float(steps.max()), 1),
            "xy_base": [[round(float(v), 5) for v in self.frame.to_base(p)] for p in xy],
        }  # bottom to top, base frame (m)

    def stationary_for(self) -> float:
        """Seconds the arm joints have not moved."""
        if not self.samples:
            return 0.0
        t_last = self.samples[-1][0]
        for t, _, _, moving in reversed(self.samples):
            if moving:
                return t_last - t
        return t_last - self.samples[0][0]

    # ---- after the run ----
    def analyse(self, scan_xyz) -> dict:
        s = self.samples
        if len(s) < 10:
            return {"error": "too few samples"}
        t = np.array([x[0] for x in s])
        p = np.array([x[1] for x in s])
        w = np.array([x[2] for x in s])
        v = np.r_[0.0, np.linalg.norm(np.diff(p, axis=0), axis=1) / np.maximum(np.diff(t), 1e-3)]
        closed = w < CLOSED_WIDTH
        edges = np.flatnonzero(np.diff(closed.astype(int)))
        grasp_z = self.control["PICK_GRASP_Z"]
        transfers = []
        for e in edges:
            start = e + 1
            off_tower = np.linalg.norm(p[start, :2] - self.tower_xy)
            if closed[start] and abs(p[start, 2] - grasp_z) < 0.01 and off_tower > 0.05:
                kind = "to tower"  # grasped in the pick area: until at rest above the tower
                arrive = lambda j: np.linalg.norm(p[j, :2] - self.tower_xy) < 0.02 and v[j] < 0.005  # noqa: E731
            elif not closed[start] and off_tower < 0.03:
                kind = "back to pick"  # released on the tower: until at rest at the pick scan pose
                arrive = lambda j: np.linalg.norm(p[j] - scan_xyz) < 0.015 and v[j] < 0.005  # noqa: E731
            else:
                continue  # the gripper opening at start-up, or closing on nothing during a test
            end = next((j for j in range(start + 5, len(s)) if arrive(j)), None)
            if end is None:
                continue
            transfers.append(self._transfer(kind, t, p, v, start, end))
        clear = [c for c in self.clearance if any(tr["t0"] <= c[0] <= tr["t1"] for tr in transfers)]
        worst = min(clear, key=lambda c: c[1]) if clear else None
        return {
            "transfers": transfers,
            "stops_in_transfers": sum(len(tr["stops"]) for tr in transfers),
            "min_clearance_mm": None if worst is None else round(worst[1] * 1000, 1),
            "min_clearance_at": None if worst is None else {"t": round(worst[0], 1), "arm": worst[2], "cube": worst[3]},
        }

    @staticmethod
    def _transfer(kind, t, p, v, a, b) -> dict:
        rest = v[a : b + 1] < 0.003
        stops, j = [], 0
        while j < len(rest):
            if rest[j]:
                k = j
                while k < len(rest) and rest[k]:
                    k += 1
                i0, i1 = a + j, a + k - 1
                where = p[i0 : i1 + 1].mean(0)
                if t[i1] - t[i0] >= 0.2 and np.linalg.norm(where - p[a]) > 0.02 and np.linalg.norm(where - p[b]) > 0.02:
                    stops.append(
                        {
                            "t": round(float(t[i0]), 1),
                            "seconds": round(float(t[i1] - t[i0]), 2),
                            "at": [round(float(x), 3) for x in where],
                        }
                    )
                j = k
            else:
                j += 1
        return {
            "kind": kind,
            "t0": round(float(t[a]), 1),
            "t1": round(float(t[b]), 1),
            "seconds": round(float(t[b] - t[a]), 1),
            "stops": stops,
        }


# --------------------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------------------
def old_worktree() -> pathlib.Path:
    path = REPO / "runs/e2e" / f"old-{OLD_COMMIT}"
    if not (path / "airbot_play_pick_place.py").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(path), OLD_COMMIT], cwd=REPO, check=True, capture_output=True
        )
    return path


def run(
    name: str,
    scene_name: str,
    env_type: str,
    program: str = "new",
    vision: str = "camera",
    record: bool = False,
    out: pathlib.Path | None = None,
    policy=None,
    timeout: float = 900.0,
    args: tuple = (),
    vision_args: tuple = (),
) -> dict:
    out = out or REPO / "runs/e2e" / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    cfg = scene.load_config(scene_name)
    control = config_file.load(REPO / f"configs/control/{env_type}.yaml")
    stack = Stack(out)
    report = {
        "scenario": name,
        "scene": scene_name,
        "control": env_type,
        "program": program,
        "vision": " ".join([vision, *vision_args]),
        "out": str(out.relative_to(REPO)),
    }
    print(f"[e2e] {name}: logs in {report['out']}", flush=True)
    monitor = None
    try:
        sim_cmd = [sys.executable, "-m", "sim.world", "--scene", scene_name, "--headless", "--seed", "1"]
        if vision != "camera":
            sim_cmd.append("--no-camera")
        stack.start("sim", sim_cmd)
        deadline = time.monotonic() + 60
        while not (out / "sim.sock").exists():
            if time.monotonic() > deadline or not stack.alive("sim"):
                raise RuntimeError("the simulation did not start (sim.log)")
            time.sleep(0.2)
        stack.start("bringup", ["bash", "scripts/bringup.sh", "use_rviz:=false"])
        if not stack.wait_for("bringup", "You can start planning now", 120):
            raise RuntimeError("bringup did not get ready (bringup.log)")
        if env_type != "sim":
            stack.rosparam("set", "/env_type", env_type)
        stack.rosparam("set", "/control_param1", 90)
        stack.rosparam("set", "/control_param2", 100)
        if vision == "camera":
            stack.start(
                "vision",
                [
                    sys.executable,
                    "Best_Contour.py",
                    "-cfg",
                    "./configs/vision/sim.yaml",
                    "-vd",
                    "/camera/color/image_raw",
                    "-ns",
                    *vision_args,
                ],
            )
            if not stack.wait_for("vision", "开始获取图像", 60):
                raise RuntimeError("the vision did not start (vision.log)")
        else:
            stack.start("vision", [sys.executable, str(REPO / "tests/e2e/zero_target.py")])
        monitor = Monitor(out / "sim.sock", cfg, control)
        monitor.start()
        cwd = old_worktree() if program == "old" else REPO
        cmd = [sys.executable, "airbot_play_pick_place.py", "-cp", "2", "-r", *args]
        if record:
            cmd = [sys.executable, str(REPO / "tests/e2e/record_motion.py"), str(out / "motion.jsonl")] + cmd[1:]
        stack.start("pick_place", cmd, cwd=cwd, pipe=True)
        state = {"answered": 0, "t_start": time.monotonic(), "freeze": False, "killed_sim": None, "stopped": False}
        while time.monotonic() - state["t_start"] < timeout:
            time.sleep(0.2)
            if policy is not None and policy(stack, monitor, state):
                break
            if not stack.alive("pick_place"):
                break
        else:
            report["timeout"] = True
        report.update({k: v for k, v in state.items() if k not in ("t_start",)})
    except Exception as e:  # noqa: BLE001
        report["error"] = str(e)
    finally:
        rc = stack.stop("pick_place", signal.SIGINT, wait=20)
        report["pick_place_exit"] = rc
        if monitor is not None:
            monitor.stop_flag.set()
            monitor.join(5)
        for proc in ("vision", "bringup"):
            stack.stop(proc, signal.SIGINT, wait=20)
        stack.stop("sim", signal.SIGTERM, wait=15)
    log = stack.output("pick_place")
    report["blocks"] = monitor.tower_count if monitor else None
    report["bad_callback"] = "bad callback" in log
    report["fault_prompts"] = log.count(PROMPT_FAULT)
    report["built_line"] = next((ln.strip() for ln in log.splitlines()[::-1] if "共搭了" in ln), None)
    report["log_tail"] = [ln for ln in log.splitlines() if ln.strip()][-6:]
    if monitor is not None:
        if monitor.samples:  # raw data, to look at a run again without repeating it
            np.savez_compressed(
                out / "samples.npz",
                t=[x[0] for x in monitor.samples],
                eef=[x[1] for x in monitor.samples],
                width=[x[2] for x in monitor.samples],
                clearance=np.array([c[:2] for c in monitor.clearance]),
                held=np.array(monitor.held),
                qpos=np.array(monitor.qpos),
            )
        scan = np.array([control["PICK_POSITION_X"], control["PICK_POSITION_Y"], control["PICK_POSITION_Z"]])
        report["events"] = monitor.events
        report["tower_disturbed"] = [e for e in monitor.events if "tower disturbed" in e[1]]
        report["tower_alignment"] = monitor.tower_alignment()
        report.update(monitor.analyse(scan))
        report["monitor_error"] = monitor.error
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    return report


# --------------------------------------------------------------------------------------
# scenario policies: return True to end the run
# --------------------------------------------------------------------------------------
def stop_after(blocks: int):
    def policy(stack, mon, st):
        if mon.tower_count >= blocks and not st["stopped"]:
            time.sleep(1.0)
            st["stopped"] = True
            return True
        return False

    return policy


def answer_enter(stack, mon, st):
    if stack.output("pick_place").count(PROMPT_FAULT) > st["answered"]:
        st["answered"] += 1
        time.sleep(1.0)
        stack.send("pick_place", "\n")
    return False


def detect_freeze(seconds: float = 25.0, need_blocks: bool = True):
    """Arm still for `seconds` without a prompt, after building started (or, with need_blocks
    False, after the arm first moved: with a zero vision offset nothing ends up on the tower)."""

    def policy(stack, mon, st):
        started = mon.tower_count >= 1 if need_blocks else any(x[3] for x in mon.samples)
        if started and mon.stationary_for() > seconds and PROMPT_FAULT not in stack.output("pick_place"):
            st["freeze"] = True
            st["freeze_after_blocks"] = mon.tower_count
            return True
        return answer_enter(stack, mon, st)

    return policy


def kill_sim_during(block: int, watch: float = 60.0):
    def policy(stack, mon, st):
        if st["killed_sim"] is None:
            if mon.tower_count >= block - 1 and mon.samples and mon.samples[-1][2] < CLOSED_WIDTH:
                time.sleep(1.5)  # carrying the block
                stack.stop("sim", signal.SIGKILL, wait=5)
                st["killed_sim"] = time.monotonic()
                mon.log("killed the simulation (SIGKILL)")
            return False
        elapsed = time.monotonic() - st["killed_sim"]
        if not stack.alive("pick_place"):
            st["pick_place_ended_after_s"] = round(elapsed, 1)
            return True
        if PROMPT_FAULT in stack.output("pick_place") and "prompt_after_s" not in st:
            st["prompt_after_s"] = round(elapsed, 1)
        return elapsed > watch

    return policy


# --------------------------------------------------------------------------------------
# golden comparison
# --------------------------------------------------------------------------------------
def load_motion(path: pathlib.Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def compare(old: list[dict], new: list[dict]) -> dict:
    """Same gripper commands at the same poses (grasp and release heights, every layer)?"""

    def grips(ev):
        return [e for e in ev if e["ev"] == "gripper"]

    go, gn = grips(old), grips(new)
    n = min(len(go), len(gn))
    diffs = []
    for i in range(n):
        a, b = go[i], gn[i]
        if a["cmd"] != b["cmd"]:
            diffs.append(f"gripper command {i}: old {a['cmd']}, new {b['cmd']}")
        elif max(abs(x - y) for x, y in zip(a["xyz"], b["xyz"])) > 1e-5:
            diffs.append(
                f"gripper command {i} ({'close' if a['cmd'] else 'open'}): old at {a['xyz']}, new at {b['xyz']}"
            )

    def segments(ev):
        """Motion events between gripper commands."""
        out, cur = [], []
        for e in ev:
            if e["ev"] == "gripper":
                out.append(cur)
                cur = []
            elif e["ev"] in ("pose", "axis", "joints", "line", "plan", "plan_joints", "error"):
                cur.append(e)
        out.append(cur)
        return out

    def fmt(e):
        if e["ev"] == "joints":
            return f"joints {e.get('name') or ''}".strip()
        if e["ev"] == "plan_joints":
            return f"plan->joints ({e['what']})"
        if e["ev"] == "error":
            return f"ERROR in {e['call']}: {e['error'][:80]}"
        label = {"pose": "pose", "axis": "axis", "line": "line", "plan": "plan"}[e["ev"]]
        what = f" ({e['what']})" if "what" in e else ""
        return f"{label} ({', '.join(f'{v:+.4f}' for v in e['xyz'])}){what}"

    so, sn = segments(old), segments(new)
    table = []
    for i in range(max(len(so), len(sn))):
        table.append(
            {
                "after_gripper_command": i,
                "old": [fmt(e) for e in (so[i] if i < len(so) else [])],
                "new": [fmt(e) for e in (sn[i] if i < len(sn) else [])],
            }
        )
    return {
        "gripper_commands": {"old": len(go), "new": len(gn)},
        "same_grasp_and_release_poses": not diffs,
        "differences": diffs,
        "moves": table,
    }


# --------------------------------------------------------------------------------------
def summary(r: dict, tower: bool = True) -> str:
    lines = [
        f"== {r['scenario']} ({r['program']} program, {r['scene']} scene, {r['control']}.yaml, vision {r['vision']})"
    ]
    if r.get("error"):
        lines.append(f"   ERROR: {r['error']}")
    lines.append(
        f"   blocks on the tower: {r.get('blocks')}; pick-place exit status: {r.get('pick_place_exit')}"
        + ("; TIMEOUT" if r.get("timeout") else "")
    )
    if r.get("freeze"):
        lines.append(
            f"   froze after {r.get('freeze_after_blocks')} blocks (arm still for 25 s, no prompt); "
            f"bad callback in the log: {r.get('bad_callback')}"
        )
    if r.get("fault_prompts"):
        lines.append(f"   failure prompts: {r['fault_prompts']}; {r.get('built_line')}")
    if r.get("killed_sim"):
        lines.append(
            f"   killed the simulation; prompt after {r.get('prompt_after_s')} s; "
            f"pick-place ended after {r.get('pick_place_ended_after_s')} s"
        )
    if not tower:
        lines.append("   last lines: " + " | ".join(r.get("log_tail", [])[-3:]))
        return "\n".join(lines)
    lines.append(
        f"   tower disturbed: {len(r.get('tower_disturbed') or [])}"
        + "".join(f"\n     {e[0]} s {e[1]}" for e in (r.get("tower_disturbed") or []))
    )
    if r.get("tower_alignment"):
        a = r["tower_alignment"]
        lines.append(
            f"   tower alignment: largest step between layers {a['max_step_mm']} mm; "
            f"offset from the bottom block, bottom to top (mm): {a['offset_from_bottom_mm']}"
        )
    if "transfers" in r:
        lines.append(
            f"   transfers: {len(r['transfers'])}, stops in the middle: {r['stops_in_transfers']}; "
            f"smallest arm-tower distance while transferring: {r['min_clearance_mm']} mm {r.get('min_clearance_at') or ''}"
        )
        for tr in r["transfers"]:
            if tr["stops"]:
                lines.append(f"     {tr['kind']} at {tr['t0']} s ({tr['seconds']} s): stops {tr['stops']}")
    lines.append("   last lines: " + " | ".join(r.get("log_tail", [])[-3:]))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("\n\n", 1)[1],
    )
    ap.add_argument("scenario", choices=["floor", "open-place", "held-cube", "limit", "baseline", "kill-sim", "golden"])
    ap.add_argument("--blocks", type=int, default=6, help="floor, open-place, held-cube: stop after this many blocks")
    ap.add_argument(
        "--place-stop", type=float, metavar="PX", help="floor, held-cube: run the program with --place-stop PX"
    )
    ap.add_argument(
        "--scene",
        action="store_true",
        help="floor, open-place, held-cube, limit, golden: run the program with --scene (the planning scene)",
    )
    args = ap.parse_args()
    scene_args = ("--scene",) if args.scene else ()
    suffix = "-scene" if args.scene else ""
    if args.scenario in ("floor", "open-place", "held-cube"):
        extra = ("--open-place",) if args.scenario == "open-place" else ()
        name = args.scenario
        if args.place_stop is not None and args.scenario != "open-place":
            extra, name = ("--place-stop", str(args.place_stop)), f"{args.scenario}-stop{args.place_stop:g}"
        vision_extra = ("--held-cube",) if args.scenario == "held-cube" else ()
        r = run(
            name + suffix,
            "floor",
            "sim",
            policy=stop_after(args.blocks),
            args=extra + scene_args,
            vision_args=vision_extra,
        )
        # exit 0 after the Ctrl+C that ends the run: it used to abort now and then (-6) when a MoveIt call
        # returned while Python shut down (the end of airbot_play_pick_place.py)
        ok = (
            r["blocks"] >= args.blocks
            and not r["tower_disturbed"]
            and (not args.scene or r["stops_in_transfers"] == 0)
            and r["pick_place_exit"] == 0
        )
    elif args.scenario == "limit":
        r = run("limit" + suffix, "tabletop", "real", policy=answer_enter, args=scene_args)
        ok = (
            r["pick_place_exit"] == 0
            and r["fault_prompts"] == 1
            and r["blocks"] == LIMIT_BLOCKS
            and not r["tower_disturbed"]
        )
    elif args.scenario == "baseline":
        r = run("baseline", "tabletop", "real", program="old", policy=detect_freeze())
        ok = bool(r.get("freeze")) and r["bad_callback"] and r.get("freeze_after_blocks") == LIMIT_BLOCKS
    elif args.scenario == "kill-sim":
        r = run("kill-sim", "floor", "sim", policy=kill_sim_during(3))
        ok = (
            bool(r.get("killed_sim"))
            and r.get("pick_place_ended_after_s") is not None
            # exit status 2: the joint-state watchdog's own exit (it prints what happened). A crash
            # (-11 when a MoveIt pose target found no current robot state) is a failure too.
            and r["pick_place_exit"] == 2
        )
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        base = REPO / "runs/e2e" / f"golden{suffix}_{stamp}"
        runs = {}
        for program in ("old", "new"):
            runs[program] = run(
                f"golden-{program}",
                "tabletop",
                "real",
                program=program,
                vision="zero",
                record=True,
                out=base / program,
                policy=detect_freeze(15, need_blocks=False),
                args=scene_args if program == "new" else (),
            )
            print(summary(runs[program], tower=False), flush=True)
        cmp = compare(load_motion(base / "old/motion.jsonl"), load_motion(base / "new/motion.jsonl"))
        (base / "compare.json").write_text(json.dumps(cmp, ensure_ascii=False, indent=1))
        print(
            f"== golden: gripper commands old {cmp['gripper_commands']['old']}, new {cmp['gripper_commands']['new']}; "
            f"same grasp and release poses: {cmp['same_grasp_and_release_poses']}"
        )
        for d in cmp["differences"]:
            print("   ", d)
        print(f"   moves side by side: {(base / 'compare.json').relative_to(REPO)}")
        return 0 if cmp["same_grasp_and_release_poses"] else 1
    print(summary(r), flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
