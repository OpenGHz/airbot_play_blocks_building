"""Does a grasped cube stay put in the gripper? (`pixi run sim-slip-test`)

Picks a cube from the pick area with the physics of `pixi run sim` (not real time),
then lifts it, carries it to the tower spot, swings the base and the wrist hard and
stops abruptly. Prints how far the cube moved relative to the gripper. Run with
--noslip 0 to see the difference the noslip solver makes.
"""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

from sim import config_file, kin, scene
from sim.world import World, width_to_finger_q


class Rig:
    def __init__(self, cfg, control):
        self.w = World(cfg)
        self.m, self.d = self.w.model, self.w.data
        self.arm = kin.Arm(self.m, mujoco.MjData(self.m), self.w.base)
        self.eef = self.m.body(scene.EEF_BODY).id
        self.control = control
        self.max_slip = 0.0
        self.max_rot = 0.0
        self.track = None

    def step(self, n=1):
        for _ in range(n):
            self.d.ctrl[self.w.act] = self.w.cmd_q
            self.d.ctrl[self.w.grip_act] = width_to_finger_q(self.w.cmd_width)
            mujoco.mj_step(self.m, self.d)
            if self.track is not None:
                self._measure()

    def _rel(self, i):
        mujoco.mj_kinematics(self.m, self.d)  # body poses for the current qpos (mj_step leaves them one step behind)
        p = self.w.cube_pos(i)
        R = self.d.xmat[self.eef].reshape(3, 3)
        a = self.w.cubes[i]["qadr"]
        q = self.d.qpos[a + 3 : a + 7]
        Rc = np.zeros(9)
        mujoco.mju_quat2Mat(Rc, q)
        return R.T @ (p - self.d.xpos[self.eef]), R.T @ Rc.reshape(3, 3)

    def _measure(self):
        i, p0, R0 = self.track
        p, R = self._rel(i)
        self.max_slip = max(self.max_slip, float(np.linalg.norm(p - p0)))
        c = np.clip((np.trace(R0.T @ R) - 1) / 2, -1, 1)
        self.max_rot = max(self.max_rot, math.degrees(math.acos(c)))

    def move(self, q, seconds):
        q0 = self.w.cmd_q.copy()
        n = max(1, int(seconds / self.m.opt.timestep))
        for k in range(1, n + 1):
            s = 0.5 - 0.5 * math.cos(math.pi * k / n)  # smooth start and stop
            self.w.cmd_q = q0 + s * (np.asarray(q) - q0)
            self.step()

    def ik(self, xyz, yaw=0.0, seed=None):
        c, s = math.cos(yaw), math.sin(yaw)
        rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        q = self.arm.ik(xyz, rot=rz @ kin.DOWN, seed=seed if seed is not None else self.w.cmd_q, restarts=40)
        if q is None:
            raise RuntimeError(f"IK failed at {np.round(xyz, 3)}")
        return q


def run(cfg, control, seconds_scale=1.0):
    r = Rig(cfg, control)
    w = r.w
    w.pick.spawn()
    i = w.pick.batch[0]
    x, y, z = w.cube_pos(i)
    yaw = w.cube_yaw(i)
    base = w.base[2]
    grasp_z = z - base  # grasp at the cube centre (PICK_GRASP_Z is "about the centre")
    r.w.cmd_width = 0.07
    r.move(r.ik([x, y, grasp_z + 0.10], yaw), 2.0)
    r.move(r.ik([x, y, grasp_z + 0.025], yaw), 1.0)
    r.move(r.ik([x, y, grasp_z], yaw), 1.0)
    r.w.cmd_width = 0.0
    r.step(int(0.8 / r.m.opt.timestep))
    p0, R0 = r._rel(i)
    r.track = (i, p0, R0)
    t = seconds_scale
    tx, ty = cfg["TOWER_XY"]
    r.move(r.ik([x, y, grasp_z + 0.035], yaw), 0.6 * t)  # lift
    r.move(r.ik([tx, ty, grasp_z + 0.30]), 1.2 * t)  # carry high
    q = w.cmd_q.copy()
    for dq1 in (0.9, -0.9, 0.0):  # swing the base
        q1 = q.copy()
        q1[0] += dq1
        r.move(q1, 0.5 * t)
    for dq6 in (1.5, -1.5, 0.0):  # spin the wrist
        q6 = q.copy()
        q6[5] += dq6
        r.move(q6, 0.4 * t)
    r.move(r.ik([tx, ty, grasp_z + 0.02]), 1.0 * t)  # down to the floor
    r.step(int(0.5 / r.m.opt.timestep))
    held = np.linalg.norm(r._rel(i)[0]) < 0.03
    # peak joint speed of the motions, for reference
    return {"max_slip_mm": 1000 * r.max_slip, "max_rot_deg": r.max_rot, "held": bool(held)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default=scene.DEFAULT_SCENE)
    ap.add_argument("--control", default="configs/control/sim.yaml")
    ap.add_argument("--noslip", type=int, help="override NOSLIP_ITERATIONS (0 = off)")
    ap.add_argument("--trials", type=int, default=5, help="different random pick-area layouts")
    ap.add_argument("--fast", type=float, default=1.0, help="time scale of the moves (0.5 = twice as fast)")
    args = ap.parse_args()
    cfg = scene.load_config(args.scene)
    if args.noslip is not None:
        cfg["NOSLIP_ITERATIONS"] = args.noslip
    control = config_file.load(args.control)
    worst = {"max_slip_mm": 0.0, "max_rot_deg": 0.0, "held": True}
    for k in range(args.trials):
        cfg.setdefault("PICK_AREA", {})["SEED"] = k
        res = run(cfg, control, args.fast)
        print(
            f"trial {k}: slip {res['max_slip_mm']:.2f} mm, rotation {res['max_rot_deg']:.2f} deg, held: {res['held']}"
        )
        worst = {
            "max_slip_mm": max(worst["max_slip_mm"], res["max_slip_mm"]),
            "max_rot_deg": max(worst["max_rot_deg"], res["max_rot_deg"]),
            "held": worst["held"] and res["held"],
        }
    print(
        f"worst: slip {worst['max_slip_mm']:.2f} mm, rotation {worst['max_rot_deg']:.2f} deg, all held: {worst['held']}"
    )


if __name__ == "__main__":
    main()
