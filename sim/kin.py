"""Inverse kinematics and clearance checks on the MuJoCo model.

Used offline (scene analysis, snapshots); the running system uses MoveIt.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np

from sim import scene

# Gripper pointing straight down, as in configs/control/*.yaml (rpy = 0, pi/2, 0).
DOWN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])


def limit_margin(cfg: dict) -> float:
    """How far (rad) IK solutions must stay inside the URDF joint limits."""
    return math.radians(cfg.get("JOINT_LIMIT_MARGIN_DEG", 2.0))


class Arm:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, base: np.ndarray, limit_margin: float = 0.0):
        self.m, self.d, self.base = model, data, np.asarray(base, dtype=float)
        self.qadr = np.array([model.joint(j).qposadr[0] for j in scene.ARM_JOINTS])
        self.dadr = np.array([model.joint(j).dofadr[0] for j in scene.ARM_JOINTS])
        self.hard_lo, self.hard_hi = model.jnt_range[[model.joint(j).id for j in scene.ARM_JOINTS]].T
        # IK only uses the range shrunk by the margin; the hard limits stay in the physics model.
        self.lo, self.hi = self.hard_lo + limit_margin, self.hard_hi - limit_margin
        self.eef = model.body(scene.EEF_BODY).id
        self.rng = np.random.default_rng(0)
        # base_link sits on the table, so it is left out of the clearance check.
        robot = {model.body(b).id for b in _robot_bodies(model) if b != "base_link"}
        self.robot_geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in robot and model.geom_contype[g]]
        env = ["table_top", "column_0", "column_1", "foot_0", "foot_1"]
        self.env_geoms = {n: model.geom(n).id for n in env}
        self.floor = model.geom("floor").id

    # --- forward ---------------------------------------------------------
    def fk(self, q):
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)  # needed by mj_jacBody
        return self.d.xpos[self.eef] - self.base, self.d.xmat[self.eef].reshape(3, 3)

    # --- inverse ---------------------------------------------------------
    def ik(self, pos, rot=DOWN, seed=None, restarts=30, iters=150):
        """Damped least squares from `seed`, then random restarts. Returns q or None."""
        seeds = [] if seed is None else [np.asarray(seed, dtype=float)]
        seeds += [self.rng.uniform(self.lo, self.hi) for _ in range(restarts)]
        jacp, jacr = np.zeros((3, self.m.nv)), np.zeros((3, self.m.nv))
        for q in seeds:
            q = q.copy()
            for _ in range(iters):
                p, r = self.fk(q)
                ep = np.asarray(pos) - p
                er = 0.5 * sum(np.cross(r[:, i], rot[:, i]) for i in range(3))
                if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                    return q
                mujoco.mj_jacBody(self.m, self.d, jacp, jacr, self.eef)
                jac = np.vstack([jacp[:, self.dadr], jacr[:, self.dadr]])
                err = np.concatenate([ep, er])
                dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), err)
                q = np.clip(q + dq, self.lo, self.hi)
        return None

    # --- wrist camera ----------------------------------------------------
    def camera(self, q):
        """Wrist camera position and orientation (columns: right, up, backwards) in the base frame."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)
        c = self.m.camera("eye").id
        return self.d.cam_xpos[c] - self.base, self.d.cam_xmat[c].reshape(3, 3)

    # --- clearance -------------------------------------------------------
    def clearance(self, q, finger_q=0.0, with_floor=False):
        """Smallest distance (m) from the arm to the table, its legs (and the floor)."""
        self.d.qpos[self.qadr] = q
        for j, s in zip(scene.FINGER_JOINTS, (1, -1)):
            self.d.qpos[self.m.joint(j).qposadr[0]] = s * finger_q
        mujoco.mj_kinematics(self.m, self.d)
        targets = dict(self.env_geoms)
        if with_floor:
            targets["floor"] = self.floor
        best, where = np.inf, ""
        for g in self.robot_geoms:
            for name, e in targets.items():
                dist = mujoco.mj_geomDistance(self.m, self.d, g, e, 0.2, None)
                if dist < best:
                    best, where = dist, f"{self.m.body(self.m.geom_bodyid[g]).name}~{name}"
        return best, where

    def gap(self, q, geoms, finger_q=0.0, distmax=0.2):
        """Smallest distance (m) from the arm to the given geoms, and the closest arm body."""
        self.d.qpos[self.qadr] = q
        for j, s in zip(scene.FINGER_JOINTS, (1, -1)):
            self.d.qpos[self.m.joint(j).qposadr[0]] = s * finger_q
        mujoco.mj_kinematics(self.m, self.d)
        best, where = np.inf, ""
        for g in self.robot_geoms:
            for e in geoms:
                dist = mujoco.mj_geomDistance(self.m, self.d, g, e, distmax, None)
                if dist < best:
                    best, where = dist, self.m.body(self.m.geom_bodyid[g]).name
        return best, where


def _robot_bodies(model):
    names = [model.body(i).name for i in range(model.nbody)]
    start = names.index("base_link")
    return [n for n in names[start:] if not n.startswith("cube_")]
