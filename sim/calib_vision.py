"""Expected image positions for configs/vision/sim.yaml (`pixi run sim-calib-vision`).

What the manual's `vision -t pi0` / `-t pl0` steps find by hand, computed in the
simulation with the detection code of Best_Contour.py (std_vision colour filter +
Contour.NFC_F):

  EXPECT_X, EXPECT_PICK_REF  centre of a cube right under the gripper at the pick scan pose
  EXPECT_PLACE_REF           upper edge of the tower's top cube with the next cube held
                             GAP_PLACE above it (the adjust pose)

--apply writes them into the vision config.
"""

from __future__ import annotations

import argparse
import math

import cv2
import mujoco
import numpy as np
from std_vision.contour import Contour
from std_vision.video import StdVideo

from sim import config_file, kin, scene

PLACE_CROP_ROW = 423  # Best_Contour.py keeps rows above this in the place stage


class Cam:
    def __init__(self, cfg):
        self.cfg = cfg
        self.m, self.d, _ = scene.build_physics(cfg)
        self.arm = kin.Arm(self.m, mujoco.MjData(self.m), scene.base_offset(cfg))
        c = cfg["CAMERA"]
        self.r = mujoco.Renderer(self.m, int(c["HEIGHT"]), int(c["WIDTH"]))
        self.cubes = [self.m.joint(self.m.body(f"cube_{i}").jntadr[0]).qposadr[0] for i in range(int(cfg.get("CUBE_POOL", 40)))]
        self.colors = scene.pool_colors(cfg)
        self.base = scene.base_offset(cfg)
        self.s = cfg["CUBE_SIZE"] / 2

    def cube(self, color, k):
        """Index of the k-th pool cube of a colour."""
        return [i for i, c in enumerate(self.colors) if c == color][k]

    def put(self, i, xyz_base, yaw=0.0):
        a = self.cubes[i]
        self.d.qpos[a : a + 3] = np.asarray(xyz_base) + self.base
        self.d.qpos[a + 3 : a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]

    def pose(self, xyz_base, finger_q, yaw=0.0):
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ kin.DOWN
        q = self.arm.ik(xyz_base, rot=rot, restarts=60)
        if q is None:
            raise RuntimeError(f"IK failed at {xyz_base}")
        scene.set_arm(self.m, self.d, q, finger_q)

    def render(self):
        mujoco.mj_forward(self.m, self.d)
        self.r.update_scene(self.d, camera="eye")
        return cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)


def place_hsv(vision: dict) -> dict:
    """Place-stage thresholds as Best_Contour.py uses them (HSV*_P when USE_PLACE_HSV is "True")."""
    p = "_P" if str(vision.get("USE_PLACE_HSV", "False")).lower() == "true" else ""
    return {1: (vision[f"HSV1_L{p}"], vision[f"HSV1_H{p}"]), 2: (vision[f"HSV2_L{p}"], vision[f"HSV2_H{p}"])}


def detect(bgr, hsv_l, hsv_h, area, ratio, ref, crop=False, crop_x=None):
    _, binary = StdVideo.color_filter(bgr, hsv_l, hsv_h)
    if crop:
        b = np.zeros_like(binary)
        b[:PLACE_CROP_ROW] = binary[:PLACE_CROP_ROW]
        binary = b
        if crop_x is not None:
            binary[:, : max(0, int(crop_x[0]))] = 0
            binary[:, int(crop_x[1]) :] = 0
    center, bias, rot, box, wh, cnt = Contour.NFC_F(binary, area, ratio, ref)
    if cnt is None:
        return None
    return {"center": center, "wh": wh, "upper_edge": center[1] - wh[0] / 2, "rot": rot}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default=scene.DEFAULT_SCENE)
    ap.add_argument("--control", default="configs/control/sim.yaml")
    ap.add_argument("--vision", default="configs/vision/sim.yaml")
    ap.add_argument("--apply", action="store_true", help="write the values into --vision")
    ap.add_argument("--save", help="save the rendered frames with this path prefix")
    args = ap.parse_args()
    cfg = scene.load_config(args.scene)
    control = config_file.load(args.control)
    vision = config_file.load(args.vision)
    cam = Cam(cfg)
    names = list(cfg["CUBE_COLORS"])
    hsv = {names[0]: (vision["HSV1_L"], vision["HSV1_H"]), names[1]: (vision["HSV2_L"], vision["HSV2_H"])}
    ph = place_hsv(vision)
    hsv_place = {names[0]: ph[1], names[1]: ph[2]}
    crop_x = vision.get("PLACE_CROP_X")
    width = int(cfg["CAMERA"]["WIDTH"])
    ref0 = (width / 2 if vision["EXPECT_X"] == "AUTO" else vision["EXPECT_X"], vision["EXPECT_PICK_REF"])
    open_q = scene.finger_q_for_gap(0.07)
    hold_q = scene.finger_q_for_gap(cfg["CUBE_SIZE"]) + 0.0005

    # Pick: one cube right under the gripper at the scan pose, for both colours.
    px, py = control["PICK_POSITION_X"], control["PICK_POSITION_Y"]
    z_floor = control["PICK_GRASP_Z"] - 0.0025 - cam.s  # the grasp point is ~2.5 mm above the cube centre
    picks = []
    for color in names:
        mujoco.mj_resetData(cam.m, cam.d)
        cam.put(cam.cube(color, 0), (px, py, z_floor + cam.s))
        cam.pose([px, py, control["PICK_POSITION_Z"]], open_q)
        img = cam.render()
        if args.save:
            cv2.imwrite(f"{args.save}_pick_{color}.png", img)
        res = detect(img, *hsv[color], vision["AREA_LIMIT_PICK"], 3.0, ref0)
        if res is None:
            raise SystemExit(f"pick: the {color} cube under the gripper was not detected")
        picks.append(res["center"])
        print(f"pick {color:6s}: centre ({res['center'][0]:.1f}, {res['center'][1]:.1f}), w_h ({res['wh'][0]:.1f}, {res['wh'][1]:.1f})")
    expect_x = float(np.mean([c[0] for c in picks]))
    pick_ref = float(np.mean([c[1] for c in picks]))

    # Place: tower of k cubes at the place spot; the next cube held GAP_PLACE above it.
    tx, ty = control["PLACE_POSITION_X"], control["PLACE_POSITION_Y"]
    h, gap = control["CUBE_HEIGHT"], control.get("GAP_PLACE", 0.003)
    edges = []
    areas = []
    for k in (1, 2, 5, 10, 15, 20, 24):
        mujoco.mj_resetData(cam.m, cam.d)
        for j in range(k):
            cam.put(cam.cube(names[j % 2], j // 2), (tx, ty, z_floor + cam.s + j * h))
        held_color = names[k % 2]
        z_grasp = control["PICK_GRASP_Z"] + k * h + gap  # adjust pose
        cam.put(cam.cube(held_color, k // 2), (tx, ty, z_grasp - 0.0025))
        try:
            cam.pose([tx, ty, z_grasp], hold_q)
        except RuntimeError:
            continue
        img = cam.render()
        if args.save:
            cv2.imwrite(f"{args.save}_place_{k}.png", img)
        top_color = names[(k - 1) % 2]
        res = detect(img, *hsv_place[top_color], vision["AREA_LIMIT_PLACE"], 5.5, (expect_x, pick_ref), crop=True, crop_x=crop_x)
        if res is None:
            print(f"place on {k:2d} blocks: top cube not detected")
            continue
        edges.append(res["upper_edge"])
        areas.append(res["wh"][0] * res["wh"][1])
        print(f"place on {k:2d} blocks: top cube centre ({res['center'][0]:.1f}, {res['center'][1]:.1f}), upper edge {res['upper_edge']:.1f}")
    place_ref = float(np.median(edges))
    print(f"place-stage tower-top box area {min(areas):.0f} ~ {max(areas):.0f} (AREA_LIMIT_PLACE {vision['AREA_LIMIT_PLACE']})")
    print(f"\nEXPECT_X {expect_x:.1f}  EXPECT_PICK_REF {pick_ref:.1f}  EXPECT_PLACE_REF {place_ref:.1f}")
    if args.apply:
        config_file.set_keys(args.vision, {"EXPECT_X": round(expect_x, 1), "EXPECT_PICK_REF": round(pick_ref, 1),
                                           "EXPECT_PLACE_REF": round(place_ref, 1)})
        print(f"written to {args.vision}")


if __name__ == "__main__":
    main()
