"""Expected image positions for configs/vision/sim.yaml (`pixi run sim-calib-vision`).

What the manual's `vision -t pi0` / `-t pl0` steps find by hand, computed in the
simulation with the detection code of Best_Contour.py (std_vision colour filter +
Contour.NFC_F):

  EXPECT_X, EXPECT_PICK_REF  centre of a cube right under the gripper at the pick scan pose
  EXPECT_PLACE_REF           upper edge in the image of the tower's top cube (the topmost
                             edge in the image; physically the cube's lower edge) with the next cube held
                             GAP_PLACE above it (the adjust pose)
  EXPECT_HELD_REF            upper edge in the image of the held cube, centred in the gripper,
                             at the adjust pose (`pixi run vision --held-cube`)
  HELD_SCALE                 pixels the tower's top cube moves in image y per pixel the held cube
                             moves, for the same shift along the finger gap (base x here)

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

from Best_Contour import find_held_cube
from sim import config_file, kin, scene

PLACE_CROP_ROW = 423  # Best_Contour.py keeps rows above this in the place stage


class Cam:
    def __init__(self, cfg):
        self.cfg = cfg
        self.m, self.d, _ = scene.build_physics(cfg)
        self.frame = scene.base_frame(cfg)
        self.arm = kin.Arm(self.m, mujoco.MjData(self.m), self.frame)
        c = cfg["CAMERA"]
        self.r = mujoco.Renderer(self.m, int(c["HEIGHT"]), int(c["WIDTH"]))
        self.cubes = [
            self.m.joint(self.m.body(f"cube_{i}").jntadr[0]).qposadr[0] for i in range(int(cfg.get("CUBE_POOL", 40)))
        ]
        self.colors = scene.pool_colors(cfg)
        self.s = cfg["CUBE_SIZE"] / 2

    def cube(self, color, k):
        """Index of the k-th pool cube of a colour."""
        return [i for i, c in enumerate(self.colors) if c == color][k]

    def put(self, i, xyz_base, yaw=0.0):
        """Cube i at xyz_base (base frame), turned yaw (rad) from the base frame's x axis."""
        a = self.cubes[i]
        self.d.qpos[a : a + 3] = self.frame.to_world(xyz_base)
        self.d.qpos[a + 3 : a + 7] = scene.yaw_quat(yaw + self.frame.yaw)

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
        print(
            f"pick {color:6s}: centre ({res['center'][0]:.1f}, {res['center'][1]:.1f}), w_h ({res['wh'][0]:.1f}, {res['wh'][1]:.1f})"
        )
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
        res = detect(
            img, *hsv_place[top_color], vision["AREA_LIMIT_PLACE"], 5.5, (expect_x, pick_ref), crop=True, crop_x=crop_x
        )
        if res is None:
            print(f"place on {k:2d} blocks: top cube not detected")
            continue
        edges.append(res["upper_edge"])
        areas.append(res["wh"][0] * res["wh"][1])
        print(
            f"place on {k:2d} blocks: top cube centre ({res['center'][0]:.1f}, {res['center'][1]:.1f}), upper edge {res['upper_edge']:.1f}"
        )
    place_ref = float(np.median(edges))
    print(
        f"place-stage tower-top box area {min(areas):.0f} ~ {max(areas):.0f} (AREA_LIMIT_PLACE {vision['AREA_LIMIT_PLACE']})"
    )

    # Held cube (--held-cube): where it is with no grasp offset, and how far the top cube has to move
    # in the image per pixel the held cube moves. Image x follows base y (finger opening), image y
    # follows base x (along the finger gap) at this orientation.
    def place_view(k, top_d=(0.0, 0.0), held_d=(0.0, 0.0)):
        mujoco.mj_resetData(cam.m, cam.d)
        for j in range(k):
            d = top_d if j == k - 1 else (0.0, 0.0)
            cam.put(cam.cube(names[j % 2], j // 2), (tx + d[0], ty + d[1], z_floor + cam.s + j * h))
        z_grasp = control["PICK_GRASP_Z"] + k * h + gap
        cam.put(cam.cube(names[k % 2], k // 2), (tx + held_d[0], ty + held_d[1], z_grasp - 0.0025))
        cam.pose([tx, ty, z_grasp], hold_q)
        img = cam.render()
        top = detect(
            img,
            *hsv_place[names[(k - 1) % 2]],
            vision["AREA_LIMIT_PLACE"],
            5.5,
            (expect_x, pick_ref),
            crop=True,
            crop_x=crop_x,
        )
        _, held_bin = StdVideo.color_filter(img, *hsv_place[names[k % 2]])
        if crop_x is not None:
            held_bin[:, : max(0, int(crop_x[0]))] = 0
            held_bin[:, int(crop_x[1]) :] = 0
        held = find_held_cube(held_bin, vision["AREA_LIMIT_PLACE"][0])
        if top is None or held is None:
            raise SystemExit(f"held cube: tower of {k}, {'top' if top is None else 'held'} cube not detected")
        return (top["center"][0], top["upper_edge"]), (held["x"], held["edge"])

    k = 3
    _, held0 = place_view(k)
    shifts = np.arange(-4, 5) * 0.001  # a slope over several shifts: one pixel is ~0.5 mm
    top, held = [], []
    for v in shifts:  # along the finger gap (base x at this orientation) -> image y
        top.append(place_view(k, top_d=(v, 0.0))[0][1])
        held.append(place_view(k, held_d=(v, 0.0))[1][1])
    top_px_mm = np.polyfit(shifts * 1000, top, 1)[0]
    held_px_mm = np.polyfit(shifts * 1000, held, 1)[0]
    scale = round(float(top_px_mm / held_px_mm), 2)
    print(f"along the finger gap: top cube {top_px_mm:+.2f} px/mm, held cube {held_px_mm:+.2f} px/mm in image y")
    print(f"held cube centred in the gripper: upper edge {held0[1]:.1f}")

    print(f"\nEXPECT_X {expect_x:.1f}  EXPECT_PICK_REF {pick_ref:.1f}  EXPECT_PLACE_REF {place_ref:.1f}")
    print(f"EXPECT_HELD_REF {held0[1]:.1f}  HELD_SCALE {scale}")
    if args.apply:
        config_file.set_keys(
            args.vision,
            {
                "EXPECT_X": round(expect_x, 1),
                "EXPECT_PICK_REF": round(pick_ref, 1),
                "EXPECT_PLACE_REF": round(place_ref, 1),
                "EXPECT_HELD_REF": round(held0[1], 1),
                "HELD_SCALE": scale,
            },
            after="EXPECT_X",
        )
        print(f"written to {args.vision}")


if __name__ == "__main__":
    main()
