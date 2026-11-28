"""Check the vision on rendered wrist-camera images (`pixi run sim-vision-check`).

Runs the detection of Best_Contour.py (std_vision colour filter + Contour.NFC_F with the same
thresholds, crops and arguments) on images rendered in the simulation, and uses MuJoCo's
segmentation image to tell which object the detection locked onto:

  pick   random pick-area layouts placed the way `pixi run sim` places them, camera at the
         pick scan pose. Every cube's top face has to be a blob of its own (no two cubes
         merged) within area_limit_pick, and the detection of each colour has to lock onto
         a cube of that colour.
  place  towers of 1 .. --levels blocks, the next block held gap_place above the top one
         (the adjust pose), the top block shifted by up to 1 cm, and a full pick area of the
         top block's colour next to the tower. The detection has to lock onto the top block.

Run it after changing the scene, the camera, the lighting or configs/vision/sim.yaml.
Exits with status 1 when a check fails.
"""

from __future__ import annotations

import argparse
import sys

import cv2
import mujoco
import numpy as np
from std_vision.contour import Contour
from std_vision.video import StdVideo

from sim import config_file, scene
from sim.calib_vision import PLACE_CROP_ROW, Cam, place_hsv
from sim.world import World


class Checker:
    def __init__(self, cfg: dict, control: dict, vision: dict):
        self.cfg, self.control, self.vision = cfg, control, vision
        self.cam = Cam(cfg)
        self.m, self.d = self.cam.m, self.cam.d
        c = cfg["camera"]
        self.seg = mujoco.Renderer(self.m, int(c["height"]), int(c["width"]))
        self.seg.enable_segmentation_rendering()
        self.names = list(cfg["cube_colors"])
        self.hsv = {
            self.names[0]: (vision["hsv1_l"], vision["hsv1_h"]),
            self.names[1]: (vision["hsv2_l"], vision["hsv2_h"]),
        }
        ph = place_hsv(vision)
        self.hsv_place = {self.names[0]: ph[1], self.names[1]: ph[2]}
        width = int(c["width"])
        ex = width / 2 if vision["expect_x"] == "AUTO" else float(vision["expect_x"])
        self.ref = (ex, float(vision["expect_pick_ref"]))  # Best_Contour.py uses it in both stages
        self.crop_x = vision.get("place_crop_x")
        self.s = self.cam.s
        self.z_floor = control["pick_grasp_z"] - 0.0025 - self.s  # cube bottom; grasp point ~2.5 mm above the centre

    def render(self):
        img = self.cam.render()
        self.seg.update_scene(self.d, camera="eye")
        return img, self.seg.render()[:, :, 0]

    def binary(self, img, color: str, place: bool):
        _, b = StdVideo.color_filter(img, *(self.hsv_place if place else self.hsv)[color])
        if place:  # Best_Contour.py, real-camera path (also taken in the simulation)
            b[PLACE_CROP_ROW:] = 0
            if self.crop_x is not None:
                b[:, : max(0, int(self.crop_x[0]))] = 0
                b[:, int(self.crop_x[1]) :] = 0
        return b

    def locked(self, b, ids, place: bool):
        """Geom ids under the contour Contour.NFC_F selects (None when nothing is selected)."""
        area = self.vision["area_limit_place" if place else "area_limit_pick"]
        *_, cnt = Contour.NFC_F(b, area, 5.5 if place else 3.0, self.ref)
        if cnt is None:
            return None
        mask = np.zeros_like(b)
        cv2.drawContours(mask, [cnt], 0, 255, -1)
        return set(np.unique(ids[(mask > 0) & (b > 0)]).tolist())

    def geom(self, i: int) -> int:
        return self.m.geom(f"cube_{i}").id

    # ---- checks ----
    def pick(self, layouts: int, log) -> int:
        world = World(self.cfg)
        open_q = scene.finger_q_for_gap(0.07)
        scan = [self.control[k] for k in ("pick_position_x", "pick_position_y", "pick_position_z")]
        problems = 0
        for seed in range(layouts):
            world.pick.rng = np.random.default_rng(seed)
            spots = world.pick._sample(world.pick.count, []) or world.pick._sample(2, [])
            colors = (self.names * world.pick.count)[: len(spots)]
            mujoco.mj_resetData(self.m, self.d)
            used = {c: 0 for c in self.names}
            placed = []
            frame = self.cam.frame
            for (x, y, yaw), c in zip(spots, colors):  # spots are world x, y, yaw; Cam.put takes the base frame
                i = self.cam.cube(c, used[c])
                used[c] += 1
                self.cam.put(i, (*frame.to_base((x, y)), self.z_floor + self.s), yaw - frame.yaw)
                placed.append((i, c))
            self.cam.pose(scan, open_q)
            img, ids = self.render()
            for c in self.names:
                b = self.binary(img, c, place=False)
                n, lab = cv2.connectedComponents(b)
                seen = set()
                for k in range(1, n):
                    blob = lab == k
                    size = int(blob.sum())
                    under = set(np.unique(ids[blob]).tolist())
                    owners = [i for i, _ in placed if self.geom(i) in under]
                    if len(owners) > 1:
                        problems += 1
                        log(f"pick layout {seed}, {c}: one blob ({size} px) covers cubes {owners}")
                    lo, hi = self.vision["area_limit_pick"]
                    if len(owners) == 1 and lo < size < hi:
                        seen.add(owners[0])
                for i, cc in placed:
                    if cc == c and i not in seen:
                        problems += 1
                        log(f"pick layout {seed}: the {c} cube_{i} is not seen as one top face within area_limit_pick")
                got = self.locked(b, ids, place=False)
                want = {self.geom(i) for i, cc in placed if cc == c}
                if want and (got is None or len(got & want) != 1):  # nothing, or not exactly one cube of this colour
                    problems += 1
                    log(f"pick layout {seed}, {c}: the detection locks onto {self.describe(got)}, not one {c} cube")
        log(f"pick: {layouts} layouts checked")
        return problems

    def place(self, levels: int, log) -> int:
        tx, ty = self.control["place_position_x"], self.control["place_position_y"]
        h, gap = self.control["cube_height"], self.control.get("gap_place", 0.003)
        hold_q = scene.finger_q_for_gap(self.cfg["cube_size"]) + 0.0005
        world = World(self.cfg)
        area = world.pick
        corners = [
            tuple(area.center + np.array([sx, sy]) * (area.half - self.s))
            for sx, sy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0))
        ]
        problems = checks = 0
        for k in range(1, levels + 1):  # placing block k+1 onto a tower of k
            top = self.names[(k - 1) % 2]
            for dx, dy in ((0, 0), (0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)):
                mujoco.mj_resetData(self.m, self.d)
                used = {c: 0 for c in self.names}
                top_id = None
                for j in range(k):
                    c = self.names[j % 2]
                    off = (dx, dy) if j == k - 1 else (0.0, 0.0)
                    i = self.cam.cube(c, used[c])
                    used[c] += 1
                    self.cam.put(i, (tx + off[0], ty + off[1], self.z_floor + self.s + j * h))
                    top_id = self.geom(i)
                for x, y in corners:  # world x, y
                    self.cam.put(
                        self.cam.cube(top, used[top]),
                        (*self.cam.frame.to_base((x, y)), self.z_floor + self.s),
                    )
                    used[top] += 1
                held = self.names[k % 2]
                z_grasp = self.control["pick_grasp_z"] + k * h + gap
                self.cam.put(self.cam.cube(held, used[held]), (tx, ty, z_grasp - 0.0025))
                try:
                    self.cam.pose([tx, ty, z_grasp], hold_q)
                except RuntimeError:
                    log(f"place: a tower of {k} is out of reach, stopping there")
                    return problems
                img, ids = self.render()
                got = self.locked(self.binary(img, top, place=True), ids, place=True)
                checks += 1
                if got is None or top_id not in got:
                    problems += 1
                    log(
                        f"place on {k} blocks, top shifted ({dx:+.2f}, {dy:+.2f}): the detection locks onto "
                        f"{self.describe(got)}, not the tower's top block"
                    )
        log(f"place: {checks} images checked (towers of 1 .. {levels}, 5 offsets each)")
        return problems

    def describe(self, ids) -> str:
        if ids is None:
            return "nothing"
        names = [self.m.geom(g).name or self.m.body(self.m.geom_bodyid[g]).name for g in sorted(ids) if g >= 0]
        return ", ".join(names) or "the background"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default=scene.DEFAULT_SCENE)
    ap.add_argument("--control", default="configs/control/sim.yaml")
    ap.add_argument("--vision", default="configs/vision/sim.yaml")
    ap.add_argument("--layouts", type=int, default=20, help="random pick-area layouts to check")
    ap.add_argument("--levels", type=int, default=25, help="highest tower to check in the place stage")
    args = ap.parse_args()
    checker = Checker(scene.load_config(args.scene), config_file.load(args.control), config_file.load(args.vision))
    problems = checker.pick(args.layouts, print) + checker.place(args.levels, print)
    print("OK" if problems == 0 else f"{problems} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
