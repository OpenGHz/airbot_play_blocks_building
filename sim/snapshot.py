"""Render the scene to PNG files (overview, side views, wrist camera).

pixi run sim-snapshot                  # pick area on the floor: runs/sim/scene_floor.png
pixi run sim-snapshot --scene table    # pick area on the table: runs/sim/scene_table.png
"""

from __future__ import annotations

import argparse

import cv2
import mujoco
import numpy as np

from sim import poses as named
from sim import scene

OUT = scene.REPO / "runs/sim"


def build_with_props(cfg, h_table, tower_blocks: int, held: bool):
    """Scene plus a static tower of `tower_blocks` cubes and an optional cube in the gripper."""
    spec = scene.build_spec(cfg, h_table)
    s = cfg["CUBE_SIZE"] / 2
    colors = list(cfg["CUBE_COLORS"].values())
    tx, ty = cfg["TOWER_XY"]
    for i in range(tower_blocks):
        spec.worldbody.add_geom(
            name=f"tower_{i}",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[s, s, s],
            pos=[tx, ty, s + 2 * s * i],
            rgba=colors[i % 2],
            contype=0,
            conaffinity=0,
        )
    if held:
        b = spec.worldbody.add_body(name="held_cube", mocap=True)
        b.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX, size=[s, s, s], rgba=colors[tower_blocks % 2], contype=0, conaffinity=0
        )
    model = spec.compile()
    return model, mujoco.MjData(model)


def render(model, data, cam, w, h):
    r = mujoco.Renderer(model, h, w)
    r.update_scene(data, camera=cam)
    img = r.render()
    r.close()
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def free_cam(lookat, distance, azimuth, elevation):
    c = mujoco.MjvCamera()
    c.type = mujoco.mjtCamera.mjCAMERA_FREE
    c.lookat[:], c.distance, c.azimuth, c.elevation = lookat, distance, azimuth, elevation
    return c


def views(cfg, H):
    """Free cameras placed relative to the mounting edge and the tower, so both MOUNTs frame well.

    phi: direction out of the table across the mounting edge. Looking along phi + 90 is
    looking along the edge; along phi + 180 is looking from outside back at the table."""
    phi = scene.edge_angle(cfg)
    tx, ty = cfg["TOWER_XY"]
    px, py = cfg["PICK_XY"]
    floor = scene.pick_on_floor(cfg)
    mid = [(tx + px) / 2, (ty + py) / 2]
    return {
        "overview": free_cam([*mid, H * (0.35 if floor else 0.6)], 1.7 if floor else 2.0, phi + 45, -22),
        "along": free_cam([tx * 0.5, ty * 0.5, H * 0.5], 1.6, phi + 90, -8),
        "outside": free_cam([tx * 0.35, ty * 0.35, H * 0.45], 1.5, phi + 180, -8),
        "top": free_cam([tx * 0.5, ty * 0.5, H * 0.75], 2.0, phi + 70, -12),
    }


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (255, 255, 255), -1)
    cv2.putText(img, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    return img


def pose_scene(cfg, res, name, tower_blocks):
    q, finger, held = res["poses"][name]
    model, data = build_with_props(cfg, res["table_height"], tower_blocks, held is not None)
    scene.set_arm(model, data, q, finger)
    if held is not None:
        data.mocap_pos[0] = scene.base_frame(cfg, res["table_height"]).to_world(held)
        mujoco.mj_forward(model, data)
    return model, data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default=scene.DEFAULT_SCENE, help="floor (default), table, or a scene file")
    ap.add_argument("--control", default="configs/control/real.yaml")
    args = ap.parse_args()
    cfg = scene.without_mount_offset(scene.load_config(args.scene))
    control = named.load_control(args.control)
    res = named.compute(cfg, control)
    H, n = res["table_height"], res["layers"]
    tx, ty = cfg["TOWER_XY"]
    OUT.mkdir(parents=True, exist_ok=True)
    W, Hh = 800, 600
    cam = cfg["CAMERA"]

    v = views(cfg, H)
    floor = scene.pick_on_floor(cfg)

    # 1. Overview: arm at the pick scan pose above the cubes.
    m, d = pose_scene(cfg, res, "pick", 0)
    a = label(
        render(m, d, v["overview"], W, Hh),
        f"{cfg['NAME']} scene, {cfg.get('MOUNT', 'side')} mount: table {H:.3f} m, "
        f"pick area on the {'floor' if floor else 'table'}",
    )
    b = label(render(m, d, "eye", cam["WIDTH"], cam["HEIGHT"]), "wrist camera at pick scan pose (640x480)")
    b = cv2.resize(b, (W, Hh))

    # 2. First block on the floor.
    m, d = pose_scene(cfg, res, "floor", 0)
    c = label(render(m, d, v["along"], W, Hh), "first block on the floor (looking along the edge)")
    e = label(render(m, d, v["outside"], W, Hh), "first block on the floor (from outside the table)")

    # 3. Highest block at this table height.
    m, d = pose_scene(cfg, res, "top", n - 1)
    f = label(render(m, d, v["top"], W, Hh), f"block {n} on top: {n} blocks without lifting the table")
    g = label(render(m, d, "eye", cam["WIDTH"], cam["HEIGHT"]), f"wrist camera placing block {n}")
    g = cv2.resize(g, (W, Hh))

    grid = np.vstack([np.hstack([a, b]), np.hstack([c, e]), np.hstack([f, g])])
    path = OUT / f"scene_{cfg['NAME']}.png"
    cv2.imwrite(str(path), grid)
    print(
        f"table height {H:.3f} m, tower at ({tx}, {ty}), first block released at z = {res['z_first']:+.3f}, "
        f"{n} blocks without lifting"
    )
    print(f"written {path.relative_to(scene.REPO)}")


if __name__ == "__main__":
    main()
