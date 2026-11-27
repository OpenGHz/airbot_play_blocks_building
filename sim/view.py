"""Open the MuJoCo viewer on the scene (physics running, arm holding a pose).

    pixi run sim-view                 # arm at the pick scan pose
    pixi run sim-view --scene table   # the scene with the pick area on the table top
    pixi run sim-view --pose floor    # placing the first block on the floor
    pixi run sim-view --pose top      # placing the highest block at this table height

Double-click a body to select it, Ctrl + right-drag to push it around.
"""

from __future__ import annotations

import argparse
import time

import mujoco
import mujoco.viewer

from sim import poses as named
from sim import scene, snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default="pick", choices=["home", "pick", "floor", "top"])
    ap.add_argument("--scene", default=scene.DEFAULT_SCENE, help="floor (default), table, or a scene file")
    ap.add_argument("--control", default="configs/control/real.yaml")
    args = ap.parse_args()

    cfg = scene.without_mount_offset(scene.load_config(args.scene))
    res = named.compute(cfg, named.load_control(args.control))
    tower = res["layers"] - 1 if args.pose == "top" else 0
    model, data = snapshot.pose_scene(cfg, res, args.pose, tower)
    print(
        f"table {res['table_height']:.3f} m, tower spot {cfg['tower_xy']}, "
        f"{res['layers']} blocks reachable without lifting; pose: {args.pose}"
    )

    with mujoco.viewer.launch_passive(model, data) as viewer:
        cam = snapshot.views(cfg, res["table_height"])["overview"]
        viewer.cam.lookat[:] = cam.lookat
        viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 2.2, cam.azimuth, -20
        while viewer.is_running():
            t0 = time.time()
            for _ in range(int(round(1 / 60 / model.opt.timestep))):
                mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(max(0.0, 1 / 60 - (time.time() - t0)))


if __name__ == "__main__":
    main()
