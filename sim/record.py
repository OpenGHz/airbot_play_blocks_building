"""Record the running simulation from outside (`pixi run sim-record`), e.g. for a demo video.

Renders a fixed view of the arm, the pick area and the tower, with the wrist camera in
the corner, the number of blocks on the tower and the elapsed time. Like the virtual
wrist camera it keeps its own copy of the model and asks the simulation for the state,
so it does not slow the physics down. Writes runs/sim/record_<time>.mp4 in real time
until Ctrl+C; speed it up afterwards, e.g.

  ffmpeg -i runs/sim/record_<time>.mp4 -vf "setpts=PTS/8" -an -c:v libx264 -crf 26 demo.mp4

--snapshot PATH writes one frame and exits (for choosing the view).
"""

from __future__ import annotations

import argparse
import pathlib
import signal
import sys
import time

import cv2
import mujoco
import numpy as np

from sim import config_file, scene
from sim.driver import DriverError, MujocoDriver

# Default view: from in front of the table, the whole arm, the pick area and the tower spot.
VIEW = {"lookat": [0.22, -0.08, 0.35], "distance": 1.1, "azimuth": 188.0, "elevation": -13.0}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default=scene.DEFAULT_SCENE)
    ap.add_argument("--control", default="configs/control/sim.yaml", help="for the tower position")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=float, default=15.0)
    ap.add_argument("--lookat", type=float, nargs=3, default=VIEW["lookat"], help="world frame (m)")
    ap.add_argument("--distance", type=float, default=VIEW["distance"])
    ap.add_argument("--azimuth", type=float, default=VIEW["azimuth"])
    ap.add_argument("--elevation", type=float, default=VIEW["elevation"])
    ap.add_argument("--no-inset", action="store_true", help="leave out the wrist camera")
    ap.add_argument("--snapshot", help="write one frame to this PNG and exit")
    ap.add_argument("--out", help="video path (default runs/sim/record_<time>.mp4)")
    args = ap.parse_args()

    cfg = scene.load_config(args.scene)
    control = config_file.load(args.control)
    width, height = (int(v) for v in args.size.split("x"))
    model, data, _ = scene.build_physics(cfg)
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
    model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    view = mujoco.Renderer(model, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = args.lookat, args.distance, args.azimuth, args.elevation
    c = cfg["camera"]
    eye = None if args.no_inset else mujoco.Renderer(model, int(c["height"]), int(c["width"]))

    place_xy = np.array([control["place_position_x"], control["place_position_y"]])  # base frame
    tower_xy = scene.base_frame(cfg).to_world(place_xy)  # world; follows the arm's mount (frame below)
    cubes = [model.body(f"cube_{i}").id for i in range(int(cfg.get("cube_pool", 40)))]
    fingers = [model.body(n).id for n in scene.FINGER_BODIES]
    finger_q = model.joint("endleft").qposadr[0]
    closed_q = scene.finger_q_for_gap(cfg["cube_size"] + 0.004)  # fingers closer than this: holding a cube

    def tower_count() -> int:
        mid = data.xpos[fingers].mean(0)
        holding = data.qpos[finger_q] > closed_q
        n = 0
        for b in cubes:
            p = data.xpos[b]
            if np.linalg.norm(p[:2] - tower_xy) > 0.025:
                continue
            if holding and np.linalg.norm(p - mid) < 0.02:
                continue  # the cube still in the gripper
            n += 1
        return n

    sim = MujocoDriver(rate=10)
    try:
        sim.connect(5.0)
    except DriverError as e:
        print(f"[sim record] {e}", file=sys.stderr)
        return 1

    def frame(t: float) -> np.ndarray:
        nonlocal tower_xy
        state = sim.request("get_qpos", timeout=1.0)
        data.qpos[:] = state["qpos"]
        if "mount" in state:  # the arm where the simulation has it (it can be moved and turned)
            yaw = state.get("mount_yaw", 0.0)
            scene.place_base(model, state["mount"], yaw)
            pos = [*state["mount"], scene.base_offset(cfg)[2]]
            tower_xy = scene.BaseFrame(pos, yaw).to_world(place_xy)
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        view.update_scene(data, camera=cam)
        img = cv2.cvtColor(view.render(), cv2.COLOR_RGB2BGR)
        if eye is not None:
            eye.update_scene(data, camera="eye")
            w = width // 4
            small = cv2.resize(
                cv2.cvtColor(eye.render(), cv2.COLOR_RGB2BGR),
                (w, w * eye.height // eye.width),
                interpolation=cv2.INTER_AREA,
            )
            m = 12
            img[m : m + small.shape[0], width - m - w : width - m] = small
            cv2.rectangle(img, (width - m - w - 1, m - 1), (width - m, m + small.shape[0]), (255, 255, 255), 2)
            cv2.putText(
                img,
                "wrist camera",
                (width - m - w + 6, m + small.shape[0] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        n = tower_count()
        mm, ss = divmod(int(t), 60)
        for text, y in ((f"blocks on tower: {n}", 36), (f"t = {mm:02d}:{ss:02d}", 68)):
            cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
        return img

    if args.snapshot:
        cv2.imwrite(args.snapshot, frame(0.0))
        print(f"[sim record] {args.snapshot}")
        sim.disconnect()
        return 0

    out = pathlib.Path(args.out or time.strftime("runs/sim/record_%Y%m%d_%H%M%S.mp4"))
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    t0 = time.time()
    print(
        f"[sim record] 写入 {out}（{width}x{height}@{args.fps:.0f}，开始于 {time.strftime('%H:%M:%S', time.localtime(t0))}）；Ctrl+C 结束",
        flush=True,
    )
    period = 1.0 / args.fps
    k = 0
    try:
        while not stop and sim.ok():
            # one frame per period of wall time; if rendering falls behind, repeat the frame so the clock stays right
            img = frame(time.time() - t0)
            while k * period <= time.time() - t0:
                writer.write(img)
                k += 1
            time.sleep(max(0.0, k * period - (time.time() - t0)))
    except DriverError:
        pass
    writer.release()
    sim.disconnect()
    print(f"[sim record] 结束：{k} 帧，{k * period:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
