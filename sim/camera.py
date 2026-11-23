"""Virtual wrist camera of `pixi run sim` (started by it; not meant to be run by hand).

Renders the "eye" camera at the scene's resolution and frame rate and publishes it
like a ROS camera driver: /camera/color/image_raw (rgb8) and /camera/color/camera_info.
Best_Contour.py and hsv_finder.py read that topic when given a topic instead of a
device number (scripts/vision.sh maps `-vd 2` to it in the simulation).

It keeps its own copy of the model and asks the simulation for the full state, so
rendering never slows the physics loop down. It waits for the ROS master (bringup
starts it) and exits when the master goes away; the simulation then restarts it.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import mujoco
import numpy as np

from sim import scene
from sim.driver import DriverError, MujocoDriver


def master_online() -> bool:
    import rosgraph

    try:
        return rosgraph.is_master_online()
    except Exception:  # noqa: BLE001
        return False


def camera_info(width: int, height: int, fovy_deg: float):
    from sensor_msgs.msg import CameraInfo

    f = height / 2 / math.tan(math.radians(fovy_deg) / 2)
    info = CameraInfo()
    info.width, info.height = width, height
    info.distortion_model = "plumb_bob"
    info.D = [0.0] * 5
    info.K = [f, 0, width / 2, 0, f, height / 2, 0, 0, 1]
    info.R = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    info.P = [f, 0, width / 2, 0, 0, f, height / 2, 0, 0, 0, 1, 0]
    info.header.frame_id = "eye"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default=scene.DEFAULT_SCENE)
    args = ap.parse_args()

    while not master_online():  # bringup not started yet
        time.sleep(1.0)

    import rospy
    from sensor_msgs.msg import CameraInfo, Image

    cfg = scene.load_config(args.scene)
    cam = cfg["CAMERA"]
    width, height, fps = int(cam["WIDTH"]), int(cam["HEIGHT"]), float(cam.get("FPS", 30))
    model, data, _ = scene.build_physics(cfg)
    root = model.body("base_link").id  # the arm's mount; follows the simulation's (it can be dragged)
    renderer = mujoco.Renderer(model, height, width)
    cam_id = model.camera("eye").id

    sim = MujocoDriver(rate=10)
    try:
        sim.connect(5.0)
    except DriverError as e:
        print(f"[sim camera] {e}", file=sys.stderr)
        return 1

    rospy.init_node("sim_wrist_camera", anonymous=False, disable_signals=True)
    pub = rospy.Publisher("/camera/color/image_raw", Image, queue_size=1)
    info_pub = rospy.Publisher("/camera/color/camera_info", CameraInfo, queue_size=1, latch=True)
    info = camera_info(width, height, cam["FOVY"])
    print(f"[sim camera] 腕部相机 {width}x{height}@{fps:.0f}Hz -> /camera/color/image_raw", flush=True)

    msg = Image()
    msg.height, msg.width, msg.encoding, msg.is_bigendian, msg.step = height, width, "rgb8", 0, width * 3
    msg.header.frame_id = "eye"
    period = 1.0 / fps
    next_t = time.monotonic()
    last_master_check = 0.0
    while sim.ok():
        now = time.monotonic()
        if now - last_master_check > 1.0:
            if not master_online():
                break  # bringup stopped; the simulation restarts us
            last_master_check = now
        try:
            state = sim.request("get_qpos", timeout=1.0)
        except DriverError:
            break
        data.qpos[:] = state["qpos"]
        if "mount" in state:
            model.body_pos[root][:2] = state["mount"]
        mujoco.mj_kinematics(model, data)
        mujoco.mj_camlight(model, data)
        renderer.update_scene(data, camera=cam_id)
        img = renderer.render()
        stamp = rospy.Time.now()
        msg.header.stamp = stamp
        msg.data = img.tobytes()
        pub.publish(msg)
        info.header.stamp = stamp
        info_pub.publish(info)
        next_t += period
        time.sleep(max(0.0, next_t - time.monotonic()))
        if time.monotonic() - next_t > 1.0:
            next_t = time.monotonic()
    sim.disconnect()
    rospy.signal_shutdown("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
