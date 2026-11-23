"""Debug overlay of the wrist camera (`pixi run sim-debug-view`): what the vision picks.

Re-runs the detection of Best_Contour.py on every frame (same colour per
/vision_attention, same place-stage crop, Contour.NFC_F with the same arguments) and
draws every candidate contour, the chosen one, the reference point / upper-edge line
and the latest /target_TF. Shows it in a window and writes runs/sim/debug_<time>.mp4.
"""

from __future__ import annotations

import argparse
import pathlib
import threading
import time

import cv2
import numpy as np
import rospy
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from std_vision.contour import Contour
from std_vision.video import StdVideo

from sim import config_file

PLACE_CROP_ROW = 423


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vision", default="configs/vision/sim.yaml")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--fps", type=float, default=10.0, help="frames per second written to the video")
    args = ap.parse_args()
    v = config_file.load(args.vision)
    hsv = {"HSV1": (v["HSV1_L"], v["HSV1_H"]), "HSV2": (v["HSV2_L"], v["HSV2_H"])}
    pfx = "_P" if str(v.get("USE_PLACE_HSV", "False")).lower() == "true" else ""
    hsv_place = {"HSV1": (v[f"HSV1_L{pfx}"], v[f"HSV1_H{pfx}"]), "HSV2": (v[f"HSV2_L{pfx}"], v[f"HSV2_H{pfx}"])}
    crop_x = v.get("PLACE_CROP_X")

    rospy.init_node("sim_debug_view", anonymous=True)
    latest = {"img": None, "tf": None, "tf_t": 0.0}
    lock = threading.Lock()

    def on_img(msg):
        img = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        with lock:
            latest["img"] = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    def on_tf(msg):
        with lock:
            latest["tf"] = (msg.transform.translation.x, msg.transform.translation.y)
            latest["tf_t"] = time.time()

    rospy.Subscriber("/camera/color/image_raw", Image, on_img, queue_size=1)
    rospy.Subscriber("/target_TF", TransformStamped, on_tf, queue_size=1)

    out_dir = pathlib.Path("runs/sim")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / time.strftime("debug_%Y%m%d_%H%M%S.mp4")
    writer = None
    period = 1.0 / args.fps
    print(f"[debug view] 写入 {path}", flush=True)
    while not rospy.is_shutdown():
        t0 = time.time()
        with lock:
            img = None if latest["img"] is None else latest["img"].copy()
            tf, tf_age = latest["tf"], time.time() - latest["tf_t"]
        if img is None:
            time.sleep(0.05)
            continue
        h, w = img.shape[:2]
        att = rospy.get_param("/vision_attention", "pause")
        ex = w / 2 if v["EXPECT_X"] == "AUTO" else float(v["EXPECT_X"])
        ref = (ex, float(v["EXPECT_PICK_REF"]))
        if att in ("pick0", "pick1", "place0", "place1"):
            key = "HSV1" if att in ("pick0", "place1") else "HSV2"
            place = att.startswith("place")
            _, binary = StdVideo.color_filter(img, *(hsv_place if place else hsv)[key])
            if place:
                b = np.zeros_like(binary)
                b[:PLACE_CROP_ROW] = binary[:PLACE_CROP_ROW]
                binary = b
                cv2.line(img, (0, PLACE_CROP_ROW), (w, PLACE_CROP_ROW), (0, 0, 255), 1)
                if crop_x is not None:
                    binary[:, : int(crop_x[0])] = 0
                    binary[:, int(crop_x[1]) :] = 0
                    for x in crop_x:
                        cv2.line(img, (int(x), 0), (int(x), h), (0, 0, 255), 1)
                y_ref = int(v["EXPECT_PLACE_REF"])
                cv2.line(img, (0, y_ref), (w, y_ref), (255, 0, 255), 1)
            area = v["AREA_LIMIT_PLACE"] if place else v["AREA_LIMIT_PICK"]
            cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                a = cv2.contourArea(c)
                x, y, bw, bh = cv2.boundingRect(c)
                ok = area[0] < a < area[1]
                cv2.rectangle(img, (x, y), (x + bw, y + bh), (0, 200, 255) if ok else (128, 128, 128), 1)
                cv2.putText(img, f"{a:.0f}", (x, max(10, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            center, bias, rot, box, wh, cnt = Contour.NFC_F(binary, area, 5.5 if place else 3.0, ref)
            if cnt is not None:
                cv2.drawContours(img, [np.int32(box)], 0, (0, 0, 255), 2)
                if place:
                    edge = center[1] - wh[0] / 2
                    cv2.putText(img, f"upper edge {edge:.0f} (ref {v['EXPECT_PLACE_REF']:.0f})", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            else:
                cv2.putText(img, "nothing picked", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        cv2.drawMarker(img, (int(ref[0]), int(ref[1])), (255, 0, 255), cv2.MARKER_CROSS, 16, 2)
        txt = f"{time.strftime('%H:%M:%S')}  {att}  "
        if tf is not None and tf_age < 0.5:
            txt += f"target_TF x {tf[0]:+.1f} y {tf[1]:+.1f}"
        cv2.putText(img, txt, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        if writer is None:
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
        writer.write(img)
        if not args.no_window:
            cv2.imshow("sim debug: what the vision picks", img)
            cv2.waitKey(1)
        time.sleep(max(0.0, period - (time.time() - t0)))
    if writer is not None:
        writer.release()


if __name__ == "__main__":
    main()
