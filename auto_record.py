import rospy
from robot_tools.roser import Launcher
from robot_tools.multi_tp import run_command
from robot_tools.pather import get_current_dir
import time
import argparse
import random


parser = argparse.ArgumentParser()
parser.add_argument("-se", "--start_episode", type=int, default=0, help="start episode")
parser.add_argument("-ee", "--end_episode", type=int, default=10, help="end episode")
parser.add_argument(
    "-pp", "--python_path", type=str, default="python3", help="python path"
)
parser.add_argument(
    "-ns", "--no_show", action="store_true", help="no show the image"
)
parser.add_argument(
    "-nr", "--no_record", action="store_true", help="no record the data"
)
args = parser.parse_args()

start_episode = args.start_episode
end_episode = args.end_episode
python_path = args.python_path
not_show = "-ns" if args.no_show else ""


print("Start auto recording.")
for ep in range(start_episode, end_episode + 1):

    # start gazebo and moveit
    print("start gazebo and moveit")
    launcher = Launcher(
        f"{get_current_dir(__file__)}/launch/gazebo_control.launch"
    )
    launcher.start()
    rospy.sleep(10)

    # start record data
    # python3 record_data.py -mts 800 -on episode_7
    print("start record data")
    p_record_data = run_command(
        f"python3 data_driven/record_data.py -mts 800 -on episode_{ep}"
    )
    time.sleep(1)

    # start pick and place
    # ./run_pick_place.sh go 76 95
    print("start pick and place")
    pick_k = random.randint(70, 85)
    place_k = random.randint(90, 110)
    p_run_pick_place = run_command(f"./run_pick_place.sh go {pick_k} {place_k}")
    time.sleep(10)

    # start vision
    # python3 Best_Contour.py -vd /usb_cam_0/image_raw -ci /usb_cam_0/camera_info -ns
    print("start vision")
    p_best_contour = run_command(
        f"{python_path} Best_Contour.py -vd /usb_cam_0/image_raw -ci /usb_cam_0/camera_info -ns"
    )

    # wait for data rocorded
    time.sleep(5)
    print("wait for data rocorded")
    p_record_data.wait()
    time.sleep(2)

    # show pid
    print(p_record_data.pid)
    print(p_run_pick_place.pid)
    print(p_best_contour.pid)

    # kill all processes
    launcher.shutdown()  # 是阻塞的
    p_record_data.kill()
    p_best_contour.kill()
    p_run_pick_place.kill()

    # wait for all processes to finish
    p_record_data.wait()
    p_run_pick_place.wait()
    p_best_contour.wait()
