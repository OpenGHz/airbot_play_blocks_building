from robot_tools.roser import Launcher
from robot_tools.multi_tp import SubCLIer
from robot_tools.pather import get_current_dir

import time
import argparse
import random
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("-se", "--start_episode", type=int, default=0, help="start episode")
parser.add_argument("-ee", "--end_episode", type=int, default=10, help="end episode")
parser.add_argument(
    "-pp", "--python_path", type=str, default="python3", help="python path"
)
parser.add_argument("-ns", "--no_show", action="store_true", help="no show the image")
parser.add_argument(
    "-nr", "--no_record", action="store_true", help="no record the data"
)
parser.add_argument(
    "-mts", "--max_time_step", type=int, default=800, help="max time step"
)
parser.add_argument(
    "-picp",
    "--pick_control_param_limit",
    nargs=2,
    type=int,
    default=[70, 85],
    help="pick control param limit",
)
parser.add_argument(
    "-plcp",
    "--place_control_param_limit",
    nargs=2,
    type=int,
    default=[90, 110],
    help="place control param limit",
)
args = parser.parse_args()

start_episode = args.start_episode
end_episode = args.end_episode
python_path = args.python_path
not_show = "-ns" if args.no_show else ""
not_record = "-nr" if args.no_record else ""
max_time_step = args.max_time_step
pick_control_param_limit = args.pick_control_param_limit
place_control_param_limit = args.place_control_param_limit


print("Start auto recording.")
for ep in range(start_episode, end_episode + 1):

    # start gazebo and moveit
    print("start gazebo and moveit")
    launcher = Launcher(f"{get_current_dir(__file__)}/launch/gazebo_control.launch")
    launcher.start(10)

    # start record data
    # python3 record_data.py -mts 800 -on episode_7
    print("start record data")
    p_record_data = SubCLIer.run(
        f"python3 data_driven/record_data.py -mts {max_time_step} -on episode_{ep}",
        sleeps=1,
    )

    # start pick and place
    # ./run_pick_place.sh go 76 95
    print("start pick and place")
    pick_k = random.randint(*pick_control_param_limit)
    place_k = random.randint(*place_control_param_limit)
    p_run_pick_place = SubCLIer.run(
        f"./run_pick_place.sh go {pick_k} {place_k}", sleeps=10
    )

    # start vision
    # python3 Best_Contour.py -vd /usb_cam_0/image_raw -ci /usb_cam_0/camera_info -ns
    print("start vision")
    p_best_contour = SubCLIer.run(
        f"{python_path} Best_Contour.py -vd /usb_cam_0/image_raw -ci /usb_cam_0/camera_info -ns"
    )

    # wait for data rocorded
    print("wait for data rocorded")
    p_record_data.wait()
    time.sleep(2)

    # show pid
    print("PIDs:", SubCLIer.get_pids())

    # kill all processes
    launcher.shutdown()
    SubCLIer.kill()

    # wait a second to choose to exit
    print("Eposide", ep, "finished.")
    print("You can choose to exit now.")
    sleep_s = 2.5
    times = 100
    duration = sleep_s / times
    for _ in tqdm(range(times)):
        time.sleep(duration)
