import os
import argparse
import time
import h5py


def main(args):
    print(args)
    dataset_dir = args["dataset_dir"]
    episode_idx = args["episode_idx"]
    dataset_name = f"episode_{episode_idx}"
    camera_names = args["camera_names"]
    compress = not args["no_compress"]
    ignore_actions = args["ignore_actions"]
    ignore_images = args["ignore_images"]
    robots_num =args["robots_num"]
    can_bias = args["can_bias"]
    urdf_path = args["urdf_path"]
    task_name = args["task_name"]
    robot_name = args["robot_name"]
    DT = 1.0 / args["control_freq"]
    vcan = "v" if args["virtual_can"] else ""
    assert camera_names is not None, "Camera names must be provided"
    # assert robots_num==2 and len(camera_names) == 4, "Two robots should have 4 cameras"
    if urdf_path == "":
        urdf_path = "/usr/local/share/airbot_play/airbot_play_v2_1/urdf/airbot_play_v2_1_with_gripper.urdf"

    # 载入数据集
    if task_name != "":
        if dataset_dir == "AUTO":
            dataset_dir = f"{os.getcwd()}/demonstrations/hdf5/"
        dataset_dir = os.path.join(dataset_dir, task_name)
    dataset_path = os.path.join(dataset_dir, dataset_name + ".hdf5")
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at \n{dataset_path}\n")
        exit()
    with h5py.File(dataset_path, "r") as root:
        if not ignore_actions:
            actions = root["/action"][()]
        if not ignore_images:
            import cv2
            images = {}
            for cam_name in camera_names:
                images[cam_name] = root[f"/observations/images/{cam_name}"][()]
                if compress:
                    # JPEG decompression
                    images_num = len(images[cam_name])
                    # print(images_num)
                    decompressed_list = [0] * images_num
                    for index, image in enumerate(images[cam_name]):
                        decompressed_list[index] = cv2.imdecode(image, 1)
                    images[cam_name] = decompressed_list

    # 动作回放
    if not ignore_actions:
        robots_list = []
        if robot_name == "airbot_play":
            # modify the path to the airbot_play urdf file
            import airbot
            # instance the airbot player
            urdf_path = airbot.AIRBOT_PLAY_URDF
            vel = 1.0
            for i in range(robots_num):
                robots_list.append(airbot.create_agent(
                    urdf_path, "down", f"{vcan}can{i+can_bias}", vel, "gripper"))
        elif "ros" in robot_name:
            from ros_robot import RosRobot
            import rospy
            rospy.init_node("replay_episodes")
            namespace = "/airbot_play"
            states_topic = f"{namespace}/joint_states"
            arm_action_topic = f"{namespace}/arm_group_position_controller/command"
            gripper_action_topic = f"{namespace}/gripper_group_position_controller/command"
            states_num = 7
            default_joints = [0.0] * 7
            for i in range(robots_num):
                robots_list.append(
                    RosRobot(
                        states_topic,
                        arm_action_topic,
                        gripper_action_topic,
                        states_num,
                        default_joints,
                    )
                )

        print(f"Moving to start pose:{actions[0]}...")
        for index, robot in enumerate(robots_list):
            first = index * 6
            last = first + 6
            robot.set_target_joint_q(actions[0][first:last], False)
            robot.set_target_end(actions[0][last])
        time.sleep(3)

        z = input("Press Enter to start or z to exit...")
        if z == "z":
            for robot in robots_list:
                del robot
            del robots_list
            exit()

        for action in actions:
            for index, robot in enumerate(robots_list):
                first = index * 6
                last = first + 6
                robot.set_target_joint_q(action[first:last])
                robot.set_target_end(action[last])
            time.sleep(DT)
        # wait key
        while True:
            key = input("Press Enter to exit...")
            if key == "":
                for robot in robots_list:
                    del robot
                del robots_list
                break

    if not ignore_images:
        # 同时显示多个窗口
        window_name = f"Camera {camera_names}"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        # 显示图片
        images_num = len(list(images.values())[0])
        dt = int(DT * 1000)
        for i in range(images_num):
            frames = [images[cam_name][i] for cam_name in camera_names]
            frame = cv2.hconcat(frames)
            cv2.imshow(window_name, frame)
            cv2.waitKey(dt)  # 40ms, 25fps
        # 关闭窗口
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-dd",
        "--dataset_dir",
        action="store",
        type=str,
        default="AUTO",
        help="Dataset dir.",
        required=False,
    )
    parser.add_argument(
        "-tn",
        "--task_name",
        action="store",
        type=str,
        help="Task name.",
        default="",
        required=False,
    )
    parser.add_argument(
        "-ei",
        "--episode_idx",
        action="store",
        type=int,
        help="Episode index.",
        required=True,
    )
    parser.add_argument(
        "-cn",
        "--camera_names",
        nargs="+", 
        default=("0", "1"),
        help="Camera names.",
    )
    parser.add_argument(
        "-nc",
        "--no_compress",
        action="store_true",
    )
    parser.add_argument(
        "-ia",
        "--ignore_actions",
        action="store_true",
    )
    parser.add_argument(
        "-ii",
        "--ignore_images",
        action="store_true",
    )
    parser.add_argument(
        "-rn",
        "--robots_num",
        action="store",
        type=int,
        default=1,
        help="Number of robots.",
    )
    parser.add_argument(
        "-cb",
        "--can_bias",
        action="store",
        type=int,
        default=0,
        help="Can is bias.",
    )
    parser.add_argument(
        "-up",
        "--urdf_path",
        action="store",
        type=str,
        default="",
        help="The urdf path to your robot.",
    )
    parser.add_argument(
        "-fq",
        "--control_freq",
        action="store",
        type=float,
        default=15.0,
        help="Control frequency.",
    )
    parser.add_argument(
        "-mt",
        "--mobile_type",
        action="store",
        type=str,
        default="",  # "slantec_ydn", "" means no mobile
        help="Mobile type.",
    )
    parser.add_argument(
        "-rna",
        "--robot_name",
        action="store",
        type=str,
        default="airbot_play",
        help="Robot name.",
    )
    parser.add_argument(
        "-vcan",
        "--virtual_can",
        action="store_true",
        help="Virtual can.",
    )
    main(vars(parser.parse_args()))
