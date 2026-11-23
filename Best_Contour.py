import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import math, time, argparse, atexit, os, threading
from typing import Union
from numpy import ndarray
import cv2
import numpy as np

from robot_tools import recorder, pather, transformations
from std_vision.video import StdVideo, Types
from std_vision.contour import Contour
from std_vision.ros_tools import RosTools
from std_vision.geometry import Geometry


# Output 窗口上叠加的识别过程（BGR）。"target" 指视觉在这一帧选中、正在对准的物块，和抓取（pick）阶段无关
OV_FILTERED = (150, 150, 150)  # 灰：被面积或长宽比筛掉的轮廓
OV_CANDIDATE = (0, 220, 255)   # 黄：通过筛选的候选轮廓
OV_TARGET = (0, 0, 255)        # 红：选中的目标（候选里离参考点最近的）
OV_REF = (255, 0, 255)         # 品红：参考点 / Place 阶段上边沿的期望位置
OV_EDGE = (255, 255, 0)        # 青：Place 阶段塔顶物块上边沿的当前位置


def _text(img, s, org, color=(255, 255, 255), scale=0.5):
    """带黑色描边的文字，在任何背景上都看得清"""
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_overlay(frame, attention, binary=None, area_limit=None, ratio_max=None, ref=None,
                 target=None, keep=None, edge_ref=None):
    """
    在 Output 窗口的图像上画出这一帧的识别过程（原地修改 frame）。
      attention  /vision_attention：pick0 / pick1 / place0 / place1 / pause
      binary     查找轮廓用的二值图（已裁剪）
      ref        参考点 (x, y)：候选轮廓里离它最近的成为目标；x 方向偏差相对它计算
      target     None，或 dict(box=四个顶点, offset=(x, y, 角度), edge=上边沿 y 或 None)
      keep       Place 阶段参与识别的区域 (x0, x1, y1)，区域外调暗
      edge_ref   Place 阶段塔顶物块上边沿的期望位置 EXPECT_PLACE_REF
    """
    h, w = frame.shape[:2]
    if attention == 'pause':
        _text(frame, 'PAUSED: vision off', (10, 24), scale=0.6)
        return
    place = attention.startswith('place')
    color = 1 if attention in ('pick0', 'place1') else 2
    if keep is not None:
        x0, x1, y1 = keep
        ignored = np.ones((h, w), bool)
        ignored[:y1, x0:x1] = False
        frame[ignored] = (frame[ignored] * 0.35).astype(frame.dtype)
    if binary is not None:  # 和 Contour.NFC_F 相同的筛选：轮廓面积、外接矩形面积、长宽比
        cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            _, (rw, rh), _ = cv2.minAreaRect(c)
            ok = (area_limit[0] < a < area_limit[1] and area_limit[0] < rw * rh < area_limit[1]
                  and rh > 0 and 1 / ratio_max < rw / rh < ratio_max)
            col = OV_CANDIDATE if ok else OV_FILTERED
            cv2.drawContours(frame, [c], -1, col, 1)
            if ok or a >= area_limit[0] / 4:  # 很小的噪点不标面积
                x, y, _, _ = cv2.boundingRect(c)
                _text(frame, f'{a:.0f}', (x, max(12, y - 4)), col, 0.4)
    if place and edge_ref is not None:
        e = int(round(edge_ref))
        cv2.line(frame, (0, e), (w, e), OV_REF, 1)
        _text(frame, 'edge ref', (w - 75, e - 4), OV_REF, 0.4)
    if ref is not None:
        cv2.drawMarker(frame, (int(round(ref[0])), int(round(ref[1]))), OV_REF, cv2.MARKER_CROSS, 18, 2)
    if target is not None:
        box = np.int32(target['box'])
        cv2.drawContours(frame, [box], 0, OV_TARGET, 2)
        _text(frame, 'target', (int(box[:, 0].min()), int(box[:, 1].min()) - 6), OV_TARGET, 0.45)
        if target.get('edge') is not None:
            e = int(round(target['edge']))
            cv2.line(frame, (int(box[:, 0].min()) - 10, e), (int(box[:, 0].max()) + 10, e), OV_EDGE, 2)
        ox, oy, rot = target['offset']
        status, status_col = f'target offset  x {ox:+.1f}  y {oy:+.1f} px  rot {rot:+.1f} deg', (255, 255, 255)
    else:
        status, status_col = 'no target', OV_TARGET
    what = 'tower top' if place else 'cube to grasp'
    _text(frame, f'{"PLACE" if place else "PICK"} stage: {what}, colour {color}', (10, 24), scale=0.6)
    _text(frame, status, (10, 48), status_col, 0.55)
    legend = 'grey: filtered out  yellow: candidate  red: target  magenta: reference'
    if place:
        legend += '  cyan: top edge'
    _text(frame, legend, (10, h - 10), scale=0.4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Best Contour")
    parser.add_argument('-r','--use_real',action='store_true',help='use sim or real robot')
    parser.add_argument('-ng','--negative',action='store_true',help='negative place')
    parser.add_argument('-ns','--not_show',action='store_true',help='show img or not')
    parser.add_argument('-t','--test',type=str, help='test mode: pi0 pi1 pl0 pl1',default='no')
    parser.add_argument('-cfg','--config_path',type=str, help='config file path',default='./configs/vision/sim.yaml')
    parser.add_argument('-dfv','--default_hsv',type=str, help='use default hsv config: real or isaac',default='')
    parser.add_argument('-dfr','--default_ref',type=str, help='use default ref config: real or isaac',default='')
    parser.add_argument('-vd', '--video_device', type=str, help='video device', default="/camera/color/image_raw")
    parser.add_argument('-ci', '--camera_info', type=str, help='camera info topic', default="/camera/color/camera_info")
    parser.add_argument('-rt', '--rotate_180', action='store_true', help='rotate 180 degree')
    parser.add_argument('--record', nargs='?', const='', default=None,
                        help='also save the Output window to a video (default runs/vision/vision_<time>.mp4)')
    args, unknown = parser.parse_known_args()

    sim_player_nums = rospy.get_param("/airbot_play/sim_player_nums", default=0)
    env_type = rospy.get_param("/env_type", default=None)
    device:str = args.video_device
    if sim_player_nums == 0:
        args.use_real = True
        # print(device)
        # if not device.isdigit():
        #     args.video_device = device = "2"
        print("Use video device: ", device)
    else:
        print("Use topic: ", device)

    # 参数配置
    if args.config_path not in ['None','none', '']:
        current_dir = pather.get_current_dir(__file__)
        if args.config_path[0] == '.':
            config_path = current_dir + args.config_path[1:]
        elif args.config_path[0] != '/':
            config_path = current_dir + args.config_path
        else:
            config_path = args.config_path
        if args.use_real and env_type is None:
            config_path = config_path.replace('gazebo','real').replace('mujoco','real').replace('/sim.', '/real.')
        print(f'载入配置文件路径：{config_path}')
        # 载入配置文件（YAML，可以写注释；.json 文件按 JSON 读）
        if config_path.endswith('.json'):
            configs = recorder.json_process(config_path)
        else:
            import yaml
            with open(config_path) as f:
                configs = yaml.safe_load(f) or {}
        # 颜色阈值设置
        HSV1_L,HSV1_H = configs['HSV1_L'],configs['HSV1_H']  # 第1个颜色的HSV参数的下限L和上限H
        HSV2_L,HSV2_H = configs['HSV2_L'],configs['HSV2_H']  # 第2个颜色的HSV参数的下限L和上限H
        # Place 阶段单独的 HSV 阈值（USE_PLACE_HSV 为 "True" 时使用 HSV*_P；否则和 Pick 阶段相同）
        if str(configs.get('USE_PLACE_HSV', 'False')).lower() == 'true':
            HSV1_L_P,HSV1_H_P = configs['HSV1_L_P'],configs['HSV1_H_P']
            HSV2_L_P,HSV2_H_P = configs['HSV2_L_P'],configs['HSV2_H_P']
        else:
            HSV1_L_P,HSV1_H_P,HSV2_L_P,HSV2_H_P = HSV1_L,HSV1_H,HSV2_L,HSV2_H
        # Place 阶段只保留的图像列 [x0, x1]（像素），不设置时保留整幅宽度
        PLACE_CROP_X = configs.get('PLACE_CROP_X')
        # 参考设置
        EXPECT_PICK_REF = configs['EXPECT_PICK_REF']  # 中心坐标
        EXPECT_PLACE_REF = configs['EXPECT_PLACE_REF']  # 上边沿
        EXPECT_X = configs['EXPECT_X']
        # 面积限制
        AREA_LIMIT_PICK = configs['AREA_LIMIT_PICK']
        AREA_LIMIT_PLACE = configs['AREA_LIMIT_PLACE']
    else:
        print('未指定配置文件，使用默认参数')
        converter = {True: 'Real', False: 'Sim'}
        args.default_hsv = converter[args.use_real]
        args.default_ref = converter[args.use_real]
        PLACE_CROP_X = None

    # 打印一些信息
    def loginfo(msg: str):
        print(msg)
        return

    # 发布像素偏差信息
    def target_publish(xyz_bias, rotation_angle):
        target = TransformStamped()
        target.header.stamp = rospy.Time.now()
        target.transform.translation.x = xyz_bias[1] # 图像y（前后方向）实际对应机械臂的x轴，dx
        target.transform.translation.y = xyz_bias[0] # 图像x（左右方向）实际对应机械臂的y轴，dy
        target.transform.translation.z = xyz_bias[2] # 图像z（垂直方向）实际对应机械臂的z轴，dz
        tf_q = transformations.quaternion_from_euler(0, 0, rotation_angle*math.pi/180)  # pick&place状态下，你以为的末端关节的roll，其实是机械臂的yaw
        target.transform.rotation.x = tf_q[0]
        target.transform.rotation.y = tf_q[1]
        target.transform.rotation.z = tf_q[2]
        target.transform.rotation.w = tf_q[3]
        target_pub.publish(target)

    # 把 Output 窗口的画面存成视频（--record）
    video_out = {'writer': None, 'closed': False}
    video_lock = threading.Lock()

    def record(frame):
        if args.record is None or frame is None:
            return
        with video_lock:
            if video_out['closed']:
                return
            if video_out['writer'] is None:
                path = args.record or time.strftime('runs/vision/vision_%Y%m%d_%H%M%S.mp4')
                if not os.path.isabs(path):
                    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                h, w = frame.shape[:2]
                video_out['writer'] = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w, h))
                print(f'录像写入：{path}')
            video_out['writer'].write(frame)

    def close_record():
        with video_lock:
            video_out['closed'] = True
            if video_out['writer'] is not None:
                video_out['writer'].release()
    atexit.register(close_record)

    # 核心图像处理函数
    def image_process(frame:Union[Image, ndarray]):
            # 根据注意力参数进行一些信息是否发送的切换
            global vision_attention
            vision_attention_new = rospy.get_param("/vision_attention", default="pause")
            if args.test != 'no':
                if 'i' in args.test: vision_attention_new =  'pick'+args.test[2]  # 查看识别效果用
                elif 'l' in args.test: vision_attention_new ='place'+args.test[2] # 查看识别效果用
                else: vision_attention_new = args.test  # 比如为‘hsv’/'t'等均可，只要不是p就行，整定HSV阈值用
            if vision_attention_new != vision_attention:
                print("模式切换为：{}".format(vision_attention_new))
                vision_attention = vision_attention_new
            
            # Image图像自动转换为cv2格式
            if isinstance(frame,Image):
                frame  = RosTools.imgmsg_to_cv2(frame)

            # 若暂停，则直接退出函数，不进行任何图像处理和话题发布
            if vision_attention == 'pause':
                draw_overlay(frame, 'pause')
                record(frame)
                StdVideo.Show('Output',frame,wait_time=1,disable=args.not_show)
                return

            # 颜色阈值确定
            if 'p' not in vision_attention:
                if args.use_real:  # 目前仅针对实机课程有意义
                    if args.test != 'no':
                        StdVideo.color_thresh_determine(device,mode='HSV',show_raw=True)  # 阻塞，按esc退出
                        exit('ESC键按下，视觉程序退出')
                    else:
                        print("当前的HSV阈值为:")  # 提示当前阈值
                        print(f"  1L={HSV_Threshold['PurpleL']},1H={HSV_Threshold['PurpleH']}")
                        print(f"  2L={HSV_Threshold['GreenL']}, 2H={HSV_Threshold['GreenH']}")
                        StdVideo.color_thresh_determine(device,mode=vision_attention,show_raw=False)  # 阻塞，按esc退出
                else: StdVideo.color_thresh_determine("camera/color/image_raw",mode=vision_attention)
                rospy.set_param("/vision_attention",'pick0')  # 回到0状态
                vision_attention = 'pick0'

            # 得到二值图
            if ColorFilter:  # 根据奇偶次和pickplace状态选择不同的颜色
                if vision_attention in ['pick0','place1']:  # 一轮中，pick和place利用的颜色不同
                    if vision_attention == 'place1':  # place和pick的参数可以要分开会好一些
                        res,binary_img = StdVideo.color_filter(frame,HSV_Threshold['PurpleL_P'],HSV_Threshold['PurpleH_P'])
                    else: res,binary_img = StdVideo.color_filter(frame,HSV_Threshold['PurpleL'],HSV_Threshold['PurpleH'])
                else:
                    if vision_attention == 'place0':
                        res,binary_img = StdVideo.color_filter(frame,HSV_Threshold['GreenL_P'],HSV_Threshold['GreenH_P'])
                    else: res,binary_img = StdVideo.color_filter(frame,HSV_Threshold['GreenL'],HSV_Threshold['GreenH'])
            # 检测得到边缘图
            if EdgeDetect:
                converter = {True: 'Real', False: 'Sim'}
                if ColorFilter:
                    binary_img = StdVideo.edge_detect(binary_img,*CannyParam[converter[args.use_real]],convert=None)
                else: binary_img = StdVideo.edge_detect(frame,*CannyParam[converter[args.use_real]])
            # 对二值图进行轮廓检测与筛选
            ov_keep = ov_edge = None  # Output 窗口叠加用：Place 阶段参与识别的区域、塔顶上边沿
            if 'pick' in vision_attention:  # pick模式
                ov_area, ov_ratio = AREA_LIMIT_PICK, 3.0
                converter = {
                    'area_limit': AREA_LIMIT_PICK, 
                    'ratio_max': {True: 3.0, False: 3.0}, 
                    'mid_xy': {True: (expect_x,expect_pick_y), False: (expect_x,expect_pick_y)}
                    }
                center_xy,bias_xyz,rotation_angle,box_int,w_h,cnt = Contour.NFC_F(binary_img,
                                                                                           converter['area_limit'],
                                                                                           3.0,
                                                                                           (expect_x,expect_pick_y))

                if args.test != 'no': StdVideo.Show('Binary',binary_img,wait_time=1)
            else:  # place模式
                # place模式下mid_y没啥意义,然后‘长宽比’要大些
                if args.use_real:
                    # 裁切图像，避免由于物块大小不均匀造成的下层物块在上层轮廓侧边的漏出误识别（也可以通过轮廓近似的方式来消除这种凸性缺陷）
                    bi = StdVideo.create_empty_frame(ImageSize[:2], 0)
                    bi[:423,:] = binary_img[:423,:]
                    binary_img = bi
                if PLACE_CROP_X is not None:  # 左右裁掉，避免识别到旁边抓取区的物块
                    x0, x1 = int(PLACE_CROP_X[0]), int(PLACE_CROP_X[1])
                    binary_img[:, :max(0, x0)] = 0
                    binary_img[:, x1:] = 0
                ov_h, ov_w = binary_img.shape[:2]
                ov_keep = (max(0, int(PLACE_CROP_X[0])) if PLACE_CROP_X is not None else 0,
                           int(PLACE_CROP_X[1]) if PLACE_CROP_X is not None else ov_w,
                           423 if args.use_real else ov_h)
                ov_area, ov_ratio = AREA_LIMIT_PLACE, 5.5
                # 轮廓识别
                converter = {
                    'area_limit': AREA_LIMIT_PLACE, 
                    'ratio_max': {True: 5.5, False: 5.5}, 
                    'mid_xy': {True: (expect_x,expect_pick_y), False: (expect_x,expect_pick_y)}
                    }
                center_xy,bias_xyz,rotation_angle,box_int,w_h,cnt = Contour.NFC_F(binary_img,
                                                                                           converter['area_limit'],
                                                                                           5.5,
                                                                                           (expect_x, expect_pick_y))
                if args.test != 'no': StdVideo.Show('Binary',binary_img,wait_time=1)
                if cnt is not None:
                    # 上边沿法（前提是物块上边是近乎平行的，否则不对）（注意摄像头下移则上边沿相对上移）
                    if not args.use_real:
                        reference_y = expect_place_y  # 参考的标准的上边沿的位置（参考值的偏差直接影响了所有物块放置的偏差（即每次放置的系统偏差），是累计偏差的根源，因此一定要尽可能精确确定，并且在放置时可以采用上下轮换逼近的方式避免累计）
                    else: reference_y = expect_place_y
                    upper_edge = center_xy[1] - w_h[0]/2  # 注意此时减的是宽，因为宽更小，符合实际情况
                    ov_edge = upper_edge
                    if args.negative:
                        bias_xyz[0] *= -1  # x方向偏差照旧，只是另一头方向要反一下
                        bias_xyz[1] = upper_edge - reference_y # 将y方向的偏差改为物块上边沿的位置(物块在)
                    else: bias_xyz[1] = reference_y - upper_edge # 将y方向的偏差改为物块上边沿的位置
                    # 不管rotation（因为不旋转才是正确的搭建姿态）
                    rotation_angle = 0

            # 若成功筛选出目标
            if cnt is not None:
                bias_xyz[2] = 0  # 不考虑z方向偏差
                # 角度按45度为界进行正逆转动的转换，保证优弧
                # minAreaRect returns (0, 90] in OpenCV 4.5.1 ~ 4.x and [-90, 0) in others (e.g. 4.13); a square is the same every 90 deg
                rotation_angle %= 90
                rotation_angle = rotation_angle - 90 if rotation_angle > 45 else rotation_angle
                rotation_angle *= -1
                target_publish(bias_xyz, rotation_angle)
                if args.test != 'no':
                    loginfo("偏差量(图像坐标系)为：x:{:.1f} y:{:.1f} r:{:.1f} ".format(bias_xyz[0],bias_xyz[1],rotation_angle))
                    loginfo(f'中心坐标为：({center_xy[0]:.1f},{center_xy[1]:.1f});w_h为：({w_h[0]:.1f},{w_h[1]:.1f})')
                    loginfo(f'上边沿为：{center_xy[1] - w_h[0]/2}')
            # 图像展示：叠加识别过程（候选、目标、参考点、裁剪区域），见 draw_overlay
            target = None if cnt is None else {'box': box_int, 'offset': (bias_xyz[0], bias_xyz[1], rotation_angle), 'edge': ov_edge}
            draw_overlay(frame, vision_attention, binary_img, ov_area, ov_ratio, (expect_x, expect_pick_y),
                         target, ov_keep, expect_place_y)
            record(frame)
            StdVideo.Show('Output',frame,wait_time=1,disable=args.not_show,show_fps=False)
            return frame

    """ ***********程序内置参考用识别参数配置************ """
    # 颜色阈值
    HSV_Color = {
        # 仿真
        'PurpleLS':[64,93,59],   'PurpleHS':[179,255,255],
        'GreenLS':[40,111,113],  'GreenHS':[107,255,255],
        # 实机
        # 'PurpleLR':[90,180,70], 'PurpleHR':[178,255,255], # 2.10+2.25晚均可
        'PurpleLR':[70,70,24], 'PurpleHR':[152,255,255],  # 2.26下午16点
        # 'PurpleLR':[90,174,0], 'PurpleHR':[179,255,255],  # 3.2晚
        # 'PurpleLR':[67,91,23], 'PurpleHR':[179,255,255],
        # 'PurpleLR':[90,125,0], 'PurpleHR':[179,255,255],  # 4.13土木

        'GreenLR':[20,94,27],  'GreenHR':[68,255,255],  # 2.10
        # 'GreenLR':[47,118,53],  'GreenHR':[85,252,161], # 2.25晚891
        # 'GreenLR':[47,150,20],  'GreenHR':[77,255,254], # 4.10土木
        # 'GreenLR':[42,140,60],  'GreenHR':[84,255,254], # 4.13土木
        # 'GreenLR':[47,171,18],  'GreenHR':[70,255,71],  # 2.26下午16点
        # 'GreenLR':[37,111,24],  'GreenHR':[73,255,255],   # 3.26下午1点半

        'PurpleLU':[67,91,23], 'PurpleHU':[179,255,255],
        'GreenLU':[37,111,24],  'GreenHU':[73,255,255],
    }

    # 期望位置
    default_ref = args.default_ref
    if default_ref == '':
        expect_pick_y = EXPECT_PICK_REF
        expect_place_y = EXPECT_PLACE_REF
        expect_x = EXPECT_X
    else:
        expect_x = 'AUTO'
        if default_ref == 'real':
            expect_pick_y = 529
            expect_place_y = 456
        elif default_ref == 'isaac':
            expect_pick_y = 529
            expect_place_y = 456

    HSV_Threshold = {}
    # HSV参数配置
    default_hsv = args.default_hsv
    if default_hsv == '':
        HSV_Threshold['PurpleL'] = HSV1_L
        HSV_Threshold['PurpleH'] = HSV1_H
        HSV_Threshold['GreenL']  = HSV2_L
        HSV_Threshold['GreenH']  = HSV2_H
        HSV_Threshold['PurpleL_P'] = HSV1_L_P
        HSV_Threshold['PurpleH_P'] = HSV1_H_P
        HSV_Threshold['GreenL_P']  = HSV2_L_P
        HSV_Threshold['GreenH_P']  = HSV2_H_P
    else:
        if default_hsv == 'real':
            HSV_Threshold['PurpleL'] = HSV_Color['PurpleLR']
            HSV_Threshold['PurpleH'] = HSV_Color['PurpleHR']
            HSV_Threshold['GreenL']  = HSV_Color['GreenLR']
            HSV_Threshold['GreenH']  = HSV_Color['GreenHR']
        elif default_hsv == 'isaac':
            HSV_Threshold['PurpleL'] = HSV_Color['PurpleLS']
            HSV_Threshold['PurpleH'] = HSV_Color['PurpleHS']
            HSV_Threshold['GreenL']  = HSV_Color['GreenLS']
            HSV_Threshold['GreenH']  = HSV_Color['GreenHS']
    for _k in ('PurpleL', 'PurpleH', 'GreenL', 'GreenH'):
        HSV_Threshold.setdefault(_k + '_P', HSV_Threshold.get(_k))

    # 算法相关
    CannyParam = {'Sim':[210,240,5,True],'Real':[0,0,0,False]}
    ColorFilter = True
    EdgeDetect = False

    """ ***********节点初始化************ """
    NODE_NAME = 'airbot_play_cube_detect'
    rospy.init_node(NODE_NAME)
    loginfo("Start {} node.".format(NODE_NAME))

    target_pub = rospy.Publisher("/target_TF",TransformStamped,queue_size=1) # queue_size=1表明只发布最新数据
    vision_attention = 'pause'  # 初始默认为pause
    print("模式初始为：{}".format(vision_attention))
    if args.not_show: print('图像显示设置为：不显示图像')
    # 接收相机数据并进行处理
    def com_print(image_size):
        print('图像大小为：',image_size)
        print('等待视频流稳定......')
        time.sleep(2)
        print('开始获取图像')
    if not device.isdigit():
        """ 获取相机参数，主要是用到了宽和高来作为参考"""
        try:
            camrera_info:CameraInfo = rospy.wait_for_message(args.camera_info, CameraInfo, timeout=2)
            ImageSize = [camrera_info.width,camrera_info.height]
        except:
            raise Exception("获取相机参数信息失败")
        else:
            expect_x = ImageSize[0]/2 if expect_x == 'AUTO' else expect_x
        """ 启动图片订阅 """
        rospy.set_param("/best_contour_started", True)
        rospy.Subscriber(device, Image, image_process, queue_size=1)
        com_print(ImageSize)
        rospy.spin()
    else:  # USB camera
        device = int(device)
        # image_size = (1280, 720)
        # image_size = (640, 480)
        # cap = StdVideo.Cap(device, wh_set=image_size)
        cap = StdVideo.Cap(device)
        ImageSize = [int(StdVideo.Info(device, Types.CAP_FRAME_WIDTH)),
                     int(StdVideo.Info(device, Types.CAP_FRAME_HEIGHT))]
        com_print(ImageSize)
        expect_x = ImageSize[0]/2 if expect_x == 'AUTO' else expect_x
        while True:
            frame = StdVideo.Read(device)
            # 将图像旋转180度
            if args.rotate_180:
                frame = Geometry.Rotation(frame, 180)
            frame = image_process(frame)
            RosTools.ros_spin_once(0.001)
