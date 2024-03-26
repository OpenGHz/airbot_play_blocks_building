
import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped
import math, time, argparse
from typing import Union
from numpy import ndarray

from robot_tools import recorder, pather, transformations
from std_vision.video import StdVideo, Types
from std_vision.contour import Contour
from std_vision.draw import Draw
from std_vision.ros_tools import RosTools
from std_vision.geometry import Geometry


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Best Contour")
    parser.add_argument('-r','--use_real',action='store_true',help='use sim or real robot')
    parser.add_argument('-ng','--negative',action='store_true',help='negative place')
    parser.add_argument('-ns','--not_show',action='store_true',help='show img or not')
    parser.add_argument('-t','--test',type=str, help='test mode: pi0 pi1 pl0 pl1',default='no')
    # parser.add_argument('-hsv','--hsv_params',type=str, help='example: 1,2,3;4,5,6',default='')
    parser.add_argument('-cfg','--config_path',type=str, help='config file path',default='./configs/vision/gazebo.json')
    parser.add_argument('-dfv','--default_hsv',type=str, help='use default hsv config: real or isaac',default='')
    parser.add_argument('-dfr','--default_ref',type=str, help='use default ref config: real or isaac',default='')
    parser.add_argument('-vd', '--video_device', type=str, help='video device', default="/camera/color/image_raw")
    parser.add_argument('-ci', '--camera_info', type=str, help='camera info topic', default="/camera/color/camera_info")
    parser.add_argument('-rt', '--rotate_180', action='store_true', help='rotate 180 degree')
    args, unknown = parser.parse_known_args()

    # 参数配置
    if args.config_path not in ['None','none', '']:
        current_dir = pather.get_current_dir(__file__)
        if args.config_path[0] == '.':
            config_path = current_dir + args.config_path[1:]
        elif args.config_path[0] != '/':
            config_path = current_dir + args.config_path
        else:
            config_path = args.config_path
        if args.use_real:
            config_path = config_path.replace('gazebo','real')
        print(f'载入配置文件路径：{config_path}')
        # 载入配置文件
        configs = recorder.json_process(config_path)
        # 颜色阈值设置
        HSV1_L,HSV1_H = configs['HSV1_L'],configs['HSV1_H']  # 第1个颜色的HSV参数的下限L和上限H
        HSV2_L,HSV2_H = configs['HSV2_L'],configs['HSV2_H']  # 第2个颜色的HSV参数的下限L和上限H
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
                        res,binary_img = StdVideo.color_filter(frame,HSV_Threshold['PurpleL'],HSV_Threshold['PurpleH'])
                    else: res,binary_img = StdVideo.color_filter(frame,HSV_Threshold['PurpleL'],HSV_Threshold['PurpleH'])
                else:
                    if vision_attention == 'place0':
                        res,binary_img = StdVideo.color_filter(frame,HSV_Threshold['GreenL'],HSV_Threshold['GreenH'])
                    else: res,binary_img = StdVideo.color_filter(frame,HSV_Threshold['GreenL'],HSV_Threshold['GreenH'])
            # 检测得到边缘图
            if EdgeDetect:
                converter = {True: 'Real', False: 'Sim'}
                if ColorFilter:
                    binary_img = StdVideo.edge_detect(binary_img,*CannyParam[converter[args.use_real]],convert=None)
                else: binary_img = StdVideo.edge_detect(frame,*CannyParam[converter[args.use_real]])
            # 对二值图进行轮廓检测与筛选
            if 'pick' in vision_attention:  # pick模式
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
                    # 裁切图像，避免由于物块大小不均匀造成的下层物块在上层轮廓侧边的漏出误识别（您也可以通过轮廓近似的方式来消除这种凸性缺陷）
                    bi = StdVideo.create_empty_frame(ImageSize[:2],0)
                    bi[:423,:] = binary_img[:423,:]
                    binary_img = bi
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
                    if args.negative:
                        bias_xyz[0] *= -1  # x方向偏差照旧，只是另一头方向要反一下
                        bias_xyz[1] = upper_edge - reference_y # 将y方向的偏差改为物块上边沿的位置(物块在)
                    else: bias_xyz[1] = reference_y - upper_edge # 将y方向的偏差改为物块上边沿的位置
                    # 不管rotation（因为不旋转才是正确的搭建姿态）
                    rotation_angle = 0

            # 若成功筛选出目标
            if cnt is not None:
                bias_xyz[2] = 0  # 不考虑z方向偏差
                Contour.draw(frame,[box_int])
                Draw.basic_draw(frame,(640,360),3,(0,0,255),-1,tips='circle')  # 绘制图像中心点（-1表示填充）
                # 角度按45度为界进行正逆转动的转换，保证优弧
                rotation_angle = rotation_angle - 90 if rotation_angle > 45 else rotation_angle
                rotation_angle *= -1
                target_publish(bias_xyz, rotation_angle)
                if args.test != 'no':
                    loginfo("偏差量(图像坐标系)为：x:{:.1f} y:{:.1f} r:{:.1f} ".format(bias_xyz[0],bias_xyz[1],rotation_angle))
                    loginfo(f'中心坐标为：({center_xy[0]:.1f},{center_xy[1]:.1f});w_h为：({w_h[0]:.1f},{w_h[1]:.1f})')
                    loginfo(f'上边沿为：{center_xy[1] - w_h[0]/2}')
            # 图像展示
            StdVideo.Show('Output',frame,wait_time=1,disable=args.not_show)

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
    device:str = args.video_device
    if not device.isdigit():
        """ 获取相机参数，主要是用到了宽和高来作为参考"""
        try:
            camrera_info:CameraInfo = rospy.wait_for_message(args.camera_info,CameraInfo,timeout=2)
            ImageSize = [camrera_info.width,camrera_info.height]
        except:
            raise Exception("获取相机参数信息失败")
        else:
            expect_x = ImageSize[0]/2 if expect_x == 'AUTO' else expect_x
        """ 启动图片订阅 """
        rospy.Subscriber(device,Image,image_process,queue_size=1)
        com_print(ImageSize)
        rospy.spin()
    else:  # 实机
        device = int(device)
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
            image_process(frame)
            RosTools.ros_spin_once(0.001)