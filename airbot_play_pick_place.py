#!/usr/bin/env python3
from geometry_msgs.msg import TransformStamped
import rospy
import tf_conversions

import numpy as np
import math,time
from threading import Thread,Event
from copy import deepcopy

from airbot_play_control.control import RoboticArmAgent, CoordinateTools, ChooseGripper


class VisualPerception(object):
    """ 视觉反馈交互类#TODO """
    def __init__(self) -> None:
        # 订阅视觉反馈
        rospy.Subscriber("/target_TF",TransformStamped,self.__feedback_callback,queue_size=1)  # queue_size=1表明只接收最新数据
        rospy.loginfo('Start visual perception (from /target_TF).')
        self.registed_func = {}

    def set_vision_attention(self, attention:str):
        """ 设置视觉注意力参数（"pick0","pick1","place0","place1"） """
        rospy.set_param("/vision_attention", attention)
        self.vision_attention = attention

    def get_raw_feedback(self):
        """ 获得原始的反馈TF数据 """
        return self._pixel_error

    def get_relative_target(self, current_yaw:float, place_same=False):
        """ 根据当前yaw，将反馈TF转换为机械臂在世界系下的(x、y)以及yaw方向的误差值 """
        position_error, rotation_error = self.tf_to_xyzrpy(self._pixel_error)  # 偏差相对于机械臂末端系
        if place_same or (self.vision_attention in ['pick0','pick1']):  # pick阶段认为第一个物块放置时没有偏转，因此后续不需要进行偏转补偿
            current_yaw_bias = (rotation_error[2] - current_yaw)
            tf_trans_x =  (position_error[0]*math.cos(current_yaw_bias) - position_error[1]*math.sin(current_yaw_bias))  # 左右移动
            position_error[1] = (position_error[0]*math.sin(current_yaw_bias) + position_error[1]*math.cos(current_yaw_bias))  # 前后移动
            position_error[0] = tf_trans_x
        return position_error[:2], rotation_error[2]

    def register_callback(self,name,func):
        """ 注册视觉反馈 """
        self.registed_func[name] = func

    def __feedback_callback(self, msg:TransformStamped):
        self._pixel_error = msg
        # 执行注册的函数
        for func in self.registed_func.values():
            func(msg)
    
    def tf_to_xyzrpy(self,tf:TransformStamped):
        """ 将TransformStamped格式的数据转换为xyz和rpy """
        xyz = [tf.transform.translation.x,tf.transform.translation.y,tf.transform.translation.z]
        rpy = list(tf_conversions.transformations.euler_from_quaternion([tf.transform.rotation.x,tf.transform.rotation.y,
                                                                tf.transform.rotation.z,tf.transform.rotation.w]))
        return xyz,rpy


class AIRbotPlayPickPlace(RoboticArmAgent):
    """ 智能搭积木 """

    def load_configs(self,file_path,log=False):
        """ 加载任务的外部配置参数 """
        # 参数加载
        configs = self.process_configs(file_path,log=log)
        # 参数预处理
        if configs["PLACE_POSITION_Z"] == "PICK_GRASP_Z":
            configs["PLACE_POSITION_Z"] = configs["PICK_GRASP_Z"]
        if configs["PLACE_ROLL"] == "PICK_ROLL":
            configs["PLACE_ROLL"] = configs["PICK_ROLL"]
        if configs["PLACE_PITCH"] == "PICK_PITCH":
            configs["PLACE_PITCH"] = configs["PICK_PITCH"]
        if configs["PLACE_YAW"] == "PICK_YAW":
            configs["PLACE_YAW"] = configs["PICK_YAW"]

        # 参数设置
        self._pick_scan_xyz_dict = {0:[configs["PICK_POSITION_X"],configs["PICK_POSITION_Y"],configs["PICK_POSITION_Z"]],
                                    1:[configs["PICK_POSITION_X"],configs["PICK_POSITION_Y"],configs["PICK_POSITION_Z"]]}
        self._pick_rpy = (configs["PICK_ROLL"],configs["PICK_PITCH"],configs["PICK_YAW"])
        self._pick_base_z = configs["PICK_GRASP_Z"]
        self._place_xy = (configs["PLACE_POSITION_X"],configs["PLACE_POSITION_Y"])
        self._place_rpy = (configs["PLACE_ROLL"],configs["PLACE_PITCH"],configs["PLACE_YAW"])
        self._place_base_z = configs["PLACE_POSITION_Z"]  # place的基础就是pick的基础
        self._cube_height = configs["CUBE_HEIGHT"]
        # 添加预置关节角位姿
        if configs['PICK_JOINT'] == "AUTO":
            self.preset_pose['PickJoint'] = self.change_pose_to_joints(list(self._pick_scan_xyz_dict[0]) + list(self._pick_rpy))
        else:
            if max(np.abs(configs['PICK_JOINT'])) > 2*np.pi:  # 自动检测degree并转换
                self.preset_pose['PickJoint'] = (np.array(configs['PICK_JOINT'])*np.pi/180).tolist()
            else:
                self.preset_pose['PickJoint'] = configs['PICK_JOINT']

    def task_pick_place_param_init(self,area_mode=0,use_tof=False,control_mode='g_i',sim_type=None):
        """ 初始化搭积木任务的参数 """
        # 基本外部参数配置
        self.use_sim = False if sim_type is None else True
        self.loginfo(f'use_sim={self.use_sim}')
        self.sim_type = sim_type
        self.pick_place_area_mode = area_mode
        # 根据load的分量组建完整pose
        self.prepose_pick_scan = tuple(self._pick_scan_xyz_dict[0]) + self._pick_rpy
        self.prepose_pick_base = (*self._pick_scan_xyz_dict[0][:2],self._pick_base_z) + self._pick_rpy
        self.prepose_place_base = (*self._place_xy,self._place_base_z) + self._place_rpy
        # 设置控制模式
        self.set_control_mode(control_mode)
        # 滤波有关变量
        self.average_count = 0
        self.pose_buf = np.zeros((7,20))
        self.wait_near_judge = 0.0005
        self.filter_times = 1  # 滤波次数
        self.filter_near_judge = 10  # 多近开启滤波
        # 确定detect位置
        self.use_tof = use_tof
        if not self.use_sim:  # 实机
            if self.use_tof: self._place_detect_x = self._place_xy[0] - 0.01
            else: self._place_detect_x = self._place_xy[0]
        else:  # 仿真
            self._place_detect_x = self._place_xy[0]  # 仿真和不使用detect时都跟self._place_xy[0]保持一致

        """ 初始化PickPlace任务相关参数"""
        self.__jiumin = 0
        self.approximate = False  # 表示给定的pose是精确的值
        self.new_target_xyz, self.new_target_rpy  = [0.0,0.0,0.0], [0.0,0.0,0.0]  # 初始化新目标存储变量
        self._pick_place_stage = 'pick'  # 首次进入始终意味着阶段初始化为pick
        self.pick_place_0_1 = 0
        self.__auto_pick_place = False
        self._gap_disget = 0.003  # 检测物块搭建的实际高度时的位置裕量
        self._gap_place  = 0.003  # 视觉调整时两个物块之间的上下距离间隙
        # 高度限制
        self._max_z = 0.253 + 0.6  # 机械臂z轴支持的最高高度，实测?
        self._max_cubes_num = int((self._max_z-self._place_base_z)/self._cube_height) + 1  # 计算得到的理论上支持的最多的物块数量
        self.loginfo(f'支持叠放的物块数量最多为：{self._max_cubes_num}')

    def set_control_mode(self,control_mode='g_i'):
        """ 控制模式初始化。注：self.target_base_mode='current'且self.start_base_mode=1不合理 """
        if control_mode in ['traditional','td','t']:  # 实机可用
            self.use_integral = False   # 是否采用偏差积分的方式进行移动
            self.target_base_mode = 'current'  # 即target设定的基准是基于当前状态还是基于上次目标，前者必须要保证偏差量要大于重力影响
            self.start_base_mode  = 0  # 即moveit轨迹规划的起点位置是当前状态0还是上次目标1
        elif control_mode in ['no-gravity_integral','ng_i']:  # 实机可用
            self.use_integral = True
            self.target_base_mode = 'current'
            self.start_base_mode  = 0
        elif control_mode in ['gravity_integral','g_i']:  # SIM可用
            self.use_integral = True
            self.target_base_mode = 'last'
            self.start_base_mode  = 1
        elif control_mode in ['local_integral','l_i']:
            self.use_integral = True
            self.target_base_mode = 'last'
            self.start_base_mode  = 0
        elif control_mode in ['constant_exe','c_e']:
            self.use_integral = True
            self.target_base_mode = 'last'
            self.start_base_mode  = 1
            self.wait_flag = True  # 该模式下必须要求执行完
        if self.use_integral is True:
            self.discrimination:int = 0  # 采用积分方式时，实际上积分判断的函数中已经加入了微小偏差的忽略机制，因此这里可以设置为0

    @staticmethod
    def process_configs(file_path,write=None,log=False):
        """ 读取/写入配置文件 """
        import json
        if write is not None:
            with open(file_path, 'w') as f_obj:
                # 使用函数json.dump()将数字列表存储到文件中
                json.dump(write ,f_obj)
            if log: print('写入配置为：',write)
        else:
            with open(file_path) as f_obj:
                # 使用函数json.load加载存储在number.json中的信息并将其存储到变量number中
                write = json.load(f_obj)
            if log: print('加载配置为：',write)
        return write

    def AutoFollow(self):
        """ 开始自动跟随任务 """
        if not hasattr(self.AutoFollow,'first'):
            self.AutoFollow.__dict__['first'] = False
            self.go_to_pick_pose(True,2)
            # 订阅视觉反馈
            rospy.Subscriber("/target_TF",TransformStamped,self.feedback_callback,queue_size=1)  # queue_size=1表明只接收最新数据
            self.loginfo("Subscribe to /target_TF to receive vision feedback.")
            # 启动跟踪线程
            self._follow_event = Event()  # 跟踪线程的event
            Thread(target=self.__follow,daemon=True).start()
            self.loginfo('Follow Tread Started.')
            # 使能视觉反馈
            rospy.set_param("/vision_attention",'pick'+str(self.pick_place_0_1))
        else: print('请勿重复调用start_auto_follow函数')

    def PickPlace(self,pick_keyboard=False,place_mode=0,start_base=0,use_gripper=True):
        """
            pick_keyboard为真时pick和place均通过键盘控制完成，为假则pick自动完成，place根据mode配置。
            place_mode为0时，采用自动闭环控制；为1时采用自动半闭环控制；为2时采用键盘控制。
            start_base表示初始化时认为已经叠放好的物块个数，默认为0。
        """
        self.__pick_keyboard = pick_keyboard
        self.__start_base = start_base
        self._first_place_satisfy = False
        self._first_pick_satisfy = False
        self._first_out = 0
        self._change_pick_place_state.__dict__['times'] = self.__start_base  # times参数可以用来记录叠放次数，同时也可用于确定place的高度，而其奇偶决定待抓取颜色（该参数实际记录的是从place切到pick的次数）
        self.pick_place_0_1 = self._change_pick_place_state.times % 2
        if start_base >= self._max_cubes_num:
            exit(f"初始值{start_base}应小于上限值{self._max_cubes_num}")
        else: print(f"初始叠放物块数：{start_base}")
        if self.__pick_keyboard:
            rospy.set_param("/vision_attention",'pause')  # 关闭视觉反馈
            if place_mode != 2:
                self.place_mode = 2
                print('pick为键控时place也需为键控，已自动调整')
        else: self.place_mode = place_mode
        """ 开始自动叠放任务 """
        self.__auto_pick_place = True
        # 进入初始化位置
        if use_gripper: self.gripper_control(0,sleep_time=0)  # 确保夹爪开启
        self.AutoFollow()
        Thread(target=self.__pick_place_test,daemon=True,args=(pick_keyboard,place_mode,0.5)).start()

    def _max_deviation(self)->float:
        """ 获得当前6D数据中的最大值，用于进行near_judge """
        self.max_dis = 0.0
        for i in range(3):
            if self.max_dis < abs(self.feedback_target_position[i]):
                self.max_dis = abs(self.feedback_target_position[i])
            if self.max_dis < abs(self.feedback_target_euler[i])/10:
                self.max_dis = abs(self.feedback_target_euler[i])/10
        return self.max_dis # 返回归一化的6D的归一化最大距离

    def _near_judge(self,start_max=0.01)->bool:
        """ start_max以xyz的距离为标准，0.01表示1cm的范畴，对应角度为0.1rad，即5.72度 """
        if self.max_dis < start_max:
            return True
        else: return False

    def _change_feedback_target_to_list(self,relative_target:TransformStamped):
        """ 获得反馈的偏差数据并进行适当转换处理 """
        self.update_current_state()  # 得到当前状态
        self.feedback_target_position = [relative_target.transform.translation.x,relative_target.transform.translation.y,relative_target.transform.translation.z]
        self.feedback_target_euler = list(tf_conversions.transformations.euler_from_quaternion([relative_target.transform.rotation.x,relative_target.transform.rotation.y,
                                                                relative_target.transform.rotation.z,relative_target.transform.rotation.w]))

        # 图像坐标系转换为机械臂末端坐标系
        position_in_robot = [-self.feedback_target_position[2], self.feedback_target_position[1], self.feedback_target_position[0]]
        euler_in_robot = [-self.feedback_target_euler[2], 0, 0]
        # print("target_in_robot:", self.feedback_target_position, self.feedback_target_euler)
        # print("current_pose", self.current_xyz,self.current_rpy)
        world_pose = CoordinateTools.to_world_coordinate((position_in_robot, euler_in_robot),
                                                         (self.current_xyz, self.current_rpy))
        # print('world_pose:', world_pose)
        # 对trans进行额外处理，转换xy为世界坐标系中的坐标
        self.feedback_target_position[0] = world_pose[0][0] - self.current_xyz[0]
        self.feedback_target_position[1] = world_pose[0][1] - self.current_xyz[1]

    def _feedback_moving_smooth(self,nums=1,near_dis=0.001)->bool:
        """ 滑动/移动均值滤波法 """
        self._max_deviation()  # 执行一次这个函数以获得当前的最大偏差
        if self._near_judge(near_dis):
            if(nums>1):
                for i in range(3):
                    self.pose_buf[i][self.average_count] = self.feedback_target_position[i]
                    self.pose_buf[i+3][self.average_count] = self.feedback_target_euler[i]
                self.average_count+=1
                if(self.average_count==nums):
                    self.average_count = 0
                    self.pose_buf[6][0] = 7
                if(self.pose_buf[6][0] == 7):
                    for i in range(3):
                        self.feedback_target_position[i]  = sum(self.pose_buf[i])/nums
                    for i in range(3):
                        self.feedback_target_euler[i] = sum(self.pose_buf[i+3])/nums
                    return True  # 滤波完成
                return False     # 滤波未完成
        else:  # 距离较远或num=1,无需滤波
            self.pose_buf[6][0] = 0
            self.average_count=0
        return True

    def _feedback_average_smooth(self,nums,near_dis=0.001)->bool:
        """ 均值滤波: 在对相机帧率要求不高，甚至需要降低帧率的时候，使用均值滤波以达到更稳定的效果 """
        self._max_deviation()  # 执行一次这个函数以获得当前的最大偏差
        if not hasattr(self._feedback_average_smooth,'times'):
            self._feedback_average_smooth.__dict__['times']=0
            self.__sum = np.zeros((6,nums))
        if self._near_judge(near_dis):
            if nums > 1:
                times = self._feedback_average_smooth.__dict__['times']
                self.__sum[:,times] = np.array(self.feedback_target_position+self.feedback_target_euler)
                self._feedback_average_smooth.__dict__['times'] += 1
                if self._feedback_average_smooth.__dict__['times'] == nums:
                    sum_:np.ndarray = np.sum(self.__sum,axis=1)/nums
                    temp = sum_.tolist()
                    self.feedback_target_position, self.feedback_target_euler = temp[:3],temp[3:6]
                    self._feedback_average_smooth.__dict__['times'] = 0
                    return True
        else: self._feedback_average_smooth.__dict__['times']=0;self.__sum = np.zeros((6,nums))
        return False

    def _set_const_bias_cmd_target(self,bias_xyz=0.0001,neardis_xyz=0.001,min_xyz=0.0002,bias_rpy=0.1,neardis_rpy=1,min_rpy=0.15):
        """
        通过定值的重复执行某个小步长以积分方式慢慢逐步逼近,const_target远距离时设定的值应为30ms左右移动距离，
        而由于一般远距离同时允许连续执行，因此，该值设的稍微大点也无所谓。近距离时，应根据精度要求设定较小的值。
        """
        self._set_const_bias_xyz_target(bias_xyz,neardis_xyz,min_xyz)
        self._set_const_bias_rpy_target(bias_rpy,neardis_rpy,min_rpy)

    def _set_const_bias_xyz_target(self,bias_xyz=0.0001,neardis_xyz=0.001,min_xyz=0.0002):
        abs_trans = [abs(self.feedback_target_position[i]) for i in range(3)]
        for i in range(3):
            if abs_trans[i] <= neardis_xyz:
                if abs_trans[i] > min_xyz:
                    if self.feedback_target_position[i]>0:
                        self.new_target_xyz[i] =  bias_xyz
                    else:
                        self.new_target_xyz[i] = -bias_xyz
                else:
                    self.new_target_xyz[i] = 0

    def _set_const_bias_rpy_target(self,bias_rpy=0.1,neardis_rpy=1,min_rpy=0.15):
        """ 这里rpy输入参数的单位均是deg，方便人的直观配置"""
        min_rpy *= 0.01745
        neardis_rpy *= 0.01745
        bias_rpy *= 0.01745
        abs_angles = [abs(self.feedback_target_euler[i]) for i in range(3)]
        for i in range(3):
            if abs_angles[i] <= neardis_rpy:
                if abs_angles[i] > bias_rpy:  # 0.4度
                    if self.feedback_target_euler[i]>0:
                        self.new_target_rpy[i] =  bias_rpy  # 以0.15度为增量
                    else:
                        self.new_target_rpy[i] = -bias_rpy
                else:
                     self.new_target_rpy[i] = 0

    def _set_pixels2meter_xyz_target(self,k,mindis=4):
        """ 当距离超过mindis时，直接设置xyz目标为像素对应的距离值，从而更快地到达目标点，并且分段代码也将更加简洁 """
        abs_trans = [abs(self.feedback_target_position[i]) for i in range(3)]
        for i in range(3):
            if abs_trans[i] > mindis:
                self.new_target_xyz[i] = self.feedback_target_position[i]/k

    def _feedback_target_dichotomia(self,near_dis=0.0003,dichotomia=2.0,test_log=None):
        """ 偏差N分（默认二分，即Kp=0.5） """
        if self._near_judge(near_dis):
            for i in range(3):
                self.new_target_xyz[i] = self.feedback_target_position[i]/dichotomia
                self.new_target_rpy[i] = self.feedback_target_euler[i]/dichotomia
            if test_log is not None:
                self.loginfo(test_log)
            return True
        return False

    def _clear_meaningless_deviation(self,min_t_error=1,min_p_error=2,precision_whole=1)->bool:  
        """
        按实际可达精度或要求精度对无效的小数位进行四舍五入处理（precision为1代表整体偏差在0.1以下时忽略，以此类推）
        # min_t_error=1,min_p_error=2一般不用再改了，这就是精度的固定水平了
        """
        if precision_whole > 0:
            self.error_abs_sum = 0.0  # 6D偏差的绝对值之和，反映了总体的偏差。单位：m
            for i in range(3):
                self.feedback_target_position[i]  = round(self.feedback_target_position[i],min_t_error+3) # 0.1mm精度，0.0001
                self.feedback_target_euler[i] = round(self.feedback_target_euler[i],min_p_error)  # 0.01rad精度，即0.572度
                self.error_abs_sum += (abs(self.feedback_target_position[i]) + abs(self.feedback_target_euler[i]))
            # 判断清洗后是否所有值均为0（为避免浮点问题，同一个很小的数进行比较）
            if self.error_abs_sum*(10**(min_t_error+3)) < precision_whole:  # 如precision=1,若求和为0.0001m，即0.1mm，则结果为1，返回为True
                self.loginfo("距离目标较近,放弃MoveIt控制!\r\n")
                return False
            return True
        else:  # 精度为0表示不需要进行clear
            self.error_abs_sum = 112233
            return True

    def _auto_vel_acc_limit(self,min_vel_factor=1.0,min_acc_factor=0.1):
        """ 自动速度、加速度限制（系数对应可能得好好调一下） """
        limit_factor =  self._max_deviation()
        if limit_factor > 1:
            limit_factor=1.0
        elif limit_factor < min_acc_factor:
            limit_factor = min_acc_factor
        self.basic_c.set_max_acceleration_scaling_factor(limit_factor) # 加速度限制
        if limit_factor < min_vel_factor:
            limit_factor = min_vel_factor
        self.basic_c.set_max_velocity_scaling_factor(limit_factor)  # 速度限制

    def _manual_vel_acc_limit(self,near_dist=0.005,vel_limit=1.0,acc_limit=0.01):
        """ 手动指定在特定距离下的速度、加速度限制(注意顺序是自上而下、由大至小) """
        if self._near_judge(near_dist):
            self.basic_c.set_max_acceleration_scaling_factor(acc_limit) # 加速度限制
            self.basic_c.set_max_velocity_scaling_factor(vel_limit)  # 速度限制

    def _set_link_vel_acc_maxval(self,vel_max):
        """ 限制link运动的速度（注意是link，而不是joint；无加速度限制接口） """
        for i in range(1,6):
            self.basic_c.limit_max_cartesian_link_speed(vel_max, link_name="link{}".format(i))
        self.basic_c.limit_max_cartesian_link_speed(vel_max, link_name="gripper_link1")
        self.basic_c.limit_max_cartesian_link_speed(vel_max, link_name="gripper_link2")

    def _motion_optimization(self,always_wait=True):
        """ 运动控制优化 """
        if not self.use_sim:  # 实机
            # 分段限速
            self._manual_vel_acc_limit(near_dist=640,vel_limit=0.407,acc_limit=1.0)
            self._manual_vel_acc_limit(near_dist=10, vel_limit=0.2,acc_limit=0.5)
            self._manual_vel_acc_limit(near_dist=5,vel_limit=0.1,acc_limit=0.1)
        else:  # 仿真
            self._manual_vel_acc_limit(near_dist=640,vel_limit=1.0,acc_limit=1.0)
            self._manual_vel_acc_limit(near_dist=320,vel_limit=0.9,acc_limit=0.5)
            self._manual_vel_acc_limit(near_dist=160,vel_limit=0.8,acc_limit=0.2)
            self._manual_vel_acc_limit(near_dist=80, vel_limit=0.5,acc_limit=0.1)
            self._manual_vel_acc_limit(near_dist=20, vel_limit=0.2,acc_limit=0.01)
            self._manual_vel_acc_limit(near_dist=10, vel_limit=0.1,acc_limit=0.001)

    def feedback_callback(self,target_message:TransformStamped):  
        if not hasattr(self.feedback_callback,'running'):
            self.feedback_callback.__dict__['running'] = False
        # 避免函数被重复运行（ROS多个话题发布时，某个订阅者的回调函数将会被开启多次，即便上次的回调函数还没执行完）
        if self.feedback_callback.__dict__['running']: return
        else: self.feedback_callback.__dict__['running'] = True
        self.feedback_callback.__dict__['time'] = time.time()  # 获得当前时间(正常有数据情况下刷新频率约为33ms)，用于执行情况检测
        # 计算获取当前与目标的相对距离
        self._change_feedback_target_to_list(target_message)
        # ******视觉反馈的偏差数据滤波********（原始数据30Hz，33ms）
        if self._feedback_moving_smooth(self.filter_times,self.filter_near_judge) and not self._follow_event.is_set():
            # 当前的最大偏差
            self._max_deviation()
            # 判断并执行
            self.judge_and_excute()
        # 声明函数退出
        self.feedback_callback.__dict__['running'] = False

    def judge_and_excute(self):
        if self.__auto_pick_place:
            if self._pick_place_stage == 'pick':
                if self.sim_type in [None,'isaac','gibson']:
                    self._Pick_and_Go(stable=True,pick_keyboard=self.__pick_keyboard,place_mode=self.place_mode)
                elif self.sim_type in ['gazebo','gazebo_fx']:
                    self._Pick_and_Go(3.7,3.5,stable=True,pick_keyboard=self.__pick_keyboard,place_mode=self.place_mode)
                else: raise Exception('sim_type参数错误')
            if self._pick_place_stage == 'place':  # 不用elif，可以直接当次进入
                if self.sim_type in [None,'isaac','gibson']:
                    self._Place_and_Go(stable=True,pick_keyboard=self.__pick_keyboard,place_mode=self.place_mode)
                elif self.sim_type in ['gazebo','gazebo_fx']:
                    self._Place_and_Go(3.7,3.5,stable=True,pick_keyboard=self.__pick_keyboard,place_mode=self.place_mode)
                else: raise Exception('sim_type参数错误')
        else: self._follow_event.set()  # 仅跟踪，不抓取

    def set_control_param(self,mode=0):
        """ 课程用 """
        self.set_control_param.__dict__['first'] = mode
        self.__jiumin = mode

    def convert_feedback_to_target(self):
        """ 像素坐标系转换机械臂坐标系+细调整 """
        if hasattr(self.set_control_param,'first'):  # 课程用参数动态调整（初始化为正无穷大，则此时理论上机械臂不会进行任何移动）
            pik = rospy.get_param('control_param1', default=math.inf)*100
            plc = rospy.get_param('control_param2', default=math.inf)*100
        else: pik = plc = 0

        # 纯像素新版参数调整（将反馈数据转换为self.new_target_xyz和rpy）
        if self.use_sim:
            if self._pick_place_stage == 'pick':
                if pik != 0:
                    self._set_pixels2meter_xyz_target(k=pik, mindis=0)  # k建议直接实测得到，并根据实际情况进行适当的调整
                else:
                    self._set_pixels2meter_xyz_target(k=5555, mindis=0)  # 粗调。仿真中k建议直接实测得到，并根据实际情况进行适当的调整
                    self._set_const_bias_xyz_target(bias_xyz=0.002, neardis_xyz=20, min_xyz=2)
                    self._set_const_bias_xyz_target(bias_xyz=0.0005,neardis_xyz=10, min_xyz=2)
                    self._set_const_bias_xyz_target(bias_xyz=0.0002,neardis_xyz=5, min_xyz=2)
            else:
                if plc != 0:
                    self._set_pixels2meter_xyz_target(k=plc,mindis=0)  # k建议直接实测得到，并根据实际情况进行适当的调整
                else:
                    self._set_const_bias_cmd_target(bias_xyz=0.015, neardis_xyz=640,min_xyz=2,bias_rpy=10,  neardis_rpy=180,min_rpy=0.2)
                    self._set_const_bias_cmd_target(bias_xyz=0.004, neardis_xyz=100,min_xyz=2,bias_rpy=5,neardis_rpy=15,min_rpy=0.2)
                    self._set_const_bias_cmd_target(bias_xyz=0.002, neardis_xyz=50,min_xyz=2,bias_rpy=2,neardis_rpy=5,min_rpy=0.2)
                    self._set_const_bias_cmd_target(bias_xyz=0.0013, neardis_xyz=20, min_xyz=2, bias_rpy=1,neardis_rpy=2,min_rpy=0.2)
                    self._set_const_bias_cmd_target(bias_xyz=0.0008,neardis_xyz=10, min_xyz=2,bias_rpy=0.5,neardis_rpy=1,min_rpy=0.2)
                    self._set_const_bias_cmd_target(bias_xyz=0.0004,neardis_xyz=5, min_xyz=2,bias_rpy=0.2,neardis_rpy=0.5,min_rpy=0.2)

            self.new_target_rpy = self.feedback_target_euler  # rpy（实际只有y有偏差值）直接一次到目标
        else:  # 实机参数
            if self._pick_place_stage == 'pick':
                if pik == 0:
                    self._set_pixels2meter_xyz_target(k=6000,mindis=20)  # 粗调。仿真中k建议直接实测得到，并根据实际情况进行适当的调整
                    self._set_const_bias_xyz_target(bias_xyz=0.002, neardis_xyz=20, min_xyz=2)
                    self._set_const_bias_xyz_target(bias_xyz=0.0005,neardis_xyz=10, min_xyz=2)
                    self._set_const_bias_xyz_target(bias_xyz=0.0002,neardis_xyz=5, min_xyz=2)
                else:
                    self._set_pixels2meter_xyz_target(k=pik,mindis=0)
            else:
                if plc != 0:
                    self._set_pixels2meter_xyz_target(k=plc,mindis=0)  # k建议直接实测得到，并根据实际情况进行适当的调整
                else:
                    self._set_const_bias_xyz_target(bias_xyz=0.002, neardis_xyz=640, min_xyz=2)
                    self._set_const_bias_xyz_target(bias_xyz=0.001, neardis_xyz=20, min_xyz=2)
                    self._set_const_bias_xyz_target(bias_xyz=0.00025,neardis_xyz=10, min_xyz=2)
                    self._set_const_bias_xyz_target(bias_xyz=0.0001,neardis_xyz=5, min_xyz=2)
            self.new_target_rpy = self.feedback_target_euler  # rpy（实际只有y有偏差值）直接一次到目标
        
        self.set_target_pose(self.new_target_xyz,self.new_target_rpy,target_base='last')

    def __follow(self):
        """ 在单独线程中执行控制执行程序 """
        if not hasattr(self.__follow,'thread'):
            self.__follow.__dict__['thread'] = True
            self.follow_wait = True  # 这个根据自己的情况灵活更改
        while True:
            if self.__follow.thread == False: break  # 退出线程
            # 等待执行
            self._follow_event.wait()
            # 反馈值转换为目标值
            self.convert_feedback_to_target()
            # 优化控制与执行
            self._motion_optimization()
            self.go_to(self.start_base_mode,self.follow_wait,self.approximate,return_enable=True)
            #print(self.target_pose_position,self.target_pose_euler)
            # clear，从而可以刷新pose
            self._follow_event.clear()

    def _Pick_and_Go(self,start_xy=5,start_yaw=0.0175*1,stable=False,pick_keyboard=False,place_mode=0):
        """
            半闭环抓起物块然后移动到待放置的位置的，但不立刻开环放置
        """
        # TODO：目前仍然有一小段开环下降抓取过程，后续可以通过两段参考的方式实现全闭环
        if place_mode: set_vision = False
        else: set_vision = True
        if pick_keyboard:
            print("可以开始手动调节夹爪位置，调节好后按'g'键进入下一阶段")
            while self.key_control(delta_task=0.01) != 'g':pass
            self.gripper_control(1,sleep_time=1)  # 然后发送抓取指令
            self._change_pick_place_state('place',place_mode=not(place_mode),set_vision=set_vision)  # 切换状态
            return
        if self._first_out != 0:
            self._first_out -= 1
        elif (abs(self.feedback_target_position[0])+abs(self.feedback_target_position[1]) < start_xy) and self.feedback_target_euler[2] < start_yaw:  # 当xyroll偏差小于一定像素值时，向下到可以抓取物体的高度
            if stable and not self._first_pick_satisfy:
                self._first_pick_satisfy = True
                rospy.sleep(0.5)  # 冷静，待机械臂稳定
                return
            rospy.set_param("/vision_attention",'pause')  # 关闭视觉反馈
            # self.joints_speed_limit(max_=0.15)  # 限制一下速度减少抖动
            self.go_to_single_axis_target(2,self._pick_base_z,sleep_time=1)  # 首先到达可抓取的高度位置(z单轴移动)
            self.gripper_control(1,sleep_time=1)  # 然后发送抓取指令
            self._change_pick_place_state('place',place_mode=place_mode,set_vision=set_vision)  # 切换状态
            self._first_pick_satisfy = False
        else:
            if self._first_pick_satisfy:
                self._first_pick_satisfy = False
            self._follow_event.set()

    def _Place_and_Go(self,x_bias=2,y_bias=2,stable=False,pick_keyboard=False,place_mode=0):
        """ 达到指定位置后，开始进行视觉闭环调节。条件满足后立刻放置物块 """
        if place_mode == 2:  # 键盘控制
            print("可以开始手动调节夹爪位置，调节好后按'g'键进入下一阶段")
            while self.key_control(delta_task=0.001) != 'g':pass
            self.gripper_control(0,sleep_time=0.5)  # 然后发送释放指令
            self._change_pick_place_state('pick',set_vision=not(pick_keyboard))  # 切换状态
        elif place_mode == 1:  # 开环控制
            self.go_to_shift_single_axis_target(2,-self._gap_place,sleep_time=1)  # 下降到恰好相接触处进行释放
            self.gripper_control(0,sleep_time=0.5)  # 然后发送释放指令
            self._change_pick_place_state('pick',set_vision=not(pick_keyboard))  # 状态切换到pick
        # 闭环控制
        elif (self._change_pick_place_state.times == 0) or (abs(self.feedback_target_position[0]) < x_bias and abs(self.feedback_target_position[1]) < y_bias and self._first_out == 0):
            if self._change_pick_place_state.times == 0: self.loginfo("放置首个物块")  # 首次Place
            else:
                if self.__jiumin == 1:
                    self._first_place_satisfy = True
                    self._follow_event.set()  # 条件不满足，继续调节
                    return
                if stable and not self._first_place_satisfy:
                    self._first_place_satisfy = True
                    rospy.sleep(0.5)  # 冷静，待机械臂稳定
                    return
            rospy.set_param("/vision_attention",'pause')  # 关闭视觉反馈
            # 下降到恰好相接触处进行释放
            if self._change_pick_place_state.times != 0: self.go_to_shift_single_axis_target(2,-self._gap_place,sleep_time=1.5)
            self.gripper_control(0,sleep_time=0.5)
            # 状态切换到pick
            self._change_pick_place_state('pick',set_vision=not(pick_keyboard))
            self._first_place_satisfy = False
        # 每一轮中首次从pick出去并且进入place，并且不是第一轮进入，那么需要刷新偏差值（为保险，刷新3遍）再重新进行判断，故在这里进行回退
        elif self._first_out != 0:
            self._first_out -= 1
            # self.loginfo(f"pick后首次place，刷新3遍视觉反馈：{3-self._first_out}")
        else:
            self._first_place_satisfy = False
            self._follow_event.set()  # 条件不满足，继续调节

    def _change_pick_place_state(self,pick_or_place:str, place_mode=True, set_vision=True):
        """ 每个阶段以夹爪的闭合/开启为结束，然后进行状态的change """
        # 内部状态信息切换
        self._pick_place_stage = pick_or_place
        # 执行动作
        if pick_or_place=='place':  # 进入place位置以备视觉修正
            if self._change_pick_place_state.__dict__['times'] == 0 and place_mode != 2:
                self.go_to_place_pose(first=True)  # 第一次特殊，不经过高度测量以及调节，直接到达待放置位置
            else: self.go_to_place_pose(use_tof=self.use_tof, place_mode=place_mode)
        elif pick_or_place=='pick':  # 进入pick位置开始新一轮pp
            self._change_pick_place_state.__dict__['times'] += 1  # 每次change到pick，意味着新一轮开始了，此时将次数进行累加
            self._place_base_z = self._pick_base_z + self._cube_height * self._change_pick_place_state.times
            self.pick_place_0_1 = self._change_pick_place_state.times % 2 # 刷新01状态
            self.go_to_pick_pose(sleep_time=2,from_place=True,use_name=True)  # 移动到pick区域
            # 如果下次的z轴已经超过最高点，则直接结束（#TODO：后续统一放在debug后的self.monitor_and_change_pick_region()函数中）
            if self._place_base_z + self._gap_disget > self._max_z:
                self.life_end_or_restart(0,info='已达到支持的最高堆叠高度，程序自动退出')
            # # 开启无反馈检测线程
            # self.monitor_and_change_pick_region()
        self._first_out = 3  # 刚出去标志
        if set_vision: rospy.set_param("/vision_attention",self._pick_place_stage+str(self.pick_place_0_1))  # 对2取余判断奇偶，0和1

    def go_to_pick_pose(self,use_name=False,sleep_time=1,from_place=False):
        """ 到达pick位姿 """
        if from_place:
            if self.pick_place_area_mode == 0:
                pose1 = deepcopy(self.last_target_pose.pose)
                pose1.position.z = self.last_xyz[2] + self._cube_height + 0.01  # 先上升
                pose2 = deepcopy(pose1)
                pose2.position.y = self.last_xyz[1] - self._cube_height - 0.01  # 然后向右移动一段距离
                self.set_and_go_to_pose_target(pose1)
                self.set_and_go_to_pose_target(pose2)
            else:  # 实机
                pose1 = deepcopy(self.last_target_pose.pose)
                pose1.position.z = self.last_xyz[2] + self._cube_height + 0.01  # 先上升
                pose2 = deepcopy(pose1)
                pose2.position.y = self.last_xyz[1] - 0.075  # 然后向右移动一段距离
                self.set_and_go_to_pose_target(pose1)
                self.set_and_go_to_pose_target(pose2)

        if use_name:
            if self.pick_place_area_mode == 0:
                self.go_to_named_or_joint_target('PickJoint')
            else:
                self.go_to_named_or_joint_target('PickJoint')
            vel_limit = acc_limit = 0.1
        else: vel_limit = acc_limit = 1.0

        # 精确的工作空间移动
        if self.pick_place_area_mode == 0:
            suc = self.set_and_go_to_pose_target(self._pick_scan_xyz_dict[self.pick_place_0_1],self._pick_rpy,sleep_time=sleep_time,return_enable=True)
            if not suc and not use_name:  # 多次规划失败则尝试先进行named移动
                self.go_to_named_or_joint_target("PickJoint")
                self.set_and_go_to_pose_target(self._pick_scan_xyz_dict[self.pick_place_0_1],self._pick_rpy,sleep_time=sleep_time)
            elif not suc: exit('执行pick姿态失败')
        else:
            suc = self.set_and_go_to_pose_target(self._pick_scan_xyz_dict[self.pick_place_0_1],self._pick_rpy,sleep_time=sleep_time,return_enable=True,vel_limit=vel_limit,acc_limit=acc_limit)
            if not suc and not use_name:  # 多次规划失败则尝试先进行named移动
                self.go_to_named_or_joint_target("PickJoint")
                self.set_and_go_to_pose_target(self._pick_scan_xyz_dict[self.pick_place_0_1],self._pick_rpy,sleep_time=sleep_time)
            elif not suc: exit('执行pick姿态失败')

    def go_to_place_pose(self,first=False,use_tof=True,place_mode=0):
        """ 多种方式到达place位姿 """
        def prepare(first):  # pick区域先上升，然后调整xy为place时的位置
            pose1 = deepcopy(self.last_target_pose.pose)
            if first: pose1.position.z = self.last_xyz[2] + self._cube_height + 0.01
            else: pose1.position.z = disget_z
            pose2 = deepcopy(pose1)
            if self.pick_place_area_mode != 1:  # 反向一波
                pose2.position.x,pose2.position.y = -self._place_xy[0],-self._place_xy[1]
            elif self.pick_place_area_mode == 1:
                pose2.position.x,pose2.position.y = self._place_xy[0],self._place_xy[1]
            self.set_and_go_to_pose_target(pose1)
            self.set_and_go_to_pose_target(pose2)
        if first:
            print("Go to fist place pose.")
            print(self._place_xy, self._place_base_z, self._place_rpy)
            prepare(first)
            if self.pick_place_area_mode != 1: self.go_to_named_or_joint_target({0:-3})  # 中间段保证轨迹单向性
            if self.use_sim:
                self.set_and_go_to_pose_target([*self._place_xy,self._place_base_z+self._gap_place],self._place_rpy,sleep_time=1)
            else:  # 实机由于各种误差因素需要更细致的控制
                # self.speed_control(0.4,0.5)
                self.set_and_go_to_pose_target([self._place_xy[0]+0.01,self._place_xy[1],self._pick_base_z+self._cube_height],self._place_rpy,sleep_time=0.5,vel_limit=0.4,acc_limit=0.2)  # 先保证xy对正，放置卡位造成xy无法到达目标引起偏差
                # self.speed_control(0.4,0.5)
                self.set_and_go_to_pose_target([self._place_xy[0]+0.01,self._place_xy[1],self._place_base_z],self._place_rpy,sleep_time=1,vel_limit=0.2,acc_limit=0.2)
            return
        else:
            disget_z = self._place_base_z + self._gap_disget
            if place_mode == 2: disget_z += 0.025  # 键控时要留够余量
            # 首先在pick的区域抬高到进行高度测量时的高度（这样可以避免移动时碰到其它物块）
            prepare(first)
            # 然后移动到place区域顶层物块的高度测量处（以全过程不碰到顶层物块为宜;由于实机转向的限位，因此分成两段，保证转向始终沿俯视的逆时针方向，这也符合大多数机械臂的实际转动情况）
            print("Go to detect pose.")
            if place_mode == 2:
                self.set_and_go_to_pose_target([*self._place_xy, disget_z], self._place_rpy)
                return
            if use_tof:
                self.set_and_go_to_pose_target([self._place_detect_x,self._place_xy[1],disget_z],self._place_rpy,sleep_time=2,vel_limit=0.4,acc_limit=0.2)
            else:
                if self.sim_type in [None,'isaac','gibson','gazebo_fx', 'gazebo']:
                    print(self._place_xy, disget_z, self._place_rpy)
                    self.set_and_go_to_pose_target([*self._place_xy, disget_z], self._place_rpy,sleep_time=1,vel_limit=0.4,acc_limit=0.2)
                # elif self.sim_type == 'gazebo':
                #     # self.set_and_go_to_pose_target([self._place_xy[0],self._place_xy[1]+0.015*(disget_z/0.02-24),disget_z],self._place_rpy,sleep_time=1,vel_limit=0.4,acc_limit=0.2)
                #     self.set_and_go_to_pose_target([self._place_xy[0]-0.01, self._place_xy[1], disget_z], self._place_rpy,sleep_time=1,vel_limit=0.4,acc_limit=0.2)
                else: raise Exception('sim_type error')
            # 停稳后（一定要保证停稳），获取物块此时的配合值（必然大于0且小于(0.025+self._above_gap_disget)，若大于0.025则上次物块搭建是失败了的，为此整轮应进行回退）
            self.distance_detect(use_sim=self.use_sim, use_tof=use_tof)
            # 然后移动到可以开始调节的位置
            print("Go to adjust pose.")
            adjust_z = disget_z - self.delta_cube_z + self._gap_place
            if self.use_sim:
                if self.sim_type in ['isaac','gibson','gazebo_fx','gazebo']:
                    self.set_and_go_to_pose_target([*self._place_xy, adjust_z],self._place_rpy, sleep_time=2)
                # elif self.sim_type == 'gazebo':
                #     # self.set_and_go_to_pose_target([self._place_xy[0],self._place_xy[1]+0.015*(disget_z/0.02-24),adjust_z],self._place_rpy,sleep_time=2,vel_limit=0.004,acc_limit=0.002)
                #     self.set_and_go_to_pose_target([self._place_xy[0]-0.01,self._place_xy[1], adjust_z],self._place_rpy,sleep_time=2,vel_limit=0.004,acc_limit=0.002)
                # elif self.sim_type == 'gazebo_fx':
                #     self.set_and_go_to_pose_target([self._place_xy[0]-0.01,self._place_xy[1], adjust_z],self._place_rpy,sleep_time=2,vel_limit=0.004,acc_limit=0.002)
                else: raise Exception("sim_type error")
            else:  # 实机当高度增加时，会出现x方向前移的问题，造成下层物块难识别，为此调节位置x方向向后偏移一定距离
                if place_mode == 0:  # 闭环自动（实机经过一些特殊处理）
                    self.set_and_go_to_pose_target([self._place_xy[0],self._place_xy[1],adjust_z],self._place_rpy,sleep_time=2)
                    self.loginfo(f'到达调节位置，开始闭环叠放第{self._change_pick_place_state.times+1}个物块')
                elif place_mode == 1:  # 开环自动
                    self.set_and_go_to_pose_target([self._place_xy[0],self._place_xy[1],adjust_z],self._place_rpy,sleep_time=2)

    def monitor_and_change_pick_region(self,__in_thread=False):
        """
            当一定次数连续没有检测到目标时，认为当前位置已经无目标，移动到新的检测位置
            首次执行时创建线程，并且当前线程未结束时，不会创建新的线程，直到结束后再次执行该函数才会创建新的线程
            并且该函数只有在其自己创建的线程中执行相应功能，其它不执行（也就是说__in_thread参数不要修改其默认值）
        """
        if not hasattr(self.monitor_and_change_pick_region,'times'):
            self.monitor_and_change_pick_region.__dict__['times'] = {0:[0,0],1:[0,0],2:False}  # key 0和1表示self.pick_place_0_1状态，元素两个值分别表示机械臂x轴和y轴的change次数
            self.monitor_and_change_pick_region.__dict__['thread'] = False
            self._pick_region_delta = 0.025
            self.cornor_region = [1,2,self._pick_scan_xyz_dict[0][2]]  # TODO:待确定边角位置（一般右下角，因为x和y增加是朝左上角移动的，具备一致性）
            self.__pick_scan_xyz_dict = deepcopy(self._pick_scan_xyz_dict)

        # 如果下次的z轴已经超过最高点，则直接结束
        if self._place_base_z+self._gap_disget > self._max_z:
            self.life_end_or_restart(0,'已达到支持的最高堆叠高度，程序自动退出')

        # 判断是否有未完成线程
        if self.monitor_and_change_pick_region.thread == False:
            self.monitor_and_change_pick_region.__dict__['thread'] = True
            feedback_time = self.feedback_callback.time
            Thread(target = self.monitor_and_change_pick_region,daemon=True,args=(True,)).start()  # 启动检测线程
        else: self.loginfo('有尚未完成的线程,本次不设置线程')

        if __in_thread:  # 线程中才会执行该部分代码
            self.loginfo('启动可夹取物块存在性监察线程')
            monitor_times=0
            sleep = 0.5  # 检测频率
            time_out = 3  # 容许时间
            func_times:list = self.monitor_and_change_pick_region.times[self.pick_place_0_1]  # 取出当前状态的列表（由于可变对象是引用赋值，所以取出后的改动将连锁到原变量）
            pick_scan_xyz = self._pick_scan_xyz_dict[self.pick_place_0_1]  # 取出当前状态
            base_axis = 1
            another_axis = base_axis^1  # ^不是幂运算，而是异或运算，可以实现0和1的不断求反切换功能
            max_convert_xy = [4,4]  # xy方向总共支持的改变的次数
            while True:  # 线程中循环执行
                rospy.sleep(sleep)
                if self.feedback_callback.time != feedback_time:  # 若发生更新，则清空计数
                    monitor_times = 0
                    feedback_time = self.feedback_callback.time
                else: monitor_times+=1  # 否则计数累加
                # 若夹爪状态为关闭，则表明已经夹到物块了，可以不必检测了，退出线程
                if self.gripper_control.state == 'closed': break
                # 若超时，则表明需要更换位置了
                elif monitor_times*sleep > time_out:
                    # 若任何一个阶段的参考方向到达终点，则回到初始位置，并提醒添加物块，物块添加好后，键盘按下某个键，可以重新开始！
                    if self.monitor_and_change_pick_region.times[2]:
                        print('已无可继续识别的相应颜色的物块，本次搭建结束，回到起点，可重新添加物块后按6键继续进行搭建，按0键重新进行搭建，或者按.键退出程序。')
                        self._pick_scan_xyz_dict = deepcopy(self.__pick_scan_xyz_dict)  # 恢复最初的起点位置
                        self.set_and_go_to_pose_target(pick_scan_xyz,self._pick_rpy,sleep_time=2)
                        while True:
                            key = self.key_get()
                            if key == '6':
                                print('开始继续搭建')
                                return
                            elif key == '0':  # 重新搭建，最简单的就是直接重启一遍程序，但实际上程序难以自动重启，因此采用软重启：清空之前的所有ros内容，然后删除实例变量，然后生成一个新的实例变量
                                self.life_end_or_restart(1,'开始重新搭建')
                                return  # 此时退出后，应该没有任何与该实例相关活动程序了
                            elif key == '.':
                                self.life_end_or_restart(0,'搭建完成进程结束')
                                return
                            else: print('请按正确的按键：6、0或.')
                    self.loginfo('长时间未检测到目标，更换检测区域。')
                    # 首次更换区域，设置右下角作为后续依次change的起点位置  # TODO:后续直接初始位置应该就是边缘位置，或者将整个pick区域通过代号划分成几个部分，然后用户可以通过代号进行选择初始化位置，而后续程序也可以根据代号选择后续调整时跳过该初始化位置的重复检测。
                    if func_times[base_axis] == 0:
                        func_times[base_axis] = 1
                        pick_scan_xyz = self.cornor_region
                    else:  # 后续开始按S型累加移动
                        func_times[base_axis] += 1
                        if func_times[base_axis] > max_convert_xy[base_axis]:  # 到达y方向的pick区域的边界处了，x增加，y不变
                            pick_scan_xyz[another_axis] += self._pick_region_delta
                            func_times[another_axis] += 1  # 另一个轴计数累加
                            func_times[base_axis] = 1  # 重新归1
                            if func_times[another_axis] >= max_convert_xy[base_axis]:
                                self.monitor_and_change_pick_region.times[2] = True
                        else:
                            if func_times[another_axis]%2 == 0:
                                pick_scan_xyz[base_axis] += self._pick_region_delta
                            else: pick_scan_xyz[base_axis] -= self._pick_region_delta
                    # 到达新的位置
                    self.set_and_go_to_pose_target(pick_scan_xyz,self._pick_rpy,sleep_time=2)
                    self._first_out = 2
                    self.loginfo(f'目前共更换了{func_times}次')
                    monitor_times = 0  # 计数清零，继续检测和更换区域，直到成功为止
            # 退出线程前，将thread标志设为false，便于下次再次启用检测
            self.monitor_and_change_pick_region.__dict__['thread'] = False
            self.loginfo('监察完毕，退出可夹取物块存在性监察线程')
        else: self.loginfo("非独立线程执行，本次执行无效")

    def distance_detect(self,use_sim=False,use_tof=False,sleep_time=0):
        """ 距离测量 """
        if sleep_time > 0: rospy.sleep(sleep_time)
        if use_sim:
            if self.sim_type == 'isaac':
                self.delta_cube_z = rospy.get_param('/cube_delta_z')  # 获得最高物块与次高物块的高度差
                self.loginfo('开始测量到顶端物块的实际距离')
            else:
                self.delta_cube_z = self._gap_disget
                use_sim = False
        else:
            if use_tof:
                self.loginfo('开始测量到顶端物块的实际距离')
                tof2cubebottom = 0.01  # 测量得到的tof至夹爪夹取的物块的下底面的距离，可通过简介测量，即测量tof到已放置的物块的距离，减去用卡尺测量到的两物块间隙。
                dis = rospy.get_param('/distance')
                self.delta_cube_z = dis - tof2cubebottom
            else: self.delta_cube_z = self._gap_disget  # 越小越不容易碰到之前的物块- 0.001
        # 有效性判断
        if self.delta_cube_z < 0 or self.delta_cube_z > (self._cube_height+self._gap_disget):
            raise Exception(f'间隙值异常,叠放失败结束。间隙：{self.delta_cube_z}，超出理论边界[0,{0.025+self._gap_disget}]')
        elif use_tof or use_sim: self.loginfo(f'得到两物块间隙高度为：{self.delta_cube_z:.3f}'+'m'+'，开始进行高度调整')

    def __pick_place_test(self,pick=True,place=True,sleep_time=0.5):
        """通过自行发送0target数据进行过程模拟"""
        target_pub = rospy.Publisher("/target_TF",TransformStamped,queue_size=1) # queue_size=1表明只发布最新数据
        target = TransformStamped()
        target.header.stamp = rospy.Time.now()
        target.transform.translation.x = 0
        target.transform.translation.y = 0
        target.transform.translation.z = 0
        tf_q = tf_conversions.transformations.quaternion_from_euler(0,0,0)
        target.transform.rotation.x = tf_q[0]
        target.transform.rotation.y = tf_q[1]
        target.transform.rotation.z = tf_q[2]
        target.transform.rotation.w = tf_q[3]
        while (pick or place):
            if pick and self._pick_place_stage == 'pick':
                target_pub.publish(target)
            if place and self._pick_place_stage == 'place':
                target_pub.publish(target)
            rospy.sleep(sleep_time)


class MagicControlMethods(object):
    """ This is just magic. """
    pass


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser("AIRbotPlay PickPlace param set.")
    parser.add_argument('-r','--use_real',action='store_true',help='use real robot instead of sim')
    parser.add_argument('-np','--not_auto_pick_place',action='store_true',help='do not auto pick and place')
    parser.add_argument('-na','--not_auto',action='store_true',help='do not auto follow')
    parser.add_argument('-cp','--control_param',type=int,default=-1,help='set control params')
    parser.add_argument('-g','--gazebo',action='store_true',help='use gazebo instead of isaac sim')
    parser.add_argument('-gb','--gibson',action='store_true',help='use gibson instead of isaac sim')
    parser.add_argument('-fx','--fixed',action='store_true',help='use gazebo with fixed base')
    parser.add_argument('-gr','--gripper',type=str,default="gripper",help='gripper type')
    args, unknown = parser.parse_known_args()

    NODE_NAME = 'airbot_play_pick_place'
    rospy.init_node(NODE_NAME)
    rospy.loginfo("Initializing {} Node.".format(NODE_NAME))

    sim = not(args.use_real)  # 是否仿真
    gripper = args.gripper  # 夹爪类型
    auto_follow = not(args.not_auto)  # 是否自动
    auto_pick_place = not(args.not_auto_pick_place)  # 是否自动夹取
    sim_type = None if not sim else 'gazebo' if args.gazebo else 'gibson' if args.gibson else 'isaac'  # 仿真类型
    control_param = args.control_param
    print(f'sim_type={sim_type}')

    # 选择夹爪类型
    gripper_sim_type = "gazebo_i" if sim_type == 'gazebo' else sim_type
    gripper_control = ChooseGripper(gripper_sim_type, gripper)()

    # 根据不同夹爪类型初始化机器人体
    gripper_type_0 = {None}
    gripper_type = 4 if sim_type not in gripper_type_0 else 0  # 0表示没有夹爪，4表示有夹爪
    # other_config = None if sim_type != 'gazebo' else ("", "airbot_play_arm")
    other_config = None
    sim_type = 'gazebo_fx' if args.fixed else sim_type  # 最后修改gazebo的fix模式
    print('new_sim_type:',sim_type)
    airbot_player = AIRbotPlayPickPlace(init_pose=None,node_name=NODE_NAME,gripper=(gripper_type, gripper_control),other_config=other_config)

    # 参数配置
    if args.use_real:
        file_path = './configs/control/real.json'
    elif args.gazebo:
        file_path = './configs/control/gazebo.json'
    elif args.gibson:
        file_path = './configs/control/gibson.json'
    else:
        file_path = './configs/control/isaac.json'
    airbot_player.load_configs(file_path)

    # 根据真机还是仿真初始化pick & place 任务
    place_mode_0 = {'isaac'}  # 0表示在一侧pick另一侧place；1表示同侧pick&place
    place_mode = 1 if sim_type not in place_mode_0 else 0
    airbot_player.task_pick_place_param_init(place_mode, sim_type=sim_type)
    # 传入args.control_param参数，则进行参数配置阻塞以及自动夹取控制
    print('args.control_param=', control_param)
    if control_param >= 0: airbot_player.set_control_param(control_param)  # 自调参数（包括pick和place）
    if control_param == 0: auto_pick_place = False  # 不自动夹取（也即相当于仅调节pick阶段）
    # 自动模式配置
    if auto_pick_place: airbot_player.PickPlace(pick_keyboard=False,place_mode=0,start_base=0)
    elif auto_follow: airbot_player.AutoFollow()
    rospy.loginfo("{} Node Ready.".format(NODE_NAME))
    rospy.spin()
