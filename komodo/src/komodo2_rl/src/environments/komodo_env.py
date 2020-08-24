#! /usr/bin/env python
# coding=utf-8

import rospy
import time
import numpy as np
from std_srvs.srv import Empty
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from gazebo_msgs.srv import SetModelState, SetModelStateRequest, SetModelConfiguration, SetModelConfigurationRequest
from gazebo_msgs.srv import GetModelState, GetModelStateRequest, GetLinkState, GetLinkStateRequest
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, SpawnModelResponse
from control_msgs.msg import JointControllerState
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf import TransformListener
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist, Vector3Stamped, WrenchStamped, PoseStamped, Point, PointStamped

from Spawner import Spawner
from matplotlib import path
import math
import pandas


class Actions:
    def __init__(self):
        self.arm_pos_pub = rospy.Publisher('/arm_position_controller/command', Float64, queue_size=10)
        self.bucket_pos_pub = rospy.Publisher('/bucket_position_controller/command', Float64, queue_size=10)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.vel_msg = Twist()
        # assumption we are moving just in x-axis
        self.vel_msg.linear.y = 0
        self.vel_msg.linear.z = 0
        self.vel_msg.angular.x = 0
        self.vel_msg.angular.y = 0
        self.vel_msg.angular.z = 0

    def move(self, cmd): # cmd [velocity , arm , bucket ]
        self.vel_msg.linear.x = cmd[0]
        arm_cmd = cmd[1]
        bucket_cmd = cmd[2]
        self.arm_pos_pub.publish(arm_cmd)
        self.bucket_pos_pub.publish(bucket_cmd)
        self.vel_pub.publish(self.vel_msg)

    def reset_move(self, cmd):
        self.vel_msg.linear.x = cmd[0]
        self.arm_pos_pub.publish(cmd[1])
        self.bucket_pos_pub.publish(cmd[2])
        self.vel_pub.publish(self.vel_msg)

class Pile:

    def __init__(self):

        self.length = 1
        self.width = 1
        self.height = 1
        self.size = 0.1
        self.radius = 0.035
        self.num_particle = 0
        self.z_max = 0.26
        self.x_min = 0
        self.sand_box_x = 0.35
        self.sand_box_y = 0.301
        self.sand_box_z = 0.0
        self.sand_box_height = 0.25
        self.center_x = self.sand_box_x / 2
        self.center_z = self.sand_box_z / 2
        self.HALF_KOMODO = 0.53 / 2
        self.spawner = Spawner()
        self.spawn_srv = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state',SetModelState)
        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state',GetModelState)


        # Spawn Box
        box_req = self.spawner.create_box_request('sand_box', self.sand_box_x, self.sand_box_y, self.sand_box_z,0.0, 0.0, 0.0)
        
        # self.spawn_srv(box_req)
        try:
            response = self.spawn_srv(box_req)
            if response.success:
                rospy.loginfo("沙盒模型已成功生成!")
            else:
                rospy.logwarn("沙盒模型生成失败: %s", response.status_message)
        except rospy.ServiceException as e:
            rospy.logerr("服务调用失败: %s", e)

        self.pile_box_req = SetModelStateRequest()
        self.pile_box_req.model_state = ModelState()
        self.pile_box_req.model_state.model_name = 'sand_box'
        self.pile_box_req.model_state.pose.position.x = self.sand_box_x
        self.pile_box_req.model_state.pose.position.y = self.sand_box_y
        self.pile_box_req.model_state.pose.position.z = self.sand_box_z
        self.pile_box_req.model_state.pose.orientation.x = 0.0
        self.pile_box_req.model_state.pose.orientation.y = 0.0
        self.pile_box_req.model_state.pose.orientation.z = 0.0
        self.pile_box_req.model_state.pose.orientation.w = 0.0
        self.pile_box_req.model_state.twist.linear.x = 0.0
        self.pile_box_req.model_state.twist.linear.y = 0.0
        self.pile_box_req.model_state.twist.linear.z = 0.0
        self.pile_box_req.model_state.twist.angular.x = 0.0
        self.pile_box_req.model_state.twist.angular.y = 0.0
        self.pile_box_req.model_state.twist.angular.z = 0.0
        self.pile_box_req.model_state.reference_frame = 'world'

    def create_pile(self):

        count = 0
        l = int(self.length/self.size)
        w = int(self.width/self.size)
        h = int(self.height/self.size)
        for k in range(h):
            #w = w - 1
            l = l - 1
            for j in range(-w/2 , w/2): # range(-w/2 + 1, w/2)
                for i in range(0,l):
                    count +=1
                    name = "particle" + str(count)
                    # pos = [i*self.size*0.25 , j*self.size*0.25 , self.radius*(1+2*k) ]
                    pos = [(2*i+1)*self.radius , (2*j+1)*self.radius, self.radius*(1+2*k) ]
                    rot = [0.0, 0.0, 0.0]

                    req = self.spawner.create_sphere_request(name, pos[0], pos[1], pos[2],
                                                                 rot[0], rot[1], rot[2],
                                                                 self.radius)
                    self.spawn_srv(req)

        self.num_particle = count

    def set_pile(self):
        count = 0
        l = int(self.length/self.size)
        w = int(self.width/self.size)
        h = int(self.height/self.size)
        self.model_state_proxy(self.pile_box_req)
        eps = 0.001

        for k in range(h):
            #w = w - 1
            l = l - 1
            for j in range(-w/2, w/2):
                for i in range(0,l):
                    count +=1
                    self.pile_state_req = SetModelStateRequest()
                    self.pile_state_req.model_state = ModelState()
                    self.pile_state_req.model_state.model_name = 'particle'+str(count)
                    # self.pile_state_req.model_state.pose.position.x = i*self.size*0.25
                    # self.pile_state_req.model_state.pose.position.y = j*self.size*0.25
                    # self.pile_state_req.model_state.pose.position.z = self.radius*(1+2*k)
                    self.pile_state_req.model_state.pose.position.x = (2*i+1)*(self.radius+eps)
                    self.pile_state_req.model_state.pose.position.y = (self.radius+ eps)*(1+2*j)
                    self.pile_state_req.model_state.pose.position.z = self.radius*(1+2*k)
                    self.pile_state_req.model_state.pose.orientation.x = 0.0
                    self.pile_state_req.model_state.pose.orientation.y = 0.0
                    self.pile_state_req.model_state.pose.orientation.z = 0.0
                    self.pile_state_req.model_state.pose.orientation.w = 0.0
                    self.pile_state_req.model_state.twist.linear.x = 0.0
                    self.pile_state_req.model_state.twist.linear.y = 0.0
                    self.pile_state_req.model_state.twist.linear.z = 0.0
                    self.pile_state_req.model_state.twist.angular.x = 0.0
                    self.pile_state_req.model_state.twist.angular.y = 0.0
                    self.pile_state_req.model_state.twist.angular.z = 0.0
                    self.pile_state_req.model_state.reference_frame = 'world'
                    self.model_state_proxy(self.pile_state_req)

    def particle_location(self,num_p):
        px_arr = np.zeros(num_p)
        py_arr = np.zeros(num_p)
        pz_arr = np.zeros(num_p)
        for i in range(1, num_p+1):
            get_particle_state_req = GetModelStateRequest()
            get_particle_state_req.model_name = 'particle'+str(i)
            get_particle_state_req.relative_entity_name = 'base_footprint' # 'world'
            particle_state = self.get_model_state_proxy(get_particle_state_req)
            x = abs(particle_state.pose.position.x) + self.HALF_KOMODO
            y = particle_state.pose.position.y
            z = particle_state.pose.position.z
            orientation = particle_state.pose.orientation
            (roll, pitch, theta) = euler_from_quaternion(
                [orientation.x, orientation.y, orientation.z, orientation.w])
            px_arr[i-1] = x
            py_arr[i-1] = y
            pz_arr[i-1] = z
        return px_arr, pz_arr, py_arr

    def in_bucket_2d(self,xq, yq, xv, yv):
        shape = xq.shape
        xq = xq.reshape(-1)
        yq = yq.reshape(-1)
        xv = xv.reshape(-1)
        yv = yv.reshape(-1)
        q = [(xq[i], yq[i]) for i in range(xq.shape[0])]
        p = path.Path([(xv[i], yv[i]) for i in range(xv.shape[0])])
        return p.contains_points(q).reshape(shape)

class KomodoEnvironment:
    def __init__(self):
        """
        初始化Komodo环境类，用于强化学习机器人控制沙盒模拟
        """
        # 初始化ROS节点
        rospy.init_node('RL_Node')
        
        # 沙堆信息
        self.pile = Pile() # (1.75, 2.8, 1.05, 0.34) - 沙堆尺寸参数
        self.pile.length = 1.75   # 沙堆长度
        self.pile.width = 2.8     # 沙堆宽度
        self.pile.height = 1.05   # 沙堆高度
        self.pile.size = 0.34     # 沙粒大小
        self.pile_flag = True     # 沙堆创建标志

        # 机器人信息
        self.bucket_init_pos = 0  # 铲斗初始位置
        self.arm_init_pos = 0     # 机械臂初始位置
        self.vel_init = 0         # 初始速度
        self.HALF_KOMODO = 0.53 / 2   # 机器人半宽
        self.particle = 0             # 沙粒计数
        self.x_tip = 0                # 铲斗尖端x坐标
        self.z_tip = 0                # 铲斗尖端z坐标
        self.bucket_link_x = 0        # 铲斗连接点x坐标
        self.bucket_link_z = 0        # 铲斗连接点z坐标
        self.velocity = 0             # 当前速度
        self.wheel_vel = 0            # 轮子速度
        self.joint_name_lst = ['arm_joint', 'bucket_joint', 'front_left_wheel_joint', 'front_right_wheel_joint',
                              'rear_left_wheel_joint', 'rear_right_wheel_joint']  # 关节名称列表
        self.last_pos = np.zeros(3)       # 上一时刻位置
        self.last_ori = np.zeros(4)       # 上一时刻方向
        self.max_limit = np.array([0.1, 0.32, 0.9])  # 动作上限
        self.min_limit = np.array([-0.1, -0.1, -0.5]) # 动作下限
        self.orientation = np.zeros(3)     # 方向角
        self.angular_vel = np.zeros(3)     # 角速度
        self.linear_acc = np.zeros(3)      # 线性加速度

        # 强化学习(RL)相关信息
        self.nb_actions = 3  # 动作维度: 底盘速度, 机械臂, 铲斗
        self.state_shape = (self.nb_actions * 2 + 6 ,)  # 状态维度
        self.action_shape = (self.nb_actions,)          # 动作维度
        self.actions = Actions()                        # 动作控制器
        self.starting_pos = np.array([self.vel_init, self.arm_init_pos, self.bucket_init_pos])  # 初始位置
        self.action_range = self.max_limit - self.min_limit  # 动作范围
        self.action_mid = (self.max_limit + self.min_limit) / 2.0  # 动作中值
        self.last_action = np.zeros(self.nb_actions)    # 上一个动作
        self.joint_state = np.zeros(self.nb_actions)    # 关节状态
        self.joint_pos = self.starting_pos              # 关节位置
        self.state = np.zeros(self.state_shape)         # 当前状态
        self.reward = 0.0                               # 奖励值
        self.done = False                               # 回合结束标志
        self.reward_done = False                        # 奖励结束标志
        self.reach_target = False                       # 到达目标标志
        self.episode_start_time = 0.0                   # 回合开始时间
        self.max_sim_time = 8.0                         # 最大模拟时间

        # 机器人信息订阅器
        self.joint_state_subscriber = rospy.Subscriber('/joint_states', JointState, self.joint_state_subscriber_callback)  # 关节状态订阅
        self.velocity_subscriber = rospy.Subscriber('/mobile_base_controller/odom', Odometry, self.velocity_subscriber_callback)  # 速度订阅
        self.imu_subscriber = rospy.Subscriber('/IMU', Imu, self.imu_subscriber_callback)  # IMU传感器订阅
        self.reward_publisher = rospy.Publisher('/reward', Float64, queue_size=10)  # 奖励发布器

        # 扭矩相关（已注释，未使用）
        # self.torque_subscriber = rospy.Subscriber('/bucket_torque_sensor',WrenchStamped,self.torque_subscriber_callback)
        # self.controller_state_sub = rospy.Subscriber('/bucket_position_controller/state',JointControllerState,self.controller_subscriber_callback)
        # self.torque_publisher = rospy.Publisher('/t1',Float64,queue_size=10)
        # self.controller_publisher = rospy.Publisher('/t2',Float64,queue_size=10)
        #
        # self.torque_sum = 0
        # self.torque_y = 0
        # self.smooth_command = 0
        # self.ft_out = WrenchStamped()
        # self.ft_out.header.stamp = rospy.Time.now()
        # self.ft_out.wrench.force.x = 0
        # self.ft_out.wrench.force.y = 0
        # self.ft_out.wrench.force.z = 0
        # self.ft_out.wrench.torque.x = 0
        # self.ft_out.wrench.torque.y = 0
        # self.ft_out.wrench.torque.z = 0

        # Gazebo相关服务
        self.pause_proxy = rospy.ServiceProxy('/gazebo/pause_physics', Empty)  # 暂停物理模拟
        self.unpause_proxy = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)  # 恢复物理模拟
        self.reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)  # 重置世界
        self.tfl = TransformListener()  # 坐标变换监听器

        # 模型配置代理
        self.model_config_proxy = rospy.ServiceProxy('/gazebo/set_model_configuration', SetModelConfiguration)
        self.model_config_req = SetModelConfigurationRequest()
        self.model_config_req.model_name = 'komodo2'  # 机器人模型名称
        self.model_config_req.urdf_param_name = 'robot_description'  # 机器人描述参数
        self.model_config_req.joint_names = self.joint_name_lst  # 关节名称列表
        self.model_config_req.joint_positions = self.starting_pos  # 关节初始位置

        # 模型状态代理
        self.model_state_proxy = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
        self.model_state_req = SetModelStateRequest()
        self.model_state_req.model_state = ModelState()
        self.model_state_req.model_state.model_name = 'komodo2'  # 机器人模型名称
        self.model_state_req.model_state.pose.position.x = 1.0  # 初始位置x
        self.model_state_req.model_state.pose.position.y = 0.0  # 初始位置y
        self.model_state_req.model_state.pose.position.z = 0.0  # 初始位置z
        self.model_state_req.model_state.pose.orientation.x = 0.0  # 初始方向四元数x
        self.model_state_req.model_state.pose.orientation.y = 0.0  # 初始方向四元数y
        self.model_state_req.model_state.pose.orientation.z = 0.0  # 初始方向四元数z
        self.model_state_req.model_state.pose.orientation.w = 0.0  # 初始方向四元数w
        self.model_state_req.model_state.twist.linear.x = 0.0  # 初始线速度x
        self.model_state_req.model_state.twist.linear.y = 0.0  # 初始线速度y
        self.model_state_req.model_state.twist.linear.z = 0.0  # 初始线速度z
        self.model_state_req.model_state.twist.angular.x = 0.0  # 初始角速度x
        self.model_state_req.model_state.twist.angular.y = 0.0  # 初始角速度y
        self.model_state_req.model_state.twist.angular.z = 0.0  # 初始角速度z
        self.model_state_req.model_state.reference_frame = 'world'  # 参考坐标系

        # 获取模型状态服务
        self.get_model_state_proxy = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)
        self.get_model_state_req = GetModelStateRequest()
        self.get_model_state_req.model_name = 'komodo2'  # 机器人模型名称
        self.get_model_state_req.relative_entity_name = 'world'  # 参考坐标系

        # 获取连接状态服务
        self.get_link_state_proxy = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        self.get_link_state_req = GetLinkStateRequest()
        self.get_link_state_req.link_name = 'bucket'  # 铲斗连接名称
        self.get_link_state_req.reference_frame = 'world'  # 参考坐标系

    def velocity_subscriber_callback(self, data):
        """
        速度订阅回调函数
        参数:
            data: 速度数据
        """
        vel = data.twist.twist.linear.x
        self.joint_state[0] = vel  # 更新速度状态
        self.velocity = vel

    def calc_torque(self):
        """
        计算扭矩函数
        """
        mass = 0.3  # 沙粒质量

        # 初始化沙粒位置数组
        px_arr = np.zeros(self.pile.num_particle)
        py_arr = np.zeros(self.pile.num_particle)
        pz_arr = np.zeros(self.pile.num_particle)

        # 遍历所有沙粒
        for i in range(1, self.pile.num_particle + 1):
            if self.particle_index[i] == 1:
                # 获取沙粒状态
                get_particle_state_req = GetModelStateRequest()
                get_particle_state_req.model_name = 'particle' + str(i)
                get_particle_state_req.relative_entity_name = 'bucket'  # 相对于铲斗的位置
                particle_state = self.get_model_state_proxy(get_particle_state_req)
                
                # 提取位置和方向信息
                x = particle_state.pose.position.x
                y = particle_state.pose.position.y
                z = particle_state.pose.position.z
                orientation = particle_state.pose.orientation
                (roll, pitch, theta) = euler_from_quaternion(
                    [orientation.x, orientation.y, orientation.z, orientation.w])
                
                # 保存沙粒位置
                px_arr[i - 1] = x
                py_arr[i - 1] = y
                pz_arr[i - 1] = z

                # 计算扭矩总和
                self.torque_sum += mass * 9.80665 * abs(x) * math.sin(roll)

    def controller_subscriber_callback(self, con_in):
        """
        控制器状态订阅回调函数
        参数:
            con_in: 控制器输入数据
        """
        e = 0.99  # 平滑因子
        control_command = con_in.command
        # 平滑控制命令
        self.smooth_command = con_in.command * (1-e) + e * self.smooth_command
        sensor_torque = self.torque_y
        self.controller_publisher.publish(self.smooth_command)

    def torque_subscriber_callback(self, ft_in):
        """
        扭矩传感器回调函数
        参数:
            ft_in: 力和扭矩输入数据
        """
        e = 0.9  # 平滑因子
        # 更新时间戳
        self.ft_out.header.stamp = rospy.Time.now()
        self.ft_out.header.frame_id = ft_in.header.frame_id
        
        # 平滑处理力和扭矩数据
        self.ft_out.wrench.force.x = ft_in.wrench.force.x * (1 - e) + self.ft_out.wrench.force.x * e
        self.ft_out.wrench.force.y = ft_in.wrench.force.y * (1 - e) + self.ft_out.wrench.force.y * e
        self.ft_out.wrench.force.z = ft_in.wrench.force.z * (1 - e) + self.ft_out.wrench.force.z * e
        self.ft_out.wrench.torque.x = ft_in.wrench.torque.x * (1 - e) + self.ft_out.wrench.torque.x * e
        self.ft_out.wrench.torque.y = ft_in.wrench.torque.y * (1 - e) + self.ft_out.wrench.torque.y * e
        self.torque_y = ft_in.wrench.torque.y
        self.ft_out.wrench.torque.z = ft_in.wrench.torque.z * (1 - e) + self.ft_out.wrench.torque.z * e
        
        # 发布处理后的扭矩数据
        self.torque_publisher.publish((-self.ft_out.wrench.torque.y))

    def joint_state_subscriber_callback(self, joint_state):
        """
        关节状态订阅回调函数
        参数:
            joint_state: 关节状态数据
        """
        self.joint_state[1] = joint_state.position[0]  # 更新机械臂关节状态
        self.joint_state[2] = joint_state.position[1]  # 更新铲斗关节状态
        self.wheel_vel = joint_state.velocity[2]       # 更新轮子速度

    def imu_subscriber_callback(self, imu):
        """
        IMU传感器回调函数
        参数:
            imu: IMU传感器数据
        """
        # 从四元数转换为欧拉角
        self.orientation = np.array(euler_from_quaternion([imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w]))
        # 更新角速度和线性加速度
        self.angular_vel = np.array([imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z])
        self.linear_acc = np.array([imu.linear_acceleration.x, imu.linear_acceleration.y, imu.linear_acceleration.z])

    def reset(self):
        """
        重置环境函数
        返回:
            state: 重置后的状态
            done: 回合结束标志
        """
        # 暂停物理模拟
        rospy.wait_for_service('/gazebo/pause_physics')
        try:
            self.pause_proxy()
        except rospy.ServiceException as e:
            print('/gazebo/pause_physics service call failed')

        # 设置机器人模型位置
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.model_state_proxy(self.model_state_req)
        except rospy.ServiceException as e:
            print('/gazebo/set_model_state call failed')

        # 设置机器人关节配置
        rospy.wait_for_service('/gazebo/set_model_configuration')
        try:
            self.model_config_proxy(self.model_config_req)
        except rospy.ServiceException as e:
            print('/gazebo/set_model_configuration call failed')

        # 重置关节位置和动作
        self.joint_pos = self.starting_pos
        self.actions.reset_move(self.starting_pos)

        # 生成沙堆
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            if self.pile_flag:
                self.pile.create_pile()  # 创建新沙堆
                self.pile_flag = False
            else:
                self.pile.set_pile()     # 重置现有沙堆
        except rospy.ServiceException as e:
            print('/gazebo/unpause_physics service call failed')

        # 恢复物理模拟
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except rospy.ServiceException as e:
            print('/gazebo/unpause_physics service call failed')

        rospy.sleep(0.5)  # 等待系统稳定
        
        # 初始化奖励和状态
        self.reward = 0.0
        self.state = np.zeros(self.state_shape)

        # 获取机器人模型状态
        rospy.wait_for_service('/gazebo/get_model_state')
        model_state = self.get_model_state_proxy(self.get_model_state_req)
        pos = np.array([model_state.pose.position.x, model_state.pose.position.y, model_state.pose.position.z])
        
        # 重置标志
        done = False
        self.reward_done = False
        self.reach_target = False

        # 等待链接状态服务
        rospy.wait_for_service('/gazebo/get_link_state')

        # 保存当前状态作为上一状态
        self.last_joint = self.joint_state
        self.last_pos = pos
        self.episode_start_time = rospy.get_time()
        self.last_action = np.zeros(self.nb_actions)

        # 计算状态差分
        diff_joint = np.zeros(self.nb_actions)
        normed_js = self.joint_state

        # 准备状态向量组件
        arm_data = np.array([self.x_tip, self.z_tip, self.bucket_link_x, self.bucket_link_z])
        model_data = np.array([pos[0]])  # 距离
        m = np.array([self.particle * 0.31])
        
        # 组合完整状态向量
        self.state = np.concatenate((m, arm_data, model_data, self.joint_state, diff_joint)).reshape(1, -1)

        return self.state, done

    def check_particle_in_bucket(self):
        """
        检查铲斗中沙粒数量
        返回:
            particle: 铲斗中的沙粒数量
        """
        # 铲斗几何参数
        dp = 0.215  # 铲斗几何尺寸
        d1 = 0.124
        d2 = 0.1
        
        # 获取铲斗状态（相对于机器人底盘）
        self.get_link_state_req.reference_frame = 'base_footprint'
        bucket_state = self.get_link_state_proxy(self.get_link_state_req)

        # 提取铲斗位置和方向
        x = bucket_state.link_state.pose.position.x
        self.bucket_link_x = abs(x) + self.HALF_KOMODO
        y = bucket_state.link_state.pose.position.y
        z = bucket_state.link_state.pose.position.z
        self.bucket_link_z = z
        orientation = bucket_state.link_state.pose.orientation
        (roll, pitch, theta) = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w])

        # 计算铲斗各关键点位置
        self.x_tip = self.bucket_link_x + dp*np.cos(roll)  # 铲斗尖端x坐标
        self.z_tip = z + dp*np.sin(roll)                   # 铲斗尖端z坐标
        x_up = self.bucket_link_x + d1*np.cos(roll + math.radians(46))    # 铲斗上部x坐标
        z_up = z + d1*np.sin(roll + math.radians(46))                     # 铲斗上部z坐标
        x_down = self.bucket_link_x + d2*np.cos(math.radians(41) - roll)  # 铲斗下部x坐标
        z_down = z - d2*np.sin(math.radians(41) - roll)                   # 铲斗下部z坐标

        # 创建铲斗边界多边形点集
        xv = np.array([self.bucket_link_x, x_up, self.x_tip, x_down, self.bucket_link_x])
        zv = np.array([z, z_up, self.z_tip, z_down, z])

        # 获取沙粒位置
        xq, zq, yq = self.pile.particle_location(self.pile.num_particle)
        index = np.where(abs(yq) >= 0.2)
        # TODO: 3D修复
        # xq = np.delete(xq, index)
        # zq = np.delete(zq, index)

        # 检测沙粒是否在铲斗内
        particle_in_bucket = self.pile.in_bucket_2d(xq, zq, xv, zv)
        self.particle = (particle_in_bucket == 1).sum()  # 统计铲斗中沙粒数量
        self.particle_index = np.where(np.array(particle_in_bucket) == 1)

        # 调试信息输出
        rospy.logdebug('BUCKET: x: '+str(round(x, 2)) +' y: '+str(round(y, 3))+' z: '+str(round(z, 2))+ ' x_tip: '+str(round(self.x_tip, 2)))
        rospy.logdebug('BUCKET: roll: '+str(round(roll,2)) +' pitch: '+str(round(pitch, 3))+' theta: '+str(round(theta, 2)))
        rospy.logdebug('xv: '+str(np.round(xv, 2)) + ' zv: ' + str(np.round(zv, 2)))
        rospy.logdebug('xq: '+str(np.round(xq, 2)) + ' zq: ' + str(np.round(zq, 2)))
        rospy.logdebug('Particle in bucket: ' + str(self.particle))

        return self.particle

    def reward_V1(self, pos, particles):
        """
        奖励函数版本1
        参数:
            pos: 当前位置
            particles: 铲斗中沙粒数量
        返回:
            reward_tot: 总奖励值
        """
        # 设置最大沙粒数和位置计算
        max_particle = 6
        bucket_link_x_pile = pos[0] - self.bucket_link_x + self.HALF_KOMODO
        x_tip = (pos[0] - self.x_tip + self.HALF_KOMODO)  # 相对于原点(0,0)的铲斗尖端位置

        bucket_pos = np.array([x_tip, self.z_tip])   # 相对于原点(0,0)的铲斗位置
        min_end_pos = np.array([self.pile.sand_box_x, self.pile.sand_box_height + 0.5])  # 目标位置

        # 计算与目标的距离
        arm_dist = math.sqrt((bucket_pos[0] - (min_end_pos[0] + 0.1))**2 + (bucket_pos[1] - min_end_pos[1])**2)
        loc_dist = math.sqrt((bucket_pos[0] - min_end_pos[0]) ** 2 + self.bucket_link_z**2)

        # 正向奖励计算
        reward_par = 0
        if self.particle:  # 如果铲斗中有沙粒
            # 沙粒数量奖励权重
            w = 1 - (abs(self.particle - max_particle) / max(max_particle, self.particle)) ** 0.4
            # 距离奖励
            reward_dist = (1 - math.tanh(arm_dist) ** 0.4)
            # 沙粒奖励
            reward_par = 0.2 * w
            # 机械臂姿态奖励
            reward_arm = - self.joint_state[2] - self.joint_state[1]
            # 总奖励
            reward_tot = reward_par + reward_arm + reward_dist
        else:  # 如果铲斗中没有沙粒
            # 距离奖励
            reward_dist = 0.25 * (1 - math.tanh(loc_dist) ** 0.4)
            # 机械臂姿态奖励
            reward_arm = -0.5 * self.bucket_link_z
            # 总奖励
            reward_tot = reward_arm + reward_dist

        # 负向奖励（惩罚）
        if (pos[2] > -0.001):  # 如果机器人离地面过近
            reward_tot += -100 * abs(pos[2])
        if (x_tip < 0):  # 如果铲斗尖端超出界限
            reward_tot += -100 * abs(x_tip)

        # 输出各项奖励
        print('Reward dist:    {:0.2f}').format(reward_dist)
        print('Reward par:     {:0.2f}').format(reward_par)
        print('Reward arm:     {:0.2f}').format(reward_arm)

        return reward_tot

    def reward_V2(self, pos, particles):
        """
        奖励函数版本2
        参数:
            pos: 当前位置
            particles: 铲斗中沙粒数量
        返回:
            reward_tot: 总奖励值
        """
        # 计算铲斗位置
        bucket_link_x_pile = pos[0] - self.bucket_link_x + self.HALF_KOMODO
        x_tip = (pos[0] - self.x_tip + self.HALF_KOMODO)  # 相对于原点(0,0)的位置

        b_tip_pos = np.array([x_tip, self.z_tip])   # 铲斗尖端位置
        b_joint_pos = np.array([bucket_link_x_pile, self.bucket_link_z])   # 铲斗连接点位置

        # 目标位置
        min_end_pos = np.array([self.pile.sand_box_x + 0.1, self.pile.sand_box_height + 0.1])

        # 计算距离
        tip_to_target_dist = math.sqrt((b_tip_pos[0] - min_end_pos[0])**2 + (b_tip_pos[1] - min_end_pos[1])**2)
        tip_to_pile_dist = math.sqrt((b_tip_pos[0] - self.pile.sand_box_x) ** 2 + b_tip_pos[1]**2)

        # 正向奖励计算
        reward_par = 0
        if self.particle:  # 如果铲斗中有沙粒
            reward_dist = (1 - math.tanh(tip_to_target_dist) ** 0.4)
            reward_par = 0  # 沙粒奖励
            reward_arm = 0  # 机械臂姿态奖励
            reward_tot = reward_par + reward_arm + reward_dist
        else:  # 如果铲斗中没有沙粒
            reward_dist = (1 - math.tanh(tip_to_pile_dist) ** 0.4)
            reward_arm = 0.5 * (self.joint_state[1]) - 0.5 * self.bucket_link_z
            reward_tot = reward_arm + reward_dist

        # 到达目标时的额外奖励
        eps = 0.05

        if tip_to_target_dist < eps:
            reward_tot += 0.02*self.particle  # 根据铲斗中沙粒数量增加奖励
            self.reach_target = True  # 设置到达目标标志

        #  负向奖励（惩罚）:
        if pos[2] > -0.0004 or pos[0] > 1.1:  # 如果机器人离地面过近或超出区域边界
            reward_tot = -1  # 给予负奖励
            self.reward_done = True  # 设置回合结束标志
        if (self.x_tip < 0):  # 如果铲斗尖端超出边界
            reward_tot = -1  # 给予负奖励
            self.reward_done = True  # 设置回合结束标志

        return reward_tot

    def reward_V3(self, pos, particles):
        """
        奖励函数版本3
        参数:
            pos: 当前位置
            particles: 铲斗中沙粒数量
        返回:
            reward_tot: 总奖励值
        """
        # 计算铲斗位置
        bucket_link_x_pile = pos[0] - self.bucket_link_x + self.HALF_KOMODO
        x_tip = (pos[0] - self.x_tip + self.HALF_KOMODO)  # 相对于原点(0,0)的位置

        b_tip_pos = np.array([x_tip, self.z_tip])  # 铲斗尖端位置
        b_joint_pos = np.array([bucket_link_x_pile, self.bucket_link_z])  # 铲斗连接点位置

        # 目标位置
        min_end_pos = np.array([self.pile.sand_box_x + 0.1, self.pile.sand_box_height + 0.1])

        # 计算距离
        tip_to_target_dist = math.sqrt((b_tip_pos[0] - min_end_pos[0])**2 + (b_tip_pos[1] - min_end_pos[1])**2)
        tip_to_pile_dist = math.sqrt((b_tip_pos[0] - self.pile.sand_box_x) ** 2 + b_tip_pos[1]**2)

        # 正向奖励计算
        reward_par = 0
        if self.particle:  # 如果铲斗中有沙粒
            reward_dist = (1 - math.tanh(tip_to_target_dist) ** 0.4)  # 距离奖励
            reward_par = 0  # 沙粒奖励
            reward_arm = 0  # 机械臂姿态奖励
            reward_tot = reward_par + reward_arm + reward_dist
        else:  # 如果铲斗中没有沙粒
            reward_dist = (1 - math.tanh(tip_to_pile_dist) ** 0.4)  # 距离奖励
            reward_arm = 0.5 * (self.joint_state[1]) - 0.5 * self.bucket_link_z  # 机械臂姿态奖励
            reward_tot = reward_arm + reward_dist

        # 到达目标时的额外奖励
        eps = 0.05
        if tip_to_target_dist < eps:
            reward_tot += 0.02 * self.particle  # 根据铲斗中沙粒数量增加奖励
            self.reach_target = True  # 设置到达目标标志

        return reward_tot

    def normalize_joint_state(self, joint_pos):
        """
        规范化关节状态
        参数:
            joint_pos: 关节位置
        返回:
            规范化后的关节位置
        """
        return joint_pos * 1  # 简单乘以1，实际上没有改变值

    def step(self, action):
        """
        环境步进函数，执行动作并返回新状态、奖励和完成标志
        参数:
            action: 动作向量 [速度, 机械臂, 铲斗]
        返回:
            state: 新状态
            reward: 奖励
            done: 回合结束标志
        """
        # 设置输出精度
        np.set_printoptions(precision=1)

        # 输出当前状态
        keys = ["par", "X_tip", "Z_tip", "Bucket_x", "Bucket_z", "Distance", "Velocity", "Arm", "Bucket", "Diff_vel", "Diff_arm", "Diff_Bucket"]
        df = pandas.DataFrame((np.round(self.state, 2)), columns=keys)
        print(df.to_string(index=False))

        # 输出动作信息
        print('Action : {:0.2f}     {:0.2f}     {:0.2f}').format(action[0], action[1], action[2])  # 动作: [速度, 机械臂, 铲斗]

        # 将动作缩放到实际范围并应用
        action = action * self.action_range
        self.joint_pos = np.clip(self.joint_pos + action, a_min=self.min_limit, a_max=self.max_limit)
        self.actions.move(self.joint_pos)

        # 等待一小段时间让动作生效
        rospy.sleep(15.0/60.0)
        
        # 获取机器人当前状态
        rospy.wait_for_service('/gazebo/get_model_state')
        model_state = self.get_model_state_proxy(self.get_model_state_req)
        pos = np.array([model_state.pose.position.x, model_state.pose.position.y, model_state.pose.position.z])

        # 检查铲斗中沙粒数量
        p_in_bucket = self.check_particle_in_bucket()

        # 计算奖励
        self.reward = self.reward_V2(pos, p_in_bucket)

        # 更新状态信息
        curr_joint = self.normalize_joint_state(self.joint_state)
        diff_joint = (curr_joint - self.last_joint)  # 关节状态变化

        arm_data = np.array([self.x_tip, self.z_tip, self.bucket_link_x, self.bucket_link_z])  # 机械臂数据
        model_data = np.array([pos[0]])  # 位置数据
        m = np.array([self.particle * 0.31])  # 沙粒数据

        # 组合完整状态向量
        self.state = np.concatenate((m, arm_data, model_data, self.joint_state, diff_joint)).reshape(1, -1)

        # 保存当前状态为上一状态
        self.last_joint = curr_joint
        self.last_pos = pos
        self.last_action = action

        # 获取当前时间
        curr_time = rospy.get_time()

        # 发布奖励信息
        self.reward_publisher.publish(self.reward)

        # 检查是否需要结束回合
        if (curr_time - self.episode_start_time) > self.max_sim_time or self.reward_done:
            self.done = True
            self.reset()  # 重置环境
        else:
            self.done = False

        # 限制奖励范围
        self.reward = np.clip(self.reward, a_min=-20, a_max=20)

        return self.state, self.reward, self.done