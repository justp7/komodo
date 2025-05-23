#! /usr/bin/env python
from __future__ import print_function
#from builtins import range
import rospy
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan, Joy
from geometry_msgs.msg import Twist, Vector3Stamped
import numpy as np
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Float32
import pandas
from geometry_msgs.msg import WrenchStamped
from scipy.signal import savgol_filter
import os
from datetime import datetime

motor_con = 4

def maprange(a, b, s):
    (a1, a2), (b1, b2) = a, b
    return  b1 + ((s - a1) * (b2 - b1) / (a2 - a1))

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

class Actions:
    def __init__(self):
        self.des_to_pub = Int32MultiArray()
        self.des_cmd = np.array([350, 350, 120, 120], dtype=np.int32)
        self.arm_bucket_pub = rospy.Publisher('/arm/des_cmd', Int32MultiArray, queue_size=10)
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
        self.des_to_pub.data = self.normalize_arm_cmd(arm_cmd, bucket_cmd)
        self.arm_bucket_pub.publish(self.des_to_pub)
        self.vel_pub.publish(self.vel_msg)
        # rospy.loginfo([self.des_to_pub.data , self.vel_msg.linear.x])

    def reset_move(self, cmd):
        self.vel_msg.linear.x = cmd[0]
        arm_cmd = cmd[1]
        bucket_cmd = cmd[2]
        self.des_to_pub.data = self.normalize_arm_cmd(arm_cmd, bucket_cmd)
        self.arm_bucket_pub.publish(self.des_to_pub)
        self.vel_pub.publish(self.vel_msg)

    def normalize_arm_cmd(self,arm_cmd , bucket_cmd):
        des_cmd = np.array([arm_cmd, arm_cmd, bucket_cmd, bucket_cmd])
        des_cmd[:2] = maprange([0.32, -0.1],[300, 780],des_cmd[:2])  # maprange([0.32, -0.2],[380, 780],des_cmd[:2])
        des_cmd[2:] = maprange([-0.5, 0.9], [10, 450],des_cmd[2:])  # maprange([-0.5, 0.548], [20, 450],des_cmd[2:])
        return des_cmd

class torque_listener:
    def __init__(self):
        self.sub = rospy.Subscriber('/arm/calibrated_torque', Float32, self.update_force)
        self.sub_measured = rospy.Subscriber('/arm/measured_force', Float32, self.update__measured_force)
        self.arr = []
        self.measured_force_arr = []
        self.current_path = os.getcwd()


    def update_force(self,  msg):
        torque = msg.data
        self.arr.append(torque)

    def update__measured_force(self,    msg):
        force = msg.data
        self.measured_force_arr.append(force)

    def torque_plot(self):
        self.sub.unregister()
        self.sub_measured.unregister()
        date_time = str(datetime.now().strftime('%d_%m_%Y_%H_%M'))
        np.save(self.current_path + '/data/real/torque_raw_real_robot_' + date_time, self.arr)
        np.save(self.current_path + '/data/real/force_raw_real_robot_' +date_time, self.measured_force_arr)

        import matplotlib.pyplot as plt
        plt.subplot(311)
        w = savgol_filter(self.arr, 501, 2)
        plt.plot(w, 'b-', lw=2)
        plt.subplot(312)
        plt.plot(smooth(self.arr, 19), 'r-', lw=2)
        plt.subplot(313)
        plt.plot(smooth(self.arr, 30), 'g-', lw=2)
        plt.xlabel('time (s)')
        plt.title('Torque over time - Simulation')
        plt.ylabel('Torque (Nm)')
        plt.show()

class Scoop_controller:
    def __init__(self):

        # TODO: Robot information
        self.flag = True
        self.bucket_init_pos = 0
        self.arm_init_pos = 0
        self.vel_init = 0
        self.HALF_KOMODO = 0.53 / 2
        self.particle = np.zeros(1)
        self.x_tip = 0
        self.z_tip = 0
        self.bucket_link_x = 0
        self.bucket_link_z = 0
        self.velocity = 0
        self.wheel_vel = 0
        self.position_from_pile = np.array([0])
        self.last_pos = np.zeros(3)
        self.last_ori = np.zeros(4)
        self.max_limit = np.array([0.2, 0.32, 0.9]) # np.array([0.1, 0.32, 0.548])
        self.min_limit = np.array([-0.2, -0.1, -0.5]) # np.array([-0.1, -0.2, -0.5])

        self.arm_data = np.array([0, 0, 0, 0], dtype=np.float)
        self.fb = np.zeros((motor_con,), dtype=np.int32)
        self.old_fb = np.zeros((motor_con,), dtype=np.int32)
        self.velocity_motor = np.zeros((motor_con,), dtype=np.int32)



        # TODO: RL information
        self.nb_actions = 3  # base , arm , bucket
        self.state_shape = (self.nb_actions * 2 + 6,)
        self.action_shape = (self.nb_actions,)
        self.actions = Actions()
        self.starting_pos = np.array([self.vel_init,self.arm_init_pos, self.bucket_init_pos])
        self.action_range = self.max_limit - self.min_limit
        self.action_mid = (self.max_limit + self.min_limit) / 2.0
        self.last_action = np.zeros(self.nb_actions)
        self.joint_state = np.zeros(self.nb_actions)
        self.joint_pos = self.starting_pos
        self.state = np.zeros(self.state_shape)
        self.last_joint = self.joint_state
        self.observation_arr = self.state


        # TODO: Robot information Subscribers
        self.joint_state_subscriber = rospy.Subscriber('/arm/pot_fb', Int32MultiArray, self.update_fb)
        self.velocity_subscriber = rospy.Subscriber('/mobile_base_controller/odom',Odometry,self.velocity_subscriber_callback)
        self.distance_subscriber = rospy.Subscriber('/scan', LaserScan, self.update_distace)
        self.arm_data_subscriber = rospy.Subscriber('/arm/data', Float32MultiArray, self.update_arm_data)
        self.force_subscriber = rospy.Subscriber('/arm/calibrated_force', Float32, self.update_force)
        self.joy_subscriber = rospy.Subscriber('/joy', Joy, self.manual_control)


    def update_fb(self, data):
        """
        :param data:
        :type data:
        :return:       range      0 - 1023
        :rtype:
        """
        dt = 0.02
        self.old_fb = np.array(self.fb)
        self.fb = np.array(data.data)
        self.velocity_motor = (self.fb - self.old_fb) / dt  # SensorValue per second
        self.joint_state[1] = maprange([300, 780], [0.32, -0.1], self.fb[0]) # maprange([380, 780], [0.32, -0.2], self.fb[0])
        self.joint_state[2] = maprange([10, 450], [-0.5,0.9], self.fb[2]) # maprange([20, 450], [-0.5,0.548], self.fb[2])

    def update_distace(self,msg):
        """
        :param data:
        :type data:
        :return:       middle value of laser sensor
        :rtype:
        """
        self.position_from_pile = np.array([msg.ranges[270]])

    def update_arm_data(self,data):
        self.arm_data = np.array(data.data, dtype=np.float) / 1000

    def update_force(self, data):
        min_w = 0
        max_w = 10
        force = abs(data.data)
        self.particle = np.array([force], dtype=np.float)

    def normalize_joint_state(self,joint_pos):
        joint_coef = 1.0
        return joint_pos * joint_coef

    def velocity_subscriber_callback(self, data):
        vel = data.twist.twist.linear.x
        self.joint_state[0] = vel # fixed velocity
        self.velocity = vel


    def dump_pile(self):
        return 'dumped'

    def reset(self):
        self.joint_pos = self.starting_pos
        self.actions.reset_move(self.starting_pos)
        rospy.sleep(0.5)
        self.state = np.zeros(self.state_shape)
        self.last_joint = self.joint_state
        diff_joint = np.zeros(self.nb_actions)
        normed_js = self.normalize_joint_state(self.joint_state)
        self.particle = np.zeros(1)

        self.state = np.concatenate((self.particle, self.arm_data, self.position_from_pile, normed_js, diff_joint)).reshape(1, -1)
        self.last_action = np.zeros(self.nb_actions)

        return self.state

    def manual_control(self, data):
        """callback for /joy
        * sends commands to the simulation
        Args:
            data (Joy): the joystick's state
            Left stick - right left - data.axes[0]
            Left stick - up down - data.axes[1]
        """
        action = np.array([data.axes[7],data.axes[1],data.axes[4]]) # range -1 to 1
        action[0] = action[0] #* self.action_range[0] * 2
        action[1] = -action[1] * self.action_range[1] * 0.002
        action[2] = -action[2] * self.action_range[2] * 0.001

        # rospy.loginfo(np.round(action, 2))

        self.joint_pos = np.clip(self.joint_pos + action, a_min=self.min_limit, a_max=self.max_limit)

        if abs(action[0]) < 0.001:
            self.joint_pos[0] *= 0

        self.actions.move(self.joint_pos)

        normed_js = self.normalize_joint_state(self.joint_state)

        diff_joint = normed_js - self.last_joint

        self.state = np.concatenate((self.particle, self.arm_data, self.position_from_pile, normed_js, diff_joint)).reshape(1, -1)

        if True:
            self.observation_arr = np.vstack((self.observation_arr, self.state))

        self.last_joint = normed_js
        self.last_action = action

        keys = ["Particle", "X_tip", "Z_tip", "Bucket_x", "Bucket_z", "Distance", "Velocity", "Arm", "Bucket", "Diff_vel", "Diff_arm", "Diff_Bucket"]
        df = pandas.DataFrame((np.round(self.state,2)), columns=keys)
        print(df.to_string(index=False))

        if data.buttons[2] == 1 and self.flag:
            self.flag = False
            np.save( '/home/osher/catkin_ws/src/komodo/komodo2_rl/src/data/obs' , self.observation_arr)
            print("Reset Simulation")
            self.reset()

        return self.state




if __name__ == '__main__':
    try:
        rospy.init_node("manual_control")
        Hz = 50
        rate = rospy.Rate(Hz)
        S = Scoop_controller()
        S.reset()

        while not rospy.is_shutdown():
            rate.sleep()

        # rospy.spin()

    except rospy.ROSInterruptException:
        pass