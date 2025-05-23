from __future__ import print_function
from environments.komodo_env import KomodoEnvironment
from agents.ddpg import DDPG
from agents.a2c import A2C

from torque_listener import TorqueListener
import numpy as np
import os
from datetime import datetime
import rospy
import matplotlib.pyplot as plt

save = 0
current_path = os.getcwd()
t_listener = TorqueListener()
env = KomodoEnvironment()
state_shape = env.state_shape
action_shape = env.action_shape

model = 'ddpg'
if model == 'ddpg':
    agent = DDPG(state_shape,action_shape,batch_size=128,gamma=0.995,tau=0.001,
                                            actor_lr=0.0005, critic_lr=0.001, use_layer_norm=True)
    print('DDPG agent configured')
    agent.load_model(agent.current_path + '/model/model.ckpt')
elif model == 'a2c':
    agent = A2C(state_shape,action_shape,gamma=0.995,actor_lr=0.0002, critic_lr=0.001, use_layer_norm=True)
    print('A2C agent configured')
    agent.load_model(agent.current_path + '/model_a2c/model.ckpt')

max_episode = 100
particle_arr = np.array([1])
time_arr = np.array([1])

for i in range(max_episode):
    print('---------------------------env reset---------------------------------------------------------')
    observation, done = env.reset()
    action = agent.act_without_noise(observation)
    observation, reward, done = env.step(action)
    step_num = 0
    observation_arr = observation
    action_arr = action
    flag = 1
    while done == False:
        step_num += 1
        action = agent.act_without_noise(observation)
        observation, reward, done = env.step(action)
        observation_arr = np.vstack((observation_arr, observation))
        action_arr= np.vstack((action_arr, action))
        print('Reward:', round(reward,3), 'Episode:', i, 'Step:', step_num)
        print('------------------------------------------------------------------------------------------')
        if observation[0,5] < 0 and observation[0,3] > 0.1 and flag:
            particle_arr = np.vstack((particle_arr, observation[0,0]))  # amount of particle at the end
            time_arr = np.vstack((time_arr, step_num))  # time elapsed from episode start
            flag = 0
    if i == 0 :
        full_observation_arr = observation_arr
    else:
        full_observation_arr = np.vstack((full_observation_arr, observation_arr))

keys = ["X_tip", "Z_tip", "Bucket_x", "Bucket_z", "Distance", "Velocity", "Arm", "Bucket", "Diff_vel", "Diff_arm",  "Diff_Bucket"]

# t_listener.force_plot()
# plt.plot(observation_arr)
# plt.legend(keys)
# plt.show()

if save:
    date_time =  str(datetime.now().strftime('%d_%m_%Y_%H_%M'))
    np.save(current_path + '/data/sim/particle_end'+ date_time, particle_arr)
    np.save(current_path + '/data/sim/time_end'+ date_time, time_arr)
    np.save(current_path + '/data/sim/full_observation' + date_time, full_observation_arr)
    np.save(current_path + '/data/sim/observation_' + date_time,observation_arr)
    np.save(current_path + '/data/sim/action_' + date_time,action_arr)
    t_listener.force_plot()
