#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from environments.komodo_env import KomodoEnvironment
from agents.ddpg import OUNoise, DDPG
from agents.a2c import A2C
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

HALF_KOMODO = 0.53 / 2
np.set_printoptions(precision=1)
current_path = os.getcwd()

def train_agent(seed=0, model='a2c', max_episode=100):
    """
    训练单个智能体并返回结果
    
    参数:
    seed -- 随机种子
    model -- 使用的模型类型 ('ddpg' 或 'a2c')
    max_episode -- 最大训练回合数
    
    返回:
    tot_rewards -- 每个回合的总奖励
    particle_arr -- 每个回合结束时的粒子数量
    time_arr -- 每个回合的步数
    """
    np.random.seed(seed)
    
    env = KomodoEnvironment()
    state_shape = env.state_shape
    action_shape = env.action_shape
    
    if model == 'ddpg':
        agent = DDPG(state_shape, action_shape, batch_size=128, gamma=0.995, tau=0.001,
                     actor_lr=0.0001, critic_lr=0.001, use_layer_norm=True)
        print('DDPG agent configured with seed {0}'.format(seed))
    elif model == 'a2c':
        agent = A2C(state_shape, action_shape, gamma=0.995, actor_lr=0.0001, 
                    critic_lr=0.001, use_layer_norm=True)
        print('A2C agent configured with seed {0}'.format(seed))
    
    tot_rewards = []
    observation, done = env.reset()
    action = agent.act(observation)
    observation, reward, done = env.step(action)
    noise_sigma = 0.15
    save_cutoff = 1
    cutoff_count = 0
    save_count = 0
    curr_highest_eps_reward = -1000.0
    
    particle_arr = np.array([1])
    time_arr = np.array([1])
    
    for i in range(max_episode):
        if i % 100 == 0 and noise_sigma > 0.03 and model == 'ddpg':
            agent.noise = OUNoise(agent.num_actions, sigma=noise_sigma)
            noise_sigma /= 2.0
        
        step_num = 0
        flag = 1
        
        while done == False:
            step_num += 1
            action = agent.step(observation, reward, done)
            observation, reward, done = env.step(action)
            
            if i % 10 == 0:  # 减少输出频率，只在每10个回合打印
                print('[Seed {0}] reward: {1}, episode: {2}, step: {3}, highest reward: {4}, saved: {5}, cutoff count: {6}'.format(
                    seed, round(reward,3), i, step_num, round(curr_highest_eps_reward, 3), save_count, cutoff_count))
                print('\n-----------------------------------------------------------------------------------------------------\n')
            
            if env.reach_target and flag:
                particle = observation[0,0]  # amount of particle at the end
                timer = step_num  # time elapsed from episode start
                if i % 10 == 0:
                    print('[Seed {0}] Particle: {1}, Step: {2}'.format(
                        seed, round(observation[0,0], 3), step_num))
                flag = 0
        
        action, eps_reward = agent.step(observation, reward, done)
        tot_rewards.append(eps_reward)
        
        if eps_reward > curr_highest_eps_reward:
            cutoff_count += 1
            curr_highest_eps_reward = eps_reward
        
        if cutoff_count >= save_cutoff:
            save_count += 1
            agent.save_model()
            agent.save_memory()
            cutoff_count = 0
        
        if flag:
            particle_arr = np.vstack((particle_arr, 0))  # amount of particle at the end
            time_arr = np.vstack((time_arr, 20))  # time elapsed from episode start
        else:
            particle_arr = np.vstack((particle_arr, particle))  # amount of particle at the end
            time_arr = np.vstack((time_arr, timer))  # time elapsed from episode start
        
        observation, done = env.reset()
    
    return np.array(tot_rewards), particle_arr, time_arr

# 主程序
def main():
    # 配置参数
    model = 'a2c'  # 可选 'ddpg' 或 'a2c'
    max_episode = 10000
    num_seeds = 5  # 使用5个不同的随机种子运行实验
    save = True
    
    # 存储多次实验的结果
    all_rewards = []
    all_particles = []
    all_times = []
    
    # 进行多次实验
    for seed in range(num_seeds):
        print("\n=== 开始实验 #{0} (随机种子: {1}) ===\n".format(seed+1, seed))
        rewards, particles, times = train_agent(seed=seed, model=model, max_episode=max_episode)
        all_rewards.append(rewards)
        all_particles.append(particles)
        all_times.append(times)
    
    # 将所有实验结果转换为numpy数组
    all_rewards = np.array(all_rewards)
    all_particles = np.array(all_particles)
    all_times = np.array(all_times)
    
    # 保存结果
    if save:
        date_time = str(datetime.now().strftime('%d_%m_%Y_%H_%M'))
        
        # 确保保存目录存在
        save_dir = current_path + '/data/sim/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 保存每个实验的原始数据
        np.save(save_dir + 'all_rewards_' + model + '_' + date_time, all_rewards)
        np.save(save_dir + 'all_particles_' + model + '_' + date_time, all_particles)
        np.save(save_dir + 'all_times_' + model + '_' + date_time, all_times)
    
    # 计算均值和95%置信区间
    rewards_mean = np.mean(all_rewards, axis=0)
    rewards_std = np.std(all_rewards, axis=0)
    rewards_upper = rewards_mean + 1.96 * rewards_std / np.sqrt(num_seeds)  # 95% 置信区间上界
    rewards_lower = rewards_mean - 1.96 * rewards_std / np.sqrt(num_seeds)  # 95% 置信区间下界
    
    # 绘制论文风格的图表
    plot_paper_style(rewards_mean, rewards_upper, rewards_lower, model, save, date_time, save_dir)

def plot_paper_style(mean, upper, lower, model_name, save, date_time, save_dir):
    """绘制论文风格的训练曲线图"""
    plt.figure(figsize=(10, 6))
    
    # 绘制均值曲线
    episodes = np.arange(1, len(mean) + 1)
    plt.plot(episodes, mean, color='#1f77b4', linewidth=2, label=model_name.upper())
    
    # 绘制置信区间
    plt.fill_between(episodes, lower, upper, alpha=0.3, facecolor='#1f77b4')
    
    # 设置图表样式
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Episode', fontsize=14)
    plt.ylabel('Total Reward', fontsize=14)
    plt.title('Training Curve with 95% Confidence Interval', fontsize=16)
    
    # 添加图例
    plt.legend(loc='lower right', fontsize=12)
    
    # 设置坐标轴样式
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # 添加描述性文本
    plt.annotate('The mean across 5 seeds is plotted and the 95% confidence interval is shown shaded.',
                xy=(0.5, 0.01), xycoords='figure fraction', ha='center', fontsize=10)
    
    # 保存图表
    if save:
        plt.tight_layout()
        plt.savefig(save_dir + 'training_curve_' + model_name + '_' + date_time + '.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_dir + 'training_curve_' + model_name + '_' + date_time + '.pdf', format='pdf', bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    main()