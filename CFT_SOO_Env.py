# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:09:27 2023

@author: User
"""

import csv
from CFT_Null import CFT_Null
from ising_viol_20 import ising_viol_20
import gym
from gym import spaces
import numpy as np
from Where_execute import Where_execute


class CFT_SOO_Env(gym.Env):
    '''
    parameters: 
        scale: int
            A huge number to enlarge the punishment of agent.
        step_size: float
            The biggest length of action. (Can be tunned during training)
        ep_steps: int
            How many steps in an episode.
        episodes: int
            How many episodes in this training process.
        neurons: int
            How many neurons. 
        (The last parameter is which I want to test and do hyperparameter tunning.)
    '''
    metadata = {'render.modes': ['console']}

    def __init__(self,
                 scale, step_size, ep_steps, episodes, neurons, retrain_number, Which_pt
                 ):
        super(CFT_SOO_Env, self).__init__()
        print(f'Neurons:{neurons}_train start')
        self.execution = 'my_computer'
        self.neurons = neurons
        self.retrain_number = retrain_number
        self.delta_scale = np.pi/2
        self.ep_best_reward = 0
        self.ep_worst_reward = 0

        self.spin_list = np.array([0., 0., 2., 4., 6.])
        # self.delta_theta_list = np.random.rand(5)
        # self.delta_list = self.spin_list + (6.5-self.spin_list)*np.sin(self.delta_theta_list*np.pi/2)**2
        # self.ope_c_list = np.random.rand(5)

        self.delta_theta_list, self.ope_c_list = CFT_Null().Discretize(self.retrain_number)
        self.delta_list = self.spin_list + (6.5-self.spin_list)*np.sin(self.delta_theta_list*np.pi/2)**2
        
        self.ex_delta = 0.125
        
        self.ep_best_obs = np.concatenate((self.delta_list, self.ope_c_list))
        
        self.n = len(self.spin_list)
        
        nx = 20
        self.Which_pt = Which_pt
        self.z_list = CFT_Null().generate_grid_pts(nx)[self.Which_pt]

        self.err_analyitc_list = ising_viol_20() if nx == 20 else CFT_Null().ising_viol(nx)
        
        self.scale = scale
        self.step_size = step_size
        self.step_num = 0
        
        self.episodes = episodes
        self.ep_counting = 0
        self.ep_steps = ep_steps

        self.Reward_collector = []

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(4 * self.n ,), dtype=np.float32)
        # Feed the agent with 'spin_list', 'delta_list', 'ope_c_list', 'reward_list'
        # -> (3n+nz) dimensions
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(3 * self.n,), dtype=np.float32)
    
    def cft_soo_viol(self, h_list, hb_list, c_list, z):
        import torch
        zb = torch.resolve_conj(z).numpy().conj()
            
        e1=0
        for h, hb, c in zip(h_list, hb_list, c_list):
            f1 = (torch.abs(z-1) ** (2*self.ex_delta)) * CFT_Null().g(h, hb, z, zb)
            f2 = (torch.abs(z) ** (2*self.ex_delta)) * CFT_Null().g(h, hb, 1-z, 1-zb)
            e1 += c * (f1 - f2)
            
        f3 = (torch.abs(z - 1) ** (2 * self.ex_delta))
        f4 = (torch.abs(z) ** (2 * self.ex_delta))
        e1 += (f3 - f4)
        return abs(e1)
    
    def Single_Ising_err(self):
        return self.err_analyitc_list[self.Which_pt]
    
    def calc_reward(self, err_list):
        return - self.scale * (err_list - abs(self.Single_Ising_err()))#.numpy()
    
    @staticmethod
    def reflected_bc(ope_c_list):
        for i in range(len(ope_c_list)):
            if ope_c_list[i] < 0:
                ope_c_list[i] = -ope_c_list[i]

        return ope_c_list
    
    def Step_size_tunning(self, reward, ep_best_reward, ep_worst_reward):
        # Extra reward
        if reward > ep_best_reward:
            ep_best_reward = reward
            self.ep_best_obs = np.concatenate((self.delta_list, self.ope_c_list))
            reward /= 10**5
        # # Extra punishment
        # if reward < ep_worst_reward:
        #     ep_worst_reward = reward
        #     self.step_size *= 1.05
        #     reward *= 10**2
        return reward, ep_best_reward, ep_worst_reward
        
    def step(self, action):
        # action = [delta_action, delta_step_size, ope_action, ope_step_size]
        self.delta_theta_list = self.delta_theta_list + action[self.n:2*self.n] * action[0:self.n]
        self.ope_c_list = self.ope_c_list + action[3*self.n:4*self.n] * action[2*self.n:3*self.n]
        self.delta_list = self.spin_list + (6.5-self.spin_list)*np.sin(self.delta_theta_list*np.pi/2)**2
        self.ope_c_list = self.reflected_bc(self.ope_c_list)

        self.step_num += 1

        h_list = 0.5 * (self.delta_list + self.spin_list)
        hb_list = 0.5 * (self.delta_list - self.spin_list)
        
        err_list = self.cft_soo_viol(h_list, hb_list, self.ope_c_list, self.z_list)

        reward = float(self.calc_reward(err_list))

        obs = np.concatenate(
                (self.spin_list/6, self.delta_list/6.5, self.ope_c_list))

        done = False

        self.Reward_collector.append(reward/self.scale)

        reward, self.ep_best_reward, self.ep_worst_reward = self.Step_size_tunning(reward, self.ep_best_reward, self.ep_worst_reward)
        #print('step_size: ',self.step_size)
        
        # If reward cross 0, then multiply it a huge hug a big reward.
        if reward > 0:
            reward *= 10 ** 5
            #done = True

        if self.step_num == self.ep_steps:
            print('sofar_best: ', self.ep_best_reward/self.scale)
            print('best obs', self.ep_best_obs)

            done = True
            with open(Where_execute(self.execution)[0] +f'Rewards_Neuron_{self.neurons}_Train.csv', 'ab') as file:
                np.savetxt(file, self.Reward_collector)
                
            file = open(f'Best_with_maxreward_Neuron:{self.neurons}_Train.csv',mode='a')
            writer = csv.DictWriter(file, ['best_reward','Best_obs'])
            writer.writerow({'best_reward':self.ep_best_reward/self.scale,'Best_obs':list(self.ep_best_obs)})
            file.close()
            self.ep_counting += 1
            print('episodes', self.ep_counting)
        info = {}

        return obs, reward, done, info

    def reset(self):
        self.step_num = 0
        self.spin_list = self.spin_list
        self.delta_theta_list = np.random.rand(5)
        self.delta_list = self.spin_list + (6.5-self.spin_list)*np.sin(self.delta_theta_list*np.pi/2)**2
        self.ope_c_list = np.random.rand(5)
        
        # self.delta_theta_list, self.ope_c_list = CFT_Null().Discretize(self.ep_counting *(1+ self.retrain_number))
        # self.delta_list = self.spin_list + (6.5-self.spin_list)*np.sin(self.delta_theta_list*np.pi/2)**2
        
        self.Reward_collector = []
        self.step_size = 1e-2
        
        h_list = 0.5 * (self.delta_list + self.spin_list)
        hb_list = 0.5 * (self.delta_list - self.spin_list)
        err_list = self.cft_soo_viol(h_list, hb_list, self.ope_c_list, self.z_list)
        reward = self.calc_reward(err_list)
        self.ep_best_reward = float(reward)
        self.ep_best_obs = np.concatenate((self.delta_list, self.ope_c_list))

        obs = np.concatenate(
                (self.spin_list/6, self.delta_list/6.5, self.ope_c_list))

        return obs