import csv
from CFT_Null import CFT_Null
from ising_viol_20 import ising_viol_20
import gym
from gym import spaces
import numpy as np



class CFT_Env(gym.Env):
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
                 scale, step_size, ep_steps, episodes, neurons
                 ):
        super(CFT_Env, self).__init__()
        print(f'Neurons:{neurons}_train start')
        self.neurons = neurons
        
        self.delta_scale = np.pi/2
        self.ep_best_reward = 0
        self.ep_worst_reward = 0

        self.spin_list = np.array([0., 0., 2., 4., 6.])
        # self.delta_theta_list = np.random.rand(5)
        # self.delta_list = self.spin_list + (6.5-self.spin_list)*np.sin(self.delta_theta_list*np.pi/2)**2
        # self.ope_c_list = np.random.rand(5)

        self.delta_theta_list, self.ope_c_list = CFT_Null().Discretize()
        self.delta_list = self.spin_list + (6.5-self.spin_list)*np.sin(self.delta_theta_list*np.pi/2)**2
        
        self.ex_delta = 0.125
        
        self.ep_best_obs = np.concatenate((self.delta_list, self.ope_c_list))
        
        self.n = len(self.spin_list)
        
        nx = 20
        self.z_list = CFT_Null().generate_grid_pts(nx)
        
        self.nz = len(self.z_list)

        self.err_analyitc_list = ising_viol_20() if nx == 20 else CFT_Null().ising_viol(nx)
        
        self.scale = scale
        self.step_size = step_size
        self.step_num = 0
        
        self.episodes = episodes
        self.ep_counting = 0
        self.ep_steps = ep_steps

        self.Reward_collector = []

        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(2 * self.n,), dtype=np.float32)
        # Feed the agent with 'spin_list', 'delta_list', 'ope_c_list', 'reward_list'
        # -> (3n+nz) dimensions
        self.observation_space = spaces.Box(low=0, high=1,
                                            shape=(3 * self.n,), dtype=np.float32)
    def calc_reward(self, err_list):
        return - self.scale * max([abs(err_list[i]) - abs(self.err_analyitc_list[i]) for i in range(len(err_list))])#.numpy()
    
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
            self.step_size /= 1.05
            self.ep_best_obs = np.concatenate((self.delta_list, self.ope_c_list))
            reward /= 10**5
        # Extra punishment
        if reward < ep_worst_reward:
            ep_worst_reward = reward
            self.step_size *= 1.05
            reward *= 10**2
        return reward, ep_best_reward, ep_worst_reward
        
    def step(self, action):
        self.delta_theta_list = self.delta_theta_list + self.step_size * action[:self.n]
        self.ope_c_list = self.ope_c_list + self.step_size * action[self.n:]
        self.delta_list = self.spin_list + (6.5-self.spin_list)*np.sin(self.delta_theta_list*np.pi/2)**2
        self.ope_c_list = self.reflected_bc(self.ope_c_list)
        
        self.step_num += 1

        h_list = 0.5 * (self.delta_list + self.spin_list)
        hb_list = 0.5 * (self.delta_list - self.spin_list)

        err_list = CFT_Null().cft_viol(h_list, hb_list, self.ope_c_list, self.ex_delta, self.z_list)

        reward = self.calc_reward(err_list)

        obs = np.concatenate(
                (self.spin_list/6, self.delta_list/6.5, self.ope_c_list))
        
        reward = float(reward)

        done = False

        self.Reward_collector.append(reward/self.scale)
        reward, self.ep_best_reward, self.ep_worst_reward = self.Step_size_tunning(reward, self.ep_best_reward, self.ep_worst_reward)

        if reward > 0:
            reward *= 10 ** 5
            done = True

        if self.step_num == self.ep_steps:
            print('sofar_best: ', self.ep_best_reward/self.scale)
            print('best obs', self.ep_best_obs)

            done = True
            with open(f'Rewards_Neuron_{self.neurons}_Train.csv', 'ab') as file:
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
        # self.delta_theta_list = np.random.rand(5)
        # self.delta_list = self.spin_list + (6.5-self.spin_list)*np.sin(self.delta_theta_list*np.pi/2)**2
        # self.ope_c_list = np.random.rand(5)
        
        self.delta_theta_list, self.ope_c_list = CFT_Null().Discretize()
        self.delta_list = self.spin_list + (6.5-self.spin_list)*np.sin(self.delta_theta_list*np.pi/2)**2
        
        self.Reward_collector = []
        self.step_size = 1e-2
        
        h_list = 0.5 * (self.delta_list + self.spin_list)
        hb_list = 0.5 * (self.delta_list - self.spin_list)
        err_list = CFT_Null().cft_viol(h_list, hb_list, self.ope_c_list, self.ex_delta, self.z_list)
        reward = self.calc_reward(err_list)
        self.ep_best_reward = float(reward)
        self.ep_best_obs = np.concatenate((self.delta_list, self.ope_c_list))

        obs = np.concatenate(
                (self.spin_list/6, self.delta_list/6.5, self.ope_c_list))

        return obs
