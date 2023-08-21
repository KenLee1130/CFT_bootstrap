# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 12:43:03 2023

@author: User
"""

from stable_baselines3 import SAC
from CFT_Env_Check import CFT_Env_Check
from CFT_Env_eval import CFT_Env_Eval
from CFT_Env import CFT_Env
from stable_baselines3.common.evaluation import evaluate_policy
import time
import math
import torch.nn as nn
import os

def Validate(scale, step_size, ep_steps, episodes, neurons):
    model_path = f'sac_cft_{neurons}'
    model = SAC.load(model_path)
    
    env = CFT_Env_Eval(scale, step_size, ep_steps, episodes, neurons)
    
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, env=env, n_eval_episodes=episodes, deterministic=True)
    
def Train(scale, step_size, ep_steps, episodes, neurons):
    policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[neurons, neurons, neurons], qf=[neurons, neurons, neurons]))
    
    model = SAC('MlpPolicy',
                CFT_Env(scale, step_size, ep_steps, episodes, neurons),
                learning_rate=learning_rate,
                policy_kwargs=policy_kwargs,
                verbose=0, device='cuda')
    
    model.learn(total_timesteps=ep_steps*episodes)
    model.save(f'sac_cft_{neurons}')

def ReTrain(scale, step_size, ep_steps, episodes, neurons):
    model_path = f'sac_cft_{neurons}'
    model = SAC.load(model_path)
    
    env = CFT_Env(scale, step_size, ep_steps, episodes)
    
    model.set_env(env)
    model.learn(total_timesteps=ep_steps*episodes)
    model.save(f'sac_cft_{neurons}')

def Check_max(Neurons):
    import pandas as pd
    data = pd.read_csv(f"Rewards_Neuron_{Neurons}_Eval.csv",
                          header=None, names=[f'N:{Neurons}'])
    Done = pd.DataFrame.max(data) > -1
    
    return Done.item()

def Check_overfit(scale, step_size, ep_steps, episodes, Neurons):
    import pandas as pd
    import numpy as np
    data_Check_path = f"C:/Users/User/Desktop/master/RL/CFT/CFT_data/Rewards_Neuron_{Neurons}_Check.xlsx"
    if os.path.isfile(data_Check_path):
        Normalize_upgrade = []
        data = pd.read_excel(data_Check_path, header=None, names=[f'N:{Neurons}'])
        data = pd.DataFrame(data)
        for i in range(10000//100):
            Normalize_upgrade.append(-(pd.DataFrame.max(data)-data[i*100+1: (i+1)*100+1].head(1))*100/ data[i*100+1: (i+1)*100+1].head(1))
        mean, std = np.mean(Normalize_upgrade), np.std(Normalize_upgrade)
        N_mean_check = mean
        N_std_check = std
        
    else:
        model_path = f'sac_cft_{neurons}'
        model = SAC.load(model_path)
        
        env = CFT_Env_Check(scale, step_size, ep_steps, episodes, neurons)
        
        # Evaluate the trained agent
        mean_reward, std_reward = evaluate_policy(model, env=env, n_eval_episodes=episodes, deterministic=True)

####################################
ep_steps_train = 200
episodes_train = 500

ep_steps_retrain = 200
episodes_retrain = 50

ep_steps_validate = 100
episodes_validate = 100

# learning rate for updating neural network
learning_rate = 1e-5

# scale of panelty and step size of action
scale = 10 ** 8
step_size = 1e-2


####################################
print("Start!!!")
start_time = time.time()

for neurons in range(64, 1344, 128):
    filepath = f'sac_cft_{neurons}.zip'
    retrain_number = 0
    
    if os.path.isfile(filepath):
        # File exist
        while not Check_max(neurons):
            ReTrain(scale, step_size, ep_steps_retrain, episodes_retrain, neurons)
            Validate(scale, step_size, ep_steps_validate, episodes_validate, neurons)
            retrain_number += 1
    else:
        # File don't exist
        Train(scale, step_size, ep_steps_train, episodes_train, neurons)
        Validate(scale, step_size, ep_steps_validate, episodes_validate, neurons)
        while not Check_max(neurons):
            ReTrain(scale, step_size, ep_steps_retrain, episodes_retrain, neurons)
            Validate(scale, step_size, ep_steps_validate, episodes_validate, neurons)
            retrain_number += 1
    
    Check_overfit(scale, step_size, ep_steps_validate, episodes_validate, neurons)

end_time = time.time()
print('It took ', end_time - start_time, ' seconds!!!')
print('That is roughly ', math.floor(100 * (end_time - start_time) / 60) / 100, ' minutes, or ',
      math.floor(100 * (end_time - start_time) / 3600) / 100, 'hours!!!')
