# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:51:06 2023

@author: User
"""

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
import time
import math
import torch.nn as nn
from CFT_SOO_Env import CFT_SOO_Env

def main():

    total_steps = 150
    episodes = 500

    # learning rate for updating neural network
    learning_rate = 1e-5

    # scale of panelty and step size of action
    scale = 10 ** 8
    step_size = 1e-2
    
    neurons = 576

    ####################################


    # Create model and Train model
    print("Start!!!")
    start_time = time.time()
    
    # Setup Checkpoint callback
    # The saved model is in the argument of 'save_path'
    #checkpoint_callback = CheckpointCallback(save_freq=total_steps//5, save_path='./logs/', name_prefix='sac2_rl_model_')

    # The saved tensorboard data is in the argument of 'tensorboard_log'
    # Use "tensorboard --logdir ./tensorboard/" in cmd/terminal to activate tensorboard localhost
    # and open browser to access to the given url given in the cmd/terminal
    # ***Note: If the GPU is Not Nvidia GPU, "device='cuda' -> device='cpu'"
    # pi: actor, qf: Q function
    policy_kwargs = dict(activation_fn=nn.ReLU, net_arch=dict(pi=[neurons, neurons, neurons], qf=[neurons, neurons, neurons]))
    
    model = SAC('MlpPolicy',
                CFT_SOO_Env(scale, step_size, total_steps, episodes),
                learning_rate=learning_rate,
                policy_kwargs=policy_kwargs,
                verbose=0, device='cuda')

    #model.learn(total_timesteps=total_steps, callback=checkpoint_callback, tb_log_name="sac2")
    model.learn(total_timesteps=total_steps)
    model.save(f'sac_cft_{neurons}')
    #print('policy kwargs: ', policy_kwargs)

    end_time = time.time()
    print('It took ', end_time - start_time, ' seconds!!!')
    print('That is roughly ', math.floor(100 * (end_time - start_time) / 60) / 100, ' minutes, or ',
          math.floor(100 * (end_time - start_time) / 3600) / 100, 'hours!!!')


if __name__ == '__main__':
    main()