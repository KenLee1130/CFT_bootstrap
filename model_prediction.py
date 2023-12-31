# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 17:45:27 2023

@author: User
"""

import numpy as np
from stable_baselines3 import SAC
from Where_execute import Where_execute 

def predict(input, model):
    path = Where_execute('my_computer')[1]
    # 載入Model
    with open(path+model+'.zip', 'rb') as f:
        Model = SAC.load(f)
        
    def reflected_bc(ope_c_list):
        for i in range(len(ope_c_list)):
            if ope_c_list[i] < 0:
                ope_c_list[i] = -ope_c_list[i]

        return ope_c_list
    pred=Model.predict(input)
    
    step_size = 0.01/pred[0][10]
    spin_list = np.array([0., 0., 2., 4., 6.])
    Spec_delta = input[5:10] + step_size*pred[0][0:5]
    Spec_ope = reflected_bc(input[10:15] + step_size*pred[0][5:10])
    Spec = np.concatenate((spin_list, Spec_delta, Spec_ope))
    
    def calc_reward(nx, err_list):
        import torch
        from CFT_Null import CFT_Null
        from ising_viol_13 import ising_viol_13
        from ising_viol_20 import ising_viol_20
        from ising_viol_30 import ising_viol_30
        err_analyitc_dict = {'13':ising_viol_13(), '20':ising_viol_20(), '30':ising_viol_30()}
        err_list = torch.tensor(err_list)
        err_analyitc = torch.tensor(err_analyitc_dict[f'{nx}'])
        
        return - torch.max(torch.subtract(err_list, err_analyitc))
    
    def reward_conparasion(Spec):
        from CFT_Null import CFT_Null
        spin_list = np.array([0., 0., 2., 4., 6.])
        delta_list = Spec[5:10]
        ope_c_list = Spec[10:15]
        h_list = 0.5 * (delta_list + spin_list)
        hb_list = 0.5 * (delta_list - spin_list)
        z_list = CFT_Null().generate_grid_pts(20)
        
        err_list = CFT_Null().cft_viol(h_list, hb_list, ope_c_list, 0.125, z_list)
        reward = calc_reward(20, err_list)
        return reward
    #print(Spec)
    Spec_reward = reward_conparasion(Spec)
    Ising_reward = reward_conparasion(input)
    improve = (Spec_reward - Ising_reward)*100 / (-Ising_reward)
    
    return [f'Delta: {Spec_delta}', f'OPE: {Spec_ope}', f'Reward_imporove: {improve}%'
            , f'Reward from {Ising_reward} to {Spec_reward}']

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return 'hello!!'

@app.route('/predict', methods=['POST'])
def postInput():
    # 取得前端傳過來的數值
    insertValues = request.get_json()
    x1=insertValues['spin']
    x2=insertValues['delta']
    x3=insertValues['ope']
    input = np.concatenate((x1, x2, x3))

    result = predict(input, 'sac_cft_546_0')

    return jsonify({'return': str(result)})

if __name__ == '__main__':
    
    app.run(host='0.0.0.0', port=3000, debug=True)

