# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:40:56 2023

@author: User
"""

from CFT_Null import CFT_Null
import numpy as np
from ising_viol_20 import ising_viol_20
import matplotlib.pyplot as plt
import time

print("Start!!!")
start_time = time.time()

def calc_reward(err_list):
    err_analyitc_list = ising_viol_20()
    return - max([abs(err_list[i]) - abs(err_analyitc_list[i]) for i in range(len(err_list))])#.numpy()

spin_list = np.array([0., 0., 2., 4., 6.])

delta_theta = [np.array([0.105, 0.52 , 0.434, 0.74 , 0.246]), np.array([0.357, 0.118, 0.365, 0.81 , 0.636]), np.array([0.15 , 0.741, 0.583, 0.767, 0.084]), np.array([0.225, 0.283, 0.02 , 0.492, 0.293]), np.array([0.86 , 0.312, 0.597, 0.581, 0.21 ]), np.array([0.451, 0.99 , 0.443, 0.49 , 0.003]), np.array([0.176, 0.029, 0.498, 0.666, 0.564]), np.array([0.107, 0.013, 0.546, 0.816, 0.074]), np.array([0.423, 0.717, 0.928, 0.97 , 0.379]), np.array([0.467, 0.596, 0.401, 0.203, 0.595]), np.array([0.313, 0.402, 0.036, 0.765, 0.048]), np.array([0.884, 0.062, 0.247, 0.029, 0.409]), np.array([0.939, 0.763, 0.331, 0.047, 0.447]), np.array([0.013, 0.557, 0.433, 0.51 , 0.728]), np.array([0.947, 0.169, 0.07 , 0.652, 0.38 ]), np.array([0.409, 0.088, 0.378, 0.048, 0.316]), np.array([0.811, 0.769, 0.041, 0.363, 0.808]), np.array([0.569, 0.337, 0.927, 0.909, 0.616]), np.array([0.047, 0.647, 0.664, 0.471, 0.936]), np.array([0.178, 0.442, 0.715, 0.548, 0.535]), np.array([0.193, 0.084, 0.533, 0.108, 0.544]), np.array([0.13 , 0.121, 0.394, 0.358, 0.049]), np.array([0.674, 0.141, 0.242, 0.765, 0.944]), np.array([0.833, 0.508, 0.249, 0.359, 0.471]), np.array([0.141, 0.708, 0.982, 0.964, 0.326]), np.array([0.04 , 0.663, 0.484, 0.196, 0.254]), np.array([0.125, 0.631, 0.21 , 0.739, 0.869]), np.array([0.416, 0.09 , 0.383, 0.33 , 0.7  ]), np.array([0.995, 0.739, 0.123, 0.457, 0.453]), np.array([0.41 , 0.149, 0.048, 0.028, 0.717]), np.array([0.024, 0.15 , 0.58 , 0.478, 0.682]), np.array([0.603, 0.883, 0.404, 0.52 , 0.096]), np.array([0.101, 0.651, 0.936, 0.876, 0.138]), np.array([0.435, 0.435, 0.246, 0.089, 0.167]), np.array([0.022, 0.384, 0.237, 0.41 , 0.838]), np.array([0.392, 0.293, 0.293, 0.904, 0.975]), np.array([0.706, 0.95 , 0.369, 0.289, 0.951]), np.array([0.666, 0.058, 0.509, 0.977, 0.765]), np.array([0.357, 0.489, 0.605, 0.191, 0.689]), np.array([0.99 , 0.225, 0.663, 0.076, 0.01 ]), np.array([0.565, 0.528, 0.732, 0.633, 0.415]), np.array([0.261, 0.96 , 0.95 , 0.671, 0.754]), np.array([0.708, 0.284, 0.9  , 0.553, 0.174]), np.array([0.984, 0.636, 0.226, 0.471, 0.436]), np.array([0.255, 0.819, 0.589, 0.4  , 0.258]), np.array([0.232, 0.789, 0.25 , 0.24 , 0.293]), np.array([0.502, 0.966, 0.107, 0.696, 0.263]), np.array([0.558, 0.847, 0.752, 0.605, 0.444]), np.array([0.565, 0.446, 0.789, 0.269, 0.107]), np.array([0.128, 0.532, 0.535, 0.12 , 0.655]), np.array([0.634, 0.476, 0.294, 0.447, 0.663]), np.array([0.384, 0.348, 0.782, 0.674, 0.988]), np.array([0.107, 0.137, 0.427, 0.898, 0.031]), np.array([0.615, 0.145, 0.669, 0.978, 0.991]), np.array([0.21 , 0.389, 0.582, 0.575, 0.739]), np.array([0.526, 0.345, 0.653, 0.501, 0.506]), np.array([0.52 , 0.013, 0.602, 0.523, 0.916]), np.array([0.457, 0.321, 0.062, 0.624, 0.747]), np.array([0.658, 0.431, 0.582, 0.498, 0.526]), np.array([0.893, 0.961, 0.179, 0.762, 0.549]), np.array([0.862, 0.295, 0.436, 0.249, 0.106]), np.array([0.222, 0.817, 0.258, 0.142, 0.313]), np.array([0.004, 0.853, 0.218, 0.5  , 0.479]), np.array([0.984, 0.228, 0.686, 0.971, 0.092]), np.array([0.174, 0.953, 0.454, 0.783, 0.695]), np.array([0.239, 0.874, 0.041, 0.332, 0.987]), np.array([0.223, 0.888, 0.205, 0.323, 0.884]), np.array([0.17 , 0.801, 0.123, 0.684, 0.671]), np.array([0.026, 0.232, 0.775, 0.808, 0.99 ]), np.array([0.11 , 0.277, 0.433, 0.276, 0.12 ]), np.array([0.979, 0.994, 0.875, 0.479, 0.824]), np.array([0.029, 0.922, 0.569, 0.578, 0.908]), np.array([0.182, 0.003, 0.868, 0.37 , 0.525]), np.array([0.131, 0.929, 0.674, 0.088, 0.87 ]), np.array([0.124, 0.516, 0.34 , 0.697, 0.073]), np.array([0.38 , 0.398, 0.569, 0.456, 0.397]), np.array([0.627, 0.258, 0.575, 0.019, 0.579]), np.array([0.338, 0.076, 0.565, 0.577, 0.991]), np.array([0.013, 0.699, 0.56 , 0.705, 0.629]), np.array([0.462, 0.018, 0.643, 0.939, 0.59 ]), np.array([0.685, 0.112, 0.817, 0.766, 0.216]), np.array([0.478, 0.003, 0.437, 0.708, 0.738]), np.array([0.941, 0.362, 0.352, 0.55 , 0.793]), np.array([0.553, 0.33 , 0.674, 0.058, 0.514]), np.array([0.455, 0.569, 0.472, 0.343, 0.938]), np.array([0.534, 0.926, 0.417, 0.865, 0.802]), np.array([0.549, 0.879, 0.411, 0.706, 0.206]), np.array([0.12 , 0.068, 0.887, 0.263, 0.228]), np.array([0.091, 0.146, 0.124, 0.805, 0.357]), np.array([0.693, 0.455, 0.305, 0.606, 0.763]), np.array([0.663, 0.301, 0.035, 0.31 , 0.355]), np.array([0.402, 0.193, 0.967, 0.877, 0.21 ]), np.array([0.738, 0.014, 0.363, 0.778, 0.973]), np.array([0.268, 0.833, 0.375, 0.839, 0.071]), np.array([0.909, 0.56 , 0.252, 0.474, 0.003]), np.array([0.743, 0.678, 0.07 , 0.109, 0.576]), np.array([0.884, 0.167, 0.082, 0.293, 0.981]), np.array([0.137, 0.119, 0.3  , 0.462, 0.593]), np.array([0.809, 0.477, 0.063, 0.786, 0.595]), np.array([0.915, 0.938, 0.756, 0.854, 0.324])]
ope_c = [np.array([0.891, 0.898, 0.688, 0.967, 0.196]), np.array([0.545, 0.653, 0.429, 0.474, 0.049]), np.array([0.215, 0.706, 0.999, 0.084, 0.789]), np.array([0.845, 0.998, 0.005, 0.49 , 0.293]), np.array([0.815, 0.021, 0.462, 0.471, 0.393]), np.array([0.203, 0.998, 0.302, 0.55 , 0.414]), np.array([0.557, 0.723, 0.267, 0.735, 0.789]), np.array([0.827, 0.313, 0.875, 0.263, 0.392]), np.array([0.236, 0.622, 0.157, 0.514, 0.903]), np.array([0.64 , 0.462, 0.336, 0.927, 0.405]), np.array([0.121, 0.786, 0.221, 0.751, 0.422]), np.array([0.367, 0.705, 0.925, 0.394, 0.685]), np.array([0.817, 0.267, 0.403, 0.06 , 0.217]), np.array([0.043, 0.309, 0.332, 0.675, 0.387]), np.array([0.191, 0.51 , 0.03 , 0.599, 0.5  ]), np.array([0.836, 0.518, 0.965, 0.369, 0.13 ]), np.array([0.64 , 0.597, 0.659, 0.704, 0.736]), np.array([0.799, 0.642, 0.309, 0.992, 0.459]), np.array([0.087, 0.942, 0.165, 0.467, 0.139]), np.array([0.331, 0.504, 0.064, 0.783, 0.968]), np.array([0.104, 0.896, 0.844, 0.737, 0.269]), np.array([0.949, 0.559, 0.52 , 0.451, 0.183]), np.array([0.06 , 0.743, 0.811, 0.936, 0.582]), np.array([0.217, 0.829, 0.578, 0.154, 0.724]), np.array([0.559, 0.443, 0.374, 0.568, 0.496]), np.array([0.441, 0.278, 0.674, 0.622, 0.413]), np.array([0.004, 0.224, 0.977, 0.708, 0.368]), np.array([0.381, 0.349, 0.344, 0.734, 0.613]), np.array([0.74 , 0.923, 0.535, 0.327, 0.644]), np.array([0.225, 0.157, 0.585, 0.062, 0.827]), np.array([0.054, 0.31 , 0.015, 0.797, 0.925]), np.array([0.357, 0.388, 0.875, 0.195, 0.675]), np.array([0.153, 0.611, 0.106, 0.22 , 0.31 ]), np.array([0.908, 0.718, 0.512, 0.761, 0.801]), np.array([0.289, 0.132, 0.484, 0.673, 0.583]), np.array([0.239, 0.739, 0.42 , 0.896, 0.751]), np.array([0.839, 0.1  , 0.19 , 0.586, 0.579]), np.array([0.639, 0.991, 0.898, 0.896, 0.148]), np.array([0.03 , 0.523, 0.538, 0.081, 0.648]), np.array([0.456, 0.912, 0.495, 0.583, 0.417]), np.array([0.278, 0.474, 0.34 , 0.95 , 0.366]), np.array([0.828, 0.562, 0.12 , 0.855, 0.278]), np.array([0.599, 0.233, 0.819, 0.075, 0.426]), np.array([0.901, 0.989, 0.676, 0.809, 0.829]), np.array([0.506, 0.216, 0.325, 0.561, 0.815]), np.array([0.485, 0.485, 0.513, 0.564, 0.345]), np.array([0.443, 0.719, 0.394, 0.977, 0.564]), np.array([0.86 , 0.589, 0.897, 0.989, 0.254]), np.array([0.564, 0.556, 0.274, 0.17 , 0.842]), np.array([0.925, 0.574, 0.024, 0.758, 0.993]), np.array([0.145, 0.832, 0.116, 0.31 , 0.385]), np.array([0.063, 0.339, 0.973, 0.399, 0.576]), np.array([0.603, 0.474, 0.788, 0.934, 0.643]), np.array([0.238, 0.395, 0.331, 0.326, 0.452]), np.array([0.58 , 0.6  , 0.973, 0.715, 0.545]), np.array([0.145, 0.51 , 0.941, 0.635, 0.524]), np.array([0.619, 0.693, 0.574, 0.416, 0.218]), np.array([0.881, 0.843, 0.428, 0.634, 0.881]), np.array([0.984, 0.213, 0.28 , 0.726, 0.511]), np.array([0.699, 0.039, 0.333, 0.15 , 0.31 ]), np.array([0.452, 0.202, 0.328, 0.592, 0.377]), np.array([0.515, 0.569, 0.241, 0.465, 0.372]), np.array([0.236, 0.847, 0.808, 0.614, 0.724]), np.array([0.5  , 0.882, 0.381, 0.007, 0.91 ]), np.array([0.409, 0.276, 0.836, 0.912, 0.742]), np.array([0.646, 0.36 , 0.823, 0.212, 0.115]), np.array([0.586, 0.274, 0.02 , 0.823, 0.025]), np.array([0.746, 0.776, 0.149, 0.939, 0.578]), np.array([0.38 , 0.016, 0.806, 0.463, 0.137]), np.array([0.839, 0.256, 0.789, 0.798, 0.407]), np.array([0.05 , 0.337, 0.316, 0.346, 0.3  ]), np.array([0.937, 0.086, 0.031, 0.244, 0.409]), np.array([0.167, 0.875, 0.661, 0.941, 0.261]), np.array([0.175, 0.061, 0.758, 0.986, 0.111]), np.array([0.716, 0.473, 0.889, 0.94 , 0.306]), np.array([0.584, 0.891, 0.185, 0.871, 0.735]), np.array([0.861, 0.759, 0.92 , 0.99 , 0.685]), np.array([0.721, 0.943, 0.584, 0.683, 0.646]), np.array([0.998, 0.695, 0.292, 0.542, 0.66 ]), np.array([0.175, 0.616, 0.008, 0.176, 0.999]), np.array([0.674, 0.246, 0.218, 0.823, 0.702]), np.array([0.106, 0.343, 0.379, 0.669, 0.796]), np.array([0.567, 0.313, 0.623, 0.357, 0.67 ]), np.array([0.816, 0.603, 0.778, 0.429, 0.634]), np.array([0.302, 0.837, 0.853, 0.537, 0.304]), np.array([0.369, 0.952, 0.01 , 0.831, 0.554]), np.array([0.084, 0.201, 0.654, 0.066, 0.331]), np.array([0.706, 0.477, 0.125, 0.109, 0.431]), np.array([0.568, 0.757, 0.751, 0.337, 0.243]), np.array([0.274, 0.103, 0.383, 0.828, 0.642]), np.array([0.446, 0.206, 0.598, 0.813, 0.592]), np.array([0.66 , 0.277, 0.59 , 0.466, 0.002]), np.array([0.062, 0.803, 0.518, 0.697, 0.535]), np.array([0.623, 0.776, 0.234, 0.163, 0.772]), np.array([0.752, 0.78 , 0.853, 0.077, 0.057]), np.array([0.277, 0.099, 0.289, 0.763, 0.811]), np.array([0.898, 0.495, 0.674, 0.869, 0.369]), np.array([0.418, 0.347, 0.349, 0.409, 0.696]), np.array([0.665, 0.887, 0.073, 0.271, 0.843]), np.array([0.36 , 0.717, 0.341, 0.344, 0.227])]

Reward = []
Delta_data = []
OPE_data = []
for i in range(100):
    delta_theta_list = delta_theta[i]
    ope_c_list = ope_c[i]
    # Delta_data.append(delta_theta_list)
    # OPE_data.append(ope_c_list)
    
    delta_list = spin_list + (6.5-spin_list)*np.sin(delta_theta_list*np.pi/2)**2
    z_list = CFT_Null().generate_grid_pts(20)
    
    h_list = 0.5 * (delta_list + spin_list)
    hb_list = 0.5 * (delta_list - spin_list)
    
    err_list = CFT_Null().cft_viol(h_list, hb_list, ope_c_list, 0.125, z_list)
    
    reward = calc_reward(err_list)
    Reward.append(reward.item())

print('delta', Delta_data)
print('ope', OPE_data)

plt.hist(Reward, bins='auto', density=True)


end_time = time.time()
print('It took ', end_time - start_time, ' seconds!!!')