# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 14:26:20 2023

@author: User
"""

class Neurons_comparison():
    def __init__(self, Neurons):
        self.Neurons = Neurons
    def Statistical(self):
        import pandas as pd
        import numpy as np
        import ast
        
        Mean = []
        STD = []
        MAX_reward = []
        Normalize_upgrade = []
        N_mean = []
        N_std = []
        Experimental_spec = []
        analytic = [4, 1, 2, 4, 6, 0.000244141, 0.25, 0.015625, 0.000219727, 0.0000136239]
        
        for neurons in range(self.Neurons[0], self.Neurons[1], 128):
            data = pd.read_excel(f"C:/Users/User/Downloads/drive-download-20230816T112113Z-001/Rewards_Neuron_{neurons}_Eval.xlsx",
                                  header=None, names=[f'N:{neurons}'])
            data = pd.DataFrame(data)
            for i in range(10000//100):
                Normalize_upgrade.append(-(pd.DataFrame.max(data)-data[i*100+1: (i+1)*100+1].head(1))*100/ data[i*100+1: (i+1)*100+1].head(1))
            mean, std = np.mean(Normalize_upgrade), np.std(Normalize_upgrade)
            N_mean.append(mean)
            N_std.append(std)
            MAX_reward.append(pd.DataFrame.max(data).item())
            
            mean, std = pd.DataFrame.mean(data), pd.DataFrame.std(data)
            Mean.append(mean.item())
            STD.append(std.item())
            data = pd.read_excel(f"C:/Users/User/Desktop/master/RL/CFT/CFT_data/Best10000_with_maxreward_Neuron_{neurons}_Eval.xlsx", 
                                  header=None, names=[f'N:{neurons}_reward', f'N:{neurons}_spec'])

            data = data.applymap(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            Experimental_spec.append(pd.DataFrame(data)[f'N:{neurons}_spec'])


        Analytic_alike_mean = [[], []]
        Analytic_alike_std = [[], []]
        for i in range(len(Experimental_spec)):
            Each_neuron = [[], []]
            for j in range(len(Experimental_spec[i])):
                diff = Experimental_spec[i][j] - np.array(analytic)
                Each_neuron[0].append(diff[0:5])
                Each_neuron[1].append(diff[5:10])
            Analytic_alike_mean[0].append(np.mean(Each_neuron[0]))
            Analytic_alike_mean[1].append(np.mean(Each_neuron[1]))
            Analytic_alike_std[0].append(np.std(Each_neuron[0]))
            Analytic_alike_std[1].append(np.std(Each_neuron[1]))
        return (Mean, STD, MAX_reward, N_mean, N_std, Analytic_alike_mean, Analytic_alike_std)
    def Plot(self):
        import numpy as np
        import matplotlib.pyplot as plt
        
        Stat = self.Statistical()
        
        x_axis = np.arange(self.Neurons[0], self.Neurons[1], 128)

        plt.subplot(3,1,1)
        plt.title('(Mean/Std) Reward in each Neurons')
        plt.errorbar(x_axis, Stat[0], yerr=Stat[1], fmt='o', ecolor='r', color='b',
                      elinewidth=2, capsize=4, linestyle=':')
        plt.grid()
        plt.xlabel('Neurons')
        plt.ylabel('Reward')
        plt.xticks(x_axis)

        plt.subplot(3,1,2)
        plt.title('Max_reward in each Neurons')
        plt.plot(x_axis, Stat[2], marker='o', linestyle=':')
        plt.grid()
        plt.xlabel('Neurons')
        plt.ylabel('Reward')
        plt.xticks(x_axis)

        plt.subplot(3,1,3)
        plt.title('Reward improvment within an episode')
        plt.errorbar(x_axis, Stat[3], yerr=Stat[4], fmt='o', ecolor='r', color='b',
                      elinewidth=2, capsize=4, linestyle=':')
        plt.grid()
        plt.xlabel('Neurons')
        plt.ylabel('Improve(%)')
        plt.xticks(x_axis)

        plt.tight_layout()
        plt.savefig('Reward_monitor.png')
        plt.show()

        plt.figure()
        plt.subplot(2,1,1)
        plt.title('Similarity of Analytic and Experimental [Delta]')
        plt.errorbar(x_axis, Stat[5][0], yerr=Stat[5][1], fmt='o', ecolor='r', color='b',
                      elinewidth=2, capsize=4, linestyle=':')
        plt.grid()
        plt.xlabel('Neurons')
        plt.ylabel('relative distance')
        plt.xticks(x_axis)

        plt.subplot(2,1,2)
        plt.title('Similarity of Analytic and Experimental [Ope]')
        plt.errorbar(x_axis, Stat[6][0], yerr=Stat[6][1], fmt='o', ecolor='r', color='b',
                      elinewidth=2, capsize=4, linestyle=':')
        plt.grid()
        plt.xlabel('Neurons')
        plt.ylabel('relative distance')
        plt.xticks(x_axis)

        plt.tight_layout()
        plt.savefig('Similarity_monitor.png')
        plt.show()
        
Neurons_comparison([64, 1088]).Plot()