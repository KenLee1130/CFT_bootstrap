# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 16:18:52 2023

@author: User
"""

#from mpmath import hyp2f1
import torch
from sympy import hyper
import numpy as np
import scipy

class CFT_Null():
    """
    Include all tools I need in this CFT task.
    
    generate_grid_pts(Nx=20): Nx: int, output: Tensor
        Discretize the whole allow region in complex z plane.
    Discretize(grid=1000): grid: int, output: np.array
        Cut the allowed region of Physical parameters into grid.
    
    """
    def g(self, h, hb, z, zb, h12=0, h34=0, h12b=0, h34b=0):
        hyp2f1 = scipy.special.hyp2f1
        #hyp2f1 = hyper
    
        coeff = 1
        if h == hb:
            coeff = .5
    
        factor = (z**h * zb**hb * hyp2f1(h-h12, h+h34, 2*h, z) * hyp2f1(hb-h12b, hb+h34b, 2*hb, zb)) + \
                 (zb**h * z**hb * hyp2f1(h-h12, h+h34, 2*h, zb) * hyp2f1(hb-h12b, hb+h34b, 2*hb, z))
    
        return torch.tensor([complex(coeff * factor)])
    
    def Rho(self, z):
        return z / (1 + np.sqrt(1 - z))**2
    
    def Lambda(self, z):
        return abs(self.Rho(z)) + abs(self.Rho(1 - z))
    
    def generate_grid_pts(self, Nx=20):
        
        x_vals = np.linspace(.5, 1.5, Nx+1)
        y_vals = np.linspace(0., 1.16, Nx+1)
    
        pts = [(x, y) for x in x_vals for y in y_vals]
        # unit circle sampling
        pts1=torch.tensor([pt for pt in pts if self.Lambda(pt[0] + 1j*pt[1]) <= 0.6 and pt[0] >= 1/2 and pt[1] > 0 and pt[0]**2+pt[1]**2<.9])
    
    
        return torch.complex(pts1[:,0],pts1[:,1])
    
    def cft_viol(self, h_list, hb_list, c_list, delta, z_list):
    
        zb_list = torch.resolve_conj(z_list).numpy().conj()
    
        violations=[]
    
        for z, zb in zip(z_list, zb_list):
            e1=0
            for h, hb, c in zip(h_list, hb_list, c_list):
                f1 = (torch.abs(z-1) ** (2*delta)) * self.g(h, hb, z, zb)
                f2 = (torch.abs(z) ** (2*delta)) * self.g(h, hb, 1-z, 1-zb)
                e1 += c * (f1 - f2)
                
            f3 = (torch.abs(z - 1) ** (2 * delta))
            f4 = (torch.abs(z) ** (2 * delta))
            e1 += (f3 - f4)
            violations.append(abs(e1))
        
        return violations
    
    def ising_viol(self, Nx=20):
        pts=self.generate_grid_pts(Nx)
        spins_ising=torch.tensor([0,0,2,4,6])
        deltas_analyt=torch.tensor([4, 1, 2, 4, 6])
        cs_analyt=torch.tensor([0.000244141, 0.25, 0.015625, 0.000219727, 0.0000136239])
        h_a = (deltas_analyt + spins_ising)/2
        hb_a = (deltas_analyt - spins_ising)/2
        return self.cft_viol(h_a, hb_a, cs_analyt, 1/8, pts)
    
    def Discretize(self, grid=1000):
        Select_delta = np.random.randint(0, grid, 5)/grid
        delta_theta_list = Select_delta
        
        Select_ope_c = np.random.randint(0, grid, 5)/grid
        ope_c_list = Select_ope_c
        
        return delta_theta_list, ope_c_list