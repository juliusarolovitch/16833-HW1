'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import torch
import math

class MotionModel:

    def __init__(self, device):
        self._alpha1 = 0.0003
        self._alpha2 = 0.0001
        self._alpha3 = 0.0005
        self._alpha4 = 0.0001
        self.device = device

    def wrapPi(self, angles):
        return angles - 2 * torch.pi * torch.floor((angles + torch.pi) / (2 * torch.pi))
    
    def update(self, u_t0, u_t1, x_t0):
        del_x = u_t1[0] - u_t0[0]
        del_y = u_t1[1] - u_t0[1]
        del_theta = u_t1[2] - u_t0[2]
        delta_rot1 = torch.atan2(del_y, del_x) - u_t0[2]
        delta_trans = torch.sqrt(del_x**2 + del_y**2)
        delta_rot2 = del_theta - delta_rot1
        std_rot1 = self._alpha1 * (delta_rot1**2) + self._alpha2 * (delta_trans**2)
        std_trans = self._alpha3 * (delta_trans**2) + self._alpha4 * (delta_rot1**2 + delta_rot2**2)
        std_rot2 = self._alpha1 * (delta_rot2**2) + self._alpha2 * (delta_trans**2)
        N = x_t0.shape[0]
        noise_rot1 = torch.normal(0, torch.sqrt(torch.tensor(std_rot1, device=self.device)), size=(N,), device=self.device)
        noise_trans = torch.normal(0, torch.sqrt(torch.tensor(std_trans, device=self.device)), size=(N,), device=self.device)
        noise_rot2 = torch.normal(0, torch.sqrt(torch.tensor(std_rot2, device=self.device)), size=(N,), device=self.device)
        sampled_rot1 = delta_rot1 - noise_rot1
        sampled_trans = delta_trans - noise_trans
        sampled_rot2 = delta_rot2 - noise_rot2
        new_x = x_t0[:, 0] + sampled_trans * torch.cos(x_t0[:, 2] + sampled_rot1)
        new_y = x_t0[:, 1] + sampled_trans * torch.sin(x_t0[:, 2] + sampled_rot1)
        new_theta = self.wrapPi(x_t0[:, 2] + sampled_rot1 + sampled_rot2)
        return torch.stack((new_x, new_y, new_theta), dim=1)
