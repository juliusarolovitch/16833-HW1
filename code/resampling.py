'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import torch


class Resampling:
    def __init__(self, device, min_resample_fraction=0.1, max_resample_fraction=0.9):
        self.min_resample_fraction = min_resample_fraction
        self.max_resample_fraction = max_resample_fraction
        self.device = device


    def compute_ess(self, weights):
        return 1.0 / torch.sum(weights**2)
    

    def adaptive_resample(self, X_bar):
        num_particles = X_bar.shape[0]
        weights = X_bar[:, 3]
        total_weight = torch.sum(weights)
        weights = torch.where(total_weight > 0, weights/total_weight, torch.ones(num_particles, device=self.device)/num_particles)
        ess = self.compute_ess(weights)
        resample_fraction = self.max_resample_fraction + (self.min_resample_fraction - self.max_resample_fraction) * (1 - ess / num_particles)
        num_resampled_particles = max(1, int(resample_fraction * num_particles))
        if torch.var(weights) > 0.01:
            X_bar_resampled = self.low_variance_sampler(X_bar, num_resampled_particles)
        else:
            X_bar_resampled = self.multinomial_sampler(X_bar, num_resampled_particles)
        return X_bar_resampled
    

    def multinomial_sampler(self, X_bar, num_resampled_particles):
        num_particles = X_bar.shape[0]
        weights = X_bar[:, 3]
        total_weight = torch.sum(weights)
        probabilities = torch.where(total_weight > 0, weights/total_weight, torch.ones(num_particles, device=self.device)/num_particles)
        indices = torch.multinomial(probabilities, num_samples=num_resampled_particles, replacement=True)
        X_bar_resampled = X_bar[indices]
        X_bar_resampled[:, 3] = 1.0 / num_resampled_particles
        return X_bar_resampled


    def low_variance_sampler(self, X_bar, num_resampled_particles):
        num_particles = X_bar.shape[0]
        weights = X_bar[:, 3]
        total_weight = torch.sum(weights)
        weights = torch.where(total_weight > 0, weights/total_weight, torch.ones(num_particles, device=self.device)/num_particles)
        step = 1.0 / num_resampled_particles
        r = torch.empty(1, device=self.device).uniform_(0, step).item()
        c = weights[0].item()
        i = 0
        X_bar_resampled = torch.empty((num_resampled_particles, 4), dtype=X_bar.dtype, device=self.device)
        for m in range(num_resampled_particles):
            U = r + m * step
            while U > c:
                i += 1
                c += weights[i].item()
            X_bar_resampled[m] = X_bar[i]
        X_bar_resampled[:, 3] = 1.0 / num_resampled_particles
        return X_bar_resampled
