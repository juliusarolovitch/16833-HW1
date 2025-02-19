'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

import torch
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import concurrent.futures


class SensorModel:
    def __init__(self, occupancy_map, device, saved_rays_precalc_path="rays.pt"):
        if occupancy_map is None:
            raise ValueError("An occupancy map must be provided!")
        self._z_hit = 1.1
        self._z_short = 0.25
        self._z_max = 0.2
        self._z_rand = 400
        self._sigma_hit = 90
        self._lambda_short = 0.1
        self._max_range = 1000
        self._min_probability = 0.35
        self._map_resolution = 10
        self._ray_step = 1
        self._num_rays = 30
        self._interpolation_num = 250
        self._occupancy_map = occupancy_map
        self._x_max = self._occupancy_map.shape[1] * self._map_resolution
        self._y_max = self._occupancy_map.shape[0] * self._map_resolution
        self.raycast_map_file = saved_rays_precalc_path
        self.device = device
        if os.path.exists(self.raycast_map_file):
            print("Loading saved raycast map from '{}'...".format(self.raycast_map_file))
            self.raycast_map = torch.load(self.raycast_map_file, map_location=self.device)
        else:
            print("No saved raycast map found. Computing raycast map...")
            self.raycast_map = self.download_ray_mapping()
            torch.save(self.raycast_map, self.raycast_map_file)
            print("Raycast map computed and saved to '{}'.".format(self.raycast_map_file))


    def _ray_cast_all(self, start_x, start_y, start_theta, angles):
        L = 25
        sensor_x = start_x + L * torch.cos(start_theta)
        sensor_y = start_y + L * torch.sin(start_theta)
        ray_angles = start_theta + angles
        t = torch.arange(0, self._max_range, self._ray_step, device=self.device, dtype=torch.float32)
        ts = t.unsqueeze(0)
        xs = sensor_x + ts * torch.cos(ray_angles)
        ys = sensor_y + ts * torch.sin(ray_angles)
        oob = (xs < 0) | (xs >= self._x_max) | (ys < 0) | (ys >= self._y_max)
        inbound = ~oob
        grid_x = torch.zeros_like(xs, dtype=torch.int64)
        grid_y = torch.zeros_like(ys, dtype=torch.int64)
        grid_x[inbound] = (xs[inbound] / self._map_resolution).to(torch.int64)
        grid_y[inbound] = (ys[inbound] / self._map_resolution).to(torch.int64)
        blocked = torch.zeros_like(xs, dtype=torch.bool)
        if inbound.any():
            blocked[inbound] = (torch.abs(self._occupancy_map[grid_y[inbound], grid_x[inbound]]) > self._min_probability)
        hit = oob | blocked
        hit_any = torch.any(hit, dim=1)
        idx = torch.argmax(hit, dim=1)
        t_hit = torch.where(hit_any, t[idx], torch.tensor(self._max_range, device=self.device, dtype=torch.float32))
        x_hit = sensor_x + t_hit * torch.cos(ray_angles.squeeze())
        y_hit = sensor_y + t_hit * torch.sin(ray_angles.squeeze())
        return x_hit, y_hit
    

    def _gaussian_density(self, z_expected, z_measure):
        valid = (z_measure >= 0) & (z_measure <= self._max_range)
        sigma = self._sigma_hit
        cdf = 0.5*(1+torch.erf((torch.tensor(self._max_range, device=self.device, dtype=torch.float32) - z_expected)/(sigma*math.sqrt(2))))
        cdf0 = 0.5*(1+torch.erf((0 - z_expected)/(sigma*math.sqrt(2))))
        eta = cdf - cdf0
        pdf = 1/(sigma*math.sqrt(2*math.pi))*torch.exp(-0.5*((z_measure-z_expected)/sigma)**2)
        prob = pdf/eta
        return prob * valid.float()
    

    def _exp_density(self, z_expected, z_measure):
        norm_factor = torch.where(z_expected > 0, 1 - torch.exp(-self._lambda_short * z_expected), torch.tensor(1.0, device=self.device))
        p = torch.where((z_measure <= z_expected) & (z_expected > 0),
                        self._lambda_short * torch.exp(-self._lambda_short * z_measure) / norm_factor, torch.tensor(0.0, device=self.device))
        return p
    

    def _uniform_density(self, z_measure):
        return torch.where(z_measure < self._max_range, 1.0 / self._max_range, torch.tensor(0.0, device=self.device))
   
   
    def _max_density(self, z_measure):
        return torch.where(z_measure >= self._max_range, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
    
    
    def get_probability(self, z_expected, z_measure):
        p_hit = self._gaussian_density(z_expected, z_measure)
        p_short = self._exp_density(z_expected, z_measure)
        p_max = self._max_density(z_measure)
        p_rand = self._uniform_density(z_measure)
        p = (self._z_hit * p_hit + self._z_short * p_short + self._z_max * p_max + self._z_rand * p_rand)
        return p
    
    
    def beam_range_finder_model(self, z_measurements, x_t1):
        L = 25
        sensor_x = x_t1[:, 0] + L * torch.cos(x_t1[:, 2])
        sensor_y = x_t1[:, 1] + L * torch.sin(x_t1[:, 2])
        grid_x = (sensor_x / self._map_resolution).to(torch.int64)
        grid_y = (sensor_y / self._map_resolution).to(torch.int64)
        full_scans = self.raycast_map[grid_y, grid_x, :]
        theta_deg = (x_t1[:, 2] * 180 / math.pi) % 360
        start_angles = ((theta_deg - 90) % 360).to(torch.int64)
        end_angles = ((theta_deg + 90) % 360).to(torch.int64)
        scans = []
        for i in range(x_t1.shape[0]):
            full_scan = full_scans[i]
            start_angle = start_angles[i].item()
            end_angle = end_angles[i].item()
            if start_angle <= end_angle:
                scan = full_scan[start_angle:end_angle+1]
            else:
                scan = torch.cat((full_scan[start_angle:], full_scan[:end_angle+1]))
            scans.append(scan)
        scans = torch.stack(scans)
        if scans.shape[1] != self._num_rays:
            scans_interp = []
            for scan in scans:
                orig_indices = torch.arange(scan.shape[0], device=self.device, dtype=torch.float32)
                target_indices = torch.linspace(0, scan.shape[0]-1, steps=self._num_rays, device=self.device)
                left = torch.floor(target_indices).long()
                right = torch.clamp(left+1, max=scan.shape[0]-1)
                weight = target_indices - left.float()
                scan_interp = scan[left]*(1-weight) + scan[right]*weight
                scans_interp.append(scan_interp)
            scans = torch.stack(scans_interp)
        z_expected = scans
        angles = torch.linspace(-math.pi/2, math.pi/2, steps=self._num_rays, device=self.device)
        x_hits = sensor_x.unsqueeze(1) + z_expected * torch.cos(x_t1[:, 2].unsqueeze(1) + angles.unsqueeze(0))
        y_hits = sensor_y.unsqueeze(1) + z_expected * torch.sin(x_t1[:, 2].unsqueeze(1) + angles.unsqueeze(0))
        if z_measurements.shape[0] != self._num_rays:
            subsample = z_measurements.shape[0] // self._num_rays
            z_meas_sub = z_measurements[::subsample][:self._num_rays]
        else:
            z_meas_sub = z_measurements
        probs = self.get_probability(z_expected, z_meas_sub.unsqueeze(0))
        log_prob_sum = torch.sum(torch.log(probs + 1e-12), dim=1)
        weight = torch.exp(log_prob_sum)
        return weight, probs, x_hits, y_hits
    
    
    def ray_casting_all(self, origin):
        angles = torch.deg2rad(torch.arange(360, device=self.device, dtype=torch.float32)).unsqueeze(1)
        dist_step = torch.linspace(0, self._max_range, self._interpolation_num, device=self.device).unsqueeze(0)
        x_points = origin[0] + dist_step * torch.cos(angles)
        y_points = origin[1] + dist_step * torch.sin(angles)
        grid_x = (x_points / self._map_resolution).to(torch.int64)
        grid_y = (y_points / self._map_resolution).to(torch.int64)
        valid = (grid_x >= 0) & (grid_x < self._occupancy_map.shape[1]) & (grid_y >= 0) & (grid_y < self._occupancy_map.shape[0])
        blocked = torch.zeros_like(valid, dtype=torch.bool)
        if valid.any():
            blocked[valid] = (torch.abs(self._occupancy_map[grid_y[valid], grid_x[valid]]) > self._min_probability)
        hit_condition = (~valid) | blocked
        has_hit = torch.any(hit_condition, dim=1)
        first_hit_idx = torch.argmax(hit_condition.to(torch.int64), dim=1)

        ds_vector = torch.linspace(0, self._max_range, self._interpolation_num, device=self.device)
        distances = ds_vector[first_hit_idx]
        distances = torch.where(has_hit, distances, torch.tensor(self._max_range, device=self.device, dtype=torch.float32))
        return distances
    
    
    def _compute_raycast_for_cell(self, cell):
        y, x = cell
        if torch.abs(self._occupancy_map[y, x]) > self._min_probability:
            return torch.full((360,), self._max_range, dtype=torch.float32, device=self.device)
        origin = (x * self._map_resolution, y * self._map_resolution)
        return self.ray_casting_all(torch.tensor(origin, device=self.device, dtype=torch.float32))
    
    
    def _compute_raycast_for_x(self, y):
        w = self._occupancy_map.shape[1]
        row_result = torch.empty((w, 360), dtype=torch.float32, device=self.device)
        for x in range(w):
            row_result[x, :] = self._compute_raycast_for_cell((y, x))
        return row_result
    
    
    def download_ray_mapping(self):
        h = self._occupancy_map.shape[0]
        w = self._occupancy_map.shape[1]
        raycast_map = torch.empty((h, w, 360), dtype=torch.float32, device=self.device)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(tqdm(executor.map(self._compute_raycast_for_x, range(h)), total=h, desc="Precomputing raycast map"))
        for i, row in enumerate(results):
            raycast_map[i, :, :] = row
        return raycast_map
