'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
    Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
'''

#!/usr/bin/env python3
import argparse
import torch
import sys, os
import matplotlib
matplotlib.use('Agg')
from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling
from matplotlib import pyplot as plt
import time

def visualize_map(occupancy_map):
    fig = plt.figure()
    plt.ion()
    plt.imshow(occupancy_map.cpu().numpy(), cmap='Greys')
    plt.axis([0, 800, 0, 800])


def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs.cpu().numpy(), y_locs.cpu().numpy(), c='r', marker='o')
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))
    plt.pause(0.00001)
    scat.remove()


def init_particles_freespace(num_particles, occupancy_map, device):
    X_bar_init = torch.empty((num_particles,4), device=device, dtype=torch.float32)
    for i in range(num_particles):
        while True:
            y0 = torch.empty(1, device=device).uniform_(0,7000).item()
            x0 = torch.empty(1, device=device).uniform_(3000,7000).item()
            theta0 = torch.empty(1, device=device).uniform_(-torch.pi, torch.pi).item()
            wt = 1.0/num_particles
            y_map = int(y0//10)
            x_map = int(x0//10)
            if occupancy_map[y_map, x_map] == 0:
                X_bar_init[i] = torch.tensor([x0, y0, theta0, wt], device=device)
                break
    return X_bar_init


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results_1')
    parser.add_argument('--num_particles', default=3000, type=int)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--path_to_raycast_map', default='rays.pt')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)
    map_obj = MapReader(src_path_map)
    occupancy_map_np = map_obj.get_map()
    occupancy_map = torch.tensor(occupancy_map_np.copy(), dtype=torch.float32, device=device)
    logfile = open(src_path_log, 'r')
    motion_model = MotionModel(device)
    sensor_model = SensorModel(occupancy_map, device, saved_rays_precalc_path=args.path_to_raycast_map)
    resampler = Resampling(device)
    num_particles = args.num_particles
    X_bar = init_particles_freespace(num_particles, occupancy_map, device)
    if not os.path.exists(args.path_to_raycast_map):
        raycast_map = sensor_model.download_ray_mapping()
        torch.save(raycast_map, args.path_to_raycast_map)
    else:
        raycast_map = torch.load(args.path_to_raycast_map, map_location=device)
        sensor_model.raycast_map = raycast_map
    if args.visualize:
        visualize_map(occupancy_map)
    start = time.time()
    first_time_idx = True
    for time_idx, line in enumerate(logfile):
        meas_type = line[0]
        meas_vals = torch.tensor(list(map(float, line[2:].strip().split())), dtype=torch.float32, device=device)
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1].item()
        if meas_type == "L":
            odometry_laser = meas_vals[3:6]
            ranges = meas_vals[6:-1]
        print("Processing time step {} at time {}s".format(time_idx, time_stamp))
        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue
        u_t1 = odometry_robot
        new_states = motion_model.update(u_t0, u_t1, X_bar[:, :3])
        if meas_type == "L":
            weight, probs, x_hits, y_hits = sensor_model.beam_range_finder_model(ranges, new_states)
            X_bar = torch.cat((new_states, weight.view(-1,1)), dim=1)
        else:
            X_bar = torch.cat((new_states, X_bar[:, 3].view(-1,1)), dim=1)
        u_t0 = u_t1
        X_bar = resampler.adaptive_resample(X_bar)
        if args.visualize:
            visualize_timestep(X_bar, time_idx, args.output)
        print("Processing each data line in {:.2f}s".format(time.time() - start))
    print("Total time: {}s".format(time.time() - start))
