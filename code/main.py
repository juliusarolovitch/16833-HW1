import argparse
import numpy as np
import os, glob, imageio
from map_reader import MapReader
from motion_model import MotionModel
from sensor_model import SensorModel
from resampling import Resampling
from matplotlib import pyplot as plt
import time

def visualize_timestep(X_bar, tstep, output_path):
    x_locs = X_bar[:, 0] / 10.0
    y_locs = X_bar[:, 1] / 10.0
    scat = plt.scatter(x_locs, y_locs, c='r', marker='o',s=10)
    plt.savefig('{}/{:04d}.png'.format(output_path, tstep))

    plt.pause(0.00001)
    scat.remove()

def init_particles_freespace(num_particles, occupancy_map):
    X_bar_init = []
    while len(X_bar_init) < num_particles:
        y_val = np.random.uniform(3500, 4500)
        x_val = np.random.uniform(4200, 5200)
        theta_val = np.random.uniform(-np.pi, np.pi)
        if abs(occupancy_map[int(np.floor(y_val/10)), int(np.floor(x_val/10))]) < 0.2:
            X_bar_init.append([x_val, y_val, theta_val, 1])
    return np.array(X_bar_init)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_map', default='../data/map/wean.dat')
    parser.add_argument('--path_to_log', default='../data/log/robotdata1.log')
    parser.add_argument('--output', default='results')
    parser.add_argument('--num_particles', default=10000, type=int)
    parser.add_argument('--visualize', action='store_true')
    args = parser.parse_args()

    src_path_map = args.path_to_map
    src_path_log = args.path_to_log
    os.makedirs(args.output, exist_ok=True)
    map_obj = MapReader(src_path_map)
    occupancy_map = map_obj.get_map()
    logfile = open(src_path_log, 'r')
    motion_model = MotionModel()
    sensor_model = SensorModel(occupancy_map)
    resampler = Resampling()
    num_particles = args.num_particles
    X_bar = init_particles_freespace(num_particles, occupancy_map)
    
    if args.visualize:
        fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
        ax.imshow(occupancy_map, cmap='Greys', origin='lower', extent=[0,800,0,800])
        ax.set_xlim(0, 800)
        ax.set_ylim(0, 800)
    
    first_time_idx = True

    for time_idx, line in enumerate(logfile):
        if time_idx >= 2218:
            break

        meas_type = line[0]
        meas_vals = np.fromstring(line[2:], dtype=np.float64, sep=' ')
        odometry_robot = meas_vals[0:3]
        time_stamp = meas_vals[-1]
        if meas_type == "L":
            odometry_laser = meas_vals[3:6]
            ranges = meas_vals[6:-1]
        print("Processing time step {} at time {}s".format(time_idx, time_stamp))
        if first_time_idx:
            u_t0 = odometry_robot
            first_time_idx = False
            continue
        
        X_bar_new = np.zeros((num_particles, 4), dtype=np.float64)
        u_t1 = odometry_robot
        for m in range(0, num_particles):
            """
            MOTION MODEL
            """
            x_t0 = X_bar[m, 0:3]
            x_t1 = motion_model.update(u_t0, u_t1, x_t0)

            xInt = int(x_t1[0]/10.0)
            yInt = int(x_t1[1]/10.0)

            if np.abs(occupancy_map[yInt, xInt]) >= 0.2 :
                w_t = 0
                probs = 0
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
                continue

            """
            SENSOR MODEL
            """
            if (meas_type == "L"):
                z_t = ranges

                w_t, probs, laserX, laserY = sensor_model.beam_range_finder_model(z_t, x_t1)
                X_bar_new[m, :] = np.hstack((x_t1, w_t))
                if args.visualize and num_particles == 1:
                    visualize_timestep(X_bar, time_idx, args.output)

            else:
                X_bar_new[m, :] = np.hstack((x_t1, X_bar[m, 3]))

        X_bar = X_bar_new
        u_t0 = u_t1

        """
        RESAMPLING
        """
        if (meas_type == "L"):
            X_bar = resampler.low_variance_sampler(X_bar)
        
        if args.visualize and num_particles > 1 and time_idx%1==0:
            visualize_timestep(X_bar, time_idx, args.output)

    logfile.close()

    if args.visualize:
        images = []
        for filename in sorted(os.listdir(args.output)):
            if filename.endswith(".png"):
                images.append(imageio.imread(os.path.join(args.output, filename)))
        imageio.mimsave(os.path.join(args.output, 'output.gif'), images, duration=0.5)
