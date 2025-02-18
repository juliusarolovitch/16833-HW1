'''
Adapted from course 16831 (Statistical Techniques).
Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
Updated by Wei Dong (weidong@andrew.cmu.edu), 2021
Modified to: move 1000 units east, turn north, move 500 units north, then turn west and move 1000 units west.
'''

import sys
import numpy as np
import math
import matplotlib.pyplot as plt
import imageio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox. Probabilistic robotics. MIT press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        # Tune these parameters to match your odometry noise characteristics
        self._alpha1 = 0.1
        self._alpha2 = 0.0005
        self._alpha3 = 1.00
        self._alpha4 = 0.001

    def update(self, u_t0, u_t1, x_t0):
        """
        param[in] u_t0 : odometry reading at time t-1 [x, y, theta] (odometry frame)
        param[in] u_t1 : odometry reading at time t [x, y, theta] (odometry frame)
        param[in] x_t0 : particle state at time t-1 [x, y, theta] (world frame)
        param[out] x_t1 : updated particle state at time t [x, y, theta] (world frame)
        """
        u = u_t1 - u_t0

        # Use the initial orientation from x_t0 for computing the first rotation
        rot1 = np.arctan2(u[1], u[0]) - x_t0[2]
        trans = np.sqrt(u[0]**2 + u[1]**2)
        rot2 = u[2] - rot1

        rot1_b = rot1 - np.random.normal(scale=np.sqrt(self._alpha1 * np.abs(rot1) +
                                                        self._alpha2 * np.abs(trans)))
        trans_b = trans - np.random.normal(scale=np.sqrt(self._alpha3 * np.abs(trans) +
                                                        self._alpha4 * np.abs(rot1 + rot2)))
        rot2_b = rot2 - np.random.normal(scale=np.sqrt(self._alpha1 * np.abs(rot2) +
                                                        self._alpha2 * np.abs(trans)))

        x_t1 = [0, 0, 0]
        x_t1[0] = x_t0[0] + trans_b * np.cos(x_t0[2] + rot1_b)
        x_t1[1] = x_t0[1] + trans_b * np.sin(x_t0[2] + rot1_b)
        x_t1[2] = x_t0[2] + rot1_b + rot2_b

        return np.array(x_t1)

# ---- Standalone Simulation ----
if __name__ == '__main__':
    np.random.seed(42)  # For reproducibility

    # --- Simulation Parameters ---
    num_particles = 500
    step_size = 50       # Forward movement in units per step
    map_width = 7000     # Map width (x-dimension)
    map_height = 4000    # Map height (y-dimension)
    
    mm = MotionModel()   # Initialize the motion model

    # Start at (500,500) facing east (0 radians)
    true_pose = np.array([500, 500, 0])
    u_prev = true_pose.copy()
    particles = np.tile(true_pose, (num_particles, 1))
    # Optionally add a small spread:
    # particles += np.random.normal(0, 2, particles.shape)

    frames = []               # For GIF frames
    true_path = [true_pose.copy()]  # To trace the true path

    def plot_state(particles, true_pose, path, title=""):
        """Plot the particles, the true pose (arrow), and the true path (blue trace),
           then return the figure as an image array."""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.set_xlim(0, map_width)
        ax.set_ylim(0, map_height)
        ax.set_title(title)
        ax.scatter(particles[:, 0], particles[:, 1], s=10, color='red', alpha=0.5, label="Particles")
        if len(path) > 1:
            path_arr = np.array(path)
            ax.plot(path_arr[:, 0], path_arr[:, 1], 'b-', lw=2, label="True Path")
        ax.arrow(true_pose[0], true_pose[1],
                 20 * np.cos(true_pose[2]), 20 * np.sin(true_pose[2]),
                 head_width=10, head_length=10, fc='blue', ec='blue', label="True Pose")
        ax.legend(loc='upper right')
        fig.tight_layout()

        canvas = FigureCanvas(fig)
        canvas.draw()
        buf, (width, height) = canvas.print_to_buffer()
        image = np.frombuffer(buf, dtype='uint8').reshape((height, width, 4))
        plt.close(fig)
        return image[:, :, :3]

    def update_particles(u_prev, u_new, particles):
        """Update each particle using the motion model with the given odometry change."""
        new_particles = []
        for i in range(num_particles):
            new_state = mm.update(u_prev, u_new, particles[i])
            new_particles.append(new_state)
        return np.array(new_particles)

    def move_forward(current_pose, step):
        """Compute new pose after moving forward one step (heading remains unchanged)."""
        new_pose = current_pose.copy()
        new_pose[0] += step * np.cos(current_pose[2])
        new_pose[1] += step * np.sin(current_pose[2])
        return new_pose

    # -------------------------
    # 1. Walk 1000 units east (20 steps)
    for step in range(20):
        new_pose = move_forward(true_pose, step_size)
        particles = update_particles(true_pose, new_pose, particles)
        true_pose = new_pose.copy()
        true_path.append(true_pose.copy())
        frames.append(plot_state(particles, true_pose, true_path, title=f"East Step {step+1}"))
        u_prev = new_pose.copy()

    # -------------------------
    # 2. Turn to face north (π/2 radians)
    new_pose = true_pose.copy()
    new_pose[2] = np.pi/2
    particles = update_particles(true_pose, new_pose, particles)
    true_pose = new_pose.copy()
    true_path.append(true_pose.copy())
    frames.append(plot_state(particles, true_pose, true_path, title="Turn North"))
    u_prev = new_pose.copy()

    # -------------------------
    # 3. Walk 500 units north (10 steps)
    for step in range(10):
        new_pose = move_forward(true_pose, step_size)
        particles = update_particles(true_pose, new_pose, particles)
        true_pose = new_pose.copy()
        true_path.append(true_pose.copy())
        frames.append(plot_state(particles, true_pose, true_path, title=f"North Step {step+1}"))
        u_prev = new_pose.copy()

    # -------------------------
    # 4. Turn to face west (π radians)
    new_pose = true_pose.copy()
    new_pose[2] = np.pi
    particles = update_particles(true_pose, new_pose, particles)
    true_pose = new_pose.copy()
    true_path.append(true_pose.copy())
    frames.append(plot_state(particles, true_pose, true_path, title="Turn West"))
    u_prev = new_pose.copy()

    # -------------------------
    # 5. Walk 1000 units west (20 steps)
    for step in range(20):
        new_pose = move_forward(true_pose, step_size)
        particles = update_particles(true_pose, new_pose, particles)
        true_pose = new_pose.copy()
        true_path.append(true_pose.copy())
        frames.append(plot_state(particles, true_pose, true_path, title=f"West Step {step+1}"))
        u_prev = new_pose.copy()

    # Save the frames as a GIF
    gif_filename = 'motion_model.gif'
    imageio.mimsave(gif_filename, frames, duration=0.5)
    print(f"GIF saved as {gif_filename}")
