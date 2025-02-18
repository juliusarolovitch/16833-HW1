import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import norm

class SensorModel:
    """
    Sensor Model based on probabilistic robotics.
    References: Thrun, Burgard, and Fox. Probabilistic Robotics. MIT Press, 2005.
    [Chapter 6.3]
    """
    def __init__(self, occupancy_map):
        self._z_hit = 180
        self._z_short = 17.5
        self._z_max = 15 
        self._z_rand = 85
        self._sigma_hit = 100
        self._lambda_short = 15
        
        self._max_range = 1000  # Maximum sensor range (in mm)
        self._min_probability = 0.3  # Minimum occupancy probability to consider a cell as blocked
        self._map_resolution = 10  # Map resolution (mm per cell)
        self._ray_step = 1     # Step size for ray casting (mm)
        self._num_rays = 180     # Number of rays used for ray casting (subsampled from 180)

        self._occupancy_map = occupancy_map
        self._x_max = self._occupancy_map.shape[0] * self._map_resolution
        self._y_max = self._occupancy_map.shape[1] * self._map_resolution

    def wrapToPi(self, angle):
        """Wrap angle to [-pi, pi]."""
        return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))

    def _ray_cast_all(self, start_x, start_y, start_theta, angles):
        """
        Vectorized ray casting.
        Args:
            start_x, start_y: robot position.
            start_theta: robot heading.
            angles: array of beam angles (relative to robot heading).
        Returns:
            x_hit, y_hit: coordinates of hit points for each ray.
        """
        # Sensor offset from robot (e.g., mounting position)
        L = 25  
        sensor_x = start_x + L * np.cos(start_theta)
        sensor_y = start_y + L * np.sin(start_theta)
        # World-frame beam angles
        ray_angles = start_theta + angles  
        # Sample points along each ray
        t = np.arange(0, self._max_range, self._ray_step)
        ts = t[None, :]  # shape (1, T)
        xs = sensor_x + ts * np.cos(ray_angles)[:, None]  # shape (num_rays, T)
        ys = sensor_y + ts * np.sin(ray_angles)[:, None]
        # Determine which sample points are out-of-bound
        oob = (xs < 0) | (xs >= self._x_max) | (ys < 0) | (ys >= self._y_max)
        inbound = ~oob
        grid_x = np.zeros_like(xs, dtype=int)
        grid_y = np.zeros_like(ys, dtype=int)
        grid_x[inbound] = (xs[inbound] / self._map_resolution).astype(int)
        grid_y[inbound] = (ys[inbound] / self._map_resolution).astype(int)
        # A cell is considered blocked if its occupancy value exceeds the minimum probability
        blocked = np.zeros_like(xs, dtype=bool)
        blocked[inbound] = (np.abs(self._occupancy_map[grid_y[inbound], grid_x[inbound]]) > self._min_probability)
        hit = oob | blocked
        # For each ray, get the first index where a hit occurs
        hit_any = np.any(hit, axis=1)
        idx = np.argmax(hit, axis=1)  # returns first True index along each ray
        t_hit = np.where(hit_any, t[idx], self._max_range)
        x_hit = sensor_x + t_hit * np.cos(ray_angles)
        y_hit = sensor_y + t_hit * np.sin(ray_angles)
        return x_hit, y_hit

    def _gaussian_density(self, z_expected, z_measure):
        """
        Gaussian (hit) probability.
        """
        valid = (z_measure >= 0) & (z_measure <= self._max_range) & (z_expected >= 0) & (z_expected <= self._max_range)
        eta = 1.0 / (self._sigma_hit * np.sqrt(2 * np.pi))
        prob = eta * np.exp(-0.5 * ((z_measure - z_expected) / self._sigma_hit) ** 2)
        return prob * valid

    def _exp_density(self, z_expected, z_measure):
        """
        Exponential (short) probability.
        Only nonzero if the measurement is less than the expected range.
        """
        return np.where(z_measure <= z_expected, self._lambda_short * np.exp(-self._lambda_short * z_measure), 0)

    def _uniform_density(self, z_measure):
        """
        Uniform (random) probability.
        """
        return np.where(z_measure < self._max_range, 1.0 / self._max_range, 0)

    def _max_density(self, z_measure):
        """
        Delta (max) probability.
        """
        return np.where(z_measure >= self._max_range, 1, 0)

    def get_probability(self, z_expected, z_measure):
        """
        Compute the likelihood of a single measurement given the expected measurement.
        Returns a tuple (p, p_hit, p_short, p_max, p_rand).
        """
        p_hit = self._gaussian_density(z_expected, z_measure)
        p_short = self._exp_density(z_expected, z_measure)
        p_max = self._max_density(z_measure)
        p_rand = self._uniform_density(z_measure)

        p = (self._z_hit * p_hit +
             self._z_short * p_short +
             self._z_max * p_max +
             self._z_rand * p_rand)
        normalizer = self._z_hit + self._z_short + self._z_max + self._z_rand
        p /= normalizer
        return p, p_hit, p_short, p_max, p_rand

    def beam_range_finder_model(self, z_measurements, x_t1):
        """
        Compute the likelihood of an entire laser scan given a particle's pose.
        
        Args:
            z_measurements: array of actual laser range readings (e.g., 180 values).
            x_t1: particle state [x, y, theta] in the world frame.
        
        Returns:
            weight: the overall likelihood (sum of log probabilities),
            probs: individual beam probabilities,
            x_hits, y_hits: expected hit locations from ray casting.
        """
        x_o, y_o, theta_o = x_t1
        L = 25
        sensor_x = x_o + L * np.cos(theta_o)
        sensor_y = y_o + L * np.sin(theta_o)
        # Assume the scan covers [-pi/2, pi/2]
        angles = np.linspace(np.pi/2, -np.pi/2, self._num_rays)
        # Get expected hit locations via ray casting
        x_hits, y_hits = self._ray_cast_all(x_o, y_o, theta_o, angles)
        # Compute expected ranges from the sensor position
        z_expected = np.sqrt((x_hits - sensor_x)**2 + (y_hits - sensor_y)**2)
        
        # Subsample the actual measurements to match _num_rays.
        z_measurements = np.array(z_measurements)
        subsample = len(z_measurements) // self._num_rays
        z_meas_sub = z_measurements[::subsample][:self._num_rays]

        probs = np.zeros(self._num_rays)
        log_prob_sum = 0
        for i in range(self._num_rays):
            p, _, _, _, _ = self.get_probability(z_expected[i], z_meas_sub[i])
            probs[i] = p
            log_prob_sum += np.log(p + 1e-12)  # Add small epsilon to avoid log(0)

        # Use sum of log probabilities for numerical stability
        weight = self._num_rays / np.abs(log_prob_sum)  # Heuristic normalization
        return weight, probs, x_hits, y_hits

    def get_weight(self, z_measurements, x_t1):
        """
        Returns the likelihood weight for a given particle state x_t1 and sensor reading.
        """
        weight, _, _, _ = self.beam_range_finder_model(z_measurements, x_t1)
        return weight

    def _coord_to_blocked(self, x, y):
        """
        Check if the given coordinate (in world units) is in a blocked cell.
        """
        grid_x = int(x / self._map_resolution)
        grid_y = int(y / self._map_resolution)
        prob = abs(self._occupancy_map[grid_y, grid_x])
        return prob > self._min_probability