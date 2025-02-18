'''
    Adapted from course 16831 (Statistical Techniques).
    Initially written by Paloma Sodhi (psodhi@cs.cmu.edu), 2018
'''

import sys
import numpy as np
import math


class MotionModel:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox.
    Probabilistic Robotics. MIT Press, 2005.
    [Chapter 5]
    """
    def __init__(self):
        """
        Tune Motion Model parameters here.
        The original numbers are for reference but may need further tuning.
        """
        self._alpha1 = 0.0005
        self._alpha2 = 0.0005
        self._alpha3 = 0.001
        self._alpha4 = 0.001


    def wrapToPi(self, angle):
        """
        Wrap an angle in radians to [-pi, pi].
        """
        return angle - 2 * np.pi * np.floor((angle + np.pi) / (2 * np.pi))

    def sample(self, mu, sigma):
        """
        Sample from a normal distribution with mean mu and standard deviation sigma.
        """
        return np.random.normal(mu, sigma)

    def update(self, u_t0, u_t1, x_t0):
        """
        Update the particle state given odometry readings.
        
        Args:
            u_t0 (np.array): Odometry reading at time t-1 [x, y, theta].
            u_t1 (np.array): Odometry reading at time t [x, y, theta].
            x_t0 (np.array): Particle state at time t-1 [x, y, theta].
        
        Returns:
            np.array: Updated particle state at time t [x, y, theta].
        """
        # Check for no motion: if odometry readings haven't changed, return current state.
        if np.allclose(u_t0, u_t1):
            return x_t0

        # Compute differences in odometry.
        delta_x = u_t1[0] - u_t0[0]
        delta_y = u_t1[1] - u_t0[1]
        delta_theta = u_t1[2] - u_t0[2]

        deltaR1 = np.arctan2(delta_y, delta_x) - u_t0[2]
        deltaR1 = self.wrapToPi(deltaR1)
        deltaTrans = np.sqrt(delta_x**2 + delta_y**2)
        deltaR2 = delta_theta - deltaR1
        deltaR2 = self.wrapToPi(deltaR2)

        Rot1 = deltaR1 - self.sample(0, np.sqrt(self._alpha1 * deltaR1**2 + self._alpha2 * deltaTrans**2))
        Trans = deltaTrans - self.sample(0, np.sqrt(self._alpha3 * deltaTrans**2 +
                                                      self._alpha4 * deltaR1**2 +
                                                      self._alpha4 * deltaR2**2))
        Rot2 = deltaR2 - self.sample(0, np.sqrt(self._alpha1 * deltaR2**2 + self._alpha2 * deltaTrans**2))

        # Wrap the angles to [-pi, pi].
        Rot1 = self.wrapToPi(Rot1)
        Rot2 = self.wrapToPi(Rot2)

        # Compute the new state.
        new_x = x_t0[0] + Trans * np.cos(x_t0[2] + Rot1)
        new_y = x_t0[1] + Trans * np.sin(x_t0[2] + Rot1)
        new_theta = self.wrapToPi(x_t0[2] + Rot1 + Rot2)

        return np.array([new_x, new_y, new_theta])
