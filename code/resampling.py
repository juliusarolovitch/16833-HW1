import numpy as np

class Resampling:
    """
    References: Thrun, Sebastian, Wolfram Burgard, and Dieter Fox.
    Probabilistic Robotics. MIT Press, 2005.
    [Chapter 4.3]
    """
    def __init__(self):
        pass

    def multinomial_sampler(self, X_bar):
        """
        Parameters:
            X_bar: [num_particles x 4] array containing [x, y, theta, wt] for all particles.
        Returns:
            X_bar_resampled: [num_particles x 4] array of resampled particles.
        """
        # TODO: Implement the multinomial sampler if needed.
        X_bar_resampled = np.zeros_like(X_bar)
        return X_bar_resampled

    def low_variance_sampler(self, X_bar):
        """
        Parameters:
            X_bar: [num_particles x 4] array containing [x, y, theta, wt] for all particles.
                   (Weights are not assumed to be normalized.)
        Returns:
            X_bar_resampled: [num_particles x 4] array of resampled particles.
        """
        num_particles = len(X_bar)
        weights = X_bar[:, 3]
        total_weight = np.sum(weights)
        step = total_weight / num_particles

        r = np.random.uniform(0, step)
        c = weights[0]
        i = 0
        X_bar_resampled = np.zeros_like(X_bar)

        for m in range(num_particles):
            U = r + m * step
            while U > c:
                i += 1
                c += weights[i]
            X_bar_resampled[m] = X_bar[i]

        X_bar_resampled[:, 3] = total_weight / num_particles
        return X_bar_resampled
