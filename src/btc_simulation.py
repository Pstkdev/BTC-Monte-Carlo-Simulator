import numpy as np
import pandas as pd


class BTCMonteCarlo:

    STEPS_PER_YEAR = 365  # BTC trades 24/7

    def __init__(self, start_price, mu, sigma, years, num_simulations, seed):
        self.start_price = start_price
        self.mu = mu
        self.sigma = sigma
        self.years = years
        self.num_simulations = num_simulations
        self.seed = seed

        # check input parameters
        if start_price <= 0:
            raise ValueError("Start price must be greater than zero.")

        if num_simulations <= 0:
            raise ValueError("Number of simulations must be greater than zero.")

        if years <= 0:
            raise ValueError("Years must be greater than zero.")

        if sigma < 0:
            raise ValueError("Sigma must be non-negative.")

        if seed is not None and (not isinstance(seed, int) or seed < 0):
            raise ValueError("Seed must be a non-negative integer or None.")

    def simulate_paths(self) -> np.ndarray:
        """
        Simulate BTC price paths using GBM model

        Returns Matrix of simulated prices with shape (num_simulations, num_steps + 1)
            - each row is one simulated path
            - column 0 is the initial price
            - last column is the final price
        """
        dt = 1 / self.STEPS_PER_YEAR
        num_steps = int(self.years * self.STEPS_PER_YEAR)
        rng = np.random.default_rng(self.seed)

        # Z ~ N(0,1): random shocks for each simulation and each day
        Z = rng.standard_normal((self.num_simulations, num_steps))

        # GBM in log-space
        # log_return = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z
        drift = (self.mu - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt) * Z
        log_r = drift + diffusion

        # Convert log returns into price paths
        # cumulative sum of log returns gives ln(St / S0)
        # then St = S0 * exp(cumsum(log_returns))
        paths = np.zeros((self.num_simulations, num_steps + 1), dtype=float)
        paths[:, 0] = self.start_price
        paths[:, 1:] = self.start_price * np.exp(np.cumsum(log_r, axis=1))

        return paths

    def compute_quantiles(self, paths: np.ndarray, qs=(0.1, 0.5, 0.9)) -> np.ndarray:
        """
        Compute quantile curves across simulations for each time step.
        """
        return np.quantile(paths, qs, axis=0)
