"""Firefly Algorithm (FA).

Based on: Yang, X.S., 2010. "Firefly algorithm, stochastic test functions 
and design optimisation." International journal of bio-inspired computation, 2(2), pp.78-84.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
from numba import njit

from bioopt.base import BaseOptimizer


@njit(fastmath=True)
def _firefly_step_njit(
    positions: np.ndarray, fitness: np.ndarray,
    alpha: float, beta_0: float, gamma: float,
    bounds: np.ndarray, rand_vals: np.ndarray,
) -> np.ndarray:
    n_agents, n_dims = positions.shape
    new_positions = positions.copy()
    idx = 0
    for i in range(n_agents):
        for j in range(n_agents):
            if j != i and fitness[j] < fitness[i]:
                dist_sq = 0.0
                for d in range(n_dims):
                    diff = positions[i, d] - positions[j, d]
                    dist_sq += diff * diff
                r = np.sqrt(dist_sq)
                beta = beta_0 * np.exp(-gamma * r * r)
                for d in range(n_dims):
                    rn = rand_vals[idx % len(rand_vals)]
                    idx += 1
                    r_ij = positions[j, d] - positions[i, d]
                    new_positions[i, d] += beta * r_ij + alpha * (rn - 0.5)
    for i in range(n_agents):
        for d in range(n_dims):
            if new_positions[i, d] < bounds[d, 0]:
                new_positions[i, d] = bounds[d, 0]
            elif new_positions[i, d] > bounds[d, 1]:
                new_positions[i, d] = bounds[d, 1]
    return new_positions


class FA(BaseOptimizer):
    """Firefly Algorithm."""

    def __init__(
        self, n_agents: int, bounds: Union[list, np.ndarray],
        alpha: float = 0.2, beta_0: float = 1.0, gamma: float = 1.0,
        alpha_decay: Optional[float] = None, seed: Optional[int] = None,
    ):
        super().__init__(n_agents, bounds, seed)
        self.alpha = alpha
        self.beta_0 = beta_0
        self.gamma = gamma
        self.alpha_decay = alpha_decay

    def step(self, positions: np.ndarray, fitness: np.ndarray, iteration: int, **kwargs) -> np.ndarray:
        alpha = kwargs.get("alpha", self.alpha)
        beta_0 = kwargs.get("beta_0", self.beta_0)
        gamma = kwargs.get("gamma", self.gamma)
        alpha_decay = kwargs.get("alpha_decay", self.alpha_decay)
        if alpha_decay is not None and iteration > 0:
            alpha = alpha * (alpha_decay ** iteration)
        n_rand = self.n_agents * self.n_agents * self.n_dims
        rand_vals = self.rng.random(n_rand)
        return _firefly_step_njit(positions, fitness, alpha, beta_0, gamma, self.bounds, rand_vals)