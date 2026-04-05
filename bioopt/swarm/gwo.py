"""Grey Wolf Optimizer (GWO) algorithm.

Based on: Mirjalili, S., Mirjalili, S.M. and Lewis, A., 2014.
"Grey wolf optimizer." Advances in engineering software, 69, pp.46-61.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
from numba import njit

from bioopt.base import BaseOptimizer


@njit(fastmath=True)
def _gwo_step_njit(
    positions: np.ndarray,
    fitness: np.ndarray,
    a_coef: float,
    bounds: np.ndarray,
    r1: np.ndarray,
    r2: np.ndarray,
) -> np.ndarray:
    n_agents, n_dims = positions.shape
    alpha_idx = 0
    beta_idx = 1
    delta_idx = 2
    for i in range(1, n_agents):
        if fitness[i] < fitness[alpha_idx]:
            delta_idx = beta_idx
            beta_idx = alpha_idx
            alpha_idx = i
        elif i != alpha_idx and (beta_idx == alpha_idx or fitness[i] < fitness[beta_idx]):
            if beta_idx == alpha_idx or fitness[i] < fitness[beta_idx]:
                delta_idx = beta_idx
                beta_idx = i
    # Ensure distinct
    for i in range(n_agents):
        if i != alpha_idx and i != beta_idx:
            delta_idx = i
            break

    alpha_pos = positions[alpha_idx]
    beta_pos = positions[beta_idx]
    delta_pos = positions[delta_idx]

    new_positions = np.empty_like(positions)
    for i in range(n_agents):
        for d in range(n_dims):
            A1 = 2.0 * a_coef * r1[i, 0] - a_coef
            A2 = 2.0 * a_coef * r1[i, 1] - a_coef
            A3 = 2.0 * a_coef * r1[i, 2] - a_coef
            C1 = 2.0 * r2[i, 0]
            C2 = 2.0 * r2[i, 1]
            C3 = 2.0 * r2[i, 2]
            D_alpha = abs(C1 * alpha_pos[d] - positions[i, d])
            D_beta = abs(C2 * beta_pos[d] - positions[i, d])
            D_delta = abs(C3 * delta_pos[d] - positions[i, d])
            X1 = alpha_pos[d] - A1 * D_alpha
            X2 = beta_pos[d] - A2 * D_beta
            X3 = delta_pos[d] - A3 * D_delta
            new_positions[i, d] = (X1 + X2 + X3) / 3.0
            if new_positions[i, d] < bounds[d, 0]:
                new_positions[i, d] = bounds[d, 0]
            elif new_positions[i, d] > bounds[d, 1]:
                new_positions[i, d] = bounds[d, 1]
    return new_positions


class GWO(BaseOptimizer):
    """Grey Wolf Optimizer."""

    def __init__(self, n_agents: int, bounds: Union[list, np.ndarray], seed: Optional[int] = None):
        super().__init__(n_agents, bounds, seed)

    def step(self, positions: np.ndarray, fitness: np.ndarray, iteration: int, **kwargs) -> np.ndarray:
        max_iter = kwargs.get("max_iterations", 100)
        a_coef = 2.0 * (1.0 - iteration / max_iter)
        r1 = self.rng.random((self.n_agents, 3))
        r2 = self.rng.random((self.n_agents, 3))
        return _gwo_step_njit(positions, fitness, a_coef, self.bounds, r1, r2)

    def optimize(self, objective_fn: Callable, iterations: int = 100, verbose: bool = False, callback: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, float]:
        return super().optimize(objective_fn, iterations=iterations, verbose=verbose, callback=callback, max_iterations=iterations, **kwargs)