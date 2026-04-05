"""Whale Optimization Algorithm (WOA).

Based on: Mirjalili, S. and Lewis, A., 2016. "The whale optimization algorithm."
Advances in engineering software, 95, pp.51-67.
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
from numba import njit

from bioopt.base import BaseOptimizer


@njit(fastmath=True)
def _woa_step_njit(
    positions: np.ndarray, fitness: np.ndarray,
    a_coef: float, a2_coef: float, bounds: np.ndarray,
    r_vals: np.ndarray,
) -> np.ndarray:
    n_agents, n_dims = positions.shape
    best_idx = np.argmin(fitness)
    best_position = positions[best_idx]
    new_positions = np.empty_like(positions)
    idx = 0
    for i in range(n_agents):
        r1 = r_vals[idx]; idx += 1
        r2 = r_vals[idx]; idx += 1
        rc = r_vals[idx]; idx += 1
        A = 2.0 * a_coef * r1 - a_coef
        C = 2.0 * r2
        if rc < 0.5:
            if abs(A) < 1.0:
                for d in range(n_dims):
                    D = abs(C * best_position[d] - positions[i, d])
                    new_positions[i, d] = best_position[d] - A * D
            else:
                rand_idx = int(r_vals[idx]) % n_agents
                idx += 1
                X_rand = positions[rand_idx]
                for d in range(n_dims):
                    D_rand = abs(C * X_rand[d] - positions[i, d])
                    new_positions[i, d] = X_rand[d] - A * D_rand
        else:
            b = 1.0
            l = (a2_coef - 1.0) * r_vals[idx] + 1.0
            idx += 1
            for d in range(n_dims):
                distance_to_best = abs(best_position[d] - positions[i, d])
                new_positions[i, d] = distance_to_best * np.exp(b * l) * np.cos(2.0 * np.pi * l) + best_position[d]
        for d in range(n_dims):
            if new_positions[i, d] < bounds[d, 0]:
                new_positions[i, d] = bounds[d, 0]
            elif new_positions[i, d] > bounds[d, 1]:
                new_positions[i, d] = bounds[d, 1]
    return new_positions


class WOA(BaseOptimizer):
    """Whale Optimization Algorithm."""

    def __init__(self, n_agents: int, bounds: Union[list, np.ndarray], seed: Optional[int] = None):
        super().__init__(n_agents, bounds, seed)

    def step(self, positions: np.ndarray, fitness: np.ndarray, iteration: int, **kwargs) -> np.ndarray:
        max_iter = kwargs.get("max_iterations", 100)
        a_coef = 2.0 * (1.0 - iteration / max_iter)
        a2_coef = -1.0 + iteration * (-1.0 / max_iter)
        n_rand = self.n_agents * 4  # 3-4 random values per agent
        r_vals = self.rng.random(n_rand)
        return _woa_step_njit(positions, fitness, a_coef, a2_coef, self.bounds, r_vals)

    def optimize(self, objective_fn: Callable, iterations: int = 100, verbose: bool = False, callback: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, float]:
        return super().optimize(objective_fn, iterations=iterations, verbose=verbose, callback=callback, max_iterations=iterations, **kwargs)