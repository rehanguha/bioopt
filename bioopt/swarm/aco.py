"""Ant Colony Optimization for Continuous Domains (ACO_R) algorithm."""

from typing import Callable, Optional, Tuple, Union

import numpy as np
from numba import njit

from bioopt.base import BaseOptimizer


@njit(fastmath=True)
def _aco_step_njit(
    archive_solutions: np.ndarray,
    archive_fitness: np.ndarray,
    n_ants: int,
    bounds: np.ndarray,
    q: float,
    xi: float,
    r_kernel: np.ndarray,
    r_dim: np.ndarray,
) -> np.ndarray:
    n_solutions, n_dims = archive_solutions.shape
    ranks = np.argsort(np.argsort(archive_fitness)) + 1
    qk = q * n_solutions
    weights = np.empty(n_solutions)
    for l in range(n_solutions):
        weights[l] = np.exp(-ranks[l] ** 2 / (2.0 * qk * qk))
    ws = np.sum(weights)
    for l in range(n_solutions):
        weights[l] /= ws

    sigmas = np.empty(n_dims)
    for d in range(n_dims):
        mean_d = 0.0
        for l in range(n_solutions):
            mean_d += weights[l] * archive_solutions[l, d]
        var_d = 0.0
        for l in range(n_solutions):
            diff = archive_solutions[l, d] - mean_d
            var_d += weights[l] * diff * diff
        sigmas[d] = np.sqrt(max(var_d, 1e-10)) * xi
        range_d = bounds[d, 1] - bounds[d, 0]
        sigmas[d] = max(sigmas[d], range_d * 1e-6)

    new_ants = np.empty((n_ants, n_dims))
    for i in range(n_ants):
        ki = int(r_kernel[i])
        if ki >= n_solutions:
            ki = n_solutions - 1
        for d in range(n_dims):
            val = archive_solutions[ki, d] + sigmas[d] * r_dim[i, d]
            if val < bounds[d, 0]:
                val = bounds[d, 0]
            elif val > bounds[d, 1]:
                val = bounds[d, 1]
            new_ants[i, d] = val
    return new_ants


class ACO(BaseOptimizer):
    """Ant Colony Optimization for Continuous Domains (ACO_R)."""

    def __init__(
        self, n_agents: int, bounds: Union[list, np.ndarray],
        archive_size: int = 100, q: float = 0.01, xi: float = 0.1,
        seed: Optional[int] = None,
    ):
        archive_size = max(archive_size, n_agents)
        super().__init__(archive_size, bounds, seed)
        self.n_ants = n_agents
        self.q = q
        self.xi = xi
        self.full_archive_size = archive_size
        self.archive_solutions: Optional[np.ndarray] = None
        self.archive_fitness: Optional[np.ndarray] = None

    def initialize_population(self) -> np.ndarray:
        self.archive_solutions = super().initialize_population()
        self.archive_fitness = np.full(self.full_archive_size, np.inf)
        return self.archive_solutions[:self.n_ants].copy()

    def step(self, positions: np.ndarray, fitness: np.ndarray, iteration: int, **kwargs) -> np.ndarray:
        q = kwargs.get("q", self.q)
        xi = kwargs.get("xi", self.xi)
        # Update archive
        all_solutions = np.vstack([self.archive_solutions, positions])
        all_fitness = np.concatenate([self.archive_fitness, fitness])
        sorted_idx = np.argsort(all_fitness)
        top_idx = sorted_idx[:self.full_archive_size]
        self.archive_solutions = all_solutions[top_idx].copy()
        self.archive_fitness = all_fitness[top_idx].copy()
        # Sample
        r_kernel = self.rng.uniform(0, self.full_archive_size, size=self.n_ants)
        r_dim = self.rng.normal(0, 1, size=(self.n_ants, self.n_dims))
        return _aco_step_njit(self.archive_solutions, self.archive_fitness, self.n_ants, self.bounds, q, xi, r_kernel, r_dim)

    def reset(self) -> None:
        super().reset()
        self.archive_solutions = None
        self.archive_fitness = None