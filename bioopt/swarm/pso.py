"""Particle Swarm Optimization (PSO) algorithm.

Implementation inspired by Kennedy & Eberhart (1995) with
inertia weight from Shi & Eberhart (1998).
"""

from typing import Callable, Optional, Tuple, Union

import numpy as np
from numba import njit

from bioopt.base import BaseOptimizer


@njit(fastmath=True)
def _pso_step_njit(
    positions: np.ndarray,
    velocities: np.ndarray,
    pbest_positions: np.ndarray,
    pbest_fitness: np.ndarray,
    fitness: np.ndarray,
    gbest_position: np.ndarray,
    bounds: np.ndarray,
    w: float,
    c1: float,
    c2: float,
    r1: np.ndarray,
    r2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Numba-accelerated PSO step."""
    n_agents, n_dims = positions.shape
    new_positions = np.empty_like(positions)
    new_velocities = np.empty_like(velocities)
    
    # Update personal best
    for i in range(n_agents):
        if fitness[i] < pbest_fitness[i]:
            pbest_fitness[i] = fitness[i]
            for d in range(n_dims):
                pbest_positions[i, d] = positions[i, d]
    
    # Update velocities and positions
    for i in range(n_agents):
        for d in range(n_dims):
            cognitive = c1 * r1[i, d] * (pbest_positions[i, d] - positions[i, d])
            social = c2 * r2[i, d] * (gbest_position[d] - positions[i, d])
            new_velocities[i, d] = w * velocities[i, d] + cognitive + social
            new_positions[i, d] = positions[i, d] + new_velocities[i, d]
            # Clip to bounds
            if new_positions[i, d] < bounds[d, 0]:
                new_positions[i, d] = bounds[d, 0]
            elif new_positions[i, d] > bounds[d, 1]:
                new_positions[i, d] = bounds[d, 1]
    
    return new_positions, new_velocities, pbest_positions, pbest_fitness


class PSO(BaseOptimizer):
    """Particle Swarm Optimization."""
    
    def __init__(
        self,
        n_agents: int,
        bounds: Union[list, np.ndarray],
        w: float = 0.7298,
        c1: float = 1.496,
        c2: float = 1.496,
        max_velocity: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(n_agents, bounds, seed)
        self.w = w
        self.c1 = c1
        self.c2 = c2
        
        if max_velocity is None:
            max_velocity = 0.5 * np.max(self.bounds[:, 1] - self.bounds[:, 0])
        self.max_velocity = max_velocity
        
        self.velocities: Optional[np.ndarray] = None
        self.pbest_positions: Optional[np.ndarray] = None
        self.pbest_fitness: Optional[np.ndarray] = None
    
    def initialize_population(self) -> np.ndarray:
        positions = super().initialize_population()
        vel_range = self.bounds[:, 1] - self.bounds[:, 0]
        self.velocities = self.rng.uniform(
            -vel_range, vel_range, size=(self.n_agents, self.n_dims)
        )
        self.velocities = np.clip(self.velocities, -self.max_velocity, self.max_velocity)
        self.pbest_positions = positions.copy()
        self.pbest_fitness = np.full(self.n_agents, np.inf)
        return positions
    
    def step(
        self,
        positions: np.ndarray,
        fitness: np.ndarray,
        iteration: int,
        **kwargs
    ) -> np.ndarray:
        w = kwargs.get("w", self.w)
        c1 = kwargs.get("c1", self.c1)
        c2 = kwargs.get("c2", self.c2)
        
        gbest_position = self.best_position.copy()
        r1 = self.rng.random((self.n_agents, self.n_dims))
        r2 = self.rng.random((self.n_agents, self.n_dims))
        
        new_positions, velocities, self.pbest_positions, self.pbest_fitness = _pso_step_njit(
            positions, self.velocities, self.pbest_positions,
            self.pbest_fitness, fitness, gbest_position,
            self.bounds, w, c1, c2, r1, r2,
        )
        
        self.velocities = np.clip(velocities, -self.max_velocity, self.max_velocity)
        return new_positions
    
    def reset(self) -> None:
        super().reset()
        self.velocities = None
        self.pbest_positions = None
        self.pbest_fitness = None