"""Artificial Bee Colony (ABC) algorithm."""

from typing import Callable, Optional, Tuple, Union

import numpy as np
from numba import njit

from bioopt.base import BaseOptimizer


@njit(fastmath=True)
def _compute_fitness(fitness_raw: np.ndarray) -> np.ndarray:
    result = np.empty_like(fitness_raw)
    for i in range(len(fitness_raw)):
        if fitness_raw[i] >= 0:
            result[i] = 1.0 / (fitness_raw[i] + 1.0)
        else:
            result[i] = 1.0 + abs(fitness_raw[i])
    return result


@njit(fastmath=True)
def _employed_bee(positions: np.ndarray, bounds: np.ndarray, r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> np.ndarray:
    n_agents, n_dims = positions.shape
    new_positions = positions.copy()
    for i in range(n_agents):
        k = int(r1[i]) % n_agents
        if k == i:
            k = (k + 1) % n_agents
        dim = int(r2[i]) % n_dims
        phi = r3[i] * 2.0 - 1.0
        val = positions[i, dim] + phi * (positions[i, dim] - positions[k, dim])
        val = max(bounds[dim, 0], min(bounds[dim, 1], val))
        new_positions[i, dim] = val
    return new_positions


@njit(fastmath=True)
def _onlooker_bee(positions: np.ndarray, fitness: np.ndarray, probabilities: np.ndarray, bounds: np.ndarray, r1: np.ndarray, r2: np.ndarray, r3: np.ndarray) -> np.ndarray:
    n_agents, n_dims = positions.shape
    new_positions = positions.copy()
    for j in range(n_agents):
        r = r1[j]
        cumulative = 0.0
        selected = 0
        for i in range(n_agents):
            cumulative += probabilities[i]
            if r <= cumulative:
                selected = i
                break
        k = int(r2[j]) % n_agents
        if k == selected:
            k = (k + 1) % n_agents
        dim = int(r3[3*j]) % n_dims
        phi = r3[3*j+1] * 2.0 - 1.0
        val = positions[selected, dim] + phi * (positions[selected, dim] - positions[k, dim])
        val = max(bounds[dim, 0], min(bounds[dim, 1], val))
        new_positions[selected, dim] = val
    return new_positions


@njit(fastmath=True)
def _scout_bee(positions: np.ndarray, trial_counters: np.ndarray, limit: int, bounds: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_agents, n_dims = positions.shape
    for i in range(n_agents):
        if trial_counters[i] >= limit:
            for d in range(n_dims):
                positions[i, d] = bounds[d, 0] + r[i, d] * (bounds[d, 1] - bounds[d, 0])
    return positions, trial_counters


class ABC(BaseOptimizer):
    """Artificial Bee Colony optimization algorithm."""

    def __init__(self, n_agents: int, bounds: Union[list, np.ndarray], limit: Optional[int] = None, seed: Optional[int] = None):
        super().__init__(n_agents, bounds, seed)
        if limit is None:
            limit = n_agents * self.n_dims
        self.limit = limit
        self.trial_counters: Optional[np.ndarray] = None

    def initialize_population(self) -> np.ndarray:
        positions = super().initialize_population()
        self.trial_counters = np.zeros(self.n_agents, dtype=np.int64)
        return positions

    def step(self, positions: np.ndarray, fitness: np.ndarray, iteration: int, **kwargs) -> np.ndarray:
        return positions

    def optimize(self, objective_fn: Callable, iterations: int = 100, verbose: bool = False, callback: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, float]:
        positions = self.initialize_population()
        fitness = self.evaluate(positions, objective_fn)
        self.update_best(positions, fitness)
        self.fitness_history = [self.best_fitness]
        if verbose:
            print(f"Iter 0: Best Fitness = {self.best_fitness:.6e}")
        if callback is not None:
            callback(0, self.best_position, self.best_fitness)

        for i in range(1, iterations + 1):
            limit = kwargs.get("limit", self.limit)
            # Employed bee
            r1 = self.rng.uniform(0, self.n_agents, self.n_agents)
            r2 = self.rng.uniform(0, self.n_dims, self.n_agents)
            r3 = self.rng.random(self.n_agents)
            candidates = _employed_bee(positions.copy(), self.bounds, r1, r2, r3)
            cand_fit = self.evaluate(candidates, objective_fn)
            improved = cand_fit < fitness
            positions[improved] = candidates[improved]
            fitness[improved] = cand_fit[improved]
            self.trial_counters[improved] = 0
            self.trial_counters[~improved] += 1

            # Onlooker bee
            prob = _compute_fitness(fitness)
            prob /= prob.sum()
            r1o = self.rng.random(self.n_agents)
            r2o = self.rng.uniform(0, self.n_agents, self.n_agents)
            r3o = self.rng.random(self.n_agents * 3)
            candidates2 = _onlooker_bee(positions.copy(), fitness, prob, self.bounds, r1o, r2o, r3o)
            cand_fit2 = self.evaluate(candidates2, objective_fn)
            improved2 = cand_fit2 < fitness
            positions[improved2] = candidates2[improved2]
            fitness[improved2] = cand_fit2[improved2]
            self.trial_counters[improved2] = 0
            self.trial_counters[~improved2] += 1

            # Scout bee
            rr = self.rng.random((self.n_agents, self.n_dims))
            positions, self.trial_counters = _scout_bee(positions, self.trial_counters, limit, self.bounds, rr)

            replaced = np.isinf(fitness)
            if np.any(replaced):
                fitness[replaced] = self.evaluate(positions[replaced], objective_fn)

            self.update_best(positions, fitness)
            self.fitness_history.append(self.best_fitness)
            if verbose:
                print(f"Iter {i}: Best Fitness = {self.best_fitness:.6e}")
            if callback is not None:
                callback(i, self.best_position, self.best_fitness)

        return self.best_position, self.best_fitness

    def reset(self) -> None:
        super().reset()
        self.trial_counters = None