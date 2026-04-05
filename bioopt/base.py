"""Base optimizer class for bio-inspired optimization algorithms."""

from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Union

import numpy as np


class BoundsError(Exception):
    """Raised when bounds are invalid."""
    pass


class OptimizationError(Exception):
    """Raised when optimization fails."""
    pass


class BaseOptimizer(ABC):
    """
    Abstract base class for all bio-inspired optimization algorithms.
    
    Provides a common interface with bounds handling, fitness tracking,
    and a standardized optimize() method.
    
    Parameters
    ----------
    n_agents : int
        Number of agents/particles/solutions in the population.
    bounds : list of tuple
        Search space bounds as [(min, max), ...] for each dimension.
    seed : int, optional
        Random seed for reproducibility.
        
    Attributes
    ----------
    best_position : ndarray
        Best position found during optimization.
    best_fitness : float
        Best fitness value found during optimization.
    fitness_history : list
        History of best fitness values per iteration.
    """

    def __init__(
        self,
        n_agents: int,
        bounds: Union[List[Tuple[float, float]], np.ndarray],
        seed: Optional[int] = None,
    ):
        self.n_agents = n_agents
        self.bounds = np.array(bounds, dtype=np.float64)
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        
        self.best_position: Optional[np.ndarray] = None
        self.best_fitness: float = np.inf
        self.fitness_history: List[float] = []
        
        self._validate_bounds()
    
    def _validate_bounds(self):
        """Validate that bounds are properly formatted."""
        if self.bounds.ndim != 2 or self.bounds.shape[1] != 2:
            raise BoundsError(
                f"Bounds must be 2D with shape (n_dims, 2), got {self.bounds.shape}"
            )
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise BoundsError("All lower bounds must be less than upper bounds.")
    
    @property
    def n_dims(self) -> int:
        """Number of dimensions in the search space."""
        return self.bounds.shape[0]
    
    def initialize_population(self) -> np.ndarray:
        """
        Initialize population uniformly within bounds.
        
        Returns
        -------
        population : ndarray of shape (n_agents, n_dims)
        """
        return self.rng.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(self.n_agents, self.n_dims)
        )
    
    def clip_to_bounds(self, positions: np.ndarray) -> np.ndarray:
        """
        Clip positions to stay within bounds.
        
        Parameters
        ----------
        positions : ndarray of shape (n_agents, n_dims)
        
        Returns
        -------
        clipped : ndarray of same shape
        """
        return np.clip(positions, self.bounds[:, 0], self.bounds[:, 1])
    
    def evaluate(
        self,
        positions: np.ndarray,
        objective_fn: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """
        Evaluate fitness for all positions.
        
        Parameters
        ----------
        positions : ndarray of shape (n_agents, n_dims)
        objective_fn : callable
            Function that takes a 1D array and returns a scalar fitness value.
            
        Returns
        -------
        fitness : ndarray of shape (n_agents,)
        """
        return np.array([objective_fn(pos) for pos in positions])
    
    def update_best(
        self,
        positions: np.ndarray,
        fitness: np.ndarray
    ) -> None:
        """
        Update global best position and fitness.
        
        Parameters
        ----------
        positions : ndarray of shape (n_agents, n_dims)
        fitness : ndarray of shape (n_agents,)
        """
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < self.best_fitness:
            self.best_fitness = float(fitness[best_idx])
            self.best_position = positions[best_idx].copy()
    
    @abstractmethod
    def step(
        self,
        positions: np.ndarray,
        fitness: np.ndarray,
        iteration: int,
        **kwargs
    ) -> np.ndarray:
        """
        Perform one iteration step of the algorithm.
        
        Parameters
        ----------
        positions : ndarray of shape (n_agents, n_dims)
            Current positions of all agents.
        fitness : ndarray of shape (n_agents,)
            Current fitness values.
        iteration : int
            Current iteration number.
        **kwargs : dict
            Additional algorithm-specific parameters.
            
        Returns
        -------
        new_positions : ndarray of shape (n_agents, n_dims)
        """
        pass
    
    def optimize(
        self,
        objective_fn: Callable[[np.ndarray], float],
        iterations: int = 100,
        verbose: bool = False,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None,
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """
        Run the optimization.
        
        Parameters
        ----------
        objective_fn : callable
            The objective function to minimize. Takes a 1D array and returns a scalar.
        iterations : int
            Number of iterations to run.
        verbose : bool
            If True, print progress information.
        callback : callable, optional
            Function called after each iteration with signature
            (iteration, best_position, best_fitness).
        **kwargs : dict
            Additional algorithm-specific parameters.
            
        Returns
        -------
        best_position : ndarray
            Best position found.
        best_fitness : float
            Best fitness value found.
        """
        # Initialize
        positions = self.initialize_population()
        fitness = self.evaluate(positions, objective_fn)
        self.update_best(positions, fitness)
        self.fitness_history = [self.best_fitness]
        
        if verbose:
            print(f"Iter 0: Best Fitness = {self.best_fitness:.6e}")
        
        if callback is not None:
            callback(0, self.best_position, self.best_fitness)
        
        # Main loop
        for i in range(1, iterations + 1):
            positions = self.step(positions, fitness, i, **kwargs)
            positions = self.clip_to_bounds(positions)
            fitness = self.evaluate(positions, objective_fn)
            self.update_best(positions, fitness)
            self.fitness_history.append(self.best_fitness)
            
            if verbose:
                print(f"Iter {i}: Best Fitness = {self.best_fitness:.6e}")
            
            if callback is not None:
                callback(i, self.best_position, self.best_fitness)
        
        return self.best_position, self.best_fitness
    
    def reset(self) -> None:
        """Reset optimizer state for a fresh optimization run."""
        self.best_position = None
        self.best_fitness = np.inf
        self.fitness_history = []
        self.rng = np.random.RandomState(self.seed)
    
    def get_state(self) -> dict:
        """Get optimizer state for checkpointing."""
        return {
            "best_position": self.best_position,
            "best_fitness": self.best_fitness,
            "fitness_history": self.fitness_history,
            "rng_state": self.rng.get_state(),
        }
    
    def set_state(self, state: dict) -> None:
        """Restore optimizer state from checkpoint."""
        self.best_position = state["best_position"]
        self.best_fitness = state["best_fitness"]
        self.fitness_history = state["fitness_history"]
        self.rng.set_state(state["rng_state"])