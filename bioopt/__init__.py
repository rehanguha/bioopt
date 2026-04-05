"""bioopt - Bio-inspired optimization algorithms.

A collection of nature-inspired optimization algorithms for
machine learning and scientific computing.

Categories:
    - swarm: Swarm intelligence algorithms (PSO, ACO, ABC, GWO, FA, WOA)
    - evolutionary: Evolutionary algorithms (coming soon)
    - physics: Physics-based algorithms (coming soon)

Usage:
    >>> from bioopt.swarm import PSO
    >>> pso = PSO(n_agents=30, bounds=[(-5, 5)] * 10)
    >>> best_pos, best_fit = pso.optimize(objective_fn, iterations=100)
"""

__version__ = "0.1.0"
__author__ = "Rehan Guha"

from bioopt.base import BaseOptimizer, BoundsError, OptimizationError
from bioopt.utils import (
    BenchmarkFunctions,
    flatten_params,
    make_bounds_uniform,
    unflatten_params,
)

__all__ = [
    "__version__",
    "BaseOptimizer",
    "BoundsError",
    "OptimizationError",
    "BenchmarkFunctions",
    "flatten_params",
    "unflatten_params",
    "make_bounds_uniform",
]