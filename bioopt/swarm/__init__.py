"""Swarm intelligence algorithms.

Available algorithms:
- PSO: Particle Swarm Optimization
- ACO: Ant Colony Optimization (Continuous, using KDE)
- ABC: Artificial Bee Colony
- GWO: Grey Wolf Optimizer
- FA: Firefly Algorithm
- WOA: Whale Optimization Algorithm

Usage:
    >>> from bioopt.swarm import PSO, ACO, GWO
    >>> pso = PSO(n_agents=30, bounds=[(-5, 5)] * 10)
    >>> best_pos, best_fit = pso.optimize(objective_fn, iterations=100)
"""

from bioopt.swarm.pso import PSO
from bioopt.swarm.aco import ACO
from bioopt.swarm.abc import ABC
from bioopt.swarm.gwo import GWO
from bioopt.swarm.fa import FA
from bioopt.swarm.woa import WOA

__all__ = [
    "PSO",
    "ACO",
    "ABC",
    "GWO",
    "FA",
    "WOA",
]