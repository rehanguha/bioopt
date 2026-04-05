"""Comprehensive tests for all biopt optimization algorithms."""

import pytest
import numpy as np

from bioopt.swarm import PSO, ACO, ABC, GWO, FA, WOA
from bioopt.utils import BenchmarkFunctions

BOUNDS_2D = [(-5.0, 5.0), (-5.0, 5.0)]
BOUNDS_10D = [(-5.0, 5.0)] * 10
N_AGENTS = 30
ITERATIONS = 50
SEED = 42

class TestPSO:
    def test_basic_functionality(self):
        pso = PSO(n_agents=N_AGENTS, bounds=BOUNDS_2D, seed=SEED)
        best_pos, best_fit = pso.optimize(BenchmarkFunctions.sphere, iterations=ITERATIONS)
        assert best_pos.shape == (2,)
        assert best_fit < 1.0

    def test_reproducibility(self):
        pso1 = PSO(n_agents=N_AGENTS, bounds=BOUNDS_2D, seed=SEED)
        pso2 = PSO(n_agents=N_AGENTS, bounds=BOUNDS_2D, seed=SEED)
        _, fit1 = pso1.optimize(BenchmarkFunctions.sphere, iterations=ITERATIONS)
        _, fit2 = pso2.optimize(BenchmarkFunctions.sphere, iterations=ITERATIONS)
        assert fit1 == fit2

class TestACO:
    def test_basic_functionality(self):
        aco = ACO(n_agents=15, bounds=BOUNDS_2D, seed=SEED)
        best_pos, best_fit = aco.optimize(BenchmarkFunctions.sphere, iterations=ITERATIONS)
        assert best_pos.shape == (2,)
        assert best_fit < 1.0

class TestABC:
    def test_basic_functionality(self):
        abc = ABC(n_agents=N_AGENTS, bounds=BOUNDS_2D, seed=SEED)
        best_pos, best_fit = abc.optimize(BenchmarkFunctions.sphere, iterations=ITERATIONS)
        assert best_pos.shape == (2,)
        assert best_fit < 1.0

class TestGWO:
    def test_basic_functionality(self):
        gwo = GWO(n_agents=N_AGENTS, bounds=BOUNDS_2D, seed=SEED)
        best_pos, best_fit = gwo.optimize(BenchmarkFunctions.sphere, iterations=ITERATIONS)
        assert best_pos.shape == (2,)
        assert best_fit < 1.0

class TestFA:
    def test_basic_functionality(self):
        fa = FA(n_agents=N_AGENTS, bounds=BOUNDS_2D, seed=SEED)
        best_pos, best_fit = fa.optimize(BenchmarkFunctions.sphere, iterations=ITERATIONS)
        assert best_pos.shape == (2,)
        assert best_fit < 1.0

class TestWOA:
    def test_basic_functionality(self):
        woa = WOA(n_agents=N_AGENTS, bounds=BOUNDS_2D, seed=SEED)
        best_pos, best_fit = woa.optimize(BenchmarkFunctions.sphere, iterations=ITERATIONS)
        assert best_pos.shape == (2,)
        assert best_fit < 1.0

class TestBenchmarkFunctions:
    def test_sphere(self):
        assert BenchmarkFunctions.sphere(np.zeros(5)) == 0.0
        assert BenchmarkFunctions.sphere(np.ones(5)) == 5.0

    def test_rosenbrock(self):
        assert BenchmarkFunctions.rosenbrock(np.ones(5)) == 0.0

    def test_rastrigin(self):
        assert BenchmarkFunctions.rastrigin(np.zeros(5)) == 0.0

class TestErrorHandling:
    def test_invalid_bounds(self):
        with pytest.raises(Exception):
            PSO(n_agents=10, bounds=[(5.0, 1.0)])

    def test_state_checkpoint(self):
        pso = PSO(n_agents=10, bounds=BOUNDS_2D, seed=SEED)
        pso.optimize(BenchmarkFunctions.sphere, iterations=5)
        state = pso.get_state()
        assert "best_position" in state
        assert "best_fitness" in state