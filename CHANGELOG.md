# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-04-04

### Added
- Base optimizer class with common interface (`BaseOptimizer`)
- PSO (Particle Swarm Optimization) with inertia weight
- ACO (Ant Colony Optimization for Continuous Domains) with Gaussian KDE from scratch
- ABC (Artificial Bee Colony) with employed, onlooker, and scout bee phases
- GWO (Grey Wolf Optimizer) with social hierarchy simulation
- FA (Firefly Algorithm) with attractiveness-based movement
- WOA (Whale Optimization Algorithm) with bubble-net hunting
- PyTorch adapter for direct model weight optimization
- Utility functions for parameter flattening/unflattening
- Benchmark functions: sphere, rosenbrock, rastrigin, ackley, griewank, schwefel, levy, michalewicz, eggholder
- Comprehensive unit tests for all algorithms
- NumPy + Numba JIT compilation for performance
- Support for interspace package for distance computations
- Checkpoint/save/restore optimizer state