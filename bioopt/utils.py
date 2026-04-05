"""Utility functions for biopt package.

Provides helpers for:
- Flattening/unflattening model parameters
- Bounds generation
- Benchmark functions for testing
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np


def flatten_params(params: Union[np.ndarray, List[np.ndarray], Dict[str, np.ndarray]]) -> np.ndarray:
    """Flatten model parameters into a 1D array.
    
    Parameters
    ----------
    params : ndarray, list of arrays, or dict of arrays
        Model parameters to flatten.
        
    Returns
    -------
    flat : ndarray of shape (n_params,)
        Flattened parameter vector.
    """
    if isinstance(params, np.ndarray):
        return params.ravel()
    
    if isinstance(params, list):
        return np.concatenate([p.ravel() for p in params])
    
    if isinstance(params, dict):
        return np.concatenate([v.ravel() for v in params.values()])
    
    raise TypeError(f"Unsupported params type: {type(params)}")


def unflatten_params(
    flat: np.ndarray,
    shapes: Union[List[Tuple[int, ...]], Dict[str, Tuple[int, ...]]]
) -> Union[List[np.ndarray], Dict[str, np.ndarray]]:
    """Unflatten a 1D array back into model parameter structure.
    
    Parameters
    ----------
    flat : ndarray of shape (n_params,)
        Flattened parameter vector.
    shapes : list or dict of tuples
        Shape specification for each parameter.
        If list: [(3, 4), (4,), (4, 2), ...]
        If dict: {'w1': (3, 4), 'b1': (4,), ...}
        
    Returns
    -------
    params : list or dict of ndarrays
        Reconstructed parameters matching the shapes structure.
        
    Raises
    ------
    ValueError
        If the flat array size doesn't match total parameter count.
    """
    if isinstance(shapes, list):
        return _unflatten_to_list(flat, shapes)
    elif isinstance(shapes, dict):
        return _unflatten_to_dict(flat, shapes)
    else:
        raise TypeError(f"shapes must be list or dict, got {type(shapes)}")


def _unflatten_to_list(flat: np.ndarray, shapes: List[Tuple[int, ...]]) -> List[np.ndarray]:
    """Unflatten to list of arrays."""
    total_size = sum(int(np.prod(s)) for s in shapes)
    if len(flat) != total_size:
        raise ValueError(
            f"Flat array size {len(flat)} doesn't match total shape size {total_size}"
        )
    
    params = []
    idx = 0
    for shape in shapes:
        size = int(np.prod(shape))
        params.append(flat[idx:idx + size].reshape(shape))
        idx += size
    
    return params


def _unflatten_to_dict(flat: np.ndarray, shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, np.ndarray]:
    """Unflatten to dict of arrays."""
    total_size = sum(int(np.prod(s)) for s in shapes.values())
    if len(flat) != total_size:
        raise ValueError(
            f"Flat array size {len(flat)} doesn't match total shape size {total_size}"
        )
    
    params = {}
    idx = 0
    for name, shape in shapes.items():
        size = int(np.prod(shape))
        params[name] = flat[idx:idx + size].reshape(shape)
        idx += size
    
    return params


def get_param_shapes_from_list(params: List[np.ndarray]) -> List[Tuple[int, ...]]:
    """Extract shapes from a list of parameter arrays."""
    return [p.shape for p in params]


def get_param_shapes_from_dict(params: Dict[str, np.ndarray]) -> Dict[str, Tuple[int, ...]]:
    """Extract shapes from a dict of parameter arrays."""
    return {k: v.shape for k, v in params.items()}


def make_bounds_uniform(
    n_dims: int,
    low: float = -5.0,
    high: float = 5.0
) -> List[Tuple[float, float]]:
    """Create uniform bounds for all dimensions.
    
    Parameters
    ----------
    n_dims : int
        Number of dimensions.
    low : float
        Lower bound for all dimensions.
    high : float
        Upper bound for all dimensions.
        
    Returns
    -------
    bounds : list of tuples
        [(low, high), (low, high), ...]
    """
    return [(low, high)] * n_dims


# --- Benchmark Functions for Testing ---

class BenchmarkFunctions:
    """Collection of standard benchmark functions for optimizer testing.
    
    All functions are designed for minimization.
    """
    
    @staticmethod
    def sphere(x: np.ndarray) -> float:
        """Sphere function. Global minimum: f(0) = 0."""
        return float(np.sum(x ** 2))
    
    @staticmethod
    def rosenbrock(x: np.ndarray) -> float:
        """Rosenbrock function. Global minimum: f(1,1,...,1) = 0."""
        return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))
    
    @staticmethod
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin function. Global minimum: f(0) = 0.
        
        Recommended bounds: [-5.12, 5.12]
        """
        A = 10
        return float(A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x)))
    
    @staticmethod
    def ackley(x: np.ndarray) -> float:
        """Ackley function. Global minimum: f(0) = 0.
        
        Recommended bounds: [-32.768, 32.768]
        """
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        sum_sq = np.sum(x ** 2)
        sum_cos = np.sum(np.cos(c * x))
        return float(-a * np.exp(-b * np.sqrt(sum_sq / d)) - np.exp(sum_cos / d) + a + np.e)
    
    @staticmethod
    def griewank(x: np.ndarray) -> float:
        """Griewank function. Global minimum: f(0) = 0.
        
        Recommended bounds: [-600, 600]
        """
        d = len(x)
        sum_sq = np.sum(x ** 2) / 4000
        prod_cos = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
        return float(sum_sq - prod_cos + 1)
    
    @staticmethod
    def schwefel(x: np.ndarray) -> float:
        """Schwefel function. Global minimum: f(420.9687, ...) = 0.
        
        Recommended bounds: [-500, 500]
        """
        return float(418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x)))))
    
    @staticmethod
    def levy(x: np.ndarray) -> float:
        """Levy function. Global minimum: f(1,1,...,1) = 0.
        
        Recommended bounds: [-10, 10]
        """
        w = 1 + (x - 1) / 4
        term1 = np.sin(np.pi * w[0]) ** 2
        term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1) ** 2))
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)
        return float(term1 + term2 + term3)

    @staticmethod
    def michalewicz(x: np.ndarray, m: float = 10.0) -> float:
        """Michalewicz function. 
        Recommended bounds: [0, pi]
        """
        return float(-np.sum(np.sin(x) * np.sin(np.arange(1, len(x) + 1) * x ** 2 / np.pi) ** (2 * m)))

    @staticmethod
    def eggholder(x: np.ndarray) -> float:
        """Eggholder function. Global minimum: f(512, 404.2319) = -959.6407.
        Only defined for 2D. Recommended bounds: [-512, 512]
        """
        if len(x) != 2:
            raise ValueError("Eggholder function only works for 2D")
        term1 = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + x[0] / 2 + 47)))
        term2 = -x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
        return float(term1 + term2)