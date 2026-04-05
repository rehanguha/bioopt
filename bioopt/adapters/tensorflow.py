"""TensorFlow adapter for bioopt optimization algorithms.

Provides integration with TensorFlow/Keras models by converting model parameters
to flat vectors and back, enabling gradient-free optimization of neural networks.

Usage:
    >>> from bioopt.swarm import PSO
    >>> from bioopt.adapters.tensorflow import TensorFlowAdapter
    >>> 
    >>> model = tf.keras.Sequential([...])
    >>> adapter = TensorFlowAdapter(model)
    >>> 
    >>> pso = PSO(n_agents=30, bounds=adapter.get_bounds())
    >>> best_weights, best_loss = adapter.optimize(
    ...     pso, loss_fn, dataset, iterations=50
    ... )
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import tensorflow as tf
    from tensorflow import keras
except ImportError:
    raise ImportError(
        "TensorFlow is required for this module. Install with: pip install bioopt[tensorflow]"
    )

from bioopt.base import BaseOptimizer
from bioopt.utils import flatten_params, unflatten_params


class TensorFlowAdapter:
    """Adapter for using biopt optimizers with TensorFlow/Keras models.
    
    Converts TensorFlow model parameters to/from flat numpy arrays
    so that swarm intelligence algorithms can optimize them.
    
    Parameters
    ----------
    model : tf.keras.Model
        TensorFlow/Keras model to optimize.
        
    Attributes
    ----------
    model : tf.keras.Model
        Reference to the model being optimized.
    param_shapes : dict
        Dictionary mapping parameter names to their shapes.
    n_params : int
        Total number of parameters in the model.
    """
    
    def __init__(self, model: keras.Model):
        self.model = model
        self.param_shapes: Dict[str, Tuple[int, ...]] = {}
        self.param_names: List[str] = []
        
        for var in model.trainable_variables:
            self.param_shapes[var.name] = var.shape
            self.param_names.append(var.name)
        
        self.n_params = sum(int(np.prod(s)) for s in self.param_shapes.values())
    
    def get_weights_list(self) -> List[np.ndarray]:
        """Get model weights as a list of numpy arrays."""
        return [w.numpy() for w in self.model.trainable_variables]
    
    def flat_to_model(self, flat_weights: np.ndarray) -> None:
        """Set model parameters from a flat array.
        
        Parameters
        ----------
        flat_weights : ndarray of shape (n_params,)
            Flattened parameter vector.
        """
        if len(flat_weights) != self.n_params:
            raise ValueError(
                f"Expected {self.n_params} parameters, got {len(flat_weights)}"
            )
        
        shapes = [self.param_shapes[name] for name in self.param_names]
        param_list = unflatten_params(flat_weights, shapes)
        
        for i, var in enumerate(self.model.trainable_variables):
            var.assign(tf.constant(param_list[i], dtype=var.dtype))
    
    def model_to_flat(self) -> np.ndarray:
        """Get current model parameters as a flat array.
        
        Returns
        -------
        flat : ndarray of shape (n_params,)
            Flattened parameter vector.
        """
        weights = self.get_weights_list()
        return flatten_params(weights)
    
    def get_bounds(
        self,
        default_low: float = -5.0,
        default_high: float = 5.0,
        per_param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> List[Tuple[float, float]]:
        """Get bounds for all model parameters.
        
        Parameters
        ----------
        default_low : float
            Default lower bound.
        default_high : float
            Default upper bound.
        per_param_bounds : dict, optional
            Per-parameter bounds as {name: (low, high)}.
            Parameters not specified use default bounds.
            
        Returns
        -------
        bounds : list of tuples
            Bounds for each parameter in flat array.
        """
        bounds = []
        for name in self.param_names:
            shape = self.param_shapes[name]
            n_elements = int(np.prod(shape))
            
            if per_param_bounds and name in per_param_bounds:
                low, high = per_param_bounds[name]
            else:
                low, high = default_low, default_high
            
            bounds.extend([(low, high)] * n_elements)
        
        return bounds
    
    def evaluate(
        self,
        flat_weights: np.ndarray,
        loss_fn: Callable,
        dataset: Optional[tf.data.Dataset] = None,
        inputs: Optional[tf.Tensor] = None,
        targets: Optional[tf.Tensor] = None,
        **kwargs
    ) -> float:
        """Evaluate the model with given weights on a loss function.
        
        Parameters
        ----------
        flat_weights : ndarray
            Flattened parameter vector.
        loss_fn : callable
            Loss function. Can be:
            - A function that takes (outputs, targets) and returns loss
            - A function that takes (model, dataset) and returns loss
        dataset : tf.data.Dataset, optional
            Dataset for evaluation.
        inputs : tf.Tensor, optional
            Input tensor for direct evaluation.
        targets : tf.Tensor, optional
            Target tensor for direct evaluation.
        **kwargs : dict
            Additional arguments passed to loss_fn.
            
        Returns
        -------
        loss : float
            Computed loss value.
        """
        self.flat_to_model(flat_weights)
        
        if dataset is not None:
            total_loss = 0.0
            n_batches = 0
            for batch_inputs, batch_targets in dataset:
                outputs = self.model(batch_inputs, training=False)
                loss = loss_fn(outputs, batch_targets, **kwargs)
                if isinstance(loss, tf.Tensor):
                    loss = loss.numpy()
                total_loss += float(loss)
                n_batches += 1
            return total_loss / n_batches if n_batches > 0 else float("inf")
        
        elif inputs is not None and targets is not None:
            outputs = self.model(inputs, training=False)
            loss = loss_fn(outputs, targets, **kwargs)
            if isinstance(loss, tf.Tensor):
                loss = loss.numpy()
            return float(loss)
        
        else:
            loss = loss_fn(self.model, dataset, **kwargs)
            if isinstance(loss, tf.Tensor):
                loss = loss.numpy()
            return float(loss)
    
    def optimize(
        self,
        optimizer: BaseOptimizer,
        loss_fn: Callable,
        dataset: Optional[tf.data.Dataset] = None,
        inputs: Optional[tf.Tensor] = None,
        targets: Optional[tf.Tensor] = None,
        iterations: int = 100,
        verbose: bool = False,
        callback: Optional[Callable] = None,
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """Run optimizer to find best model weights.
        
        Parameters
        ----------
        optimizer : BaseOptimizer
            A biopt optimizer (PSO, GWO, etc.).
        loss_fn : callable
            Loss function for evaluation.
        dataset : tf.data.Dataset, optional
            Dataset for evaluation.
        inputs : tf.Tensor, optional
            Input tensor for direct evaluation.
        targets : tf.Tensor, optional
            Target tensor for direct evaluation.
        iterations : int
            Number of optimization iterations.
        verbose : bool
            Print progress.
        callback : callable, optional
            Callback function called with (iteration, best_weights, best_loss).
        **kwargs : dict
            Additional arguments passed to optimizer.optimize().
            
        Returns
        -------
        best_weights : ndarray
            Best parameter vector found.
        best_loss : float
            Best loss achieved.
        """
        def objective_fn(flat_weights: np.ndarray) -> float:
            return self.evaluate(flat_weights, loss_fn, dataset, inputs, targets)
        
        best_weights, best_loss = optimizer.optimize(
            objective_fn, iterations=iterations, verbose=verbose, callback=callback, **kwargs
        )
        
        self.flat_to_model(best_weights)
        
        return best_weights, best_loss
    
    def get_weight_statistics(self) -> Dict[str, Any]:
        """Get statistics about current model weights."""
        stats = {}
        for var in self.model.trainable_variables:
            w = var.numpy()
            stats[var.name] = {
                "mean": float(np.mean(w)),
                "std": float(np.std(w)),
                "min": float(np.min(w)),
                "max": float(np.max(w)),
                "norm": float(np.linalg.norm(w)),
            }
        return stats