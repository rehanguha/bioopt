"""PyTorch adapter for biopt optimization algorithms."""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
except ImportError:
    raise ImportError("PyTorch is required. Install with: pip install bioopt[pytorch]")

from bioopt.base import BaseOptimizer
from bioopt.utils import flatten_params, unflatten_params


class PyTorchAdapter:
    """Adapter for using biopt optimizers with PyTorch models."""

    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.param_shapes: Dict[str, Tuple[int, ...]] = {}
        self.param_names: List[str] = []
        for name, param in model.named_parameters():
            self.param_shapes[name] = param.shape
            self.param_names.append(name)
        self.n_params = sum(int(np.prod(s)) for s in self.param_shapes.values())

    def flat_to_model(self, flat_weights: np.ndarray) -> None:
        if len(flat_weights) != self.n_params:
            raise ValueError(f"Expected {self.n_params} parameters, got {len(flat_weights)}")
        shapes = [self.param_shapes[name] for name in self.param_names]
        param_list = unflatten_params(flat_weights, shapes)
        with torch.no_grad():
            for i, (name, param) in enumerate(self.model.named_parameters()):
                param.copy_(torch.from_numpy(param_list[i]).to(dtype=param.dtype, device=self.device))

    def model_to_flat(self) -> np.ndarray:
        params = [p.detach().cpu().numpy() for _, p in self.model.named_parameters()]
        return flatten_params(params)

    def get_bounds(self, default_low: float = -5.0, default_high: float = 5.0,
                   per_param_bounds: Optional[Dict[str, Tuple[float, float]]] = None) -> List[Tuple[float, float]]:
        bounds = []
        for name in self.param_names:
            n_elements = int(np.prod(self.param_shapes[name]))
            if per_param_bounds and name in per_param_bounds:
                low, high = per_param_bounds[name]
            else:
                low, high = default_low, default_high
            bounds.extend([(low, high)] * n_elements)
        return bounds

    def evaluate(self, flat_weights: np.ndarray, loss_fn: Callable,
                 dataloader: Optional[DataLoader] = None,
                 inputs: Optional[torch.Tensor] = None,
                 targets: Optional[torch.Tensor] = None, **kwargs) -> float:
        self.flat_to_model(flat_weights)
        self.model.eval()
        with torch.no_grad():
            if dataloader is not None:
                total_loss, n_batches = 0.0, 0
                for batch_inputs, batch_targets in dataloader:
                    outputs = self.model(batch_inputs.to(self.device))
                    loss = loss_fn(outputs, batch_targets.to(self.device), **kwargs)
                    total_loss += loss.item() if isinstance(loss, torch.Tensor) else float(loss)
                    n_batches += 1
                return total_loss / n_batches if n_batches > 0 else float("inf")
            elif inputs is not None and targets is not None:
                outputs = self.model(inputs.to(self.device))
                loss = loss_fn(outputs, targets.to(self.device), **kwargs)
                return loss.item() if isinstance(loss, torch.Tensor) else float(loss)
            else:
                loss = loss_fn(self.model, dataloader, **kwargs)
                return loss.item() if isinstance(loss, torch.Tensor) else float(loss)

    def optimize(self, optimizer: BaseOptimizer, loss_fn: Callable,
                 dataloader: Optional[DataLoader] = None,
                 inputs: Optional[torch.Tensor] = None,
                 targets: Optional[torch.Tensor] = None,
                 iterations: int = 100, verbose: bool = False,
                 callback: Optional[Callable] = None, **kwargs) -> Tuple[np.ndarray, float]:
        def objective_fn(flat_weights: np.ndarray) -> float:
            return self.evaluate(flat_weights, loss_fn, dataloader, inputs, targets)
        best_weights, best_loss = optimizer.optimize(objective_fn, iterations=iterations, verbose=verbose, callback=callback, **kwargs)
        self.flat_to_model(best_weights)
        return best_weights, best_loss