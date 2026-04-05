# bioopt

Bio-inspired optimization algorithms for machine learning and scientific computing.

## Features

- **Swarm Intelligence Algorithms**: PSO, ACO (continuous with KDE), ABC, GWO, FA, WOA
- **Pure NumPy + Numba**: No scipy dependency, fast JIT-compiled code
- **PyTorch & TensorFlow Integration**: Direct model weight optimization without gradients
- **Unified API**: All optimizers share the same interface
- **Extensible**: Easy to add new algorithms and categories

## Installation

```bash
pip install bioopt
```

For PyTorch support:

```bash
pip install bioopt[pytorch]
```

For TensorFlow support:

```bash
pip install bioopt[tensorflow]
```

For development:

```bash
pip install bioopt[dev]
```

## Quick Start

### Standalone Usage

```python
from bioopt.swarm import PSO, GWO, FA

def sphere(x):
    """Sphere function: f(x) = sum(x^2), minimum at x=0."""
    return sum(x ** 2)

# Create optimizer
pso = PSO(
    n_agents=30,
    bounds=[(-5.0, 5.0)] * 10,  # 10-dimensional problem
    w=0.7298, c1=1.496, c2=1.496,
    seed=42
)

# Run optimization
best_position, best_fitness = pso.optimize(sphere, iterations=100, verbose=True)
print(f"Best fitness: {best_fitness:.6e}")
```

### All Available Algorithms

```python
from bioopt.swarm import PSO, ACO, ABC, GWO, FA, WOA

algorithms = [
    ("PSO", PSO(n_agents=30, bounds=bounds, seed=42)),
    ("ACO", ACO(n_agents=15, bounds=bounds, seed=42)),  # Continuous KDE-based
    ("ABC", ABC(n_agents=30, bounds=bounds, seed=42)),
    ("GWO", GWO(n_agents=30, bounds=bounds, seed=42)),
    ("FA",  FA(n_agents=30, bounds=bounds, seed=42)),
    ("WOA", WOA(n_agents=30, bounds=bounds, seed=42)),
]

for name, opt in algorithms:
    pos, fit = opt.optimize(objective_fn, iterations=100)
    print(f"{name}: best_fitness = {fit:.6e}")
```

### TensorFlow Integration

```python
import tensorflow as tf
from bioopt.swarm import PSO
from bioopt.adapters.tensorflow import TensorFlowAdapter

# Define a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

adapter = TensorFlowAdapter(model)

# Create optimizer with model parameter bounds
pso = PSO(n_agents=20, bounds=adapter.get_bounds(default_low=-1.0, default_high=1.0))

# Define loss function
def loss_fn(outputs, targets):
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(targets, outputs))

# Optimize model weights
best_weights, best_loss = adapter.optimize(
    pso, loss_fn,
    inputs=sample_inputs,
    targets=sample_targets,
    iterations=50,
    verbose=True
)
```

### PyTorch Integration

```python
import torch
import torch.nn as nn
from bioopt.swarm import PSO
from bioopt.adapters.pytorch import PyTorchAdapter

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()
adapter = PyTorchAdapter(model)

# Create optimizer with model parameter bounds
pso = PSO(n_agents=20, bounds=adapter.get_bounds(default_low=-1.0, default_high=1.0))

# Define loss function
def loss_fn(outputs, targets):
    return nn.functional.cross_entropy(outputs, targets)

# Optimize model weights (use a small subset for demo)
best_weights, best_loss = adapter.optimize(
    pso, loss_fn,
    inputs=sample_inputs,
    targets=sample_targets,
    iterations=50,
    verbose=True
)
```

### Benchmark Functions

```python
from bioopt.utils import BenchmarkFunctions

# Available benchmarks (all for minimization)
fns = {
    "sphere":     BenchmarkFunctions.sphere,      # min at 0
    "rosenbrock": BenchmarkFunctions.rosenbrock,   # min at 1
    "rastrigin":  BenchmarkFunctions.rastrigin,    # min at 0
    "ackley":     BenchmarkFunctions.ackley,       # min at 0
    "griewank":   BenchmarkFunctions.griewank,     # min at 0
    "schwefel":   BenchmarkFunctions.schwefel,     # min at 420.9687
    "levy":       BenchmarkFunctions.levy,         # min at 1
}
```

## API Reference

### Base Interface

All optimizers inherit from `BaseOptimizer` and share this interface:

```python
class BaseOptimizer:
    def __init__(self, n_agents: int, bounds, seed: Optional[int] = None):
        ...
    
    def optimize(
        self,
        objective_fn: Callable,  # Function to minimize
        iterations: int = 100,
        verbose: bool = False,
        callback: Optional[Callable] = None,  # (iter, pos, fit)
        **kwargs
    ) -> Tuple[np.ndarray, float]:
        """Run optimization. Returns (best_position, best_fitness)."""
    
    def reset(self) -> None:
        """Reset optimizer state."""
    
    def get_state(self) -> dict:
        """Get checkpoint state."""
    
    def set_state(self, state: dict) -> None:
        """Restore from checkpoint."""
```

### Algorithm-Specific Parameters

| Algorithm | Key Parameters | Default |
|-----------|---------------|---------|
| **PSO** | `w`, `c1`, `c2`, `max_velocity` | w=0.7298, c1=c2=1.496 |
| **ACO** | `archive_size`, `q`, `xi` | q=0.01, xi=0.1 |
| **ABC** | `limit` | n_agents * n_dims |
| **GWO** | (none) | - |
| **FA** | `alpha`, `beta_0`, `gamma`, `alpha_decay` | alpha=0.2, beta_0=1.0, gamma=1.0 |
| **WOA** | (none) | - |

## Project Structure

```
bioopt/
├── bioopt/
│   ├── __init__.py          # Package exports
│   ├── base.py              # BaseOptimizer class
│   ├── utils.py             # Utilities, benchmarks
│   ├── swarm/               # Swarm intelligence algorithms
│   │   ├── pso.py           # Particle Swarm Optimization
│   │   ├── aco.py           # Ant Colony Optimization (KDE)
│   │   ├── abc.py           # Artificial Bee Colony
│   │   ├── gwo.py           # Grey Wolf Optimizer
│   │   ├── fa.py            # Firefly Algorithm
│   │   └── woa.py           # Whale Optimization Algorithm
│   └── adapters/            # DL framework adapters
│       ├── pytorch.py       # PyTorch integration
│       └── tensorflow.py    # TensorFlow integration
└── tests/
    └── test_all.py          # Algorithm tests
```

## Future Categories

- `bioopt.evolutionary` - Genetic Algorithms, Differential Evolution
- `bioopt.physics` - Simulated Annealing, Gravitational Search, Black Hole
- `bioopt.plant` - Flower Pollination, Invasive Weed Optimization

## Contributing

Contributions welcome! Please open issues or PRs.

## License

MIT License