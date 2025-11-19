# Shape Environment

A real-time RL environment for navigating through different colored shapes using the RealTimeEnvironment framework.

## Overview

ShapeEnv is a visual environment where you can navigate through a grid of shapes with different:
- **Colors**: Red, Green, Blue, Black (4 colors)
- **Shapes**: Circle, Triangle, Square, Pentagon (4 shapes)
- **Sizes**: Small, Medium, Large (3 sizes)

The environment generates images of shapes using matplotlib and shapely, providing a total of 48 unique shape combinations (4 colors × 4 shapes × 3 sizes).

## Features

- **Real-time interface**: Inherits from `RealTimeEnvironment` with time-scaled progression
- **PyTorch tensors**: All observations returned as tensors (shape: `[C, H, W]`)
- **Interactive pygame interface**: Play with the environment using keyboard controls
- **Benchmarking**: Built-in policy and simulation benchmarking

## Installation

The environment requires:
```bash
pip install torch numpy matplotlib shapely geopandas pygame
```

## Usage

### Python API

```python
from shape_env import ShapeEnv

# Create environment
env = ShapeEnv(noisy=False, noise_value=0.1, device='cpu', time_scaling=1.0)

# Benchmark a policy
def simple_policy(obs):
    return torch.tensor([1])  # Always increment color

env.benchmark_policy(simple_policy, num_trials=10)
env.benchmark_simulation(num_trials=100)

# Run steps
observations, info = env.step(simple_policy)

# Reset
initial_state = env.reset()
```

### Pygame Interface

Run the interactive interface:

```bash
python pygame_interface.py
```

With options:
```bash
# Enable noise
python pygame_interface.py --noisy --noise_value 0.2

# Custom window size
python pygame_interface.py --width 1200 --height 800
```

### Controls

**Navigation:**
- `←` `→`: Change color (previous/next)
- `↑` `↓`: Change shape (next/previous)

**Stepping:**
- `P`: Pause/Resume auto-stepping
- `ENTER`: Single step (when paused)
- `+` `-`: Adjust auto-step speed

**Other:**
- `R`: Reset environment
- `Q` or `ESC`: Quit

## Actions

The environment has 4 discrete actions:

| Action | Description |
|--------|-------------|
| 0 | Decrease color index |
| 1 | Increase color index |
| 2 | Decrease shape index |
| 3 | Increase shape index |

Actions are expected as `torch.Tensor` with shape `(1,)`.

## Observations

Observations are PyTorch tensors with shape `[4, 200, 200]` (RGBA image in CHW format).

## Environment Parameters

- `noisy` (bool): Enable noise in shape generation
- `noise_value` (float): Standard deviation of Gaussian noise (default: 0.1)
- `device` (str): PyTorch device ('cpu' or 'cuda')
- `time_scaling` (float): Time scaling factor for real-time progression
- `return_states` (bool): Whether to return full states along with observations

## Shape Details

### Colors
- Red (`r`)
- Green (`g`)
- Blue (`b`)
- Black (`k`)

### Shapes
Each shape is centered and sized based on the radius parameter:
- **Circle**: Standard circle with radius r
- **Triangle**: Regular triangle with side length `r * sqrt(2)`
- **Square**: Square with side length `r * sqrt(3)`
- **Pentagon**: Regular pentagon with radius r

### Sizes
- Small: radius = 15
- Medium: radius = 30
- Large: radius = 45

## Examples

### Basic Usage

```python
import torch
from shape_env import ShapeEnv

# Create environment
env = ShapeEnv()

# Define a policy that cycles through colors
def cycle_colors(obs):
    return torch.tensor([1])  # Color+

# Benchmark and run
env.benchmark_policy(cycle_colors)
env.benchmark_simulation()

# Run for 10 steps
for i in range(10):
    observations, info = env.step(cycle_colors)
    print(f"Step {i}: {len(observations)} env steps, total: {info['step_count']}")
```

### With State Tracking

```python
# Create environment that returns states
env = ShapeEnv(return_states=True)

env.benchmark_policy(lambda obs: torch.tensor([1]))

observations, states, info = env.step(lambda obs: torch.tensor([1]))

print(f"Observations: {len(observations)}")
print(f"States: {len(states)}")
```

### Visualization

```python
import matplotlib.pyplot as plt

env = ShapeEnv()
env.reset()

# Get current state and render
fig = env.render(figsize=(6, 6))
plt.show()
```

## Testing

Run tests:
```bash
# Full test suite
python test_shape_env.py

# Quick import test
python test_pygame_import.py
```

## Architecture

ShapeEnv inherits from `RealTimeEnvironment` and implements:

- `_get_initial_state()`: Returns initial shape as tensor
- `_get_state()`: Returns current state tensor
- `_step_simulation(action)`: Applies action and generates new shape

The environment uses matplotlib and shapely for shape generation, converting the rendered images to PyTorch tensors for compatibility with the framework.

## Notes

- Shape indices wrap around (modulo arithmetic)
- Color indices wrap around independently
- Each shape is randomly positioned within bounds
- Noise can be added for robustness testing
- The environment is deterministic given the same color and shape indices
