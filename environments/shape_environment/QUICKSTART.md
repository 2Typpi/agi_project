# ShapeEnv Quick Start

## Launch the Interactive Interface

```bash
cd /home/fabi/agi_environments/environments/shape_environment
python pygame_interface.py
```

## Controls

```
┌─────────────────────────────────────────┐
│         SHAPE ENVIRONMENT CONTROLS       │
├─────────────────────────────────────────┤
│  Navigation:                            │
│    ←  →   Change color                  │
│    ↑  ↓   Change shape/size             │
│                                         │
│  Stepping:                              │
│    P       Pause/Resume                 │
│    ENTER   Single step                  │
│    +  -    Speed control                │
│                                         │
│  Other:                                 │
│    R       Reset                        │
│    Q/ESC   Quit                         │
└─────────────────────────────────────────┘
```

## What You'll See

- **Main Display**: Large colored shape in the center
- **Right Panel**:
  - Current color, shape, and size
  - Index information
  - Controls reference
  - Current action indicator

## The Environment

- **4 Colors**: Red, Green, Blue, Black
- **4 Shapes**: Circle, Triangle, Square, Pentagon
- **3 Sizes**: Small (r=15), Medium (r=30), Large (r=45)
- **Total**: 48 unique combinations

## Try These Patterns

1. **Cycle all colors**: Press → repeatedly to see all 4 colors
2. **Cycle all shapes**: Press ↑ repeatedly to see all 12 shape/size combos
3. **Auto-run**: Press P to start auto-stepping, then use arrows while it runs
4. **Speed up**: Press + to increase stepping speed

## Command Line Options

```bash
# Enable noise for variation
python pygame_interface.py --noisy --noise_value 0.2

# Larger window
python pygame_interface.py --width 1200 --height 800

# Faster refresh rate
python pygame_interface.py --fps 120
```

## Python API Example

```python
from shape_env import ShapeEnv
import torch

# Create environment
env = ShapeEnv(device='cpu')

# Simple policy: cycle colors
def policy(obs):
    return torch.tensor([1])  # Color+

# Benchmark and step
env.benchmark_policy(policy)
observations, info = env.step(policy)

# Visualize
env.render()
```

## Files

- `shape_env.py` - Main environment class
- `pygame_interface.py` - Interactive interface
- `test_shape_env.py` - Test suite
- `README.md` - Full documentation
