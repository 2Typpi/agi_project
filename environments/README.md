# Real-Time RL Environments 

A flexible framework for real-time reinforcement learning environments where simulation progression is dynamically tied to policy computation time. This creates a realistic training scenario where faster policies experience fewer environment changes, while slower policies must handle more environmental progression. Various environments are provided ranging from simple switch-board logic rules to full 3D environments based on [PyBullet](https://pybullet.org/).

## Table of Contents

- [Overview](#overview)
- [Real-Time Environment Mechanics](#real-time-environment-mechanics)
- [Installation](#installation)

---

## Overview

This repository implements a standardized interface for real-time RL environments with three distinct implementations:

1. **BlocksWorld** - 3D physics simulation with egocentric vision (PyBullet)
2. **ShapeEnv** - 2D shape manipulation environment
3. **Switchboard** - Programmable rule-based environment with customizable logic

All environments inherit from the `RealTimeEnvironment` base class and return observations as PyTorch tensors.

---

## Real-Time Environment Mechanics

Unlike traditional RL environments where one policy decision = one environment step, real-time environments tie progression to actual computation time. Slower policies result in more steps taken in the environment during policy execution. Consequently, the agent receives a sequence of observation instead of a single observation.

### How It Works
1. **Benchmarking Phase**: Establish baseline timing
   ```python
   env.benchmark_policy(reference_policy, num_trials=10)
   env.benchmark_simulation(num_trials=100)
   ```

2. **Real-Time Stepping**: Environment adapts to policy speed
   ```python
   observations, info = env.step(policy_fn)
   # observations: List of all observations during policy execution
   # info: Dict with timing statistics, step counts and performed actions
   ```

3. **Time Scaling**: Adjust environment speed relative to policy
   ```python
   env = RealTimeEnvironment(time_scaling=2.0)  # 2x faster environment
   ```
**Note**: Using ```time_scaling=0.0``` results in turn-taking where the environment waits for agent interaction like in normal RL environments.
### Key Interface Methods

- `_get_initial_state()` - Initialize environment (abstract, must implement)
- `_step_simulation(action)` - Advance simulation by one step (abstract, must implement)
- `_get_state()` - Return complete environment state (abstract, must implement)
- `step(policy_fn)` - Execute policy and advance environment based on computation time
- `reset()` - Reset environment to initial state
- `benchmark_policy()` - Measure baseline policy timing
- `benchmark_simulation()` - Measure simulation step performance

### Time-Based Progression

The number of simulation steps per policy execution is computed as:

```
num_steps = max(1, int((actual_policy_time / benchmark_policy_time) * time_scaling))
```

This creates a "catch-up" mechanism where slower policies must handle more environmental changes.

---

## Installation

### Requirements

```bash
# Core dependencies
pip install torch numpy

# Environment-specific dependencies
pip install pybullet              # For BlocksWorld
pip install matplotlib shapely geopandas  # For ShapeEnv
pip install dill                  # For Switchboard rule serialization

# Notebooks
pip install jupyterlab 
```

