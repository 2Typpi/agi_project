# Switchboard Environment

A flexible real-time RL environment where actions (button presses) control observations (slot activations) through programmable rules. This guide provides a complete overview of the Switchboard environment's design, from basic concepts to advanced customization. It's intended for developers and researchers who want to understand, customize, or extend the environment.

## Getting Started: Core Concepts & Basic Usage

Before diving into the details, let's cover the fundamental components of the Switchboard environment and how to interact with it.

### Actions and Observations

- **Actions** are binary `torch.Tensor` vectors representing button presses. A value of `1` means a button is pressed, and `0` means it's not. The length of this vector is determined by `action_dim`.
- **Observations** are `torch.Tensor` vectors representing the activation level of each slot, with values ranging from `0.0` (off) to `1.0` (fully on). The length of this vector is `obs_dim`.

```python
from environments.switchboard.switchboard import Switchboard
import torch

# Create an environment with 10 buttons and 10 slots
env = Switchboard(action_dim=10, obs_dim=10)

# Get the dimensions
assert env.action_dim == 10
assert env.obs_dim == 10
```

### The Policy

A **policy** is a Python function that decides which action to take based on the current observation. It takes an `observation` tensor and must return an `action` tensor of the correct dimension.

```python
def random_policy(observation: torch.Tensor) -> torch.Tensor:
    """A simple policy that presses one random button."""
    action = torch.zeros(env.action_dim)
    random_button = torch.randint(0, env.action_dim, (1,)).item()
    action[random_button] = 1.0
    return action
```

### Stepping the Environment

The `env.step()` method is the heart of the interaction. It takes your policy function as an argument and advances the simulation.

```python
# The policy is passed as a function
obs_list, info = env.step(random_policy)

# The last observation in the returned list is the most recent state
final_obs = obs_list[-1]

# The info dictionary contains metadata about the step
print(info)
```

Crucially, the `info` dictionary contains the `performed_action`, which is the action that was *actually* executed by the environment during that step. This can be different from what your policy returned, especially in real-time mode.

## Real-Time vs. Turn-Based Mode

The environment can run in two distinct modes, controlled by the `time_scaling` parameter.

### Turn-Based Mode (`time_scaling=0.0`)

This is the standard mode for reinforcement learning. The environment waits indefinitely for your policy to compute an action, executes it for a single simulation step, and returns the result. It's a predictable, lock-step interaction.

```python
env = Switchboard(time_scaling=0.0) # Turn-based mode

# The environment will wait for this to complete
obs_list, info = env.step(random_policy) # Use your defined policy like random_policy

# This will always be 1 in turn-based mode
assert info['num_environment_steps'] == 1
```

### Real-Time Mode (`time_scaling > 0.0`)

This mode *simulates* a real-time environment by advancing the simulation by `N` steps, where `N` depends on how long your policy takes to compute an action. The `time_scaling` factor influences this calculation.

-   The environment still waits for your policy to return an action.
-   However, it then calculates how many simulation steps would have occurred in "real-time" during your policy's computation duration.
-   The `env.step()` method then executes these `N` simulation steps before returning the observation corresponding to the end of that simulated period.

This is where the `num_environment_steps` in the `info` dictionary becomes vital. It tells you how many simulation steps actually occurred since the last time your policy was called.

```python
env = Switchboard(time_scaling=1.0) # Real-time mode

# Benchmark to get realistic timing info
env.benchmark_policy(random_policy)
env.benchmark_simulation()

# Run the simulation
obs_list, info = env.step(random_policy)

# This could be > 1 if the policy was slow
print(f"Environment took {info['num_environment_steps']} steps.")
```

## Rules: The Engine of the Environment

Rules define the causal link between actions (button presses) and observations (slot activations). You can add rules manually, load predefined scenarios, or create complex custom logic.

### Loading and Inspecting Rules

You can load sets of rules, called **scenarios**, either at initialization or later.

```python
# Load a scenario by name during initialization
env = Switchboard(action_dim=10, obs_dim=10, scenario="direct_rules")

# Or load it after creation
env.load_rules("direct_rules", replace=True) # replace=True clears existing rules first

# You can inspect the loaded rules
for rule in env.list_rules():
    print(f"Rule ID: {rule['id']}, Description: {rule['description']}")
```

### The RuleBuilder

The `RuleBuilder` provides a convenient way to create common rule types without writing a full class.

```python
from environments.switchboard.switchboard import RuleBuilder

# Direct mapping: Button 0 activates Slot 0
env.add_rule(RuleBuilder.direct(0, 0))

# AND combination: Buttons 1 and 2 must be pressed to activate Slot 3
env.add_rule(RuleBuilder.and_combo([1, 2], 3))

# Toggle: Button 4 toggles Slot 5 on and off
env.add_rule(RuleBuilder.toggle(4, 5))

# More complex rules are available:
# - delayed: Activates after a delay
# - sequence: Requires a specific order of presses
# - hold: Requires a button to be held down for a duration
# - decay: Activation fades over time
# - conditional: Activates based on a custom condition function
```

### Custom Rules: `LambdaRule`

For simple, stateless custom logic, `LambdaRule` is perfect. You provide a function that calculates the observation update based on current actions and observations.

```python
from environments.switchboard.switchboard import LambdaRule

def my_logic(actions, obs, state, step):
    obs_update = torch.zeros_like(obs)
    # If button 0 is pressed AND slot 1 is already active...
    if actions[0] > 0 and obs[1] > 0.5:
        obs_update[2] = 1.0 # ...activate slot 2
    return obs_update

rule = LambdaRule("my_custom_rule", my_logic)
env.add_rule(rule)
```

### Custom Rules: `StatefulRule`

For complex behaviors that require memory or internal state (e.g., counters, timers, sequential logic), you can create a class that inherits from `StatefulRule`.

This allows you to implement rules with complex temporal dynamics, like oscillators, pulse generators, or state machines.

```python
from environments.switchboard.switchboard import StatefulRule

class OscillatingRule(StatefulRule):
    """A rule where a button press triggers a sine wave oscillation in a slot."""
    def __init__(self, rule_id, action_idx, obs_idx, frequency=0.1):
        super().__init__(rule_id, f"Oscillate {action_idx} -> {obs_idx}")
        self.action_idx = action_idx
        self.obs_idx = obs_idx
        self.frequency = frequency
        self.phase = 0.0
        self.is_active = False

    def evaluate(self, actions, observations, step):
        obs_update = torch.zeros_like(observations)

        # Activate on button press
        if actions[self.action_idx] > 0:
            self.is_active = True

        # Generate oscillation if active
        if self.is_active:
            value = (torch.sin(torch.tensor(self.phase)).item() + 1.0) / 2.0
            obs_update[self.obs_idx] = value
            self.phase += self.frequency

        return obs_update

    def reset(self):
        super().reset()
        self.phase = 0.0
        self.is_active = False

# Add it to the environment
env.add_rule(OscillatingRule("osc1", 7, 6, frequency=0.2))
```

## Creating Scenarios

A **scenario** is a collection of rules that defines a specific task or puzzle. To create one:

1.  **Create a Scenario Class:** In `environments/switchboard/scenarios/scenarios.py` (or a new file like `environments/switchboard/scenarios/custom_scenarios.py`), create a class that inherits from `Scenario`.
2.  **Implement `get_rules()`:** This method should return a list of `Rule` objects that constitute your scenario.
3.  **Register the Scenario:** The environment will automatically discover scenarios in defined files when `env.load_rules()` is called or `scenario` argument is used during `Switchboard` initialization.

```python
# In .../scenarios/custom_scenarios.py
from .scenarios import Scenario
from environments.switchboard.switchboard import RuleBuilder

class MyPuzzle(Scenario):
    name = "my_puzzle" # This name will be used to load the scenario

    def get_rules(self):
        return [
            RuleBuilder.direct(0, 0, rule_id="button_0_to_slot_0"),
            RuleBuilder.and_combo([1, 2], 3, rule_id="buttons_1_2_to_slot_3"),
            # ... add more rules for your puzzle
        ]
```

You can then load your scenario by its `name`:

```bash
# From the command line using the PyGame interface
python environments/switchboard/pygame_interface.py --scenario my_puzzle

# Or in your Python code
env = Switchboard(scenario="my_puzzle")
```

This structured approach allows for creating complex, reusable, and easily shareable environments for a wide range of research applications.

## PyGame Interface for Visualization

The environment provides an interactive PyGame interface for visualization and manual interaction. This is useful for debugging, understanding rule behaviors, and manually testing policies.

If you want to try out the environment yourself use the provided PyGame interface for visualization:

```bash
# Load a scenario (e.g., "direct_rules")
python pygame_interface.py --scenario direct_rules

# Custom window size
python pygame_interface.py --scenario direct_rules --width 1400 --height 800

# Show all options
python pygame_interface.py --help
```

### Controls

**Buttons:**
- `0-9 Keys`: Press buttons 0-9
- `Mouse Click`: Toggle buttons on/off

**Navigation:**
- `Scroll Wheel`: Navigate rules panel
- `TAB`: Hide/show rules panel

**Stepping:**
- `P`: Pause/Resume auto-stepping
- `ENTER`: Single step (when paused)
- `+/-`: Adjust auto-step speed

**Other:**
- `R`: Reset environment
- `Q/ESC`: Quit
