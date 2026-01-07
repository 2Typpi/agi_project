import torch
import time
import sys
import os
import copy
from typing import List, Dict, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from realtime_environment import RealTimeEnvironment


class Rule(ABC):
    """
    Base class for rules with programmable behavior.

    Rules can be defined by providing custom functions for:
    - evaluate: Compute observation updates from actions and current state
    - reset: Reset rule-specific state

    All rules automatically track statistics including:
    - Active state (on/off)
    - Total triggers
    - Time active
    - Activation history
    """

    def __init__(self, rule_id: str, description: str = ""):
        self.rule_id = rule_id
        self.description = description
        self.state = {}  # User-defined state storage

    @abstractmethod
    def evaluate(self, actions: torch.Tensor, observations: torch.Tensor, step: int) -> torch.Tensor:
        """
        Evaluate rule and return observation updates.

        Args:
            actions: Current action vector [action_dim]
            observations: Current observation state [obs_dim]

        Returns:
            obs_update: Observation changes to apply [obs_dim]
        """
        pass

    def reset(self):
        """Reset rule state for new episode"""
        self.state = {}


class LambdaRule(Rule):
    """
    Rule defined by a lambda function or callable.

    The evaluate_fn should have signature:
        fn(actions, observations, rule_state, step_count) -> obs_update

    Example:
        # Simple direct mapping
        rule = LambdaRule(
            "button_0_to_slot_0",
            lambda actions, obs, state, step: torch.tensor([actions[0], 0, 0, ...])
        )

        # Delayed activation
        rule = LambdaRule(
            "delayed",
            lambda actions, obs, state, step: delayed_logic(actions, obs, state, step)
        )
    """

    def __init__(
        self,
        rule_id: str,
        evaluate_fn: Callable[[torch.Tensor, torch.Tensor, Dict, int], torch.Tensor],
        reset_fn: Optional[Callable[[Dict], None]] = None,
        description: str = ""
    ):
        super().__init__(rule_id, description or rule_id)
        self.evaluate_fn = evaluate_fn
        self.reset_fn = reset_fn

    def evaluate(self, actions: torch.Tensor, observations: torch.Tensor, step: int) -> torch.Tensor:
        return self.evaluate_fn(actions, observations, self.state, step)

    def reset(self):
        super().reset()  # Reset stats
        if self.reset_fn:
            self.reset_fn(self.state)




class RuleBuilder:
    """
    Helper class for building common rule patterns with a fluent API.

    Example usage:
        # Direct mapping
        rule = RuleBuilder.direct(action_idx=0, obs_idx=0)

        # Delayed mapping
        rule = RuleBuilder.delayed(action_idx=1, obs_idx=1, delay=5)

        # AND combination
        rule = RuleBuilder.and_combo([0, 1], obs_idx=2)

        # Custom rule with builder pattern
        rule = (RuleBuilder()
            .when(lambda a, o: a[0] > 0.5)
            .then(lambda a, o, s: torch.tensor([1.0, 0, 0, ...]))
            .build("custom_rule"))
    """

    @staticmethod
    def direct(action_idx: int, obs_idx: int, strength: float = 1.0, rule_id: Optional[str] = None):
        """Button[i] -> Slot[j] (immediate)"""
        rule_id = rule_id or f"direct_{action_idx}_to_{obs_idx}"

        def evaluate_fn(actions, obs, state, step):
            update = torch.zeros_like(obs)
            if action_idx < len(actions) and actions[action_idx] > 0:
                update[obs_idx] = strength
            return update

        return LambdaRule(rule_id, evaluate_fn, description=f"Button {action_idx} -> Slot {obs_idx}")

    @staticmethod
    def delayed(action_idx: int, obs_idx: int, delay: int, strength: float = 1.0, rule_id: Optional[str] = None):
        """Button[i] -> Slot[j] (with N-step delay)"""
        rule_id = rule_id or f"delayed_{action_idx}_to_{obs_idx}_delay{delay}"

        def evaluate_fn(actions, obs, state, step):
            update = torch.zeros_like(obs)

            # Initialize pending list if needed
            if 'pending' not in state:
                state['pending'] = []

            # Add new activation if button pressed
            if action_idx < len(actions) and actions[action_idx] > 0:
                state['pending'].append((step + delay, strength))

            # Check for due activations
            due_activations = [s for t, s in state['pending'] if t == step]
            if due_activations:
                update[obs_idx] = sum(due_activations)
                state['pending'] = [(t, s) for t, s in state['pending'] if t != step]

            return update

        def reset_fn(state):
            state['pending'] = []

        return LambdaRule(rule_id, evaluate_fn, reset_fn,
                         description=f"Button {action_idx} -> Slot {obs_idx} (delay {delay})")

    @staticmethod
    def and_combo(action_indices: List[int], obs_idx: int, strength: float = 1.0, rule_id: Optional[str] = None):
        """Buttons [i, j, ...] -> Slot[k] (simultaneous press)"""
        rule_id = rule_id or f"and_{','.join(map(str, action_indices))}_to_{obs_idx}"

        def evaluate_fn(actions, obs, state, step):
            update = torch.zeros_like(obs)
            all_pressed = all(idx < len(actions) and actions[idx] > 0 for idx in action_indices)
            if all_pressed:
                update[obs_idx] = strength
            return update

        return LambdaRule(rule_id, evaluate_fn,
                         description=f"Buttons {action_indices} -> Slot {obs_idx} (AND)")

    @staticmethod
    def sequence(action_sequence: List[int], obs_idx: int, time_window: int = 10, hold: int=10,
                strength: float = 1.0, rule_id: Optional[str] = None):
        """Buttons [i, j, ...] -> Slot[k] (in sequence within time window)"""
        rule_id = rule_id or f"seq_{','.join(map(str, action_sequence))}_to_{obs_idx}"

        def evaluate_fn(actions, obs, state, step):
            update = torch.zeros_like(obs)

            # Initialize state
            if 'progress' not in state:
                state['progress'] = 0
                state['last_action_step'] = -1
                state["hold"] = 0
            
            # Check if time window expired
            if step - state['last_action_step'] > time_window:
                state['progress'] = 0

            # Check for next expected action
            if state['progress'] < len(action_sequence):
                expected = action_sequence[state['progress']]
                if expected < len(actions) and actions[expected] > 0:
                    state['progress'] += 1
                    state['last_action_step'] = step

                    # Sequence completed
                    if state['progress'] == len(action_sequence):

                        state['progress'] = 0
                        state["hold"] = step + hold
            if state["hold"] > step:
                update[obs_idx] = strength
            return update

        def reset_fn(state):
            state['progress'] = 0
            state['last_action_step'] = -1
            state['hold'] = 0

        return LambdaRule(rule_id, evaluate_fn, reset_fn,
                         description=f"Sequence {action_sequence} -> Slot {obs_idx}")

    @staticmethod
    def hold(action_idx: int, obs_idx: int, duration: int, strength: float = 1.0, rule_id: Optional[str] = None):
        """Hold Button[i] for N steps -> Slot[j]"""
        rule_id = rule_id or f"hold_{action_idx}_for_{duration}_to_{obs_idx}"

        def evaluate_fn(actions, obs, state, step):
            update = torch.zeros_like(obs)

            if 'hold_count' not in state:
                state['hold_count'] = 0

            if action_idx < len(actions) and actions[action_idx] > 0:
                state['hold_count'] += 1
                if state['hold_count'] >= duration:
                    update[obs_idx] = strength
            else:
                state['hold_count'] = 0

            return update

        def reset_fn(state):
            state['hold_count'] = 0

        return LambdaRule(rule_id, evaluate_fn, reset_fn,
                         description=f"Hold Button {action_idx} for {duration} -> Slot {obs_idx}")

    @staticmethod
    def toggle(action_idx: int, obs_idx: int, strength: float = 1.0, rule_id: Optional[str] = None):
        """Button[i] -> Toggle Slot[j]"""
        rule_id = rule_id or f"toggle_{action_idx}_to_{obs_idx}"

        def evaluate_fn(actions, obs, state, step):
            update = torch.zeros_like(obs)

            if 'is_on' not in state:
                state['is_on'] = False
                state['last_step'] = -1

            # Trigger on new press (not hold)
            if action_idx < len(actions) and actions[action_idx] > 0 and step != state['last_step']:
                state['is_on'] = not state['is_on']
                state['last_step'] = step

            update[obs_idx] = strength if state['is_on'] else 0.0
            return update

        def reset_fn(state):
            state['is_on'] = False
            state['last_step'] = -1

        return LambdaRule(rule_id, evaluate_fn, reset_fn,
                         description=f"Toggle Button {action_idx} -> Slot {obs_idx}")

    @staticmethod
    def conditional(condition_fn: Callable[[torch.Tensor, torch.Tensor], bool],
                   action_idx: int, obs_idx: int, strength: float = 1.0, rule_id: Optional[str] = None):
        """If condition(actions, obs) then Button[i] -> Slot[j]"""
        rule_id = rule_id or f"cond_{action_idx}_to_{obs_idx}"

        def evaluate_fn(actions, obs, state, step):
            update = torch.zeros_like(obs)
            if condition_fn(actions, obs):
                if action_idx < len(actions) and actions[action_idx] > 0:
                    update[obs_idx] = strength
            return update

        return LambdaRule(rule_id, evaluate_fn,
                         description=f"Conditional Button {action_idx} -> Slot {obs_idx}")

    @staticmethod
    def decay(action_idx: int, obs_idx: int, decay_rate: float = 0.1, strength: float = 1.0, rule_id: Optional[str] = None):
        """Button[i] -> Slot[j] (with decay)"""
        rule_id = rule_id or f"decay_{action_idx}_to_{obs_idx}"

        def evaluate_fn(actions, obs, state, step):
            update = torch.zeros_like(obs)

            if 'activation' not in state:
                state['activation'] = 0.0

            # New press sets activation to full
            if action_idx < len(actions) and actions[action_idx] > 0:
                state['activation'] = strength

            # Output current activation
            update[obs_idx] = state['activation']

            # Decay for next step
            state['activation'] = max(0.0, state['activation'] - decay_rate)

            return update

        def reset_fn(state):
            state['activation'] = 0.0

        return LambdaRule(rule_id, evaluate_fn, reset_fn,
                         description=f"Decay Button {action_idx} -> Slot {obs_idx}")


class Switchboard(RealTimeEnvironment):
    """
    Switchboard environment with programmable rule API.

    This version allows you to define rules using:
    1. Lambda functions (LambdaRule)
    2. Custom classes (inherit from Rule)
    3. Rule builder helpers (RuleBuilder.*)

    Example usage:
        env = Switchboard(action_dim=10, obs_dim=10)

        # Add rules using builder
        env.add_rule(RuleBuilder.direct(0, 0))
        env.add_rule(RuleBuilder.delayed(1, 1, delay=5))
        env.add_rule(RuleBuilder.and_combo([2, 3], 2))

    """

    def __init__(
        self,
        action_dim: int = 10,
        obs_dim: int = 10,
        rules: Optional[List[Rule]] = None,
        scenario: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            action_dim: Number of buttons/actions available
            obs_dim: Number of observation slots
            rules: List of rules. If None, starts with empty ruleset
            scenario: Name of the scenario to load from rules.py.
            **kwargs: Arguments passed to RealTimeEnvironment
        """
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.rules = rules if rules is not None else []

        if scenario:
            self.load_rules(scenario, replace=True)

        # Initialize environment state
        self.current_observations = torch.zeros(obs_dim)

        super().__init__(**kwargs)

    def __copy__(self):
        """
        Creates a shallow copy. 
        The new environment will reference the same rules and the same tensor object.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo):
        """
        Creates a deep copy.
        Useful for 'imagination' or branching simulations.
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        for k, v in self.__dict__.items():
            # Special handling for PyTorch tensors to ensure gradients 
            # and device placement are handled correctly during the deepcopy
            if isinstance(v, torch.Tensor):
                setattr(result, k, v.clone())
            else:
                setattr(result, k, copy.deepcopy(v, memo))
                
        return result

    def _get_initial_state(self) -> torch.Tensor:
        """Get initial observation state (all slots off)"""
        initial_obs = torch.zeros(self.obs_dim, device=self.device)
        self.current_observations = initial_obs.clone()
        return initial_obs

    def _step_simulation(self, action: torch.Tensor) -> torch.Tensor:
        """
        Perform one simulation step by evaluating all rules.

        Args:
            action: Action vector [action_dim] with values in [0, 1]

        Returns:
            observation: New observation state [obs_dim]
        """
        # Ensure action is proper size and range
        
        if action.numel() < self.action_dim:
            padded_action = torch.zeros(self.action_dim, device=self.device)
            padded_action[:action.numel()] = action.flatten()
            action = padded_action
        else:
            action = action.flatten()[:self.action_dim]

        action = torch.clamp(action, 0.0, 1.0)

        # Evaluate all rules and update statistics
        rule_updates = []
        for rule in self.rules:
            try:
                update = rule.evaluate(action, self.current_observations, self.step_count)
                if update is not None and update.numel() == self.obs_dim:
                    rule_updates.append(update)

            except Exception as e:
                print(f"Warning: Rule {rule.rule_id} failed: {e}")
                continue

        # Combine rule updates
        if len(rule_updates) == 0:
            new_obs = self.current_observations.clone()
        else:
            combined_update = torch.stack(rule_updates).max(dim=0)[0]
            new_obs = torch.clamp(combined_update, 0.0, 1.0)

        self.current_observations = new_obs.clone()
        self.step_count += 1

        return new_obs

    def _get_state(self):
        """Get current environment state (for compatibility)"""
        return {
            'observations': self.current_observations.clone(),
            'step_count': self.step_count,
            'rules': [
                {
                    'id': rule.rule_id,
                    'description': rule.description,
                    'state': rule.state
                }
                for rule in self.rules
            ]
        }
    
    def set_obs(self, current_state):
        """Set current environment state (for continous thinking)"""
        self.current_observations = current_state

    def reset(self) -> torch.Tensor:
        """Reset environment and all rules"""
        for rule in self.rules:
            rule.reset()

        self.step_count = 0
        initial_state = self._get_initial_state()
        self.state = initial_state
        return initial_state

    def add_rule(self, rule: Rule):
        """Add a new rule to the switchboard"""
        self.rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID. Returns True if found and removed."""
        for i, rule in enumerate(self.rules):
            if rule.rule_id == rule_id:
                del self.rules[i]
                return True
        return False

    def get_rule(self, rule_id: str) -> Optional[Rule]:
        """Get a rule by ID"""
        for rule in self.rules:
            if rule.rule_id == rule_id:
                return rule
        return None

    def list_rules(self) -> List[Dict[str, Any]]:
        """Get information about all rules"""
        return [
            {
                "id": rule.rule_id,
                "description": rule.description,
                "state": rule.state
            }
            for rule in self.rules
        ]

    def clear_rules(self):
        """Remove all rules"""
        self.rules = []

   
    def load_rules(self, scenario_name: str, replace: bool = False):
        """
        Load rules from a scenario class in `rules.py`.

        Args:
            scenario_name: Name of the scenario to load (e.g., "direct_rules").
            replace: If True, replace existing rules. If False, append to existing rules.

        Example:
            env = Switchboard(action_dim=10, obs_dim=10)
            env.load_rules("direct_rules")
        """
        try:
            # This will trigger the __init__.py in the scenarios package,
            # which will load all the scenario files.
            from . import scenarios
            from .scenarios.scenarios import Scenario
        except ImportError:
            raise ImportError(
                "Could not import the scenarios package or the Scenario class."
            )

        scenario_instance = Scenario.from_name(scenario_name)
        loaded_rules = scenario_instance.get_rules()

        if replace:
            self.rules = loaded_rules
        else:
            self.rules.extend(loaded_rules)

        print(f"✓ Loaded {len(loaded_rules)} rules from {scenario_instance.__class__.__name__}")
        if not replace and len(self.rules) > len(loaded_rules):
            print(f"  Total rules now: {len(self.rules)}")

    def render_state(self) -> str:
        """Create a text representation of the current state"""
        lines = []
        lines.append("=== SWITCHBOARD STATE ===")
        lines.append(f"Step: {self.step_count}")
        lines.append("")

        # Show observations
        lines.append("OBSERVATION SLOTS:")
        obs_str = ""
        for i, val in enumerate(self.current_observations):
            status = "█" if val > 0.5 else "░" if val > 0.1 else "·"
            obs_str += status
            if (i + 1) % 10 == 0:
                obs_str += " "
        lines.append(f"[{obs_str}]")
        lines.append(f"Values: {[f'{x:.2f}' for x in self.current_observations.tolist()]}")
        lines.append("")

        # Show rules
        lines.append(f"RULES ({len(self.rules)}):")
        for rule in self.rules:
            lines.append(f"  • {rule.description} (steps: {rule.step_count})")

        return "\n".join(lines)
