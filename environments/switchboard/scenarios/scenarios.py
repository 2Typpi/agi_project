
import torch
from environments.switchboard.switchboard import Rule, RuleBuilder, LambdaRule, Switchboard

class Scenario(object):
    name = None

    def get_rules(self):
        raise NotImplementedError

    @classmethod
    def from_name(cls, name):
        for scenario_class in cls.__subclasses__():
            if scenario_class.name == name:
                return scenario_class()
        raise ValueError(f"Scenario '{name}' not found.")

class DirectRules(Scenario):
    name = "direct_rules"
    def get_rules(self):
        rules = []
        # ========== Single Button Mappings ==========
        rules.append(RuleBuilder.direct(action_idx=0, obs_idx=0, strength=1.0, rule_id="single_0"))
        rules.append(RuleBuilder.direct(action_idx=1, obs_idx=1, strength=1.0, rule_id="single_1"))
        rules.append(RuleBuilder.direct(action_idx=2, obs_idx=2, strength=1.0, rule_id="single_2"))
        # ========== AND Combinations ==========
        rules.append(RuleBuilder.and_combo(action_indices=[3, 4], obs_idx=3, strength=1.0, rule_id="and_3_4"))
        rules.append(RuleBuilder.and_combo(action_indices=[5, 6], obs_idx=4, strength=1.0, rule_id="and_5_6"))
        rules.append(RuleBuilder.and_combo(action_indices=[7, 8, 9], obs_idx=5, strength=1.0, rule_id="and_7_8_9"))
        # ========== Overlapping Usage ==========
        return rules

class MoveAndDecayRules(Scenario):
    name = "move_and_decay_rules"
    def get_rules(self):
        rules = []
        # ========== Single Button Mappings ==========
        def evaluate_fn(actions, obs, state, step) -> torch.Tensor:
            obs_update = obs.clone()
            obs_update = obs_update * obs_update*0.99 # decay
            obs_update[obs_update <= 0.05] = 0.0
            idx = obs.argmax()
            if actions[0] == 1:
                idx = (idx + 1) % obs.shape[0]
            elif actions[1] == 1:
                idx = (idx - 1) % obs.shape[0]
            obs_update[idx] = 1.0
            return obs_update
        custom_rule = LambdaRule("move_and_decay", evaluate_fn=evaluate_fn, description="Move active slot, decay inactive slots.")
        rules.append(custom_rule)
        return rules

class TemporalRules(Scenario):
    name = "temporal_rules"
    def get_rules(self):
        env = Switchboard(action_dim=10, obs_dim=10)
        # Must press in sequence
        env.add_rule(RuleBuilder.sequence([0, 1, 2], 0, time_window=20))
        # Must hold button 3 for 15 steps
        env.add_rule(RuleBuilder.hold(3, 1, duration=15))
        # Toggle mechanics
        env.add_rule(RuleBuilder.toggle(4, 2))
        env.add_rule(RuleBuilder.toggle(5, 3))
        # Decay signal
        env.add_rule(RuleBuilder.decay(6, 4, decay_rate=0.05))
        return env.rules

class ChallengingRules(Scenario):
    name = "challenging_rules"
    def get_rules(self):
        rules = []
        # ========== Rule 1: Mutually Exclusive (Button 0 works ONLY if Button 1 is OFF) ==========
        def exclusive_rule_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            if actions[0] > 0.5 and actions[1] < 0.5:
                obs_update[0] = 1.0
            return obs_update
        rules.append(LambdaRule("exclusive_0_not_1", evaluate_fn=exclusive_rule_fn, description="Button 0 (only if 1 is OFF) -> Slot 0"))
        # ========== Rule 2: XOR Pattern (Exactly ONE of buttons 2,3,4 must be pressed) ==========
        def xor_rule_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            count = sum([1 for i in [2, 3, 4] if actions[i] > 0.5])
            if count == 1:
                obs_update[1] = 1.0
            return obs_update
        rules.append(LambdaRule("xor_2_3_4", evaluate_fn=xor_rule_fn, description="Exactly ONE of [2,3,4] pressed -> Slot 1"))
        # ========== Rule 3: Exactly N buttons (Exactly 3 buttons from 5,6,7,8,9) ==========
        def count_three_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            count = sum([1 for i in range(5, 10) if actions[i] > 0.5])
            if count == 3:
                obs_update[2] = 1.0
            return obs_update
        rules.append(LambdaRule("count_three_5to9", evaluate_fn=count_three_fn, description="Exactly 3 of buttons [5,6,7,8,9] -> Slot 2"))
        # ========== Rule 4: Sequential Dependency (Button 0 in step 1, then button 1 in step 2) ==========
        def sequential_dep_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            if 'saw_button_0' not in state:
                state['saw_button_0'] = False
            if actions[0] > 0.5:
                state['saw_button_0'] = True
            if state['saw_button_0'] and actions[1] > 0.5:
                obs_update[3] = 1.0
            return obs_update
        def reset_seq_fn(state):
            state['saw_button_0'] = False
        rules.append(LambdaRule("seq_0_then_1", evaluate_fn=sequential_dep_fn, reset_fn=reset_seq_fn, description="Button 0 earlier, then Button 1 -> Slot 3"))
        # ========== Rule 5: Negative Precondition (Button 7 works only if slot 0 is OFF) ==========
        def negative_precond_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            if actions[7] > 0.5 and obs[0] < 0.5:
                obs_update[4] = 1.0
            return obs_update
        rules.append(LambdaRule("button7_if_obs0_off", evaluate_fn=negative_precond_fn, description="Button 7 (only if Slot 0 OFF) -> Slot 4"))
        # ========== Rule 6: Parity Check (Even number of buttons pressed) ==========
        def parity_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            count = sum([1 for i in range(10) if actions[i] > 0.5])
            if count > 0 and count % 2 == 0:
                obs_update[5] = 1.0
            return obs_update
        rules.append(LambdaRule("even_parity", evaluate_fn=parity_fn, description="Even number of buttons pressed -> Slot 5"))
        # ========== Rule 7: Two-step combo (Button 8 in step 1, button 9 in step 2) ==========
        def two_step_combo_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            if 'step1_button8' not in state:
                state['step1_button8'] = False
            if actions[8] > 0.5:
                state['step1_button8'] = True
            if state['step1_button8'] and actions[9] > 0.5:
                obs_update[6] = 1.0
            return obs_update
        def reset_two_step_fn(state):
            state['step1_button8'] = False
        rules.append(LambdaRule("seq_8_then_9", evaluate_fn=two_step_combo_fn, reset_fn=reset_two_step_fn, description="Button 8 earlier, then Button 9 -> Slot 6"))
        # ========== Rule 8: Complex combination (obs 1 AND obs 2 must both be active) ==========
        def obs_dependency_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            if obs[1] > 0.5 and obs[2] > 0.5:
                obs_update[7] = 1.0
            return obs_update
        rules.append(LambdaRule("obs1_and_obs2_active", evaluate_fn=obs_dependency_fn, description="Slot 1 AND Slot 2 active -> Slot 7"))
        return rules

class HardRules(Scenario):
    name = "hard_rules"
    def get_rules(self):
        rules = []
        # ========== Rule 1: 3-step sequence (Button 0 → 1 → 2) ==========
        rules.append(RuleBuilder.sequence([0, 1, 2], obs_idx=0, time_window=10, hold=3))
        # ========== Rule 2: 4-step sequence (Button 3 → 4 → 5 → 6) ==========
        rules.append(RuleBuilder.sequence([3, 4, 5, 6], obs_idx=1, time_window=10, hold=3))
        # ========== Rule 3: 5-step sequence (Button 5 → 6 → 7 → 8 → 9) ==========
        rules.append(RuleBuilder.sequence([5, 6, 7, 8, 9], obs_idx=2, time_window=10, hold=3))
        # ========== Rule 4: Specific action pattern (exactly buttons 0,3,7) ==========
        def exact_pattern_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            expected = torch.zeros_like(actions)
            expected[[0, 3, 7]] = 1.0
            if torch.all(actions == expected):
                obs_update[3] = 1.0
            return obs_update
        rules.append(LambdaRule("exact_0_3_7", evaluate_fn=exact_pattern_fn, description="Exactly buttons [0,3,7], no others -> Slot 3"))
        # ========== Rule 5: Rare combo (buttons 1,4,9 pressed together) ==========
        rules.append(RuleBuilder.and_combo([1, 4, 9], obs_idx=4))
        # ========== Rule 6: Two-stage dependency (activate slot 0, then press button 8) ==========
        def two_stage_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            if obs[0] > 0.5 and actions[8] > 0.5:
                obs_update[5] = 1.0
            return obs_update
        rules.append(LambdaRule("slot0_then_button8", evaluate_fn=two_stage_fn, description="Slot 0 active + Button 8 → Slot 5"))
        # ========== Rule 7: Three-stage chain (slot 1 active, then slot 2 active, then button 9) ==========
        def three_stage_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            if obs[1] > 0.5 and obs[2] > 0.5 and actions[9] > 0.5:
                obs_update[6] = 1.0
            return obs_update
        rules.append(LambdaRule("slot1_and_slot2_then_button9", evaluate_fn=three_stage_fn, description="Slots 1&2 active + Button 9 → Slot 6"))
        # ========== Rule 8: Never together (Button 2 activates ONLY if button 5 has NEVER been pressed) ==========
        def never_together_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            if 'button5_ever_pressed' not in state:
                state['button5_ever_pressed'] = False
            if actions[5] > 0.5:
                state['button5_ever_pressed'] = True
            if actions[2] > 0.5 and not state['button5_ever_pressed']:
                obs_update[7] = 1.0
            return obs_update
        def reset_never_fn(state):
            state['button5_ever_pressed'] = False
        rules.append(LambdaRule("button2_never5", evaluate_fn=never_together_fn, reset_fn=reset_never_fn, description="Button 2 (only if 5 never pressed in episode) → Slot 7"))
        # ========== Rule 9: Alternating pattern (press 0, then NOT 0, then 0, then NOT 0) ==========
        def alternating_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            if 'pattern' not in state:
                state['pattern'] = []
            state['pattern'].append(actions[0] > 0.5)
            if len(state['pattern']) >= 4:
                recent = state['pattern'][-4:]
                if recent == [True, False, True, False]:
                    obs_update[8] = 1.0
            return obs_update
        def reset_alt_fn(state):
            state['pattern'] = []
        rules.append(LambdaRule("alternating_0", evaluate_fn=alternating_fn, reset_fn=reset_alt_fn, description="Alternating: Button 0 ON, OFF, ON, OFF → Slot 8"))
        # ========== Rule 10: Ultimate challenge (combine multiple requirements) ==========
        def ultimate_fn(actions, obs, state, step):
            obs_update = torch.zeros_like(obs)
            if obs[0] > 0.5 and obs[1] > 0.5 and obs[2] > 0.5:
                button_count = sum(1 for i in range(10) if actions[i] > 0.5)
                if actions[0] > 0.5 and button_count % 2 == 0 and button_count > 0:
                    obs_update[9] = 1.0
            return obs_update
        rules.append(LambdaRule("ultimate_challenge", evaluate_fn=ultimate_fn, description="Slots 0,1,2 active + Button 0 + Even buttons → Slot 9"))
        return rules
