from environments.switchboard.scenarios.scenarios import Scenario, RuleBuilder

class MyCustomScenario(Scenario):
    name = "my_custom_scenario"
    def get_rules(self):
        rules = []
        rules.append(RuleBuilder.direct(action_idx=0, obs_idx=0))
        return rules

class TemporalPPORules(Scenario):
    name = "temporal_ppo"
    def get_rules(self):
        rules = []
        # Direct rule (button 0, slot 0): Immediate feedback for baseline learning
        rules.append(RuleBuilder.direct(action_idx=0, obs_idx=0, strength=1.0))

        # Delayed rules (buttons 2-4, slot 2-4): Testing temporal learning
        rules.append(RuleBuilder.delayed(action_idx=2, obs_idx=2, delay=3, strength=1.0))  # 3-step delay
        rules.append(RuleBuilder.delayed(action_idx=3, obs_idx=3, delay=5, strength=1.0))  # 5-step delay
        rules.append(RuleBuilder.delayed(action_idx=4, obs_idx=4, delay=8, strength=1.0))  # 8-step delay

        # AND combination (buttons 5&6, slot 5): Multi-button coordination
        rules.append(RuleBuilder.and_combo(action_indices=[5, 6], obs_idx=5, strength=1.0))
        # HOLD AND combination (buttons 1&8, slot 1): Multi-button coordination
        rules.append(RuleBuilder.hold_and_combo(action_indices=[1, 8], duration=3, obs_idx=1, strength=1.0))

        # Sequence rule (buttons 7-8-9, slot 6): Sequential pattern learning
        rules.append(RuleBuilder.sequence(action_sequence=[7, 8, 9], obs_idx=6, time_window=20, hold=3))

        # Hold rule (button 9, slot 7): Duration-based activation
        rules.append(RuleBuilder.hold(action_idx=9, obs_idx=7, duration=5, strength=1.0))

        # Hold rule (button 6, slot 8): Duration-based activation
        rules.append(RuleBuilder.hold(action_idx=6, obs_idx=8, duration=3, strength=1.0))

        # Hold rule (button 6, slot 9): Duration-based activation
        rules.append(RuleBuilder.hold(action_idx=6, obs_idx=9, duration=15, strength=1.0))

        return rules
