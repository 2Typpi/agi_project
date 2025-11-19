from environments.switchboard.scenarios.scenarios import Scenario, RuleBuilder

class MyCustomScenario(Scenario):
    name = "my_custom_scenario"
    def get_rules(self):
        rules = []
        rules.append(RuleBuilder.direct(action_idx=0, obs_idx=0))
        return rules
