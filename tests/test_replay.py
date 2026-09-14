import pytest
from robot_ci.simulation.mock import MockSimulator
from robot_ci.replay.engine import ReplayEngine
from robot_ci.policies.registry import get_policy
from robot_ci.scenarios.manager import ScenarioManager

def test_replay_engine(scenarios_dir):
    manager = ScenarioManager()
    scenarios = manager.load_scenarios(str(scenarios_dir))
    scenario = scenarios[0]
    
    sim = MockSimulator()
    engine = ReplayEngine(sim)
    policy = get_policy("baseline")
    
    rollout = engine.run(policy, scenario, seed=42)
    assert rollout.scenario_id == scenario.scenario_id
    assert rollout.policy_version == policy.version
    assert rollout.seed == 42
    assert len(rollout.observations) > 0
    assert len(rollout.actions) == len(rollout.observations) - 1
