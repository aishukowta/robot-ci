import pytest
import numpy as np
from robot_ci.simulation.mock import MockSimulator
from robot_ci.simulation.pybullet_sim import PyBulletSimulator
from robot_ci.scenarios.manager import ScenarioManager

def test_mock_simulator_determinism(scenarios_dir):
    manager = ScenarioManager()
    scenarios = manager.load_scenarios(str(scenarios_dir))
    scenario = scenarios[0] # reach_simple
    
    sim1 = MockSimulator()
    sim1.seed(42)
    obs1 = sim1.reset(scenario)
    act = np.array([0.01, 0.01, 0.01, 1.0])
    next_obs1, r1, d1, info1 = sim1.step(act)
    
    sim2 = MockSimulator()
    sim2.seed(42)
    obs2 = sim2.reset(scenario)
    next_obs2, r2, d2, info2 = sim2.step(act)
    
    np.testing.assert_array_equal(obs1, obs2)
    np.testing.assert_array_equal(next_obs1, next_obs2)
    assert r1 == r2
    assert d1 == d2


def test_pybullet_simulator_runs_one_step(scenarios_dir):
    pytest.importorskip("pybullet")
    manager = ScenarioManager()
    scenarios = manager.load_scenarios(str(scenarios_dir))
    scenario = next(s for s in scenarios if s.scenario_id == "reach_simple")

    sim = PyBulletSimulator(gui=False, action_repeat=1)
    try:
        sim.seed(42)
        obs = sim.reset(scenario)
        action = np.array([0.01, 0.01, 0.0, 1.0])
        next_obs, reward, done, info = sim.step(action)
    finally:
        sim.close()

    assert obs.shape == (10,)
    assert next_obs.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert info["simulator"] == "pybullet"


def test_pybullet_simulator_seeded_step_is_deterministic(scenarios_dir):
    pytest.importorskip("pybullet")
    manager = ScenarioManager()
    scenarios = manager.load_scenarios(str(scenarios_dir))
    scenario = next(s for s in scenarios if s.scenario_id == "reach_simple")
    action = np.array([0.02, 0.01, 0.0, 1.0], dtype=np.float32)

    def one_step():
        sim = PyBulletSimulator(gui=False, action_repeat=2)
        try:
            sim.seed(42)
            obs = sim.reset(scenario)
            next_obs, reward, done, info = sim.step(action)
            return obs, next_obs, reward, done, info["distance_to_target"]
        finally:
            sim.close()

    first = one_step()
    second = one_step()
    np.testing.assert_allclose(first[0], second[0], atol=1e-5)
    np.testing.assert_allclose(first[1], second[1], atol=1e-5)
    assert first[2] == second[2]
    assert first[3] == second[3]
    assert first[4] == second[4]
