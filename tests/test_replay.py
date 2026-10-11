import pytest
import numpy as np
import pandas as pd
from robot_ci.replay.rollout import Rollout
from robot_ci.replay.engine import ReplayEngine
from robot_ci.simulation.mock import MockSimulator
from robot_ci.policies.registry import get_policy
from robot_ci.scenarios.manager import ScenarioManager
from robot_ci.scenarios.models import Scenario


def make_dummy_rollout(num_steps=2):
    obs0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    obs1 = np.array([0.3, 0.4, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    obs2 = np.array([0.6, 0.8, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    observations = [obs0, obs1, obs2][: num_steps + 1]

    act1 = np.array([0.03, 0.04, 0.0, 1.0], dtype=np.float32)
    act2 = np.array([0.06, 0.08, 0.0, 1.0], dtype=np.float32)
    actions = [act1, act2][:num_steps]

    rewards = [1.5, 2.5][:num_steps]
    states = [{"step": 0}, {"step": 1}, {"step": 2}][: num_steps + 1]

    return Rollout(
        scenario_id="reach_simple",
        policy_version="v1.0-baseline",
        seed=42,
        observations=observations,
        actions=actions,
        states=states,
        rewards=rewards,
        success=True,
        collision_count=1,
        total_steps=num_steps,
    )


def test_rollout_serialization_roundtrip():
    rollout = make_dummy_rollout(num_steps=2)

    # 1. to_dict serialization
    d = rollout.to_dict()
    assert d["scenario_id"] == "reach_simple"
    assert d["policy_version"] == "v1.0-baseline"
    assert d["seed"] == 42
    assert d["success"] is True
    assert d["collision_count"] == 1
    assert d["total_steps"] == 2
    assert isinstance(d["observations"], list)
    assert isinstance(d["actions"], list)
    assert len(d["observations"]) == 3
    assert len(d["actions"]) == 2

    # 2. from_dict deserialization
    reconstructed = Rollout.from_dict(d)
    assert reconstructed.scenario_id == rollout.scenario_id
    assert reconstructed.seed == rollout.seed
    assert reconstructed.success == rollout.success
    assert reconstructed.collision_count == rollout.collision_count
    assert len(reconstructed.observations) == 3
    assert isinstance(reconstructed.observations[0], np.ndarray)
    np.testing.assert_array_equal(reconstructed.observations[0], rollout.observations[0])
    np.testing.assert_array_equal(reconstructed.actions[0], rollout.actions[0])

    # 3. Missing optional collision_count fallback in dict
    del d["collision_count"]
    reconstructed_fallback = Rollout.from_dict(d)
    assert reconstructed_fallback.collision_count == 0


def test_telemetry_dataframe_hand_calculable():
    rollout = make_dummy_rollout(num_steps=2)

    # Test 1: Telemetry with target from observation [obs[7:10] = (1.0, 0.0, 0.0)]
    df = rollout.get_telemetry_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == [
        "Step",
        "Reward",
        "EE_X",
        "EE_Y",
        "EE_Z",
        "Distance_to_Target",
        "Action_Magnitude",
    ]
    assert len(df) == 2
    assert list(df["Step"]) == [0, 1]
    assert list(df["Reward"]) == [1.5, 2.5]
    assert list(df["EE_X"]) == pytest.approx([0.3, 0.6], abs=1e-5)
    assert list(df["EE_Y"]) == pytest.approx([0.4, 0.8], abs=1e-5)
    assert list(df["EE_Z"]) == pytest.approx([0.0, 0.0], abs=1e-5)

    # Hand calculations:
    # Step 0: obs1 position = (0.3, 0.4, 0.0), target = (1.0, 0.0, 0.0)
    # dist = sqrt((0.3-1.0)^2 + (0.4-0.0)^2) = sqrt(0.49 + 0.16) = sqrt(0.65) ~ 0.80622577
    assert pytest.approx(df["Distance_to_Target"][0], abs=1e-5) == float(np.sqrt(0.65))

    # Step 0: action1 = [0.03, 0.04, 0.0, 1.0] -> delta_pos mag = sqrt(0.03^2 + 0.04^2) = 0.05
    assert pytest.approx(df["Action_Magnitude"][0], abs=1e-5) == 0.05

    # Test 2: Telemetry with explicit target position [0.3, 0.0, 0.0]
    df_explicit = rollout.get_telemetry_dataframe(target_position=[0.3, 0.0, 0.0])
    # Step 0: obs1 position = (0.3, 0.4, 0.0), explicit target = (0.3, 0.0, 0.0) -> dist = 0.4
    assert pytest.approx(df_explicit["Distance_to_Target"][0], abs=1e-5) == 0.4


def test_telemetry_dataframe_empty_rollout():
    rollout = Rollout(
        scenario_id="s1",
        policy_version="v1.0",
        seed=42,
        observations=[np.zeros(10)],
        actions=[],
        states=[{}],
        rewards=[],
        success=False,
        collision_count=0,
        total_steps=0,
    )
    df = rollout.get_telemetry_dataframe()
    assert len(df) == 0
    assert list(df.columns) == [
        "Step",
        "Reward",
        "EE_X",
        "EE_Y",
        "EE_Z",
        "Distance_to_Target",
        "Action_Magnitude",
    ]


def test_telemetry_dataframe_mismatched_sequence_lengths():
    # If observations has 1 item and actions has 1 item, observations[1:] is empty (len 0),
    # while steps/rewards/actions have len 1. Pandas rejects un-equal length arrays.
    rollout = Rollout(
        scenario_id="s1",
        policy_version="v1.0",
        seed=42,
        observations=[np.zeros(10)], # len 1
        actions=[np.zeros(4)],       # len 1
        states=[{}, {}],
        rewards=[1.0],               # len 1
        success=False,
        collision_count=0,
        total_steps=1,
    )
    with pytest.raises(ValueError, match="All arrays must be of the same length"):
        rollout.get_telemetry_dataframe()


def test_replay_engine_single_run(scenarios_dir):
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
    assert len(rollout.observations) == rollout.total_steps + 1
    assert len(rollout.actions) == rollout.total_steps
    assert len(rollout.rewards) == rollout.total_steps
    assert isinstance(rollout.success, bool)


def test_replay_engine_max_steps_cutoff():
    # Scenario with 3 max steps
    scenario = Scenario(
        scenario_id="short_scenario",
        task_name="reach",
        initial_state={"ee_position": [0.0, 0.0, 0.0]},
        target_state={"position": [10.0, 10.0, 10.0]}, # unreachable target
        environment_params={},
        max_episode_steps=3,
        success_criteria={"distance_threshold": 0.01},
        safety_criteria={},
    )
    sim = MockSimulator()
    engine = ReplayEngine(sim)
    policy = get_policy("baseline")

    rollout = engine.run(policy, scenario, seed=42)

    assert rollout.total_steps == 3
    assert not rollout.success


def test_replay_engine_multi_trial_batching(scenarios_dir):
    manager = ScenarioManager()
    scenarios = manager.load_scenarios(str(scenarios_dir))

    sim = MockSimulator()
    engine = ReplayEngine(sim)
    baseline = get_policy("baseline")
    candidate = get_policy("candidate")

    # 1. Single trial per scenario (trials=1)
    batch_single = engine.run_batch(baseline, scenarios, global_seed=100, trials=1)
    assert len(batch_single) == len(scenarios)
    assert "reach_simple" in batch_single

    # 2. Multi-trial per scenario (trials=3)
    b_batch = engine.run_batch(baseline, scenarios, global_seed=100, trials=3)
    c_batch = engine.run_batch(candidate, scenarios, global_seed=100, trials=3)

    assert len(b_batch) == len(scenarios) * 3
    assert "reach_simple__trial_1" in b_batch
    assert "reach_simple__trial_2" in b_batch
    assert "reach_simple__trial_3" in b_batch

    # Verify seed propagation and matched-seed design between baseline & candidate
    # Trial index 0 seed for reach_simple: global_seed (100) + scenario.seed (42) + 0 = 142
    assert b_batch["reach_simple__trial_1"].seed == 142
    assert c_batch["reach_simple__trial_1"].seed == 142

    # Trial index 1 seed for reach_simple: 100 + 42 + 1 = 143
    assert b_batch["reach_simple__trial_2"].seed == 143
    assert c_batch["reach_simple__trial_2"].seed == 143


def test_replay_engine_invalid_trials_raises():
    sim = MockSimulator()
    engine = ReplayEngine(sim)
    policy = get_policy("baseline")

    with pytest.raises(ValueError, match="trials must be >= 1"):
        engine.run_batch(policy, [], global_seed=42, trials=0)


def test_replay_engine_determinism(scenarios_dir):
    manager = ScenarioManager()
    scenarios = manager.load_scenarios(str(scenarios_dir))
    scenario = scenarios[0]

    policy = get_policy("baseline")

    def execute():
        sim = MockSimulator()
        engine = ReplayEngine(sim)
        return engine.run(policy, scenario, seed=777)

    run1 = execute()
    run2 = execute()

    assert run1.scenario_id == run2.scenario_id
    assert run1.total_steps == run2.total_steps
    assert run1.success == run2.success
    assert run1.rewards == run2.rewards

    for obs1, obs2 in zip(run1.observations, run2.observations):
        np.testing.assert_array_equal(obs1, obs2)

    for act1, act2 in zip(run1.actions, run2.actions):
        np.testing.assert_array_equal(act1, act2)

