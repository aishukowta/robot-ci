import pytest
import numpy as np
from robot_ci.replay.rollout import Rollout
from robot_ci.evaluation.evaluator import EvaluationEngine
from robot_ci.evaluation.metrics.success_rate import SuccessRate
from robot_ci.evaluation.metrics.completion_time import CompletionTime
from robot_ci.evaluation.metrics.collision import CollisionCount
from robot_ci.evaluation.metrics.reward_difference import RewardDifference
from robot_ci.evaluation.metrics.action_difference import ActionDifference, _resample_actions
from robot_ci.evaluation.metrics.trajectory_deviation import TrajectoryDeviation, _resample_trajectory, _rms_euclidean_deviation


def make_rollout(
    scenario_id="s1",
    policy_version="v1.0",
    seed=42,
    ee_positions=None,
    actions=None,
    rewards=None,
    success=True,
    collision_count=0,
):
    if ee_positions is None:
        ee_positions = [[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]
    if actions is None:
        actions = [[0.01, 0.01, 0.01, 1.0]]
    total_steps = len(actions)
    if rewards is None:
        rewards = [1.0] * total_steps

    observations = []
    for pos in ee_positions:
        obs = np.zeros(10, dtype=np.float32)
        obs[0:3] = pos
        observations.append(obs)

    return Rollout(
        scenario_id=scenario_id,
        policy_version=policy_version,
        seed=seed,
        observations=observations,
        actions=[np.array(a, dtype=np.float32) for a in actions],
        states=[{}] * (total_steps + 1),
        rewards=[float(r) for r in rewards],
        success=success,
        collision_count=collision_count,
        total_steps=total_steps,
    )


def test_success_rate_metric():
    metric = SuccessRate()
    assert metric.name == "success_rate"

    # Single scenario: baseline pass (1.0), candidate fail (0.0)
    b_rollouts = {"s1": make_rollout("s1", success=True)}
    c_rollouts = {"s1": make_rollout("s1", success=False)}

    res = metric.compute(b_rollouts, c_rollouts)
    assert res.metric_name == "success_rate"
    assert res.baseline_value == 1.0
    assert res.candidate_value == 0.0
    assert res.delta == -1.0
    assert res.per_scenario["s1"] == {"baseline": 1.0, "candidate": 0.0, "delta": -1.0}

    # Multiple scenarios: average success rate
    b_rollouts["s2"] = make_rollout("s2", success=True)
    c_rollouts["s2"] = make_rollout("s2", success=True)
    res_multi = metric.compute(b_rollouts, c_rollouts)
    assert res_multi.baseline_value == 1.0
    assert res_multi.candidate_value == 0.5
    assert res_multi.delta == -0.5


def test_completion_time_metric():
    metric = CompletionTime()
    assert metric.name == "completion_time"

    b_rollouts = {"s1": make_rollout("s1", actions=[[0, 0, 0, 1]] * 20)}
    c_rollouts = {"s1": make_rollout("s1", actions=[[0, 0, 0, 1]] * 50)}

    res = metric.compute(b_rollouts, c_rollouts)
    assert res.baseline_value == 20.0
    assert res.candidate_value == 50.0
    assert res.delta == 30.0
    assert res.per_scenario["s1"]["delta"] == 30.0


def test_collision_count_metric():
    metric = CollisionCount()
    assert metric.name == "collision_count"

    b_rollouts = {"s1": make_rollout("s1", collision_count=0)}
    c_rollouts = {"s1": make_rollout("s1", collision_count=3)}

    res = metric.compute(b_rollouts, c_rollouts)
    assert res.baseline_value == 0.0
    assert res.candidate_value == 3.0
    assert res.delta == 3.0


def test_reward_difference_metric():
    metric = RewardDifference()
    assert metric.name == "reward_difference"

    b_rollouts = {"s1": make_rollout("s1", rewards=[1.0, 2.0, 3.0])}
    c_rollouts = {"s1": make_rollout("s1", rewards=[-1.0, 0.0, 1.0])}

    res = metric.compute(b_rollouts, c_rollouts)
    assert res.baseline_value == 6.0
    assert res.candidate_value == 0.0
    assert res.delta == -6.0


def test_action_difference_helpers_and_metric():
    # Helper test: _resample_actions
    empty_resampled = _resample_actions(np.zeros((0, 4)))
    assert empty_resampled.shape == (100, 4)
    assert np.all(empty_resampled == 0)

    single_resampled = _resample_actions(np.array([[0.1, 0.2, 0.3, 1.0]]))
    assert single_resampled.shape == (100, 4)
    assert np.allclose(single_resampled[50], [0.1, 0.2, 0.3, 1.0])

    metric = ActionDifference()
    assert metric.name == "action_difference"

    # Known offset: baseline action = [0.1, 0.0, 0.0, 1.0], candidate action = [0.4, 0.0, 0.0, 1.0]
    b_rollouts = {"s1": make_rollout("s1", actions=[[0.1, 0.0, 0.0, 1.0]] * 10)}
    c_rollouts = {"s1": make_rollout("s1", actions=[[0.4, 0.0, 0.0, 1.0]] * 10)}

    res = metric.compute(b_rollouts, c_rollouts)
    assert res.baseline_value == 0.0
    assert pytest.approx(res.candidate_value, abs=1e-5) == 0.3
    assert pytest.approx(res.delta, abs=1e-5) == 0.3


def test_trajectory_deviation_helpers_and_metric():
    # 1. Resampling edge cases
    empty_traj = _resample_trajectory(np.zeros((0, 3)))
    assert empty_traj.shape == (100, 3)

    single_point = _resample_trajectory(np.array([[1.0, 2.0, 3.0]]))
    assert single_point.shape == (100, 3)
    assert np.allclose(single_point[0], [1.0, 2.0, 3.0])

    # 2. Hand-calculable 3D RMS deviation
    # Traj A: constant at (0, 0, 0)
    # Traj B: constant at (0.3, 0.4, 0.0) -> distance per point = sqrt(0.09 + 0.16) = 0.5
    traj_a = np.zeros((10, 3))
    traj_b = np.tile([0.3, 0.4, 0.0], (10, 1))

    res_a = _resample_trajectory(traj_a)
    res_b = _resample_trajectory(traj_b)
    rms_dev = _rms_euclidean_deviation(res_a, res_b)
    assert pytest.approx(rms_dev, abs=1e-5) == 0.5

    # 3. Metric computation with different trajectory lengths along same geometric line
    line10 = [[t, t, t] for t in np.linspace(0, 1, 10)]
    line50 = [[t, t, t] for t in np.linspace(0, 1, 50)]

    b_rollouts = {"s1": make_rollout("s1", ee_positions=line10)}
    c_rollouts = {"s1": make_rollout("s1", ee_positions=line50)}

    metric = TrajectoryDeviation()
    assert metric.name == "trajectory_deviation"

    res = metric.compute(b_rollouts, c_rollouts)
    assert res.baseline_value == 0.0
    assert pytest.approx(res.candidate_value, abs=1e-4) == 0.0
    assert pytest.approx(res.delta, abs=1e-4) == 0.0


def test_evaluation_engine_runs_all_default_metrics():
    engine = EvaluationEngine()
    b_rollouts = {"s1": make_rollout("s1")}
    c_rollouts = {"s1": make_rollout("s1")}

    results = engine.evaluate(b_rollouts, c_rollouts)
    assert len(results) == 6
    names = [r.metric_name for r in results]
    assert "success_rate" in names
    assert "completion_time" in names
    assert "trajectory_deviation" in names
    assert "collision_count" in names
    assert "action_difference" in names
    assert "reward_difference" in names


def test_evaluation_mismatched_and_empty_scenarios():
    engine = EvaluationEngine()

    # 1. Empty rollouts
    res_empty = engine.evaluate({}, {})
    for r in res_empty:
        assert r.baseline_value == 0.0
        assert r.candidate_value == 0.0
        assert r.delta == 0.0
        assert r.per_scenario == {}

    # 2. Mismatched scenario IDs
    b_rollouts = {"s1": make_rollout("s1"), "s2": make_rollout("s2")}
    c_rollouts = {"s1": make_rollout("s1"), "s3": make_rollout("s3")}

    results = engine.evaluate(b_rollouts, c_rollouts)
    for r in results:
        assert set(r.per_scenario.keys()) == {"s1"}

