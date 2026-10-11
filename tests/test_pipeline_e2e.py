import pytest
from robot_ci.pipeline.runner import PipelineRunner
from robot_ci.results.store import ResultStore


def test_pipeline_e2e_mock_backend(config_dir, scenarios_dir, tmp_path):
    runner = PipelineRunner(config_dir=str(config_dir), results_dir=str(tmp_path))

    result = runner.run(
        baseline_name="baseline",
        candidate_name="candidate",
        scenarios_dir=str(scenarios_dir),
        seed=42,
        simulator_name="mock",
        trials=1,
    )

    assert result.simulator_name == "mock"
    assert result.seed == 42
    assert len(result.scenario_ids) == 5
    assert result.overall_status in {"PASS", "REGRESSION_DETECTED"}
    assert result.rollout_summary is not None
    assert len(result.rollout_summary) == 5

    # Check JSON result persistence
    results_files = list(tmp_path.glob("run_*.json"))
    assert len(results_files) == 1

    # Reload via ResultStore and verify integrity
    store = ResultStore(results_dir=str(tmp_path))
    loaded = store.load(str(results_files[0]))
    assert loaded.run_id == result.run_id
    assert loaded.overall_status == result.overall_status
    assert loaded.metric_results == result.metric_results


def test_pipeline_e2e_pybullet_backend(config_dir, scenarios_dir, tmp_path):
    pytest.importorskip("pybullet")
    runner = PipelineRunner(config_dir=str(config_dir), results_dir=str(tmp_path))

    result = runner.run(
        baseline_name="baseline",
        candidate_name="candidate",
        scenarios_dir=str(scenarios_dir),
        seed=42,
        simulator_name="pybullet",
        trials=1,
    )

    assert result.simulator_name == "pybullet"
    assert result.overall_status in {"PASS", "REGRESSION_DETECTED"}
    assert result.rollout_summary is not None
    assert len(result.rollout_summary) == 5

    # Verify JSON result persistence
    results_files = list(tmp_path.glob("run_*.json"))
    assert len(results_files) == 1

    store = ResultStore(results_dir=str(tmp_path))
    loaded = store.load(str(results_files[0]))
    assert loaded.run_id == result.run_id
    assert loaded.simulator_name == "pybullet"


def test_pipeline_e2e_multi_trial(config_dir, scenarios_dir, tmp_path):
    runner = PipelineRunner(config_dir=str(config_dir), results_dir=str(tmp_path))

    result = runner.run(
        baseline_name="baseline",
        candidate_name="candidate",
        scenarios_dir=str(scenarios_dir),
        seed=42,
        simulator_name="mock",
        trials=2,
    )

    assert result.trial_count == 2
    assert result.rollout_summary is not None
    assert len(result.rollout_summary) == 10  # 5 scenarios * 2 trials


def test_pipeline_e2e_empty_scenarios_raises(config_dir, tmp_path):
    empty_scenarios = tmp_path / "empty_scenarios"
    empty_scenarios.mkdir()
    runner = PipelineRunner(config_dir=str(config_dir), results_dir=str(tmp_path))

    with pytest.raises(ValueError, match="No valid scenarios found"):
        runner.run(
            baseline_name="baseline",
            candidate_name="candidate",
            scenarios_dir=str(empty_scenarios),
            seed=42,
            simulator_name="mock",
        )


def test_pipeline_e2e_unknown_simulator_raises(config_dir, scenarios_dir, tmp_path):
    runner = PipelineRunner(config_dir=str(config_dir), results_dir=str(tmp_path))

    with pytest.raises(ValueError, match="Unknown simulator: unknown_sim"):
        runner.run(
            baseline_name="baseline",
            candidate_name="candidate",
            scenarios_dir=str(scenarios_dir),
            seed=42,
            simulator_name="unknown_sim",
        )

