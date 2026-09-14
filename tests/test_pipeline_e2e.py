import pytest
from robot_ci.pipeline.runner import PipelineRunner

def test_pipeline_e2e_pybullet_improvement(config_dir, scenarios_dir, tmp_path):
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
    
    success_metric = next(m for m in result.metric_results if m["metric_name"] == "success_rate")
    assert success_metric["candidate_value"] >= success_metric["baseline_value"]
    
    # Check that results json was created
    results_files = list(tmp_path.glob("run_*.json"))
    assert len(results_files) == 1
