import pytest
import json
from robot_ci.results.models import TestRunResult
from robot_ci.results.store import ResultStore


def make_test_result(run_id="run1", timestamp="2026-10-09T10:00:00Z", overall_status="PASS"):
    return TestRunResult(
        run_id=run_id,
        timestamp=timestamp,
        baseline_policy_version="v1.0-baseline",
        candidate_policy_version="v2.0-candidate",
        seed=42,
        scenario_ids=["reach_simple", "pick_and_place_easy"],
        metric_results=[
            {
                "metric_name": "success_rate",
                "baseline_value": 1.0,
                "candidate_value": 1.0,
                "delta": 0.0,
                "per_scenario": {"reach_simple": {"baseline": 1.0, "candidate": 1.0, "delta": 0.0}},
            }
        ],
        regression_findings=[
            {
                "metric_name": "success_rate",
                "baseline_value": 1.0,
                "candidate_value": 1.0,
                "delta": 0.0,
                "threshold": 0.1,
                "regression_detected": False,
                "severity": "none",
                "reason": "PASS",
            }
        ],
        overall_status=overall_status,
        simulator_name="pybullet",
        trial_count=1,
        rollout_summary={
            "reach_simple": {
                "baseline": {"success": True, "steps": 25, "collisions": 0, "reward": 8.5},
                "candidate": {"success": True, "steps": 28, "collisions": 0, "reward": 8.2},
            }
        },
    )


def test_test_run_result_serialization():
    # 1. Full model roundtrip
    result = make_test_result()
    result.sim_to_real_gap = {"gap_score": 0.05}
    result.confidence_score = 0.95
    result.deployment_recommendation = "DEPLOY"
    result.hardware_validation = {"verified": False}

    d = result.to_dict()
    assert d["run_id"] == "run1"
    assert d["sim_to_real_gap"] == {"gap_score": 0.05}
    assert d["confidence_score"] == 0.95
    assert d["deployment_recommendation"] == "DEPLOY"

    reconstructed = TestRunResult.from_dict(d)
    assert reconstructed == result

    # 2. Backward compatibility (dict missing optional extension fields)
    minimal_dict = {
        "run_id": "min1",
        "timestamp": "2026-10-09T08:00:00Z",
        "baseline_policy_version": "v1.0",
        "candidate_policy_version": "v2.0",
        "scenario_ids": ["reach_simple"],
        "metric_results": [],
        "regression_findings": [],
        "overall_status": "PASS",
    }
    min_obj = TestRunResult.from_dict(minimal_dict)
    assert min_obj.seed == 42
    assert min_obj.simulator_name == "mock"
    assert min_obj.trial_count == 1
    assert min_obj.rollout_summary is None
    assert min_obj.sim_to_real_gap is None


def test_result_store_save_and_load(tmp_path):
    store = ResultStore(results_dir=str(tmp_path))
    result = make_test_result(run_id="abc12345")

    saved_path = store.save(result)
    assert f"run_abc12345.json" in saved_path
    assert (tmp_path / "run_abc12345.json").exists()

    loaded = store.load(saved_path)
    assert loaded.run_id == "abc12345"
    assert loaded.overall_status == "PASS"
    assert loaded.metric_results == result.metric_results
    assert loaded.rollout_summary == result.rollout_summary


def test_result_store_load_all_and_latest_sorting(tmp_path):
    store = ResultStore(results_dir=str(tmp_path))

    r1 = make_test_result(run_id="001", timestamp="2026-10-09T10:00:00Z")
    r2 = make_test_result(run_id="002", timestamp="2026-10-09T12:00:00Z")
    r3 = make_test_result(run_id="003", timestamp="2026-10-09T11:00:00Z")

    store.save(r1)
    store.save(r2)
    store.save(r3)

    all_results = store.load_all()
    assert len(all_results) == 3
    # Verified ordering: descending ISO timestamp
    assert [r.run_id for r in all_results] == ["002", "003", "001"]

    latest = store.load_latest()
    assert latest is not None
    assert latest.run_id == "002"
    assert latest.timestamp == "2026-10-09T12:00:00Z"


def test_result_store_empty_directory_handling(tmp_path):
    new_dir = tmp_path / "non_existent_dir"
    store = ResultStore(results_dir=str(new_dir))

    assert new_dir.exists()
    assert store.load_all() == []
    assert store.load_latest() is None


def test_result_store_malformed_and_incomplete_file_handling(tmp_path):
    store = ResultStore(results_dir=str(tmp_path))

    # 1. Direct load of malformed JSON raises JSONDecodeError
    bad_json_file = tmp_path / "run_bad.json"
    bad_json_file.write_text("{corrupt json content", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        store.load(str(bad_json_file))

    # 2. Direct load of incomplete dict raises KeyError
    incomplete_file = tmp_path / "run_incomplete.json"
    incomplete_file.write_text(json.dumps({"run_id": "inc1"}), encoding="utf-8")
    with pytest.raises(KeyError):
        store.load(str(incomplete_file))

    # 3. Direct load of non-existent file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        store.load(str(tmp_path / "run_ghost.json"))

    # 4. load_all behavior: skips corrupt / incomplete run_*.json files silently
    valid_res = make_test_result(run_id="valid001")
    store.save(valid_res)

    all_loaded = store.load_all()
    assert len(all_loaded) == 1
    assert all_loaded[0].run_id == "valid001"


def test_dashboard_helpers(tmp_path):
    from dashboard.app import load_results, load_comparison_reports, _is_regression

    # Empty directory
    assert load_results(str(tmp_path / "empty")) == []
    assert load_comparison_reports(str(tmp_path / "empty_comp")) == []

    # Valid results
    res_dir = tmp_path / "results"
    res_dir.mkdir()
    (res_dir / "run_01.json").write_text(json.dumps({"run_id": "01", "overall_status": "PASS"}), encoding="utf-8")
    (res_dir / "run_corrupt.json").write_text("{bad json", encoding="utf-8")

    loaded = load_results(str(res_dir))
    assert len(loaded) == 1
    assert loaded[0]["run_id"] == "01"

    # Regression status check
    assert _is_regression("REGRESSION_DETECTED") is True
    assert _is_regression("FAIL") is True
    assert _is_regression("PASS") is False

