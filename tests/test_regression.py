import pytest
from robot_ci.evaluation.metrics.base import MetricResult
from robot_ci.regression.detector import RegressionDetector, RegressionFinding

def test_regression_detector_default_config(config_dir):
    config_path = str(config_dir / "regression_thresholds.yaml")
    detector = RegressionDetector(config_path=config_path)
    assert detector.thresholds is not None
    assert "success_rate" in detector.thresholds
    assert "completion_time" in detector.thresholds


def test_regression_success_rate_severities(config_dir):
    detector = RegressionDetector(config_path=str(config_dir / "regression_thresholds.yaml"))
    
    # 1. Candidate improvement (0.8 vs 1.0): No regression
    res_pass = MetricResult("success_rate", baseline_value=0.8, candidate_value=1.0, delta=0.2, per_scenario={})
    findings = detector.analyze([res_pass])
    assert len(findings) == 1
    assert not findings[0].regression_detected
    assert findings[0].severity == "none"

    # 2. Minor regression (drop of 0.15 >= 0.10)
    res_minor = MetricResult("success_rate", baseline_value=1.0, candidate_value=0.85, delta=-0.15, per_scenario={})
    finding_minor = detector.analyze([res_minor])[0]
    assert finding_minor.regression_detected
    assert finding_minor.severity == "minor"
    assert finding_minor.threshold == 0.10

    # 3. Major regression (drop of 0.30 >= 0.25)
    res_major = MetricResult("success_rate", baseline_value=1.0, candidate_value=0.70, delta=-0.30, per_scenario={})
    finding_major = detector.analyze([res_major])[0]
    assert finding_major.regression_detected
    assert finding_major.severity == "major"

    # 4. Critical regression (drop of 0.60 >= 0.50)
    res_crit = MetricResult("success_rate", baseline_value=1.0, candidate_value=0.40, delta=-0.60, per_scenario={})
    finding_crit = detector.analyze([res_crit])[0]
    assert finding_crit.regression_detected
    assert finding_crit.severity == "critical"


def test_regression_completion_time_relative(config_dir):
    detector = RegressionDetector(config_path=str(config_dir / "regression_thresholds.yaml"))
    # completion_time: direction=increase, comparison=relative, threshold=0.20
    # severity: minor=0.20, major=0.50, critical=1.00

    # Baseline 100, Candidate 110 (+10%) -> magnitude 0.10 < 0.20 -> PASS
    res1 = MetricResult("completion_time", baseline_value=100.0, candidate_value=110.0, delta=10.0, per_scenario={})
    f1 = detector.analyze([res1])[0]
    assert not f1.regression_detected
    assert f1.severity == "none"

    # Baseline 100, Candidate 125 (+25%) -> magnitude 0.25 >= 0.20 -> minor
    res2 = MetricResult("completion_time", baseline_value=100.0, candidate_value=125.0, delta=25.0, per_scenario={})
    f2 = detector.analyze([res2])[0]
    assert f2.regression_detected
    assert f2.severity == "minor"

    # Baseline 100, Candidate 160 (+60%) -> magnitude 0.60 >= 0.50 -> major
    res3 = MetricResult("completion_time", baseline_value=100.0, candidate_value=160.0, delta=60.0, per_scenario={})
    f3 = detector.analyze([res3])[0]
    assert f3.regression_detected
    assert f3.severity == "major"

    # Baseline 100, Candidate 210 (+110%) -> magnitude 1.10 >= 1.00 -> critical
    res4 = MetricResult("completion_time", baseline_value=100.0, candidate_value=210.0, delta=110.0, per_scenario={})
    f4 = detector.analyze([res4])[0]
    assert f4.regression_detected
    assert f4.severity == "critical"


def test_regression_monitor_direction(config_dir):
    detector = RegressionDetector(config_path=str(config_dir / "regression_thresholds.yaml"))
    # trajectory_deviation & action_difference are "monitor" metrics
    res_traj = MetricResult("trajectory_deviation", baseline_value=0.0, candidate_value=5.0, delta=5.0, per_scenario={})
    f_traj = detector.analyze([res_traj])[0]
    assert not f_traj.regression_detected
    assert f_traj.severity == "none"
    assert "monitored only" in f_traj.reason

    res_act = MetricResult("action_difference", baseline_value=0.0, candidate_value=1.5, delta=1.5, per_scenario={})
    f_act = detector.analyze([res_act])[0]
    assert not f_act.regression_detected
    assert f_act.severity == "none"


def test_regression_collision_exact_boundaries(config_dir):
    detector = RegressionDetector(config_path=str(config_dir / "regression_thresholds.yaml"))
    # collision_count: direction=increase, comparison=absolute
    # minor: 1.0, major: 3.0, critical: 5.0

    # Exactly at threshold 1.0
    res_exact_minor = MetricResult("collision_count", baseline_value=0.0, candidate_value=1.0, delta=1.0, per_scenario={})
    f_minor = detector.analyze([res_exact_minor])[0]
    assert f_minor.regression_detected
    assert f_minor.severity == "minor"

    # Exactly at 3.0
    res_exact_major = MetricResult("collision_count", baseline_value=0.0, candidate_value=3.0, delta=3.0, per_scenario={})
    f_major = detector.analyze([res_exact_major])[0]
    assert f_major.regression_detected
    assert f_major.severity == "major"


def test_regression_edge_cases(config_dir, tmp_path):
    detector = RegressionDetector(config_path=str(config_dir / "regression_thresholds.yaml"))

    # 1. Zero baseline relative comparison (e.g. reward_difference)
    res_zero = MetricResult("reward_difference", baseline_value=0.0, candidate_value=-5.0, delta=-5.0, per_scenario={})
    f_zero = detector.analyze([res_zero])[0]
    assert f_zero.regression_detected
    assert f_zero.severity == "critical"

    # 2. Missing metric name in thresholds
    res_unknown = MetricResult("unknown_custom_metric", baseline_value=0.0, candidate_value=1.0, delta=1.0, per_scenario={})
    assert detector.analyze([res_unknown]) == []

    # 3. Empty input list
    assert detector.analyze([]) == []

    # 4. Malformed / empty YAML threshold file
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("", encoding="utf-8")
    det_empty = RegressionDetector(config_path=str(empty_yaml))
    assert det_empty.analyze([MetricResult("success_rate", 1.0, 0.5, -0.5, {})]) == []


def test_regression_finding_serialization():
    finding = RegressionFinding(
        metric_name="success_rate",
        baseline_value=1.0,
        candidate_value=0.5,
        delta=0.5,
        threshold=0.1,
        regression_detected=True,
        severity="critical",
        reason="critical regression detected",
    )
    d = finding.to_dict()
    assert d["metric_name"] == "success_rate"
    reconstructed = RegressionFinding.from_dict(d)
    assert reconstructed == finding



def test_threshold_config_validation_rejects_bad_direction(tmp_path):
    import pytest
    from robot_ci.regression.detector import RegressionDetector
    bad = tmp_path / "bad.yaml"
    bad.write_text("success_rate:\n  threshold: 0.1\n  direction: sideways\n  comparison: absolute\n"
                   "  severity_levels: {minor: 0.1, major: 0.2, critical: 0.5}\n")
    with pytest.raises(ValueError, match="invalid direction"):
        RegressionDetector(config_path=str(bad))


def test_threshold_config_validation_rejects_missing_keys(tmp_path):
    import pytest
    from robot_ci.regression.detector import RegressionDetector
    bad = tmp_path / "bad.yaml"
    bad.write_text("success_rate:\n  threshold: 0.1\n")
    with pytest.raises(ValueError, match="missing keys"):
        RegressionDetector(config_path=str(bad))
