"""Focused tests for evidence-based diagnosis panel (Phase 4).

Covers:
- Policy regression evidence without physical video.
- Physical video evidence without policy regression data.
- Synthetic fixture incorrectly being presented as physical evidence.
- Manual teleoperation incorrectly being presented as same-policy validation.
- Both evidence sources indicating potential issues (BOTH_ISSUES_LIKELY).
- Missing and malformed reports.
- Low visual tracking coverage fail-safe.
- Raw versus aligned metric interpretation (alignment masking warning).
- Inconclusive cases with insufficient evidence.
"""

import pytest

from robot_ci.diagnosis.models import (
    DiagnosisCategory,
    DiagnosisReport,
    EvidenceItem,
)
from robot_ci.diagnosis.engine import evaluate_diagnosis


@pytest.fixture
def sample_ci_regression_result() -> dict:
    """Fixture providing a CI test run with a detected regression."""
    return {
        "run_id": "run_reg_01",
        "overall_status": "REGRESSION_DETECTED",
        "baseline_policy_version": "v1.0-baseline",
        "candidate_policy_version": "v2.0-candidate",
        "simulator_name": "pybullet",
        "seed": 42,
        "regression_findings": [
            {
                "metric_name": "completion_time",
                "baseline_value": 15.0,
                "candidate_value": 30.0,
                "delta": 15.0,
                "threshold": 0.20,
                "regression_detected": True,
                "severity": "major",
                "reason": "completion_time exceeded threshold by +15.0s",
            },
            {
                "metric_name": "success_rate",
                "baseline_value": 1.0,
                "candidate_value": 1.0,
                "delta": 0.0,
                "threshold": 0.10,
                "regression_detected": False,
                "severity": "none",
                "reason": "success_rate delta 0.0 <= threshold 0.1",
            },
        ],
    }


@pytest.fixture
def sample_ci_passing_result() -> dict:
    """Fixture providing a clean CI test run without regressions."""
    return {
        "run_id": "run_pass_01",
        "overall_status": "PASS",
        "baseline_policy_version": "v1.0-baseline",
        "candidate_policy_version": "v2.0-candidate",
        "simulator_name": "pybullet",
        "seed": 42,
        "regression_findings": [
            {
                "metric_name": "completion_time",
                "baseline_value": 15.0,
                "candidate_value": 15.5,
                "delta": 0.5,
                "threshold": 0.20,
                "regression_detected": False,
                "severity": "none",
                "reason": "completion_time within threshold",
            },
        ],
    }


@pytest.fixture
def sample_physical_video_report() -> dict:
    """Fixture providing a genuine physical robot video comparison report."""
    return {
        "report_id": "vid_rep_phys_01",
        "comparison_mode": "CALIBRATED_PROJECTION",
        "comparison_type": "matched_policy_verification",
        "scenario_id": "reach_simple",
        "sim_policy_id": "candidate",
        "physical_controller_id": "candidate",
        "units": "pixels (px)",
        "metrics": {
            "tracking_coverage": 0.95,
            "video_duration_seconds": 2.0,
            "sim_duration_seconds": 2.0,
            "rms_trajectory_deviation": 55.0,  # High deviation (> 30 px)
            "raw_rms_trajectory_deviation": 55.0,
            "mean_trajectory_deviation": 50.0,
        },
        "provenance": {
            "video_file": "records/physical_ur5e_run.mp4",
            "video_source_is_synthetic": False,
            "data_source": "robot_phone_recording",
            "calibration_source": "measured",
            "calibration_matches_video_size": True,
        },
    }


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------

def test_policy_regression_without_physical_video(sample_ci_regression_result):
    """Verify diagnosis correctly reports POLICY_REGRESSION_LIKELY when video is missing."""
    report = evaluate_diagnosis(ci_result=sample_ci_regression_result, video_report=None)

    assert report.category == DiagnosisCategory.POLICY_REGRESSION_LIKELY
    assert any("completion_time" in f for f in report.supporting_findings)
    assert any("Video-to-simulation trajectory comparison report is missing" in m for m in report.missing_evidence)
    assert any("Physical sim-to-real transfer cannot be evaluated" in u for u in report.unsupported_conclusions)


def test_physical_video_evidence_without_policy_regression(sample_physical_video_report):
    """Verify diagnosis reports SIM_TO_REAL_MISMATCH_LIKELY when simulation CI data is missing."""
    report = evaluate_diagnosis(ci_result=None, video_report=sample_physical_video_report)

    assert report.category == DiagnosisCategory.SIM_TO_REAL_MISMATCH_LIKELY
    assert any("Calibrated reprojection error" in f for f in report.supporting_findings)
    assert any("Baseline-vs-candidate simulation policy regression results are missing" in m for m in report.missing_evidence)
    assert any("Algorithmic policy degradation" in u for u in report.unsupported_conclusions)


def test_synthetic_fixture_never_qualifies_as_physical_evidence(sample_ci_passing_result, sample_physical_video_report):
    """Verify that a synthetic video fixture is never treated as physical robot evidence."""
    # Mark video as synthetic fixture
    synth_report = dict(sample_physical_video_report)
    synth_report["provenance"] = {
        "video_file": "tests/fixtures/synthetic_phone_recording.mp4",
        "video_source_is_synthetic": True,
    }

    report = evaluate_diagnosis(ci_result=sample_ci_passing_result, video_report=synth_report)

    # Must NOT claim sim-to-real mismatch because video is synthetic
    assert report.category != DiagnosisCategory.SIM_TO_REAL_MISMATCH_LIKELY
    assert report.category == DiagnosisCategory.NO_REGRESSION_DETECTED
    assert any("Synthetic video fixture detected" in w for w in report.data_quality_warnings)
    assert any("synthetic test fixture" in u for u in report.unsupported_conclusions)


def test_manual_teleoperation_not_same_policy_validation(sample_ci_passing_result, sample_physical_video_report):
    """Verify that teleoperated recording is not treated as validation of candidate policy."""
    teleop_report = dict(sample_physical_video_report)
    teleop_report["physical_controller_id"] = "human_teleop"
    teleop_report["comparison_type"] = "cross_controller_behavioural_comparison"

    report = evaluate_diagnosis(ci_result=sample_ci_passing_result, video_report=teleop_report)

    assert report.category != DiagnosisCategory.SIM_TO_REAL_MISMATCH_LIKELY
    assert any("Cross-controller / teleoperated recording" in w for w in report.data_quality_warnings)
    assert any("manual teleoperation" in u for u in report.unsupported_conclusions)


def test_both_issues_likely(sample_ci_regression_result, sample_physical_video_report):
    """Verify diagnosis reports BOTH_ISSUES_LIKELY when both sources indicate problems."""
    report = evaluate_diagnosis(ci_result=sample_ci_regression_result, video_report=sample_physical_video_report)

    assert report.category == DiagnosisCategory.BOTH_ISSUES_LIKELY
    assert "Both a simulation policy regression and a physical sim-to-real mismatch" in report.summary_explanation
    assert "Address simulation policy regressions first" in report.recommended_next_step


def test_no_regression_detected_both_pass(sample_ci_passing_result, sample_physical_video_report):
    """Verify NO_REGRESSION_DETECTED when CI passes and physical trajectory matches."""
    matching_report = dict(sample_physical_video_report)
    matching_report["metrics"] = dict(sample_physical_video_report["metrics"])
    matching_report["metrics"]["rms_trajectory_deviation"] = 5.0  # Within 30 px threshold

    report = evaluate_diagnosis(ci_result=sample_ci_passing_result, video_report=matching_report)

    assert report.category == DiagnosisCategory.NO_REGRESSION_DETECTED
    assert "No policy regression or sim-to-real mismatch is indicated" in report.summary_explanation


def test_missing_and_malformed_reports():
    """Verify graceful handling and fail-safe for missing or empty inputs."""
    # Both None
    rep_none = evaluate_diagnosis(ci_result=None, video_report=None)
    assert rep_none.category == DiagnosisCategory.INCONCLUSIVE_INSUFFICIENT_EVIDENCE
    assert len(rep_none.missing_evidence) == 2

    # Malformed empty dicts
    rep_empty = evaluate_diagnosis(ci_result={"run_id": "bad"}, video_report={"report_id": "bad"})
    assert rep_empty.category == DiagnosisCategory.INCONCLUSIVE_INSUFFICIENT_EVIDENCE
    assert len(rep_empty.data_quality_warnings) >= 1


def test_low_tracking_coverage_fails_safe(sample_ci_passing_result, sample_physical_video_report):
    """Verify low tracking coverage suppresses sim-to-real claims and triggers warning."""
    low_cov_report = dict(sample_physical_video_report)
    low_cov_report["metrics"] = dict(sample_physical_video_report["metrics"])
    low_cov_report["metrics"]["tracking_coverage"] = 0.20  # 20% coverage (< 50%)

    report = evaluate_diagnosis(ci_result=sample_ci_passing_result, video_report=low_cov_report)

    assert report.category != DiagnosisCategory.SIM_TO_REAL_MISMATCH_LIKELY
    assert any("Low visual tracking coverage" in w for w in report.data_quality_warnings)


def test_raw_vs_aligned_metric_interpretation(sample_ci_passing_result):
    """Verify that Procrustes alignment reducing raw error triggers alignment masking notice."""
    report_dict = {
        "report_id": "shape_masking_rep",
        "comparison_mode": "SHAPE_ONLY_2D",
        "comparison_type": "matched_policy_verification",
        "scenario_id": "reach_simple",
        "sim_policy_id": "candidate",
        "physical_controller_id": "candidate",
        "units": "normalized_units (dimensionless)",
        "metrics": {
            "tracking_coverage": 0.90,
            "rms_trajectory_deviation": 0.15,       # Within threshold after Procrustes
            "raw_rms_trajectory_deviation": 0.85,   # High raw mismatch (> 5x aligned)
        },
        "provenance": {
            "video_file": "records/phys_robot.mp4",
            "video_source_is_synthetic": False,
            "data_source": "robot_phone_recording",
        },
    }

    report = evaluate_diagnosis(ci_result=sample_ci_passing_result, video_report=report_dict)

    # Alignment masking notice should alert user that raw deviation was much larger
    assert any("Alignment masking notice" in w for w in report.data_quality_warnings)


def test_genuinely_inconclusive_case():
    """Verify that synthetic video without CI data is correctly classified as inconclusive."""
    synth_report = {
        "report_id": "synth_only",
        "comparison_mode": "SHAPE_ONLY_2D",
        "comparison_type": "matched_policy_verification",
        "metrics": {"tracking_coverage": 0.90, "rms_trajectory_deviation": 0.50},
        "provenance": {"video_file": "synthetic.mp4", "video_source_is_synthetic": True},
    }

    report = evaluate_diagnosis(ci_result=None, video_report=synth_report)

    assert report.category == DiagnosisCategory.INCONCLUSIVE_INSUFFICIENT_EVIDENCE
    assert any("synthetic test fixtures cannot qualify" in w for w in report.data_quality_warnings)
    assert any("Baseline-vs-candidate simulation policy regression results are missing" in m for m in report.missing_evidence)


# ---------------------------------------------------------------------------
# Provenance hardening (audit findings B1.2, B1.5, B1.6)
# ---------------------------------------------------------------------------

def test_renamed_synthetic_report_is_not_physical(sample_ci_regression_result, sample_physical_video_report):
    """A report without a declared robot data source is never physical, whatever the file is called."""
    rep = dict(sample_physical_video_report)
    rep["provenance"] = {"video_file": "results/videos/lab_run_01.mp4", "video_source_is_synthetic": False}
    report = evaluate_diagnosis(ci_result=sample_ci_regression_result, video_report=rep)
    assert report.category == DiagnosisCategory.POLICY_REGRESSION_LIKELY
    assert all(not e.is_physical_hardware for e in report.evidence_sources)


def test_handheld_marker_is_not_robot_evidence(sample_ci_passing_result, sample_physical_video_report):
    rep = dict(sample_physical_video_report)
    rep["provenance"] = dict(rep["provenance"], data_source="handheld_marker_recording")
    report = evaluate_diagnosis(ci_result=sample_ci_passing_result, video_report=rep)
    assert report.category == DiagnosisCategory.NO_REGRESSION_DETECTED
    assert any("Hand-held marker" in w for w in report.data_quality_warnings)


def test_synthetic_calibration_not_accepted(sample_ci_passing_result, sample_physical_video_report):
    rep = dict(sample_physical_video_report)
    rep["provenance"] = dict(rep["provenance"], calibration_source="synthetic_fixture")
    report = evaluate_diagnosis(ci_result=sample_ci_passing_result, video_report=rep)
    assert report.category != DiagnosisCategory.SIM_TO_REAL_MISMATCH_LIKELY
    assert any("calibration source" in w for w in report.data_quality_warnings)


def test_scenario_mismatch_not_combined(sample_physical_video_report):
    ci = {
        "run_id": "r1", "overall_status": "PASS", "simulator_name": "mock",
        "baseline_policy_version": "v1.0-baseline", "candidate_policy_version": "v2.0-candidate",
        "scenario_ids": ["pick_and_place_easy"],
        "regression_findings": [{"metric_name": "success_rate", "regression_detected": False}],
    }
    report = evaluate_diagnosis(ci_result=ci, video_report=sample_physical_video_report)
    assert report.category == DiagnosisCategory.NO_REGRESSION_DETECTED
    assert any("not part of CI run scenarios" in w for w in report.data_quality_warnings)
    assert report.evidence_sources[0].provenance == "MOCK_SIMULATION"
