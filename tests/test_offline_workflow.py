"""Unit tests for offline telemetry workflow, report serialization, and CLI integration.

All fixtures are explicitly marked synthetic. No physical hardware execution is performed.
"""

import json
import subprocess
import sys
from pathlib import Path
import pytest
import numpy as np

from robot_ci.telemetry.models import (
    RecordedTelemetry,
    TelemetryRecord,
    TelemetryProvenance,
    TelemetryParseError,
    TelemetryValidationError,
)
from robot_ci.telemetry.workflow import (
    OfflineAnalysisReport,
    OfflineReportStore,
    OfflineTelemetryWorkflow,
)


@pytest.fixture
def synthetic_telemetry_file(tmp_path):
    """Create a temporary synthetic JSON telemetry file with known trajectory."""
    data = {
        "session_id": "synthetic_arm_run_01",
        "provenance": {
            "data_origin": "synthetic",
            "source_id": "synthetic_ur5e_sim",
            "recording_format": "json",
            "is_synthetic": True,
            "time_basis": "UTC_EPOCH_SECONDS",
            "notes": "Synthetic test fixture for offline analysis.",
        },
        "records": [
            {
                "timestamp": 0.0,
                "timestamp_iso": "2026-10-11T12:00:00.000Z",
                "ee_position": [0.5, 0.0, 0.3],
                "joint_positions": [0.0, -0.5, 0.5, 0.0, 0.5, 0.0],
                "gripper_state": 1.0,
                "action": [0.0, 0.0, 0.0, 1.0],
            },
            {
                "timestamp": 0.1,
                "timestamp_iso": "2026-10-11T12:00:00.100Z",
                "ee_position": [0.52, 0.01, 0.31],
                "joint_positions": [0.02, -0.48, 0.51, 0.0, 0.5, 0.0],
                "gripper_state": 1.0,
                "action": [0.02, 0.01, 0.01, 1.0],
            },
            {
                "timestamp": 0.2,
                "timestamp_iso": "2026-10-11T12:00:00.200Z",
                "ee_position": [0.55, 0.02, 0.32],
                "joint_positions": [0.05, -0.45, 0.53, 0.0, 0.5, 0.0],
                "gripper_state": 0.0,
                "action": [0.03, 0.01, 0.01, 0.0],
                "events": {"contact": True},
            },
        ],
        "metadata": {"test_scenario": "reach_synthetic"},
    }
    file_path = tmp_path / "synthetic_log.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return file_path


@pytest.fixture
def synthetic_reference_file(tmp_path):
    """Create a synthetic reference file with an exact 0.05m offset on Z."""
    data = {
        "session_id": "synthetic_reference_run",
        "provenance": {
            "data_origin": "synthetic",
            "source_id": "synthetic_reference_baseline",
            "recording_format": "json",
            "is_synthetic": True,
        },
        "records": [
            {"timestamp": 0.0, "ee_position": [0.5, 0.0, 0.35]},
            {"timestamp": 0.1, "ee_position": [0.52, 0.01, 0.36]},
            {"timestamp": 0.2, "ee_position": [0.55, 0.02, 0.37]},
        ],
    }
    file_path = tmp_path / "synthetic_reference.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")
    return file_path


# ==============================================================================
# 1. Report Model & Persistence Tests
# ==============================================================================

def test_offline_report_persistence_roundtrip(tmp_path):
    """Verify OfflineAnalysisReport serialization and file roundtrip."""
    report = OfflineAnalysisReport(
        report_id="rep_test_01",
        created_at="2026-10-11T12:00:00Z",
        telemetry_source="synthetic_test",
        input_file="path/to/log.json",
        provenance={"data_origin": "synthetic", "is_synthetic": True},
        sample_count=10,
        time_range={"start_time": 0.0, "end_time": 1.0, "duration_seconds": 1.0},
        available_fields=["timestamp", "ee_position"],
        missing_fields=["joint_positions", "action"],
        validation_summary={"is_valid": True, "errors": [], "warnings": []},
        supported_metrics=["trajectory_deviation", "completion_time"],
        unsupported_metrics=["reward_difference", "collision_count"],
        unsupported_reasons={"reward_difference": "Simulation artifact only"},
        trajectory_comparison={"rms_euclidean_distance": 0.02},
        reference_source="ref.json",
        notes="Test report",
        is_synthetic=True,
        is_live_execution=False,
    )

    out_file = tmp_path / "report.json"
    saved_path = report.save(out_file)
    assert Path(saved_path).exists()

    loaded = OfflineAnalysisReport.load(out_file)
    assert loaded.report_id == "rep_test_01"
    assert loaded.is_live_execution is False
    assert loaded.is_synthetic is True
    assert loaded.sample_count == 10
    assert loaded.supported_metrics == ["trajectory_deviation", "completion_time"]
    assert loaded.unsupported_metrics == ["reward_difference", "collision_count"]
    assert loaded.trajectory_comparison["rms_euclidean_distance"] == pytest.approx(0.02)


def test_offline_report_store(tmp_path):
    """Verify OfflineReportStore save and load_all behaviors."""
    store = OfflineReportStore(tmp_path / "reports")
    assert store.load_all() == []

    report = OfflineAnalysisReport(
        report_id="rep_02",
        created_at="2026-10-11T12:05:00Z",
        telemetry_source="test_source",
        input_file="test.json",
        provenance={"data_origin": "synthetic"},
        sample_count=5,
        time_range={"duration_seconds": 0.5},
        available_fields=["timestamp"],
        missing_fields=["ee_position"],
        validation_summary={"is_valid": True},
        supported_metrics=["completion_time"],
        unsupported_metrics=["trajectory_deviation"],
        unsupported_reasons={"trajectory_deviation": "Missing EE position"},
    )
    store.save(report)

    loaded_reports = store.load_all()
    assert len(loaded_reports) == 1
    assert loaded_reports[0].report_id == "rep_02"


# ==============================================================================
# 2. Workflow Integration Tests
# ==============================================================================

def test_workflow_analyze_single_file(synthetic_telemetry_file):
    """Test offline workflow analyzing a single recorded file without reference."""
    workflow = OfflineTelemetryWorkflow()
    report = workflow.analyze(input_path=synthetic_telemetry_file, notes="Single file test")

    assert report.sample_count == 3
    assert report.telemetry_source == "synthetic_ur5e_sim"
    assert report.time_range["duration_seconds"] == pytest.approx(0.2)
    assert "timestamp" in report.available_fields
    assert "ee_position" in report.available_fields
    assert "trajectory_deviation" in report.supported_metrics
    assert "reward_difference" in report.unsupported_metrics
    assert report.trajectory_comparison is None
    assert report.is_live_execution is False


def test_workflow_analyze_with_reference_file(synthetic_telemetry_file, synthetic_reference_file):
    """Test offline workflow comparing telemetry against a reference file."""
    workflow = OfflineTelemetryWorkflow()
    report = workflow.analyze(
        input_path=synthetic_telemetry_file,
        reference_path=synthetic_reference_file,
    )

    assert report.trajectory_comparison is not None
    tc = report.trajectory_comparison
    # Reference trajectory has constant 0.05m offset along Z
    assert tc["rms_euclidean_distance"] == pytest.approx(0.05, abs=1e-4)
    assert tc["max_euclidean_distance"] == pytest.approx(0.05, abs=1e-4)
    assert tc["dtw_average_distance"] == pytest.approx(0.05, abs=1e-4)
    assert "normalized_path_progress_resampling" in tc["alignment_method"]


def test_workflow_analyze_with_sim_scenario(synthetic_telemetry_file):
    """Test offline workflow comparing telemetry against simulated scenario rollout."""
    workflow = OfflineTelemetryWorkflow()
    report = workflow.analyze(
        input_path=synthetic_telemetry_file,
        sim_scenario="reach_simple",
        sim_policy="baseline",
        simulator_name="mock",
    )

    assert report.trajectory_comparison is not None
    assert "sim:mock:reach_simple:baseline" in report.reference_source
    assert report.trajectory_comparison["rms_euclidean_distance"] >= 0.0


def test_workflow_nonexistent_file_rejected(tmp_path):
    """Ensure non-existent files raise FileNotFoundError."""
    workflow = OfflineTelemetryWorkflow()
    with pytest.raises(FileNotFoundError):
        workflow.analyze(tmp_path / "nonexistent.json")


def test_workflow_missing_ee_positions_reports_unsupported(tmp_path):
    """Ensure telemetry lacking EE positions marks trajectory comparison as unsupported."""
    no_ee_data = {
        "session_id": "no_ee_session",
        "provenance": {"data_origin": "synthetic", "is_synthetic": True},
        "records": [
            {"timestamp": 0.0, "joint_positions": [0.1, 0.2]},
            {"timestamp": 0.1, "joint_positions": [0.1, 0.2]},
        ],
    }
    log_file = tmp_path / "no_ee.json"
    log_file.write_text(json.dumps(no_ee_data), encoding="utf-8")

    workflow = OfflineTelemetryWorkflow()
    report = workflow.analyze(input_path=log_file)

    assert "ee_position" in report.missing_fields
    assert "trajectory_deviation" in report.unsupported_metrics


# ==============================================================================
# 3. CLI Integration Tests
# ==============================================================================

def test_cli_analyze_telemetry_success(synthetic_telemetry_file, tmp_path):
    """Verify running python -m robot_ci analyze-telemetry produces report file and exits 0."""
    out_file = tmp_path / "cli_out.json"
    cmd = [
        sys.executable,
        "-m",
        "robot_ci",
        "analyze-telemetry",
        "--input",
        str(synthetic_telemetry_file),
        "--output",
        str(out_file),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI stderr: {res.stderr}"
    assert "Analysis completed successfully" in res.stdout
    assert out_file.exists()

    report = OfflineAnalysisReport.load(out_file)
    assert report.sample_count == 3
    assert report.is_live_execution is False


def test_cli_analyze_telemetry_with_reference(synthetic_telemetry_file, synthetic_reference_file, tmp_path):
    """Verify CLI comparison against reference telemetry."""
    out_file = tmp_path / "cli_comp_out.json"
    cmd = [
        sys.executable,
        "-m",
        "robot_ci",
        "analyze-telemetry",
        "--input",
        str(synthetic_telemetry_file),
        "--reference",
        str(synthetic_reference_file),
        "--output",
        str(out_file),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0
    assert "Trajectory Comparison" in res.stdout

    report = OfflineAnalysisReport.load(out_file)
    assert report.trajectory_comparison is not None
    assert report.trajectory_comparison["rms_euclidean_distance"] == pytest.approx(0.05, abs=1e-4)


def test_cli_analyze_telemetry_invalid_input_fails_closed(tmp_path):
    """Verify CLI fails closed with exit code 1 when input does not exist or is corrupt."""
    bad_cmd = [
        sys.executable,
        "-m",
        "robot_ci",
        "analyze-telemetry",
        "--input",
        str(tmp_path / "missing_file.json"),
    ]
    res = subprocess.run(bad_cmd, capture_output=True, text=True)
    assert res.returncode == 1
    assert "Offline telemetry analysis failed" in res.stderr
