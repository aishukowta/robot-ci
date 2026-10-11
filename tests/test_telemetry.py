"""Unit tests for offline robot telemetry logging, ingestion, and trajectory comparison.

All test fixtures are explicitly synthetic and do not represent physical hardware execution.
"""

import json
import math
import tempfile
from pathlib import Path
import pytest
import numpy as np

from robot_ci.telemetry.models import (
    RecordedTelemetry,
    TelemetryRecord,
    TelemetryProvenance,
    TelemetryValidationError,
    TelemetryParseError,
    UnsupportedFormatError,
)
from robot_ci.telemetry.reader import (
    JsonTelemetryAdapter,
    JsonlTelemetryAdapter,
    McapTelemetryAdapter,
    Rosbag2TelemetryAdapter,
    load_telemetry,
)
from robot_ci.telemetry.replay import OfflineTelemetrySession
from robot_ci.telemetry.comparator import (
    TrajectoryComparator,
    compute_dtw,
    resample_trajectory,
    TRAJECTORY_SAFETY_DISCLAIMER,
)
from robot_ci.replay.rollout import Rollout


@pytest.fixture
def synthetic_records():
    """Deterministic synthetic fixture with known timestamps and positions."""
    return [
        TelemetryRecord(
            timestamp=0.0,
            timestamp_iso="2026-10-11T12:00:00.000Z",
            ee_position=[0.5, 0.0, 0.3],
            joint_positions=[0.0, -0.5, 0.5, 0.0, 0.5, 0.0],
            gripper_state=1.0,
            action=[0.0, 0.0, 0.0, 1.0],
        ),
        TelemetryRecord(
            timestamp=0.1,
            timestamp_iso="2026-10-11T12:00:00.100Z",
            ee_position=[0.52, 0.01, 0.31],
            joint_positions=[0.02, -0.48, 0.51, 0.0, 0.5, 0.0],
            gripper_state=1.0,
            action=[0.02, 0.01, 0.01, 1.0],
        ),
        TelemetryRecord(
            timestamp=0.2,
            timestamp_iso="2026-10-11T12:00:00.200Z",
            ee_position=[0.55, 0.02, 0.32],
            joint_positions=[0.05, -0.45, 0.53, 0.0, 0.5, 0.0],
            gripper_state=0.0,
            action=[0.03, 0.01, 0.01, 0.0],
            events={"contact": True},
        ),
    ]


@pytest.fixture
def synthetic_session(synthetic_records):
    prov = TelemetryProvenance(
        data_origin="synthetic",
        source_id="synthetic_fixture_arm_01",
        recording_format="json",
        is_synthetic=True,
    )
    return RecordedTelemetry(
        session_id="synthetic_test_session_01",
        provenance=prov,
        records=synthetic_records,
        metadata={"scenario": "reach_target", "test_fixture": True},
    )


# ==============================================================================
# 1. Schema, Provenance, and Offline Isolation Tests
# ==============================================================================

def test_provenance_rejects_live_hardware_origin():
    """Ensure offline telemetry cannot falsely claim to be live hardware execution."""
    with pytest.raises(TelemetryValidationError, match="Offline telemetry must never claim to be live"):
        TelemetryProvenance(data_origin="live_hardware")

    with pytest.raises(TelemetryValidationError, match="Offline telemetry must never claim to be live"):
        TelemetryProvenance(data_origin="simulator_live")


def test_offline_session_rejects_simulator_methods(synthetic_session):
    """Ensure OfflineTelemetrySession cannot be used as a fake SimulatorBackend."""
    session = OfflineTelemetrySession(synthetic_session)

    with pytest.raises(TypeError, match="not an execution backend"):
        session.step([0, 0, 0, 0])

    with pytest.raises(TypeError, match="cannot reset a physical robot"):
        session.reset(None)

    assert session.get_summary()["session_type"] == "OFFLINE_RECORDED_TELEMETRY"
    assert session.provenance["is_synthetic"] is True


def test_offline_session_supported_metrics(synthetic_session):
    """Ensure session honestly exposes only supported metrics and rejects unsupported ones."""
    session = OfflineTelemetrySession(synthetic_session)
    metrics_info = session.get_supported_metrics()

    assert "trajectory_deviation" in metrics_info["supported"]
    assert "completion_time" in metrics_info["supported"]
    assert "action_difference" in metrics_info["supported"]
    assert "collision_count" in metrics_info["supported"]  # Since 'contact' was present in synthetic record 3

    # Rewards must never be claimed
    assert "reward_difference" in metrics_info["unsupported"]
    assert "simulation artifacts" in metrics_info["reasons"]["reward_difference"]


def test_to_rollout_adapter_does_not_invent_rewards(synthetic_session):
    """Ensure conversion to Rollout leaves rewards empty and marks negative seed."""
    session = OfflineTelemetrySession(synthetic_session)
    rollout = session.to_rollout_adapter(scenario_id="synthetic_eval")

    assert rollout.scenario_id == "synthetic_eval"
    assert rollout.seed == -1
    assert rollout.rewards == []  # Never fabricated
    assert len(rollout.observations) == 3
    assert len(rollout.actions) == 3
    assert rollout.collision_count == 1
    assert "offline_record" in rollout.policy_version


# ==============================================================================
# 2. Ingestion Tests (JSON & JSONL)
# ==============================================================================

def test_json_roundtrip_ingestion(synthetic_session, tmp_path):
    """Test serializing and reading valid synthetic JSON telemetry."""
    file_path = tmp_path / "synthetic_log.json"
    file_path.write_text(json.dumps(synthetic_session.to_dict()), encoding="utf-8")

    loaded = load_telemetry(file_path)
    assert loaded.session_id == synthetic_session.session_id
    assert loaded.sample_count == 3
    assert loaded.provenance.is_synthetic is True
    assert loaded.duration_seconds == pytest.approx(0.2)

    traj = loaded.get_ee_trajectory()
    assert traj.shape == (3, 3)
    assert np.allclose(traj[0], [0.5, 0.0, 0.3])


def test_jsonl_ingestion_with_header(synthetic_records, tmp_path):
    """Test reading line-delimited JSONL telemetry containing header line."""
    file_path = tmp_path / "synthetic_log.jsonl"
    lines = [
        json.dumps({
            "_header": True,
            "session_id": "synthetic_jsonl_run",
            "provenance": {"data_origin": "synthetic", "is_synthetic": True, "source_id": "synthetic_file"},
            "metadata": {"test": True},
        })
    ]
    for r in synthetic_records:
        lines.append(json.dumps(r.to_dict()))
    file_path.write_text("\n".join(lines), encoding="utf-8")

    loaded = load_telemetry(file_path)
    assert loaded.sample_count == 3
    assert loaded.provenance.is_synthetic is True
    assert loaded.provenance.source_id == "synthetic_file"
    assert loaded.get_timestamps().tolist() == [0.0, 0.1, 0.2]


def test_ingestion_empty_file_rejected(tmp_path):
    """Test that empty JSON and JSONL files raise TelemetryValidationError."""
    empty_json = tmp_path / "empty.json"
    empty_json.write_text("", encoding="utf-8")
    with pytest.raises(TelemetryValidationError, match="empty"):
        load_telemetry(empty_json)

    empty_jsonl = tmp_path / "empty.jsonl"
    empty_jsonl.write_text("   \n  \n", encoding="utf-8")
    with pytest.raises(TelemetryValidationError, match="empty"):
        load_telemetry(empty_jsonl)


def test_ingestion_malformed_json_rejected(tmp_path):
    """Test that corrupt JSON syntax raises TelemetryParseError."""
    bad_json = tmp_path / "corrupt.json"
    bad_json.write_text("{\"records\": [ {timestamp: 0.1", encoding="utf-8")
    with pytest.raises(TelemetryParseError, match="Malformed JSON"):
        load_telemetry(bad_json)


def test_ingestion_missing_timestamp_rejected(tmp_path):
    """Test that records missing required 'timestamp' fail validation."""
    invalid_data = {
        "records": [
            {"ee_position": [0.1, 0.2, 0.3]}  # missing timestamp
        ]
    }
    bad_file = tmp_path / "no_ts.json"
    bad_file.write_text(json.dumps(invalid_data), encoding="utf-8")
    with pytest.raises(TelemetryValidationError, match="timestamp"):
        load_telemetry(bad_file)


# ==============================================================================
# 3. Validation, Missing Fields, and Sanity Rules
# ==============================================================================

def test_validation_detects_non_monotonic_timestamps():
    """Ensure timestamps that jump backward are flagged as errors."""
    session = RecordedTelemetry(
        session_id="unordered_ts",
        provenance=TelemetryProvenance(is_synthetic=True),
        records=[
            TelemetryRecord(timestamp=0.0, ee_position=[0.0, 0.0, 0.0]),
            TelemetryRecord(timestamp=1.0, ee_position=[0.1, 0.0, 0.0]),
            TelemetryRecord(timestamp=0.5, ee_position=[0.2, 0.0, 0.0]),  # Backwards
        ],
    )
    report = session.validate()
    assert not report.is_valid
    assert any("Timestamp decreased" in err for err in report.errors)

    with pytest.raises(TelemetryValidationError, match="Timestamp decreased"):
        session.validate(raise_on_error=True)


def test_validation_flags_duplicate_timestamps():
    """Ensure duplicate timestamps produce a validation warning."""
    session = RecordedTelemetry(
        session_id="dup_ts",
        provenance=TelemetryProvenance(is_synthetic=True),
        records=[
            TelemetryRecord(timestamp=0.0, ee_position=[0.0, 0.0, 0.0]),
            TelemetryRecord(timestamp=0.1, ee_position=[0.1, 0.0, 0.0]),
            TelemetryRecord(timestamp=0.1, ee_position=[0.1, 0.0, 0.0]),  # Duplicate
        ],
    )
    report = session.validate()
    assert report.is_valid
    assert any("duplicate timestamp" in w for w in report.warnings)


def test_missing_optional_fields_valid():
    """Ensure optional fields (joints, actions, events) can be omitted cleanly."""
    session = RecordedTelemetry(
        session_id="minimal_record",
        provenance=TelemetryProvenance(is_synthetic=True),
        records=[
            TelemetryRecord(timestamp=0.0, ee_position=[0.1, 0.2, 0.3]),
            TelemetryRecord(timestamp=0.1, ee_position=[0.1, 0.2, 0.3]),
        ],
    )
    report = session.validate()
    assert report.is_valid
    assert report.has_ee_positions is True
    assert report.has_joint_positions is False
    assert report.has_actions is False
    assert report.has_events is False


def test_missing_ee_positions_handling():
    """Ensure sessions without EE positions report missing field and reject trajectory extraction."""
    session = RecordedTelemetry(
        session_id="no_ee_session",
        provenance=TelemetryProvenance(is_synthetic=True),
        records=[
            TelemetryRecord(timestamp=0.0, joint_positions=[0.1, 0.2]),
            TelemetryRecord(timestamp=0.1, joint_positions=[0.1, 0.2]),
        ],
    )
    with pytest.raises(TelemetryValidationError, match="does not contain ee_position"):
        session.get_ee_trajectory()

    offline_session = OfflineTelemetrySession(session)
    metrics_info = offline_session.get_supported_metrics()
    assert "trajectory_deviation" in metrics_info["unsupported"]


def test_invalid_numeric_values_rejected():
    """Ensure NaN and Inf coordinates in ee_position are rejected."""
    session = RecordedTelemetry(
        session_id="nan_session",
        provenance=TelemetryProvenance(is_synthetic=True),
        records=[
            TelemetryRecord(timestamp=0.0, ee_position=[0.0, float("nan"), 0.0]),
        ],
    )
    report = session.validate()
    assert not report.is_valid
    assert any("non-finite" in err for err in report.errors)


# ==============================================================================
# 4. Trajectory Comparison & DTW Hand-Calculated Tests
# ==============================================================================

def test_trajectory_comparison_identical():
    """Hand-calculated check: Identical trajectories must yield exactly 0 deviation."""
    comparator = TrajectoryComparator()
    traj = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ])

    result = comparator.compare(traj, traj, num_resample_points=3)
    assert result.rms_euclidean_distance == pytest.approx(0.0)
    assert result.mean_euclidean_distance == pytest.approx(0.0)
    assert result.max_euclidean_distance == pytest.approx(0.0)
    assert result.dtw_cost == pytest.approx(0.0)
    assert result.dtw_average_distance == pytest.approx(0.0)
    assert TRAJECTORY_SAFETY_DISCLAIMER in result.disclaimer


def test_trajectory_comparison_constant_offset():
    """Hand-calculated check: Constant 0.1m offset along Z must yield 0.1m across all metrics."""
    comparator = TrajectoryComparator()
    sim_traj = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    rec_traj = np.array([
        [0.0, 0.0, 0.1],
        [1.0, 0.0, 0.1],
    ])

    result = comparator.compare(sim_traj, rec_traj, num_resample_points=10)
    assert result.rms_euclidean_distance == pytest.approx(0.1, abs=1e-5)
    assert result.mean_euclidean_distance == pytest.approx(0.1, abs=1e-5)
    assert result.max_euclidean_distance == pytest.approx(0.1, abs=1e-5)
    assert result.dtw_average_distance == pytest.approx(0.1, abs=1e-5)


def test_dtw_hand_calculated_points():
    """Hand-calculated check for DTW on small synthetic trajectories.
    
    A = [[0, 0, 0], [2, 0, 0]]
    B = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    
    Cost matrix:
      C(0,0)=0, C(0,1)=1, C(0,2)=2
      C(1,0)=2, C(1,1)=1, C(1,2)=0
    
    Accumulated DP:
      (0,0) -> 0
      (0,1) -> 1
      (1,2) -> 1
    Total DTW cost = 1.0, path length = 3 -> average distance = 1.0 / 3.
    """
    traj_a = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    traj_b = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    cost, avg, path_len = compute_dtw(traj_a, traj_b)
    assert cost == pytest.approx(1.0)
    assert path_len == 3
    assert avg == pytest.approx(1.0 / 3.0)


def test_trajectory_comparison_empty_and_invalid_inputs():
    """Ensure empty or invalid inputs raise clear ValueError."""
    comparator = TrajectoryComparator()

    with pytest.raises(ValueError, match="zero points"):
        comparator.compare(np.zeros((0, 3)), np.zeros((5, 3)))

    with pytest.raises(ValueError, match="zero points"):
        comparator.compare(np.zeros((5, 3)), np.zeros((0, 3)))

    with pytest.raises(ValueError, match="shape"):
        comparator.compare(np.zeros((5, 2)), np.zeros((5, 3)))  # 2D instead of 3D

    with pytest.raises(ValueError, match="NaN"):
        comparator.compare(np.array([[0.0, float("nan"), 0.0]]), np.zeros((5, 3)))


def test_trajectory_comparison_with_rollout_object(synthetic_session):
    """Ensure TrajectoryComparator accepts Rollout objects directly."""
    comparator = TrajectoryComparator()
    obs = [
        np.array([0.5, 0.0, 0.3, 1.0, 0, 0, 0, 0, 0, 0]),
        np.array([0.52, 0.01, 0.31, 1.0, 0, 0, 0, 0, 0, 0]),
        np.array([0.55, 0.02, 0.32, 0.0, 0, 0, 0, 0, 0, 0]),
    ]
    sim_rollout = Rollout(
        scenario_id="synthetic_sim",
        policy_version="baseline",
        seed=42,
        observations=obs,
        actions=[np.zeros(4), np.zeros(4)],
        states=[{}, {}, {}],
        rewards=[0.1, 0.2],
        success=True,
        collision_count=0,
        total_steps=2,
    )

    result = comparator.compare(sim_rollout, synthetic_session)
    assert result.sim_points_count == 3
    assert result.recorded_points_count == 3
    # Same points: deviation should be zero
    assert result.rms_euclidean_distance == pytest.approx(0.0, abs=1e-5)
    assert result.dtw_cost == pytest.approx(0.0, abs=1e-5)


# ==============================================================================
# 5. Optional Dependency & Format Failure Tests
# ==============================================================================

def test_mcap_and_rosbag2_adapters_fail_safely(tmp_path, monkeypatch):
    """Verify that uninstalled MCAP/rosbag2 formats fail closed with descriptive guidance."""
    import sys
    mcap_adapter = McapTelemetryAdapter()
    bag_adapter = Rosbag2TelemetryAdapter()

    dummy_mcap = tmp_path / "recording.mcap"
    dummy_mcap.write_bytes(b"MCAP0")

    dummy_bag = tmp_path / "recording.db3"
    dummy_bag.write_bytes(b"SQLITE3")

    # When mcap is simulated as unavailable, reading raises UnsupportedFormatError
    with monkeypatch.context() as m:
        m.setitem(sys.modules, "mcap.reader", None)
        with pytest.raises(UnsupportedFormatError, match="MCAP telemetry parsing requires"):
            mcap_adapter.read(dummy_mcap)

    # When mcap is installed, corrupt bytes raise TelemetryParseError
    with pytest.raises(TelemetryParseError, match="Failed to parse MCAP file"):
        mcap_adapter.read(dummy_mcap)

    with pytest.raises(UnsupportedFormatError, match="rosbag2.*parsing requires"):
        bag_adapter.read(dummy_bag)

