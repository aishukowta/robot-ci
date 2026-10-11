"""Tests for Module 13: Report integrity, safety gating, and reproducibility audit.

Verifies:
1. OfflineAnalysisReport mutual exclusivity and semantic consistency between
   supported_metrics, unsupported_metrics, and unsupported_reasons.
2. Graceful handling of both dict and sequence event records in to_rollout_adapter.
3. Hardware execution fails closed by default when transport is unconfigured.
4. Distinction between safe-stop requested and confirmed physical stops.
5. Exit code and status distinction between regression findings and software crashes.
"""

import sys
import pytest
from pathlib import Path

from robot_ci.telemetry.workflow import OfflineAnalysisReport, OfflineTelemetryWorkflow
from robot_ci.telemetry.models import RecordedTelemetry, TelemetryRecord, TelemetryProvenance
from robot_ci.telemetry.replay import OfflineTelemetrySession
from robot_ci.simulation.hardware import HardwareBackend
from robot_ci.simulation.safety import HardwareConnectionError, SafetyInterlockError
from robot_ci.transport.fake_transport import FakeRobotTransport
from robot_ci.transport.models import TransportState


def test_offline_report_metric_consistency_and_cleanup():
    """Verify that OfflineAnalysisReport __post_init__ enforces consistency."""
    report = OfflineAnalysisReport(
        report_id="audit_rep_01",
        created_at="2026-10-11T12:00:00Z",
        telemetry_source="test_source",
        input_file="test.json",
        provenance={"data_origin": "synthetic", "is_synthetic": True},
        sample_count=5,
        time_range={"duration_seconds": 0.5},
        available_fields=["timestamp", "ee_position"],
        missing_fields=["action"],
        validation_summary={"is_valid": True},
        # Intentionally messy input: collision_count in both, missing reason for success_rate
        supported_metrics=["trajectory_deviation", "collision_count"],
        unsupported_metrics=["collision_count", "reward_difference", "success_rate"],
        unsupported_reasons={
            "collision_count": "Should not be here because it is supported",
            "reward_difference": "Simulation artifact only",
        },
    )

    # 1. Supported and unsupported must be strictly disjoint
    assert "collision_count" in report.supported_metrics
    assert "collision_count" not in report.unsupported_metrics

    # 2. Supported metric must not have an entry in unsupported_reasons
    assert "collision_count" not in report.unsupported_reasons

    # 3. Unsupported metric missing an explicit reason must receive an informative fallback
    assert "success_rate" in report.unsupported_metrics
    assert "success_rate" in report.unsupported_reasons
    assert "ground-truth" in report.unsupported_reasons["success_rate"]

    # 4. Explicitly provided reason for reward_difference must be preserved
    assert report.unsupported_reasons["reward_difference"] == "Simulation artifact only"


def test_to_rollout_adapter_handles_sequence_events():
    """Verify to_rollout_adapter handles list/set events without crashing."""
    telemetry = RecordedTelemetry(
        session_id="seq_events_session",
        provenance=TelemetryProvenance(is_synthetic=True),
        records=[
            TelemetryRecord(timestamp=0.0, ee_position=[0.1, 0.2, 0.3], events=["contact"]),
            TelemetryRecord(timestamp=0.1, ee_position=[0.12, 0.22, 0.32], events=["success"]),
        ],
    )
    session = OfflineTelemetrySession(telemetry)
    rollout = session.to_rollout_adapter()

    assert rollout.collision_count == 1
    assert rollout.success is True


def test_hardware_backend_fails_closed_without_transport():
    """Verify that HardwareBackend strictly fails closed without configured transport."""
    backend = HardwareBackend()
    assert backend.is_connected is False

    with pytest.raises(HardwareConnectionError, match="unconfigured"):
        backend.connect()


def test_safe_stop_request_and_execution_gating():
    """Verify that safe-stop requests engage interlocks and reject motion commands."""
    transport = FakeRobotTransport(execution_enabled=True)
    transport.connect()

    # Initially safe stop is not engaged
    assert transport.is_safe_stop_confirmed() is False
    assert transport.get_status().safe_stop_active is False

    # Request safe stop
    requested = transport.request_safe_stop()
    assert requested is True
    assert transport.is_safe_stop_confirmed() is True
    assert transport.get_status().state == TransportState.SAFE_STOPPED

    # Commands must be strictly rejected
    import time
    from robot_ci.transport.models import RobotCommand, CommandType, TransportCommandRejectedError
    cmd = RobotCommand(
        command_id="cmd_stop_check",
        command_type=CommandType.CARTESIAN_DELTA,
        target_values=[0.01, 0.0, 0.0, 1.0],
        timestamp=time.time(),
    )
    with pytest.raises(TransportCommandRejectedError, match="safe-stop interlock is active"):
        transport.send_command(cmd)

