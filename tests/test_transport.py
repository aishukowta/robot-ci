"""Unit tests for robot transport contracts, fake transport double, and ROS 2 availability gating.

All tests operate strictly offline without physical robot hardware or live ROS 2 daemon.
"""

import time
import pytest
import numpy as np

from robot_ci.transport.models import (
    TransportStatus,
    TransportState,
    MeasurementType,
    CommandType,
    RobotMeasurement,
    RobotCommand,
    CommandExecutionReport,
    TransportConnectionError,
    TransportCommandRejectedError,
    ROS2UnavailableError,
)
from robot_ci.transport.base import RobotTransport
from robot_ci.transport.fake_transport import FakeRobotTransport
from robot_ci.transport.ros2_adapter import (
    ROS2RobotTransport,
    ROS2TransportConfig,
    is_ros2_available,
)
from robot_ci.simulation.hardware import HardwareBackend
from robot_ci.simulation.safety import (
    HardwareConfig,
    HardwareSafetyValidator,
    HardwareConnectionError,
    SafetyValidationError,
    SafetyInterlockError,
)
from robot_ci.scenarios.models import Scenario


# ==============================================================================
# 1. Fake Transport Lifecycle & Status Tests
# ==============================================================================

def test_fake_transport_connection_and_status():
    """Verify connection lifecycle and status reporting of FakeRobotTransport."""
    transport = FakeRobotTransport(transport_name="test_arm_transport")
    assert transport.is_connected() is False

    status = transport.get_status()
    assert status.connected is False
    assert status.state == TransportState.DISCONNECTED
    assert status.transport_name == "test_arm_transport"

    # Connect
    transport.connect()
    assert transport.is_connected() is True
    connected_status = transport.get_status()
    assert connected_status.connected is True
    assert connected_status.state == TransportState.CONNECTED
    assert connected_status.last_heartbeat_timestamp is not None

    # Disconnect
    transport.disconnect()
    assert transport.is_connected() is False
    assert transport.get_status().state == TransportState.DISCONNECTED


def test_fake_transport_simulated_connection_failure():
    """Verify connection failure injection raises TransportConnectionError."""
    transport = FakeRobotTransport()
    transport.simulate_connection_failure = True

    with pytest.raises(TransportConnectionError, match="Failed to connect"):
        transport.connect("robot_controller_ip:50051")

    assert transport.is_connected() is False
    status = transport.get_status()
    assert status.state == TransportState.FAULTED
    assert status.fault_code == "ERR_CONNECT_TIMEOUT"


# ==============================================================================
# 2. ROS 2 Availability and Safe Fail-Closed Tests
# ==============================================================================

def test_missing_ros2_runtime_handled_clearly():
    """Verify ROS2RobotTransport fails closed with clear diagnostic when rclpy is missing."""
    # In Windows test environment, rclpy is not installed
    assert is_ros2_available() is False

    transport = ROS2RobotTransport(ROS2TransportConfig(node_name="test_robot_ci_node"))
    assert transport.is_connected() is False

    status = transport.get_status()
    assert status.connected is False
    assert status.metadata["ros2_available"] is False

    with pytest.raises(ROS2UnavailableError) as exc_info:
        transport.connect()

    err_msg = str(exc_info.value)
    assert "rclpy" in err_msg
    assert "Windows Python 3.11" in err_msg
    assert "Ubuntu 22.04 LTS" in err_msg
    assert "FakeRobotTransport" in err_msg


# ==============================================================================
# 3. Measurement Semantics & Sensor vs Derived Distinction
# ==============================================================================

def test_joint_state_messages_units_and_timestamps():
    """Verify measurements preserve radians, meters, and ISO timestamps."""
    transport = FakeRobotTransport(
        default_joint_names=["shoulder", "elbow", "wrist"],
        initial_joint_positions=[0.1, -0.5, 1.2],
        initial_ee_position=[0.45, 0.1, 0.25],
    )
    transport.connect()

    measurement = transport.read_state()
    assert measurement is not None
    assert measurement.measurement_type == MeasurementType.ACTUAL_SENSOR
    assert measurement.joint_names == ["shoulder", "elbow", "wrist"]
    assert measurement.joint_positions == [0.1, -0.5, 1.2]
    assert measurement.ee_position == [0.45, 0.1, 0.25]
    assert measurement.timestamp > 0
    assert measurement.timestamp_iso is not None
    assert "Z" in measurement.timestamp_iso or "+00:00" in measurement.timestamp_iso


def test_measurement_derived_estimate_distinction():
    """Verify explicit distinction between actual sensor telemetry and derived estimates."""
    transport = FakeRobotTransport()
    transport.connect()

    derived = RobotMeasurement(
        timestamp=time.time(),
        measurement_type=MeasurementType.DERIVED_ESTIMATE,
        joint_names=["j1"],
        joint_positions=[0.0],
        ee_position=[0.5, 0.0, 0.3],
        metadata={"estimator": "forward_kinematics_ekf"},
    )
    transport.set_measurement(derived)

    m = transport.read_state()
    assert m.measurement_type == MeasurementType.DERIVED_ESTIMATE
    assert m.metadata["estimator"] == "forward_kinematics_ekf"


# ==============================================================================
# 4. Fault Handling and Disconnect Interlocks
# ==============================================================================

def test_fault_and_disconnect_handling():
    """Verify fault injection stops command transmission until cleared."""
    transport = FakeRobotTransport(execution_enabled=True)
    transport.connect()

    cmd = RobotCommand(
        command_id="cmd_01",
        command_type=CommandType.CARTESIAN_DELTA,
        target_values=[0.01, 0.0, 0.0, 1.0],
        timestamp=time.time(),
    )

    # Valid command succeeds when connected
    report = transport.send_command(cmd)
    assert report.receipt_acknowledged is True

    # Inject hardware fault
    transport.trigger_fault("JOINT_OVER_TEMPERATURE", "Joint 2 temp exceeded 80C")
    status = transport.get_status()
    assert status.state == TransportState.FAULTED
    assert status.fault_code == "JOINT_OVER_TEMPERATURE"

    with pytest.raises(TransportCommandRejectedError, match="FAULTED"):
        transport.send_command(cmd)

    # Clear fault restores normal operation
    transport.clear_fault()
    assert transport.is_connected() is True
    report2 = transport.send_command(cmd)
    assert report2.receipt_acknowledged is True

    # Disconnect rejects commands
    transport.disconnect()
    with pytest.raises(TransportCommandRejectedError, match="disconnected"):
        transport.send_command(cmd)


# ==============================================================================
# 5. Invalid Command Rejection & Execution Disabled by Default
# ==============================================================================

def test_execution_disabled_by_default():
    """Ensure FakeRobotTransport has execution disabled by default for safety."""
    transport = FakeRobotTransport()  # default execution_enabled=False
    transport.connect()

    cmd = RobotCommand(
        command_id="cmd_01",
        command_type=CommandType.CARTESIAN_DELTA,
        target_values=[0.01, 0.0, 0.0, 1.0],
        timestamp=time.time(),
    )

    with pytest.raises(TransportCommandRejectedError, match="physical execution is disabled"):
        transport.send_command(cmd)


def test_invalid_command_non_finite_rejection():
    """Ensure NaN, Inf, and non-numeric target values are rejected."""
    transport = FakeRobotTransport(execution_enabled=True)
    transport.connect()

    nan_cmd = RobotCommand(
        command_id="nan_cmd",
        command_type=CommandType.CARTESIAN_DELTA,
        target_values=[float("nan"), 0.0, 0.0, 1.0],
        timestamp=time.time(),
    )
    with pytest.raises(TransportCommandRejectedError, match="non-finite"):
        transport.send_command(nan_cmd)

    inf_cmd = RobotCommand(
        command_id="inf_cmd",
        command_type=CommandType.CARTESIAN_DELTA,
        target_values=[0.0, float("inf"), 0.0, 1.0],
        timestamp=time.time(),
    )
    with pytest.raises(TransportCommandRejectedError, match="non-finite"):
        transport.send_command(inf_cmd)


# ==============================================================================
# 6. Safe-Stop Requests and Physical Acknowledgment Semantics
# ==============================================================================

def test_safe_stop_request_and_confirmation():
    """Verify safe-stop request gating and confirmation semantics."""
    transport = FakeRobotTransport(execution_enabled=True)
    transport.connect()

    assert transport.get_status().safe_stop_active is False
    assert transport.is_safe_stop_confirmed() is False

    # Request safe stop
    requested = transport.request_safe_stop()
    assert requested is True
    assert transport.get_status().safe_stop_active is True
    assert transport.is_safe_stop_confirmed() is True
    assert transport.get_status().state == TransportState.SAFE_STOPPED

    # Commands must be rejected while safe stop is latched
    cmd = RobotCommand(
        command_id="c1",
        command_type=CommandType.CARTESIAN_DELTA,
        target_values=[0.01, 0.0, 0.0, 1.0],
        timestamp=time.time(),
    )
    with pytest.raises(TransportCommandRejectedError, match="safe-stop interlock"):
        transport.send_command(cmd)

    # Reset requires explicit clearance
    transport.reset_safe_stop()
    assert transport.get_status().safe_stop_active is False
    assert transport.is_connected() is True


def test_command_receipt_acknowledgment_not_task_success():
    """Ensure message receipt acknowledgment is never conflated with physical completion."""
    transport = FakeRobotTransport(execution_enabled=True)
    transport.connect()
    transport.simulate_physical_completion = False  # Message received, motion in progress

    cmd = RobotCommand(
        command_id="cmd_in_flight",
        command_type=CommandType.CARTESIAN_DELTA,
        target_values=[0.01, 0.0, 0.0, 1.0],
        timestamp=time.time(),
    )
    report = transport.send_command(cmd)

    assert report.receipt_acknowledged is True
    assert report.execution_status == "RECEIVED"
    assert report.is_physically_completed is False  # Acknowledgment != completion!


# ==============================================================================
# 7. HardwareBackend Integration with Fake Transport
# ==============================================================================

def test_hardware_backend_with_fake_transport():
    """Verify HardwareBackend dispatches commands to transport and enforces safety."""
    fake_transport = FakeRobotTransport(execution_enabled=True)
    config = HardwareConfig(
        max_step_size=0.05,
        workspace_bounds=[[0.0, -0.5, 0.0], [1.0, 0.5, 1.0]],
    )
    backend = HardwareBackend(config=config, transport=fake_transport)

    assert backend.is_connected is False

    # Connect establishes transport connection
    backend.connect()
    assert backend.is_connected is True

    sc = Scenario(
        scenario_id="sc_reach",
        task_name="reach",
        initial_state={"ee_position": [0.5, 0.0, 0.3]},
        target_state={"position": [0.6, 0.0, 0.3]},
        environment_params={"workspace_bounds": [[0.0, -0.5, 0.0], [1.0, 0.5, 1.0]]},
        max_episode_steps=5,
        success_criteria={},
        safety_criteria={},
    )
    obs = backend.reset(sc)
    assert np.allclose(obs[0:3], [0.5, 0.0, 0.3])

    # Step dispatches command to transport
    obs, reward, done, info = backend.step(np.array([0.02, 0.0, 0.0, 1.0]))
    assert info["receipt_acknowledged"] is True
    assert info["success"] is False  # Motion acknowledgment is not physical completion
    assert info["is_simulation"] is False
    assert info["simulator"] == "hardware"
    assert len(fake_transport.sent_commands) == 1

    # Safe stop engages on both backend and transport
    backend.safe_stop()
    assert backend.safe_stop_active is True
    assert fake_transport.is_safe_stop_confirmed() is True

    # Step while safe-stopped fails closed
    with pytest.raises(SafetyInterlockError):
        backend.step(np.array([0.01, 0.0, 0.0, 1.0]))

    backend.close()
    assert backend.is_connected is False
    assert fake_transport.is_connected() is False


def test_missing_robot_limits_not_assumed_safe():
    """Verify unconfigured joint limits raise SafetyValidationError when limit enforcement is enabled."""
    config = HardwareConfig(
        joint_names=["joint_1", "joint_2"],
        joint_limits={"joint_1": (-1.57, 1.57)},  # joint_2 is missing!
        enforce_joint_limits=True,
    )
    validator = HardwareSafetyValidator(config)

    with pytest.raises(SafetyValidationError, match="no configured safety limits"):
        validator.validate_joint_positions(np.array([0.0, 0.5]))
