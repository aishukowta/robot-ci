"""Deterministic fake robot transport for hardware testing without physical equipment.

Implements the RobotTransport contract with configurable failure injection,
sensor playback, acknowledgment tracking, and safe-stop verification.
Explicitly labeled as fake/simulated to prevent any confusion with physical hardware.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
import numpy as np

from .base import RobotTransport
from .models import (
    TransportStatus,
    TransportState,
    RobotMeasurement,
    MeasurementType,
    RobotCommand,
    CommandType,
    CommandExecutionReport,
    TransportConnectionError,
    TransportCommandRejectedError,
)


class FakeRobotTransport(RobotTransport):
    """Deterministic test double for RobotTransport.
    
    Provides controllable connection states, fault injection, measurement synthesis,
    and command gating to verify communication architecture without physical robots.
    """

    def __init__(
        self,
        transport_name: str = "fake_robot_transport",
        default_joint_names: Optional[List[str]] = None,
        initial_joint_positions: Optional[List[float]] = None,
        initial_ee_position: Optional[List[float]] = None,
        execution_enabled: bool = False,
    ):
        self.transport_name = transport_name
        self.is_fake: bool = True
        self.is_physical: bool = False
        
        self.execution_enabled = execution_enabled
        self.simulate_physical_completion = False
        self.simulate_connection_failure = False

        self._connected = False
        self._state = TransportState.DISCONNECTED
        self._fault_code: Optional[str] = None
        self._fault_message: Optional[str] = None
        self._safe_stop_active = False
        self._safe_stop_confirmed = False
        self._last_heartbeat = None

        self.joint_names = default_joint_names or ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.current_joint_positions = initial_joint_positions or [0.0, -0.5, 0.5, 0.0, 0.5, 0.0]
        self.current_ee_position = initial_ee_position or [0.5, 0.0, 0.3]
        self.current_gripper_position = 1.0

        self.sent_commands: List[RobotCommand] = []
        self._injected_measurement: Optional[RobotMeasurement] = None

    def connect(self, target_address: Optional[str] = None) -> None:
        if self.simulate_connection_failure or (target_address and "unreachable" in target_address.lower()):
            self._state = TransportState.FAULTED
            self._connected = False
            self._fault_code = "ERR_CONNECT_TIMEOUT"
            self._fault_message = f"Failed to connect to target address '{target_address}'"
            raise TransportConnectionError(self._fault_message)

        self._connected = True
        self._state = TransportState.CONNECTED
        self._fault_code = None
        self._fault_message = None
        self._last_heartbeat = datetime.now(timezone.utc).timestamp()

    def disconnect(self) -> None:
        self._connected = False
        self._state = TransportState.DISCONNECTED
        self._fault_code = None
        self._fault_message = None

    def is_connected(self) -> bool:
        return self._connected and self._state != TransportState.DISCONNECTED

    def trigger_fault(self, code: str, message: str) -> None:
        """Inject a hardware fault condition."""
        self._state = TransportState.FAULTED
        self._fault_code = code
        self._fault_message = message

    def clear_fault(self) -> None:
        """Clear active fault condition."""
        if self._connected:
            self._state = TransportState.CONNECTED
        else:
            self._state = TransportState.DISCONNECTED
        self._fault_code = None
        self._fault_message = None

    def get_status(self) -> TransportStatus:
        return TransportStatus(
            connected=self.is_connected(),
            state=self._state,
            transport_name=self.transport_name,
            fault_code=self._fault_code,
            fault_message=self._fault_message,
            safe_stop_active=self._safe_stop_active,
            safe_stop_confirmed=self._safe_stop_confirmed,
            last_heartbeat_timestamp=self._last_heartbeat,
            metadata={"is_fake": True, "execution_enabled": self.execution_enabled},
        )

    def set_measurement(self, measurement: RobotMeasurement) -> None:
        """Inject a specific measurement packet."""
        self._injected_measurement = measurement

    def read_state(self) -> Optional[RobotMeasurement]:
        if not self.is_connected():
            return None

        if self._injected_measurement is not None:
            return self._injected_measurement

        now = datetime.now(timezone.utc).timestamp()
        return RobotMeasurement(
            timestamp=now,
            measurement_type=MeasurementType.ACTUAL_SENSOR,
            joint_names=list(self.joint_names),
            joint_positions=list(self.current_joint_positions),
            joint_velocities=[0.0] * len(self.joint_names),
            joint_efforts=[0.0] * len(self.joint_names),
            ee_position=list(self.current_ee_position),
            ee_orientation=[0.0, 0.0, 0.0, 1.0],
            gripper_position=self.current_gripper_position,
            metadata={"source": "fake_driver_sensor_stream"},
        )

    def request_safe_stop(self) -> bool:
        self._safe_stop_active = True
        self._state = TransportState.SAFE_STOPPED
        # In a real robot, confirmation occurs after deceleration. We confirm here.
        self._safe_stop_confirmed = True
        return True

    def is_safe_stop_confirmed(self) -> bool:
        return self._safe_stop_active and self._safe_stop_confirmed

    def reset_safe_stop(self) -> None:
        self._safe_stop_active = False
        self._safe_stop_confirmed = False
        if self._connected:
            self._state = TransportState.CONNECTED

    def send_command(self, command: RobotCommand) -> CommandExecutionReport:
        # 1. Connection check
        if not self._connected or self._state == TransportState.DISCONNECTED:
            raise TransportCommandRejectedError("Cannot send command: transport is disconnected.")

        # 2. State & Interlock checks
        if self._state == TransportState.FAULTED:
            raise TransportCommandRejectedError(
                f"Cannot send command: transport is in FAULTED state ({self._fault_code}: {self._fault_message})."
            )

        if self._safe_stop_active or self._state == TransportState.SAFE_STOPPED:
            raise TransportCommandRejectedError("Cannot send command: safe-stop interlock is active.")

        # 3. Execution gating check
        if not self.execution_enabled:
            raise TransportCommandRejectedError(
                "Cannot send command: physical execution is disabled by safety configuration."
            )

        # 4. Numerical validation
        command.validate_finite()

        # 5. Record command
        self.sent_commands.append(command)

        # 6. Apply command delta to state
        if command.command_type == CommandType.CARTESIAN_DELTA:
            if len(command.target_values) >= 3:
                self.current_ee_position = [
                    self.current_ee_position[0] + command.target_values[0],
                    self.current_ee_position[1] + command.target_values[1],
                    self.current_ee_position[2] + command.target_values[2],
                ]
            if len(command.target_values) >= 4:
                self.current_gripper_position = command.target_values[3]

        elif command.command_type == CommandType.JOINT_POSITION:
            self.current_joint_positions = list(command.target_values)

        now = datetime.now(timezone.utc).timestamp()
        return CommandExecutionReport(
            command_id=command.command_id,
            receipt_acknowledged=True,
            execution_status="COMPLETED" if self.simulate_physical_completion else "RECEIVED",
            is_physically_completed=self.simulate_physical_completion,
            timestamp=now,
            metadata={"transport": self.transport_name, "simulated": True},
        )
