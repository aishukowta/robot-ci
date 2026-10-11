"""Robot communication data models, status types, and contracts.

Provides transport-independent data models for communication with real or simulated robot arms.
Clearly distinguishes:
- Joint-space commands from Cartesian commands
- Actual sensor measurements from derived estimates
- Message receipt from successful physical execution
- Safe-stop requests from physical stop confirmations
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import List, Dict, Any, Optional, Union
import math
import numpy as np


class TransportError(Exception):
    """Base exception for robot transport and communication errors."""
    pass


class TransportConnectionError(TransportError):
    """Raised when transport connection fails or is disconnected."""
    pass


class TransportCommandRejectedError(TransportError):
    """Raised when a command is rejected due to safety, connection, or state."""
    pass


class ROS2UnavailableError(TransportError):
    """Raised when ROS 2 runtime or rclpy is requested but unavailable."""
    pass


class TransportState(str, Enum):
    """Lifecycle states of a robot transport."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    FAULTED = "FAULTED"
    SAFE_STOPPED = "SAFE_STOPPED"


class MeasurementType(str, Enum):
    """Distinguishes direct physical sensor measurements from derived estimates."""
    ACTUAL_SENSOR = "ACTUAL_SENSOR"
    DERIVED_ESTIMATE = "DERIVED_ESTIMATE"


class CommandType(str, Enum):
    """Distinguishes control spaces for robotic actuation."""
    CARTESIAN_DELTA = "CARTESIAN_DELTA"     # [dx, dy, dz, grip] in meters
    CARTESIAN_POSE = "CARTESIAN_POSE"       # [x, y, z, qx, qy, qz, qw, grip] in meters/quaternion
    JOINT_POSITION = "JOINT_POSITION"       # [q1, ..., qn] in radians
    SAFE_STOP = "SAFE_STOP"                 # Immediate brake / deceleration request


@dataclass
class TransportStatus:
    """Current health, connection, and safety state of the robot transport."""
    connected: bool
    state: TransportState
    transport_name: str
    fault_code: Optional[str] = None
    fault_message: Optional[str] = None
    safe_stop_active: bool = False          # Safe stop was requested
    safe_stop_confirmed: bool = False       # Physical confirmation received from hardware
    last_heartbeat_timestamp: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connected": self.connected,
            "state": self.state.value if isinstance(self.state, TransportState) else str(self.state),
            "transport_name": self.transport_name,
            "fault_code": self.fault_code,
            "fault_message": self.fault_message,
            "safe_stop_active": self.safe_stop_active,
            "safe_stop_confirmed": self.safe_stop_confirmed,
            "last_heartbeat_timestamp": self.last_heartbeat_timestamp,
            "metadata": self.metadata,
        }


@dataclass
class RobotMeasurement:
    """Telemetry packet received from robot controller or state estimator.
    
    Units:
    - joint_positions: Radians (revolute) or Meters (prismatic)
    - joint_velocities: Rad/s or m/s
    - joint_efforts: Nm or N
    - ee_position: Meters [x, y, z]
    - ee_orientation: Unit quaternion [x, y, z, w]
    - timestamp: UNIX epoch seconds
    """
    timestamp: float
    measurement_type: MeasurementType
    joint_names: List[str] = field(default_factory=list)
    joint_positions: List[float] = field(default_factory=list)
    joint_velocities: Optional[List[float]] = None
    joint_efforts: Optional[List[float]] = None
    ee_position: Optional[List[float]] = None
    ee_orientation: Optional[List[float]] = None
    gripper_position: Optional[float] = None
    timestamp_iso: Optional[str] = None
    is_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.timestamp_iso is None and self.timestamp is not None:
            try:
                self.timestamp_iso = datetime.fromtimestamp(self.timestamp, tz=timezone.utc).isoformat()
            except Exception:
                pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "timestamp_iso": self.timestamp_iso,
            "measurement_type": (
                self.measurement_type.value
                if isinstance(self.measurement_type, MeasurementType)
                else str(self.measurement_type)
            ),
            "joint_names": self.joint_names,
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "joint_efforts": self.joint_efforts,
            "ee_position": self.ee_position,
            "ee_orientation": self.ee_orientation,
            "gripper_position": self.gripper_position,
            "is_valid": self.is_valid,
            "metadata": self.metadata,
        }


@dataclass
class RobotCommand:
    """Action or motion command targeted at the robot transport.
    
    Target values depend explicitly on command_type:
    - CARTESIAN_DELTA: [dx, dy, dz, grip]
    - CARTESIAN_POSE: [x, y, z, qx, qy, qz, qw, grip]
    - JOINT_POSITION: [q1, ..., qn]
    """
    command_id: str
    command_type: CommandType
    target_values: List[float]
    timestamp: float
    timeout_seconds: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate_finite(self) -> None:
        """Validate that all target values are finite numeric floats."""
        for idx, val in enumerate(self.target_values):
            if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                raise TransportCommandRejectedError(
                    f"Command {self.command_id} contains non-finite target value at index {idx}: {val}"
                )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "command_type": (
                self.command_type.value
                if isinstance(self.command_type, CommandType)
                else str(self.command_type)
            ),
            "target_values": self.target_values,
            "timestamp": self.timestamp,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata,
        }


@dataclass
class CommandExecutionReport:
    """Acknowledgment and execution tracking for a command.
    
    Explicitly distinguishes message receipt/acknowledgment from physical completion.
    A receipt_acknowledged=True means the driver received the message;
    is_physically_completed=True means the robot reached goal tolerances.
    """
    command_id: str
    receipt_acknowledged: bool
    execution_status: str                   # "RECEIVED", "IN_PROGRESS", "COMPLETED", "REJECTED", "FAULTED"
    is_physically_completed: bool           # True ONLY with explicit completion signal from controller
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "command_id": self.command_id,
            "receipt_acknowledged": self.receipt_acknowledged,
            "execution_status": self.execution_status,
            "is_physically_completed": self.is_physically_completed,
            "error_message": self.error_message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }
