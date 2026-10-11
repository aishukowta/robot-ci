"""Abstract base class for robot hardware transport communication.

Defines the contract for communicating with physical or simulated robot arms.
Supports connection lifecycle, state acquisition, command dispatch, and safe-stop gating.
"""

from abc import ABC, abstractmethod
from typing import Optional

from .models import (
    TransportStatus,
    RobotMeasurement,
    RobotCommand,
    CommandExecutionReport,
    TransportConnectionError,
    TransportCommandRejectedError,
)


class RobotTransport(ABC):
    """Abstract interface for robot communication transports (e.g. ROS 2, gRPC, serial, or fake)."""

    @abstractmethod
    def connect(self, target_address: Optional[str] = None) -> None:
        """Establish connection to robot controller, endpoint, or ROS 2 node.
        
        Raises:
            TransportConnectionError: If transport address is invalid or unreachable.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Gracefully disconnect from transport and clean up communication resources."""
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Return True if active transport session is established."""
        ...

    @abstractmethod
    def get_status(self) -> TransportStatus:
        """Retrieve current health, connection, and safety status."""
        ...

    @abstractmethod
    def read_state(self) -> Optional[RobotMeasurement]:
        """Read the most recent sensor state or derived state from robot.
        
        Returns:
            RobotMeasurement containing joint states and/or EE poses, or None if no packet received.
        """
        ...

    @abstractmethod
    def request_safe_stop(self) -> bool:
        """Send an immediate deceleration / safe-stop request to the robot controller.
        
        Returns:
            True if request was successfully transmitted and acknowledged by the transport.
        """
        ...

    @abstractmethod
    def is_safe_stop_confirmed(self) -> bool:
        """Return True only if physical robot confirmation indicates velocities are zero / stopped."""
        ...

    @abstractmethod
    def reset_safe_stop(self) -> None:
        """Clear latched safe stop after manual operator clearance."""
        ...

    @abstractmethod
    def send_command(self, command: RobotCommand) -> CommandExecutionReport:
        """Send motion command to robot transport.
        
        Raises:
            TransportCommandRejectedError: If disconnected, faulted, safe-stop latched, or values invalid.
        """
        ...
