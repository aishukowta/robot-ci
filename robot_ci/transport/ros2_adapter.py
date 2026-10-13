"""Optional ROS 2 hardware transport adapter for Robot CI.

Enables communication with a ROS 2 robot controller (e.g. ros2_control or driver node)
when a native ROS 2 runtime is present.
Fails closed with clear, actionable diagnostics when ROS 2 (rclpy) is unavailable.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .base import RobotTransport
from .models import (
    TransportStatus,
    TransportState,
    RobotMeasurement,
    MeasurementType,
    RobotCommand,
    CommandExecutionReport,
    ROS2UnavailableError,
    TransportConnectionError,
    TransportCommandRejectedError,
)


@dataclass
class ROS2TransportConfig:
    """Configurable ROS 2 transport parameters.
    
    Topic and service names are explicitly decoupled from any specific robot vendor.
    """
    node_name: str = "robot_ci_transport"
    joint_states_topic: str = "/joint_states"
    cartesian_pose_topic: Optional[str] = "/ee_pose"
    command_topic: str = "/joint_commands"
    safe_stop_service: Optional[str] = "/robot_safety/safe_stop"
    domain_id: Optional[int] = None
    qos_depth: int = 10


def is_ros2_available() -> bool:
    """Check if rclpy and core ROS 2 message modules can be imported."""
    try:
        import rclpy  # noqa: F401
        import sensor_msgs.msg  # noqa: F401
        return True
    except (ImportError, ModuleNotFoundError):
        return False


SCAFFOLD_NOTICE = (
    "ROS 2 adapter is a SCAFFOLD: it can create a ROS 2 node, but no subscribers, publishers "
    "or service clients are implemented. It cannot read robot state, send commands or "
    "request a stop."
)


class ROS2RobotTransport(RobotTransport):
    """ROS 2 transport SCAFFOLD (no live communication implemented).

    Only node creation is implemented. read_state / send_command raise
    NotImplementedError and request_safe_stop returns False, so nothing can
    mistake this class for a working robot connection.
    
    Communicates with robot drivers via standard ROS 2 publishers, subscribers,
    and services. Never executes on Windows without native ROS 2.
    """

    def __init__(self, config: Optional[ROS2TransportConfig] = None):
        self.config = config or ROS2TransportConfig()
        self.transport_name = "ros2_robot_transport"
        self._connected = False
        self._node = None
        self._executor = None
        self._safe_stop_active = False
        self._safe_stop_confirmed = False

    def connect(self, target_address: Optional[str] = None) -> None:
        """Establish ROS 2 transport connection.
        
        Raises:
            ROS2UnavailableError: If rclpy or ROS 2 dependencies are not installed in the active environment.
        """
        if not is_ros2_available():
            raise ROS2UnavailableError(
                "ROS 2 Python client library ('rclpy') and message packages are not available in the current environment. "
                "Operating system: Windows Python 3.11. "
                "ROS 2 communication requires a compatible ROS 2 runtime (e.g., Ubuntu 22.04 LTS with ROS 2 Humble/Iron "
                "or configured Windows ROS 2 deployment) with sourced ROS environment variables. "
                "Use FakeRobotTransport for development, testing, and continuous integration."
            )

        # Execution path when ROS 2 runtime is genuinely present:
        try:
            import rclpy  # type: ignore
            from rclpy.node import Node  # type: ignore

            if not rclpy.ok():
                rclpy.init()

            self._node = rclpy.create_node(self.config.node_name)
            self._connected = True
        except Exception as e:
            self._connected = False
            raise TransportConnectionError(f"Failed to initialize ROS 2 node '{self.config.node_name}': {e}")

    def disconnect(self) -> None:
        if self._node is not None:
            try:
                import rclpy  # type: ignore
                self._node.destroy_node()
                if rclpy.ok():
                    rclpy.shutdown()
            except Exception:
                pass
            self._node = None
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self._node is not None

    def get_status(self) -> TransportStatus:
        return TransportStatus(
            connected=self.is_connected(),
            state=TransportState.CONNECTED if self.is_connected() else TransportState.DISCONNECTED,
            transport_name=self.transport_name,
            safe_stop_active=self._safe_stop_active,
            safe_stop_confirmed=self._safe_stop_confirmed,
            metadata={"node_name": self.config.node_name, "ros2_available": is_ros2_available()},
        )

    is_scaffold = True

    def read_state(self) -> Optional[RobotMeasurement]:
        if not self.is_connected():
            return None
        raise NotImplementedError(SCAFFOLD_NOTICE + " (read_state: no JointState subscriber)")

    def request_safe_stop(self) -> bool:
        """Records the request locally; NO service is called. Returns False (not delivered)."""
        self._safe_stop_active = True
        return False

    def is_safe_stop_confirmed(self) -> bool:
        return self._safe_stop_active and self._safe_stop_confirmed

    def reset_safe_stop(self) -> None:
        self._safe_stop_active = False
        self._safe_stop_confirmed = False

    def send_command(self, command: RobotCommand) -> CommandExecutionReport:
        if not self.is_connected():
            raise TransportCommandRejectedError("Cannot send command: ROS 2 node is not connected.")
        if self._safe_stop_active:
            raise TransportCommandRejectedError("Cannot send command: safe stop is active.")

        command.validate_finite()
        raise NotImplementedError(SCAFFOLD_NOTICE + " (send_command: no publisher)")
