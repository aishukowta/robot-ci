"""Robot communication, transport abstraction, and safety gating package."""

from .models import (
    TransportError,
    TransportConnectionError,
    TransportCommandRejectedError,
    ROS2UnavailableError,
    TransportState,
    MeasurementType,
    CommandType,
    TransportStatus,
    RobotMeasurement,
    RobotCommand,
    CommandExecutionReport,
)
from .base import RobotTransport
from .fake_transport import FakeRobotTransport
from .ros2_adapter import (
    ROS2RobotTransport,
    ROS2TransportConfig,
    is_ros2_available,
)
from .ros2_mapping import (
    MappingValidationError,
    ros_joint_state_to_measurement,
    ros_pose_stamped_to_measurement,
    measurement_to_ros_joint_state_dict,
    measurement_to_ros_pose_stamped_dict,
)

__all__ = [
    "TransportError",
    "TransportConnectionError",
    "TransportCommandRejectedError",
    "ROS2UnavailableError",
    "TransportState",
    "MeasurementType",
    "CommandType",
    "TransportStatus",
    "RobotMeasurement",
    "RobotCommand",
    "CommandExecutionReport",
    "RobotTransport",
    "FakeRobotTransport",
    "ROS2RobotTransport",
    "ROS2TransportConfig",
    "is_ros2_available",
    "MappingValidationError",
    "ros_joint_state_to_measurement",
    "ros_pose_stamped_to_measurement",
    "measurement_to_ros_joint_state_dict",
    "measurement_to_ros_pose_stamped_dict",
]

