"""ROS 2 message mapping and validation functions for Robot CI.

Provides explicit, safe mapping between standard ROS message types
(sensor_msgs/JointState, geometry_msgs/PoseStamped) and Robot CI's internal
RobotMeasurement models.

Enforces:
- Frame ID validation (never silently assumes unknown frames are the robot base frame)
- Timestamp and unit preservation (radians, meters, UTC seconds)
- Dimension and numeric finiteness validation
- Sensor vs derived estimate provenance distinction
- 100% optional ROS 2 imports so tests run cleanly in any environment.
"""

from typing import Dict, Any, List, Optional, Union
import math
import numpy as np

from .models import (
    RobotMeasurement,
    MeasurementType,
)


class MappingValidationError(ValueError):
    """Raised when incoming ROS 2 message fails schema, dimension, or frame validation."""
    pass


DEFAULT_VALID_BASE_FRAMES = [
    "base_link",
    "base",
    "robot_base",
    "world",
    "base_link_inertia",
    "table_base",
]


def _extract_header(msg: Any) -> Dict[str, Any]:
    """Extract header stamp and frame_id from a ROS message or dictionary."""
    if isinstance(msg, dict):
        header = msg.get("header", {})
        stamp = header.get("stamp", {})
        if isinstance(stamp, (int, float)):
            ts = float(stamp)
        elif isinstance(stamp, dict):
            sec = stamp.get("sec", 0)
            nanosec = stamp.get("nanosec", stamp.get("nsec", 0))
            ts = float(sec) + float(nanosec) * 1e-9
        else:
            ts = 0.0
        frame_id = header.get("frame_id", "")
        return {"timestamp": ts, "frame_id": str(frame_id)}

    # Object attributes (e.g. actual ROS 2 message or mock object)
    header = getattr(msg, "header", None)
    if header is None:
        return {"timestamp": 0.0, "frame_id": ""}

    stamp = getattr(header, "stamp", None)
    if stamp is None:
        ts = 0.0
    elif isinstance(stamp, (int, float)):
        ts = float(stamp)
    else:
        sec = getattr(stamp, "sec", 0)
        nanosec = getattr(stamp, "nanosec", getattr(stamp, "nsec", 0))
        ts = float(sec) + float(nanosec) * 1e-9

    frame_id = getattr(header, "frame_id", "")
    return {"timestamp": ts, "frame_id": str(frame_id)}


def ros_joint_state_to_measurement(
    msg: Any,
    valid_base_frames: Optional[List[str]] = None,
    allow_empty_frame: bool = True,
) -> RobotMeasurement:
    """Map a sensor_msgs/JointState message or dictionary to a RobotMeasurement.
    
    Args:
        msg: sensor_msgs.msg.JointState or representative dictionary.
        valid_base_frames: List of accepted reference frames.
        allow_empty_frame: Whether empty frame_id in JointState is allowed (standard in ROS 2).
    
    Returns:
        RobotMeasurement with measurement_type=ACTUAL_SENSOR.
    
    Raises:
        MappingValidationError: If dimensions mismatch, numbers are non-finite, or frame is unknown.
    """
    valid_frames = valid_base_frames or DEFAULT_VALID_BASE_FRAMES
    header_info = _extract_header(msg)
    ts = header_info["timestamp"]
    frame_id = header_info["frame_id"]

    if ts <= 0 or not math.isfinite(ts):
        raise MappingValidationError(f"JointState has invalid or non-finite timestamp: {ts}")

    if frame_id:
        if frame_id not in valid_frames:
            raise MappingValidationError(
                f"Unrecognized coordinate frame '{frame_id}' in JointState header. "
                f"Cannot assume robot base frame without explicit transformation. Accepted frames: {valid_frames}"
            )

    # Extract names, positions, velocities, effort
    if isinstance(msg, dict):
        names = list(msg.get("name", []))
        positions = list(msg.get("position", []))
        velocities = list(msg.get("velocity", [])) if "velocity" in msg else None
        efforts = list(msg.get("effort", [])) if "effort" in msg else None
    else:
        names = list(getattr(msg, "name", []))
        positions = list(getattr(msg, "position", []))
        vel_attr = getattr(msg, "velocity", None)
        velocities = list(vel_attr) if vel_attr is not None and len(vel_attr) > 0 else None
        eff_attr = getattr(msg, "effort", None)
        efforts = list(eff_attr) if eff_attr is not None and len(eff_attr) > 0 else None

    if len(names) == 0:
        raise MappingValidationError("JointState message contains zero joint names.")

    if len(names) != len(positions):
        raise MappingValidationError(
            f"Mismatched JointState dimensions: {len(names)} joint names vs {len(positions)} positions."
        )

    for i, val in enumerate(positions):
        if not isinstance(val, (int, float)) or not math.isfinite(val):
            raise MappingValidationError(f"Joint '{names[i]}' has non-finite position value: {val}")

    if velocities is not None:
        if len(velocities) != len(names):
            raise MappingValidationError(
                f"Mismatched JointState velocities dimension: expected {len(names)}, got {len(velocities)}."
            )
        for i, val in enumerate(velocities):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise MappingValidationError(f"Joint '{names[i]}' has non-finite velocity value: {val}")

    if efforts is not None:
        if len(efforts) != len(names):
            raise MappingValidationError(
                f"Mismatched JointState efforts dimension: expected {len(names)}, got {len(efforts)}."
            )
        for i, val in enumerate(efforts):
            if not isinstance(val, (int, float)) or not math.isfinite(val):
                raise MappingValidationError(f"Joint '{names[i]}' has non-finite effort value: {val}")

    return RobotMeasurement(
        timestamp=ts,
        measurement_type=MeasurementType.ACTUAL_SENSOR,
        joint_names=[str(n) for n in names],
        joint_positions=[float(p) for p in positions],
        joint_velocities=[float(v) for v in velocities] if velocities else None,
        joint_efforts=[float(e) for e in efforts] if efforts else None,
        is_valid=True,
        metadata={
            "source_type": "sensor_msgs/msg/JointState",
            "frame_id": frame_id,
        },
    )


def ros_pose_stamped_to_measurement(
    msg: Any,
    expected_frame_id: Optional[str] = None,
    valid_base_frames: Optional[List[str]] = None,
    measurement_type: MeasurementType = MeasurementType.DERIVED_ESTIMATE,
) -> RobotMeasurement:
    """Map a geometry_msgs/PoseStamped message or dictionary to a RobotMeasurement.
    
    Args:
        msg: geometry_msgs.msg.PoseStamped or representative dictionary.
        expected_frame_id: Explicit required frame_id. If frame does not match, rejects conversion.
        valid_base_frames: List of accepted reference frames if expected_frame_id is not given.
        measurement_type: Distinction flag (default DERIVED_ESTIMATE for forward kinematics).
    
    Returns:
        RobotMeasurement with ee_position and ee_orientation in meters/quaternion.
    """
    valid_frames = valid_base_frames or DEFAULT_VALID_BASE_FRAMES
    header_info = _extract_header(msg)
    ts = header_info["timestamp"]
    frame_id = header_info["frame_id"]

    if ts <= 0 or not math.isfinite(ts):
        raise MappingValidationError(f"PoseStamped has invalid or non-finite timestamp: {ts}")

    if not frame_id:
        raise MappingValidationError("PoseStamped header missing required 'frame_id'.")

    if expected_frame_id is not None:
        if frame_id != expected_frame_id:
            raise MappingValidationError(
                f"PoseStamped frame_id '{frame_id}' does not match expected target frame '{expected_frame_id}'. "
                f"Cannot assume equivalent frames without explicit coordinate transform."
            )
    else:
        if frame_id not in valid_frames:
            raise MappingValidationError(
                f"Unrecognized coordinate frame '{frame_id}' in PoseStamped header. "
                f"Do not assume robot base frame without explicit transformation. Accepted frames: {valid_frames}"
            )

    # Extract position and orientation
    if isinstance(msg, dict):
        pose = msg.get("pose", {})
        pos = pose.get("position", {})
        ori = pose.get("orientation", {})
        px, py, pz = pos.get("x", 0.0), pos.get("y", 0.0), pos.get("z", 0.0)
        qx, qy, qz, qw = ori.get("x", 0.0), ori.get("y", 0.0), ori.get("z", 0.0), ori.get("w", 1.0)
    else:
        pose = getattr(msg, "pose", None)
        if pose is None:
            raise MappingValidationError("PoseStamped missing 'pose' attribute.")
        pos = getattr(pose, "position", None)
        ori = getattr(pose, "orientation", None)
        if pos is None or ori is None:
            raise MappingValidationError("PoseStamped missing position or orientation attributes.")
        px, py, pz = getattr(pos, "x", 0.0), getattr(pos, "y", 0.0), getattr(pos, "z", 0.0)
        qx, qy, qz, qw = (
            getattr(ori, "x", 0.0),
            getattr(ori, "y", 0.0),
            getattr(ori, "z", 0.0),
            getattr(ori, "w", 1.0),
        )

    coords = [px, py, pz, qx, qy, qz, qw]
    for c in coords:
        if not isinstance(c, (int, float)) or not math.isfinite(c):
            raise MappingValidationError(f"PoseStamped contains non-finite coordinate values: {coords}")

    quat_norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if quat_norm < 1e-4:
        raise MappingValidationError("PoseStamped quaternion has near-zero magnitude (degenerate rotation).")

    return RobotMeasurement(
        timestamp=ts,
        measurement_type=measurement_type,
        ee_position=[float(px), float(py), float(pz)],
        ee_orientation=[float(qx), float(qy), float(qz), float(qw)],
        is_valid=True,
        metadata={
            "source_type": "geometry_msgs/msg/PoseStamped",
            "frame_id": frame_id,
            "quat_norm": round(quat_norm, 6),
        },
    )


def measurement_to_ros_joint_state_dict(
    measurement: RobotMeasurement,
    frame_id: str = "base_link",
) -> Dict[str, Any]:
    """Serialize RobotMeasurement to a sensor_msgs/JointState compatible dictionary."""
    sec = int(measurement.timestamp)
    nanosec = int((measurement.timestamp - sec) * 1e9)
    return {
        "header": {
            "stamp": {"sec": sec, "nanosec": nanosec},
            "frame_id": frame_id,
        },
        "name": list(measurement.joint_names),
        "position": list(measurement.joint_positions),
        "velocity": list(measurement.joint_velocities) if measurement.joint_velocities else [],
        "effort": list(measurement.joint_efforts) if measurement.joint_efforts else [],
    }


def measurement_to_ros_pose_stamped_dict(
    measurement: RobotMeasurement,
    frame_id: str = "base_link",
) -> Dict[str, Any]:
    """Serialize RobotMeasurement to a geometry_msgs/PoseStamped compatible dictionary."""
    if measurement.ee_position is None:
        raise ValueError("Cannot serialize to PoseStamped: measurement lacks ee_position.")

    sec = int(measurement.timestamp)
    nanosec = int((measurement.timestamp - sec) * 1e9)
    ori = measurement.ee_orientation or [0.0, 0.0, 0.0, 1.0]

    return {
        "header": {
            "stamp": {"sec": sec, "nanosec": nanosec},
            "frame_id": frame_id,
        },
        "pose": {
            "position": {
                "x": float(measurement.ee_position[0]),
                "y": float(measurement.ee_position[1]),
                "z": float(measurement.ee_position[2]),
            },
            "orientation": {
                "x": float(ori[0]),
                "y": float(ori[1]),
                "z": float(ori[2]),
                "w": float(ori[3]),
            },
        },
    }
