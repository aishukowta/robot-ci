"""Tests for ROS 2 message mapping and genuine MCAP recorded ingestion (Module 12).

Verifies:
1. JointState conversion (names, positions, velocities, efforts, radians, timestamps).
2. PoseStamped conversion (frame IDs, coordinates in meters, quaternions).
3. Rejection of non-finite values (NaN, Inf) and mismatched array lengths.
4. Rejection of unknown coordinate frames (no silent fallback to robot base frame).
5. Sensor vs derived estimate distinction (ACTUAL_SENSOR vs DERIVED_ESTIMATE).
6. Genuine MCAP container file writing and reading using the official 'mcap' library.
7. Unsupported rosbag2 (.db3/SQLite) safe fail-closed behavior.
8. Preservation of recorded provenance (data_origin='recorded_offline', is_synthetic=True).
"""

import math
import tempfile
from pathlib import Path
import pytest

from robot_ci.transport.models import (
    MeasurementType,
    RobotMeasurement,
)
from robot_ci.transport.ros2_mapping import (
    MappingValidationError,
    ros_joint_state_to_measurement,
    ros_pose_stamped_to_measurement,
    measurement_to_ros_joint_state_dict,
    measurement_to_ros_pose_stamped_dict,
)
from robot_ci.telemetry.models import (
    RecordedTelemetry,
    TelemetryRecord,
    TelemetryProvenance,
    UnsupportedFormatError,
)
from robot_ci.telemetry.reader import (
    load_telemetry,
    McapTelemetryAdapter,
    Rosbag2TelemetryAdapter,
    write_mcap_telemetry,
)


# ==============================================================================
# 1. JointState Mapping Tests
# ==============================================================================

class MockHeader:
    def __init__(self, sec: int = 1700000000, nanosec: int = 250000000, frame_id: str = "base_link"):
        self.stamp = type("Stamp", (), {"sec": sec, "nanosec": nanosec})()
        self.frame_id = frame_id


class MockJointStateMsg:
    def __init__(self, names, positions, velocities=None, efforts=None, header=None):
        self.header = header or MockHeader()
        self.name = names
        self.position = positions
        self.velocity = velocities or []
        self.effort = efforts or []


def test_joint_state_dict_mapping_valid():
    """Verify conversion of valid JointState dictionary to RobotMeasurement."""
    msg = {
        "header": {
            "stamp": {"sec": 1700000000, "nanosec": 500000000},
            "frame_id": "base_link",
        },
        "name": ["joint_1", "joint_2", "joint_3"],
        "position": [0.1, -0.5, 1.2],
        "velocity": [0.01, -0.02, 0.05],
        "effort": [1.5, -2.0, 0.8],
    }

    meas = ros_joint_state_to_measurement(msg)

    assert isinstance(meas, RobotMeasurement)
    assert meas.is_valid is True
    assert meas.measurement_type == MeasurementType.ACTUAL_SENSOR
    assert math.isclose(meas.timestamp, 1700000000.5, rel_tol=1e-9)
    assert meas.joint_names == ["joint_1", "joint_2", "joint_3"]
    assert meas.joint_positions == [0.1, -0.5, 1.2]
    assert meas.joint_velocities == [0.01, -0.02, 0.05]
    assert meas.joint_efforts == [1.5, -2.0, 0.8]
    assert meas.metadata["frame_id"] == "base_link"
    assert meas.metadata["source_type"] == "sensor_msgs/msg/JointState"


def test_joint_state_object_mapping_valid():
    """Verify conversion of mock ROS 2 JointState object with object attributes."""
    mock_msg = MockJointStateMsg(
        names=["shoulder_pan_joint", "shoulder_lift_joint"],
        positions=[0.523, -1.047],  # ~30 deg and -60 deg in radians
        velocities=[0.1, -0.1],
        efforts=[3.2, 4.5],
        header=MockHeader(sec=1690000000, nanosec=100000000, frame_id="base"),
    )

    meas = ros_joint_state_to_measurement(mock_msg)

    assert meas.measurement_type == MeasurementType.ACTUAL_SENSOR
    assert math.isclose(meas.timestamp, 1690000000.1, rel_tol=1e-9)
    assert meas.joint_names == ["shoulder_pan_joint", "shoulder_lift_joint"]
    assert meas.joint_positions == [0.523, -1.047]
    assert meas.joint_velocities == [0.1, -0.1]
    assert meas.joint_efforts == [3.2, 4.5]


def test_joint_state_bidirectional_roundtrip():
    """Verify RobotMeasurement to ROS JointState dictionary serialization roundtrip."""
    original_meas = RobotMeasurement(
        timestamp=1700000000.25,
        measurement_type=MeasurementType.ACTUAL_SENSOR,
        joint_names=["j1", "j2"],
        joint_positions=[0.35, -0.75],
        joint_velocities=[0.05, -0.05],
        joint_efforts=[2.1, 1.8],
    )

    ros_dict = measurement_to_ros_joint_state_dict(original_meas, frame_id="base_link")
    assert ros_dict["header"]["frame_id"] == "base_link"
    assert ros_dict["header"]["stamp"]["sec"] == 1700000000
    assert ros_dict["name"] == ["j1", "j2"]

    reconstructed = ros_joint_state_to_measurement(ros_dict)
    assert reconstructed.joint_names == original_meas.joint_names
    assert reconstructed.joint_positions == original_meas.joint_positions
    assert reconstructed.joint_velocities == original_meas.joint_velocities
    assert reconstructed.joint_efforts == original_meas.joint_efforts
    assert math.isclose(reconstructed.timestamp, original_meas.timestamp, rel_tol=1e-6)


# ==============================================================================
# 2. PoseStamped Mapping Tests
# ==============================================================================

class MockPoseStampedMsg:
    def __init__(self, position, orientation, header=None):
        self.header = header or MockHeader()
        self.pose = type("Pose", (), {
            "position": type("Point", (), position)(),
            "orientation": type("Quaternion", (), orientation)(),
        })()


def test_pose_stamped_dict_mapping_valid():
    """Verify conversion of valid PoseStamped dictionary to RobotMeasurement."""
    msg = {
        "header": {
            "stamp": {"sec": 1700000010, "nanosec": 125000000},
            "frame_id": "base_link",
        },
        "pose": {
            "position": {"x": 0.45, "y": -0.12, "z": 0.38},
            "orientation": {"x": 0.0, "y": 0.7071068, "z": 0.0, "w": 0.7071068},
        },
    }

    meas = ros_pose_stamped_to_measurement(msg)

    assert isinstance(meas, RobotMeasurement)
    assert meas.is_valid is True
    # By default, kinematic poses are classified as DERIVED_ESTIMATE
    assert meas.measurement_type == MeasurementType.DERIVED_ESTIMATE
    assert math.isclose(meas.timestamp, 1700000010.125, rel_tol=1e-9)
    assert meas.ee_position == [0.45, -0.12, 0.38]
    assert meas.ee_orientation == [0.0, 0.7071068, 0.0, 0.7071068]
    assert meas.metadata["frame_id"] == "base_link"


def test_pose_stamped_object_mapping_valid():
    """Verify conversion of mock ROS 2 PoseStamped object."""
    mock_msg = MockPoseStampedMsg(
        position={"x": 0.5, "y": 0.2, "z": 0.4},
        orientation={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        header=MockHeader(sec=1690000050, nanosec=0, frame_id="world"),
    )

    meas = ros_pose_stamped_to_measurement(mock_msg, valid_base_frames=["world", "base_link"])
    assert meas.ee_position == [0.5, 0.2, 0.4]
    assert meas.ee_orientation == [0.0, 0.0, 0.0, 1.0]
    assert meas.metadata["frame_id"] == "world"


def test_pose_stamped_bidirectional_roundtrip():
    """Verify RobotMeasurement to ROS PoseStamped dictionary serialization."""
    meas = RobotMeasurement(
        timestamp=1700000005.5,
        measurement_type=MeasurementType.DERIVED_ESTIMATE,
        ee_position=[0.3, -0.1, 0.6],
        ee_orientation=[0.0, 0.0, 0.7071, 0.7071],
    )

    ros_dict = measurement_to_ros_pose_stamped_dict(meas, frame_id="robot_base")
    assert ros_dict["header"]["frame_id"] == "robot_base"
    assert ros_dict["pose"]["position"]["x"] == 0.3
    assert ros_dict["pose"]["orientation"]["z"] == 0.7071

    reconstructed = ros_pose_stamped_to_measurement(ros_dict, expected_frame_id="robot_base")
    assert reconstructed.ee_position == [0.3, -0.1, 0.6]
    assert reconstructed.ee_orientation == [0.0, 0.0, 0.7071, 0.7071]


# ==============================================================================
# 3. Validation Edge Cases & Non-Finite Numbers
# ==============================================================================

def test_joint_state_rejects_empty_joint_names():
    """Empty joint names must fail validation."""
    msg = {
        "header": {"stamp": 100.0, "frame_id": "base_link"},
        "name": [],
        "position": [],
    }
    with pytest.raises(MappingValidationError, match="contains zero joint names"):
        ros_joint_state_to_measurement(msg)


def test_joint_state_rejects_mismatched_dimensions():
    """Mismatched name vs position vs velocity lengths must fail validation."""
    msg = {
        "header": {"stamp": 100.0, "frame_id": "base_link"},
        "name": ["j1", "j2"],
        "position": [0.1],  # 1 position vs 2 names
    }
    with pytest.raises(MappingValidationError, match="Mismatched JointState dimensions"):
        ros_joint_state_to_measurement(msg)

    msg_vel = {
        "header": {"stamp": 100.0, "frame_id": "base_link"},
        "name": ["j1", "j2"],
        "position": [0.1, 0.2],
        "velocity": [0.01],  # 1 velocity vs 2 names
    }
    with pytest.raises(MappingValidationError, match="Mismatched JointState velocities dimension"):
        ros_joint_state_to_measurement(msg_vel)


def test_joint_state_rejects_nan_and_inf():
    """Non-finite positions, velocities, or efforts must fail validation."""
    msg_nan = {
        "header": {"stamp": 100.0, "frame_id": "base_link"},
        "name": ["j1"],
        "position": [float("nan")],
    }
    with pytest.raises(MappingValidationError, match="non-finite position value"):
        ros_joint_state_to_measurement(msg_nan)

    msg_inf = {
        "header": {"stamp": 100.0, "frame_id": "base_link"},
        "name": ["j1"],
        "position": [0.0],
        "velocity": [float("inf")],
    }
    with pytest.raises(MappingValidationError, match="non-finite velocity value"):
        ros_joint_state_to_measurement(msg_inf)


def test_pose_stamped_rejects_nan_coordinates():
    """Pose with NaN or Inf coordinates must fail validation."""
    msg_nan_pos = {
        "header": {"stamp": 100.0, "frame_id": "base_link"},
        "pose": {
            "position": {"x": float("nan"), "y": 0.0, "z": 0.0},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    with pytest.raises(MappingValidationError, match="non-finite coordinate values"):
        ros_pose_stamped_to_measurement(msg_nan_pos)


def test_pose_stamped_rejects_degenerate_quaternion():
    """All-zero quaternion is degenerate and must fail validation."""
    msg_zero_quat = {
        "header": {"stamp": 100.0, "frame_id": "base_link"},
        "pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 0.0},
        },
    }
    with pytest.raises(MappingValidationError, match="degenerate rotation"):
        ros_pose_stamped_to_measurement(msg_zero_quat)


# ==============================================================================
# 4. Coordinate Frame Validation (No Silent Assumption of Base Frame)
# ==============================================================================

def test_joint_state_rejects_unknown_frame_id():
    """Unrecognized coordinate frames must not be silently treated as the base frame."""
    msg = {
        "header": {"stamp": 100.0, "frame_id": "wrist_camera_optical_frame"},
        "name": ["j1"],
        "position": [0.0],
    }
    with pytest.raises(MappingValidationError, match="Unrecognized coordinate frame 'wrist_camera_optical_frame'"):
        ros_joint_state_to_measurement(msg)


def test_pose_stamped_rejects_unknown_or_mismatched_frame_id():
    """PoseStamped must reject unknown frame or frames differing from expected_frame_id."""
    msg_unknown = {
        "header": {"stamp": 100.0, "frame_id": "untracked_sensor_frame"},
        "pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    with pytest.raises(MappingValidationError, match="Unrecognized coordinate frame"):
        ros_pose_stamped_to_measurement(msg_unknown)

    # Frame mismatch against expected frame
    msg_base = {
        "header": {"stamp": 100.0, "frame_id": "base_link"},
        "pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    with pytest.raises(MappingValidationError, match="does not match expected target frame 'tool0'"):
        ros_pose_stamped_to_measurement(msg_base, expected_frame_id="tool0")


def test_pose_stamped_rejects_missing_frame_id():
    """PoseStamped must explicitly specify frame_id; empty frame_id is rejected."""
    msg_no_frame = {
        "header": {"stamp": 100.0, "frame_id": ""},
        "pose": {
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        },
    }
    with pytest.raises(MappingValidationError, match="missing required 'frame_id'"):
        ros_pose_stamped_to_measurement(msg_no_frame)


# ==============================================================================
# 5. Genuine MCAP Reading & Writing
# ==============================================================================

def test_genuine_mcap_roundtrip_with_telemetry_records(tmp_path):
    """Write genuine MCAP container with official 'mcap' library and read back with load_telemetry."""
    mcap_path = tmp_path / "valid_run.mcap"

    input_records = [
        TelemetryRecord(
            timestamp=1700000000.0 + i * 0.1,
            ee_position=[0.4 + i * 0.01, 0.0, 0.2],
            joint_positions=[0.1 * i, -0.2 * i, 0.05 * i],
        )
        for i in range(5)
    ]
    input_telemetry = RecordedTelemetry(
        session_id="genuine_mcap_test_session",
        provenance=TelemetryProvenance(
            data_origin="recorded_offline",
            source_id="mcap_test_run",
            recording_format="mcap",
            is_synthetic=True,
        ),
        records=input_records,
    )

    # 1. Write genuine MCAP file
    written_path = write_mcap_telemetry(mcap_path, input_telemetry)
    assert Path(written_path).exists()
    assert Path(written_path).stat().st_size > 0

    # 2. Ingest with load_telemetry
    loaded = load_telemetry(mcap_path)

    assert isinstance(loaded, RecordedTelemetry)
    assert len(loaded.records) == 5
    assert loaded.provenance.recording_format == "mcap"
    assert loaded.provenance.data_origin == "recorded_offline"
    assert loaded.provenance.is_synthetic is True

    # 3. Verify decoded records match input values
    for i in range(5):
        assert math.isclose(loaded.records[i].timestamp, input_records[i].timestamp, rel_tol=1e-6)
        assert loaded.records[i].ee_position == input_records[i].ee_position
        assert loaded.records[i].joint_positions == input_records[i].joint_positions


def test_genuine_mcap_reading_ros_joint_state_messages(tmp_path):
    """Write an MCAP containing sensor_msgs/JointState JSON payloads and decode into TelemetryRecords."""
    from mcap.writer import Writer
    import json

    mcap_path = tmp_path / "joint_states.mcap"
    with open(mcap_path, "wb") as f:
        writer = Writer(f)
        writer.start()
        schema_id = writer.register_schema(name="sensor_msgs/msg/JointState", encoding="jsonschema", data=b"{}")
        channel_id = writer.register_channel(topic="/joint_states", message_encoding="json", schema_id=schema_id)

        for i in range(3):
            js_msg = {
                "header": {"stamp": {"sec": 1700000000 + i, "nanosec": 0}, "frame_id": "base_link"},
                "name": ["j1", "j2"],
                "position": [0.1 * i, 0.2 * i],
            }
            log_time_ns = int((1700000000 + i) * 1e9)
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time_ns,
                data=json.dumps(js_msg).encode("utf-8"),
                publish_time=log_time_ns,
            )
        writer.finish()

    loaded = load_telemetry(mcap_path)
    assert len(loaded.records) == 3
    assert loaded.records[0].joint_positions == [0.0, 0.0]
    assert loaded.records[1].joint_positions == [0.1, 0.2]
    assert loaded.records[2].joint_positions == [0.2, 0.4]


# ==============================================================================
# 6. rosbag2 Fails Closed Safely
# ==============================================================================

def test_rosbag2_fails_closed_with_unsupported_format(tmp_path):
    """Attempting to parse .db3 SQLite rosbag without Linux/ROS 2 runtime raises UnsupportedFormatError."""
    db3_file = tmp_path / "rosbag2_test.db3"
    db3_file.write_bytes(b"SQLite format 3\x00")

    adapter = Rosbag2TelemetryAdapter()
    assert adapter.can_read(db3_file) is True

    with pytest.raises(UnsupportedFormatError, match="rosbag2.*parsing requires"):
        adapter.read(db3_file)
