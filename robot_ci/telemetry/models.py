"""Normalized telemetry data models and schema for Robot CI.

Provides hardware-independent telemetry models for offline analysis.
Guarantees explicit provenance and prevents offline recorded data from being
misidentified as live simulator or hardware execution.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Union
import math
import numpy as np
import pandas as pd


class TelemetryError(Exception):
    """Base class for telemetry errors."""
    pass


class TelemetryValidationError(TelemetryError):
    """Raised when recorded telemetry fails schema or sanity validation."""
    pass


class TelemetryParseError(TelemetryError):
    """Raised when parsing telemetry log files fails."""
    pass


class UnsupportedFormatError(TelemetryError):
    """Raised when an requested telemetry format is unsupported or missing dependencies."""
    pass


@dataclass
class TelemetryProvenance:
    """Explicit provenance metadata for recorded robot telemetry.
    
    Guarantees that telemetry origin is recorded, distinguishing offline logs
    from live execution.
    """
    data_origin: str = "recorded_offline"  # "recorded_offline" or "synthetic", NEVER "live_hardware"
    source_id: str = "offline_robot_log"
    recording_format: str = "json"
    is_synthetic: bool = True
    recorded_at: Optional[str] = None
    time_basis: str = "UTC_EPOCH_SECONDS"
    notes: Optional[str] = None

    def __post_init__(self):
        forbidden_origins = {"live_hardware", "hardware_live", "live_execution", "simulator_live"}
        if self.data_origin.lower() in forbidden_origins:
            raise TelemetryValidationError(
                f"Invalid data_origin '{self.data_origin}'. Offline telemetry must never claim to be live execution."
            )
        if self.recorded_at is None:
            self.recorded_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_origin": self.data_origin,
            "source_id": self.source_id,
            "recording_format": self.recording_format,
            "is_synthetic": self.is_synthetic,
            "recorded_at": self.recorded_at,
            "time_basis": self.time_basis,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryProvenance":
        return cls(
            data_origin=data.get("data_origin", "recorded_offline"),
            source_id=data.get("source_id", "offline_robot_log"),
            recording_format=data.get("recording_format", "json"),
            is_synthetic=data.get("is_synthetic", True),
            recorded_at=data.get("recorded_at"),
            time_basis=data.get("time_basis", "UTC_EPOCH_SECONDS"),
            notes=data.get("notes"),
        )


@dataclass
class TelemetryRecord:
    """A single normalized telemetry sample.
    
    Units:
    - timestamp: Seconds (float epoch or relative)
    - timestamp_iso: ISO 8601 UTC string when available
    - ee_position: End-effector 3D Cartesian coordinates in meters [x, y, z]
    - joint_positions: Joint angles in radians (list or dict)
    - gripper_state: Gripper open/close status or aperture in meters
    - action: Commanded action if recorded
    - events: Explicit events (e.g. contact, success, alarm) observed in data
    - raw_metadata: Unparsed driver or channel metadata
    """
    timestamp: float
    timestamp_iso: Optional[str] = None
    ee_position: Optional[List[float]] = None
    joint_positions: Optional[Union[List[float], Dict[str, float]]] = None
    gripper_state: Optional[float] = None
    action: Optional[List[float]] = None
    events: Optional[Dict[str, Any]] = None
    raw_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"timestamp": self.timestamp}
        if self.timestamp_iso is not None:
            d["timestamp_iso"] = self.timestamp_iso
        if self.ee_position is not None:
            d["ee_position"] = [float(v) for v in self.ee_position]
        if self.joint_positions is not None:
            if isinstance(self.joint_positions, dict):
                d["joint_positions"] = {k: float(v) for k, v in self.joint_positions.items()}
            else:
                d["joint_positions"] = [float(v) for v in self.joint_positions]
        if self.gripper_state is not None:
            d["gripper_state"] = float(self.gripper_state)
        if self.action is not None:
            d["action"] = [float(v) for v in self.action]
        if self.events is not None:
            d["events"] = self.events
        if self.raw_metadata is not None:
            d["raw_metadata"] = self.raw_metadata
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelemetryRecord":
        if "timestamp" not in data:
            raise TelemetryValidationError("Record missing required 'timestamp' field.")
        return cls(
            timestamp=float(data["timestamp"]),
            timestamp_iso=data.get("timestamp_iso"),
            ee_position=data.get("ee_position"),
            joint_positions=data.get("joint_positions"),
            gripper_state=data.get("gripper_state"),
            action=data.get("action"),
            events=data.get("events"),
            raw_metadata=data.get("raw_metadata"),
        )


@dataclass
class TelemetryValidationReport:
    """Result of validating a recorded telemetry session."""
    is_valid: bool
    record_count: int
    duration_seconds: float
    has_ee_positions: bool
    has_joint_positions: bool
    has_actions: bool
    has_events: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "record_count": self.record_count,
            "duration_seconds": self.duration_seconds,
            "has_ee_positions": self.has_ee_positions,
            "has_joint_positions": self.has_joint_positions,
            "has_actions": self.has_actions,
            "has_events": self.has_events,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class RecordedTelemetry:
    """Normalized recorded telemetry session.
    
    Represents an offline dataset captured from robot operation or simulation logging.
    Maintains clean separation from live SimulatorBackend.
    """
    session_id: str
    provenance: TelemetryProvenance
    records: List[TelemetryRecord] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.records)

    @property
    def start_time(self) -> Optional[float]:
        return self.records[0].timestamp if self.records else None

    @property
    def end_time(self) -> Optional[float]:
        return self.records[-1].timestamp if self.records else None

    @property
    def duration_seconds(self) -> float:
        if len(self.records) < 2:
            return 0.0
        return float(self.records[-1].timestamp - self.records[0].timestamp)

    def validate(self, raise_on_error: bool = False) -> TelemetryValidationReport:
        """Validate recorded telemetry against schema and sanity rules."""
        errors: List[str] = []
        warnings: List[str] = []

        if not self.records:
            errors.append("Telemetry session contains zero records (empty file/session).")

        # Verify provenance
        if self.provenance.data_origin in {"live_hardware", "simulator_live"}:
            errors.append(f"Illegal provenance data_origin: {self.provenance.data_origin}")

        prev_ts = None
        has_ee = False
        has_joints = False
        has_actions = False
        has_events = False
        duplicate_ts_count = 0

        for i, rec in enumerate(self.records):
            # Timestamp check
            if not isinstance(rec.timestamp, (int, float)) or math.isnan(rec.timestamp) or math.isinf(rec.timestamp):
                errors.append(f"Record {i}: Invalid numeric timestamp '{rec.timestamp}'.")
            elif rec.timestamp < 0:
                warnings.append(f"Record {i}: Negative timestamp value {rec.timestamp}.")

            if prev_ts is not None and isinstance(rec.timestamp, (int, float)):
                if rec.timestamp < prev_ts:
                    errors.append(
                        f"Record {i}: Timestamp decreased from {prev_ts} to {rec.timestamp} (non-monotonic ordering)."
                    )
                elif rec.timestamp == prev_ts:
                    duplicate_ts_count += 1
            prev_ts = rec.timestamp

            # End-effector positions check
            if rec.ee_position is not None:
                has_ee = True
                if not isinstance(rec.ee_position, (list, tuple, np.ndarray)) or len(rec.ee_position) != 3:
                    errors.append(f"Record {i}: ee_position must be 3D Cartesian coordinates [x, y, z]. Got {rec.ee_position}")
                else:
                    for dim_idx, val in enumerate(rec.ee_position):
                        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                            errors.append(f"Record {i}: ee_position[{dim_idx}] contains non-finite value '{val}'.")

            # Joint positions check
            if rec.joint_positions is not None:
                has_joints = True
                if isinstance(rec.joint_positions, dict):
                    for k, val in rec.joint_positions.items():
                        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                            errors.append(f"Record {i}: joint '{k}' has non-finite value '{val}'.")
                elif isinstance(rec.joint_positions, (list, tuple, np.ndarray)):
                    for j_idx, val in enumerate(rec.joint_positions):
                        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                            errors.append(f"Record {i}: joint[{j_idx}] has non-finite value '{val}'.")
                else:
                    errors.append(f"Record {i}: joint_positions must be list or dict.")

            # Action check
            if rec.action is not None:
                has_actions = True
                if not isinstance(rec.action, (list, tuple, np.ndarray)):
                    errors.append(f"Record {i}: action must be a list/array.")
                else:
                    for a_idx, val in enumerate(rec.action):
                        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                            errors.append(f"Record {i}: action[{a_idx}] contains non-finite value '{val}'.")

            # Events check
            if rec.events is not None:
                has_events = True

        if duplicate_ts_count > 0:
            warnings.append(f"Detected {duplicate_ts_count} duplicate timestamp records.")

        is_valid = len(errors) == 0
        report = TelemetryValidationReport(
            is_valid=is_valid,
            record_count=len(self.records),
            duration_seconds=self.duration_seconds,
            has_ee_positions=has_ee,
            has_joint_positions=has_joints,
            has_actions=has_actions,
            has_events=has_events,
            errors=errors,
            warnings=warnings,
        )

        if not is_valid and raise_on_error:
            raise TelemetryValidationError("; ".join(errors))

        return report

    def get_ee_trajectory(self) -> np.ndarray:
        """Extract Nx3 end-effector trajectory array in meters.
        
        Raises TelemetryValidationError if ee_position is missing or incomplete.
        """
        if not self.records:
            return np.zeros((0, 3), dtype=np.float32)

        positions = []
        for i, rec in enumerate(self.records):
            if rec.ee_position is None:
                raise TelemetryValidationError(f"Record {i} does not contain ee_position data.")
            if len(rec.ee_position) != 3:
                raise TelemetryValidationError(f"Record {i} ee_position is not 3D: {rec.ee_position}")
            positions.append([float(rec.ee_position[0]), float(rec.ee_position[1]), float(rec.ee_position[2])])

        return np.array(positions, dtype=np.float64)

    def get_timestamps(self) -> np.ndarray:
        """Extract 1D array of sample timestamps in seconds."""
        return np.array([rec.timestamp for rec in self.records], dtype=np.float64)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert telemetry records to a pandas DataFrame for inspection."""
        rows = []
        for i, rec in enumerate(self.records):
            row: Dict[str, Any] = {
                "step": i,
                "timestamp": rec.timestamp,
            }
            if rec.timestamp_iso:
                row["timestamp_iso"] = rec.timestamp_iso
            if rec.ee_position and len(rec.ee_position) == 3:
                row["ee_x"] = rec.ee_position[0]
                row["ee_y"] = rec.ee_position[1]
                row["ee_z"] = rec.ee_position[2]
            if rec.gripper_state is not None:
                row["gripper_state"] = rec.gripper_state
            if rec.action is not None:
                row["action_len"] = len(rec.action)
            rows.append(row)
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "provenance": self.provenance.to_dict(),
            "records": [r.to_dict() for r in self.records],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RecordedTelemetry":
        if not isinstance(data, dict):
            raise TelemetryValidationError("Telemetry data must be a dictionary.")
        if "records" not in data:
            raise TelemetryValidationError("Telemetry payload missing 'records' field.")

        prov_data = data.get("provenance", {})
        provenance = TelemetryProvenance.from_dict(prov_data) if isinstance(prov_data, dict) else TelemetryProvenance()

        records = [TelemetryRecord.from_dict(r) for r in data["records"]]
        return cls(
            session_id=data.get("session_id", "unnamed_session"),
            provenance=provenance,
            records=records,
            metadata=data.get("metadata", {}),
        )
