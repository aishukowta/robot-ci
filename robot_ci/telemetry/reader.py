"""Offline telemetry ingestion adapters and readers for Robot CI.

Supports format-neutral ingestion of offline robot telemetry logs.
Prioritizes deterministic JSON and JSONL formats.
Provides extensible adapter architecture with optional hooks for MCAP and rosbag2.
"""

from abc import ABC, abstractmethod
import json
import os
from pathlib import Path
from typing import Dict, Type, Union, Optional, List, Any

from .models import (
    RecordedTelemetry,
    TelemetryRecord,
    TelemetryProvenance,
    TelemetryParseError,
    TelemetryValidationError,
    UnsupportedFormatError,
)


class TelemetryReaderAdapter(ABC):
    """Abstract base class for telemetry ingestion adapters."""

    @abstractmethod
    def can_read(self, path: Path) -> bool:
        """Return True if this adapter supports the given file format/extension."""
        ...

    @abstractmethod
    def read(self, path: Union[str, Path]) -> RecordedTelemetry:
        """Parse file into a normalized RecordedTelemetry instance."""
        ...


class JsonTelemetryAdapter(TelemetryReaderAdapter):
    """Ingests telemetry from standard JSON files."""

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() == ".json"

    def read(self, path: Union[str, Path]) -> RecordedTelemetry:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Telemetry log file not found: {file_path}")

        try:
            content = file_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            raise TelemetryParseError(f"Failed to read file {file_path}: {e}")

        if not content:
            raise TelemetryValidationError(f"Telemetry file is empty: {file_path}")

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            raise TelemetryParseError(f"Malformed JSON in {file_path}: {e}")

        if isinstance(data, dict):
            # Check if it has 'records' key
            if "records" in data:
                return RecordedTelemetry.from_dict(data)
            else:
                raise TelemetryValidationError(f"JSON object in {file_path} missing 'records' key.")
        elif isinstance(data, list):
            # Array of records
            records = [TelemetryRecord.from_dict(r) for r in data]
            prov = TelemetryProvenance(
                data_origin="recorded_offline",
                source_id=file_path.stem,
                recording_format="json",
                is_synthetic=True,
            )
            return RecordedTelemetry(
                session_id=file_path.stem,
                provenance=prov,
                records=records,
                metadata={"source_file": str(file_path)},
            )
        else:
            raise TelemetryValidationError(f"Unexpected JSON root type {type(data)} in {file_path}")


class JsonlTelemetryAdapter(TelemetryReaderAdapter):
    """Ingests line-delimited JSON (JSONL) telemetry logs."""

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() in {".jsonl", ".ndjson"}

    def read(self, path: Union[str, Path]) -> RecordedTelemetry:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Telemetry log file not found: {file_path}")

        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except Exception as e:
            raise TelemetryParseError(f"Failed to read file {file_path}: {e}")

        non_empty_lines = [line.strip() for line in lines if line.strip()]
        if not non_empty_lines:
            raise TelemetryValidationError(f"JSONL telemetry file is empty: {file_path}")

        records: List[TelemetryRecord] = []
        provenance = TelemetryProvenance(
            data_origin="recorded_offline",
            source_id=file_path.stem,
            recording_format="jsonl",
            is_synthetic=True,
        )
        metadata: Dict[str, Any] = {"source_file": str(file_path)}

        for line_num, line in enumerate(non_empty_lines, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise TelemetryParseError(f"Malformed JSON on line {line_num} of {file_path}: {e}")

            if not isinstance(item, dict):
                raise TelemetryValidationError(
                    f"Line {line_num} in {file_path} must be a JSON object, got {type(item)}"
                )

            # Check if this line is a metadata/provenance header
            if "_header" in item or "session_id" in item and "records" not in item and "timestamp" not in item:
                if "provenance" in item and isinstance(item["provenance"], dict):
                    provenance = TelemetryProvenance.from_dict(item["provenance"])
                if "metadata" in item and isinstance(item["metadata"], dict):
                    metadata.update(item["metadata"])
                continue

            records.append(TelemetryRecord.from_dict(item))

        if not records:
            raise TelemetryValidationError(f"JSONL file contained headers but zero telemetry records: {file_path}")

        return RecordedTelemetry(
            session_id=file_path.stem,
            provenance=provenance,
            records=records,
            metadata=metadata,
        )


class McapTelemetryAdapter(TelemetryReaderAdapter):
    """Adapter for MCAP binary container format.
    
    Decodes genuine MCAP files using the official 'mcap' package.
    Parses robot telemetry and ROS 2 JSON message streams into normalized RecordedTelemetry.
    """

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() == ".mcap"

    def read(self, path: Union[str, Path]) -> RecordedTelemetry:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"MCAP telemetry log file not found: {file_path}")

        try:
            from mcap.reader import make_reader
        except ImportError:
            raise UnsupportedFormatError(
                "MCAP telemetry parsing requires the 'mcap' package ('pip install mcap'). "
                "The current environment does not have 'mcap' installed. "
                "Use JSON or JSONL format, or install 'mcap' to enable this adapter."
            )

        try:
            with open(file_path, "rb") as f:
                reader = make_reader(f)
                records: List[TelemetryRecord] = []
                channels_found = []
                metadata: Dict[str, Any] = {"source_file": str(file_path)}

                for schema, channel, message in reader.iter_messages():
                    if channel.topic not in channels_found:
                        channels_found.append(channel.topic)

                    ts = float(message.log_time) * 1e-9

                    encoding = str(getattr(channel, "message_encoding", "") or "").lower()
                    if encoding in ("cdr", "ros1", "protobuf", "flatbuffer"):
                        raise UnsupportedFormatError(
                            f"MCAP channel '{channel.topic}' uses '{encoding}' message encoding. Only JSON-encoded "
                            "MCAP messages are supported. Real 'ros2 bag record -s mcap' files are CDR-encoded and "
                            "need a ROS 2 decoder (e.g. mcap-ros2-support), which is not implemented."
                        )

                    payload = None
                    try:
                        payload = json.loads(message.data.decode("utf-8"))
                    except Exception:
                        pass

                    if isinstance(payload, dict):
                        if "timestamp" in payload or "ee_position" in payload or "joint_positions" in payload:
                            if "timestamp" not in payload:
                                payload["timestamp"] = ts
                            records.append(TelemetryRecord.from_dict(payload))
                        elif "position" in payload and "name" in payload:
                            from robot_ci.transport.ros2_mapping import ros_joint_state_to_measurement
                            if "header" not in payload:
                                payload["header"] = {"stamp": ts}
                            else:
                                if "stamp" not in payload["header"]:
                                    payload["header"]["stamp"] = ts
                                # frame_id is left missing rather than assumed
                            measurement = ros_joint_state_to_measurement(payload)
                            records.append(TelemetryRecord(
                                timestamp=measurement.timestamp,
                                joint_positions=measurement.joint_positions,
                                raw_metadata={"topic": channel.topic, "schema": schema.name if schema else None},
                            ))
                        elif "pose" in payload:
                            from robot_ci.transport.ros2_mapping import ros_pose_stamped_to_measurement
                            if "header" not in payload:
                                payload["header"] = {"stamp": ts}
                            else:
                                if "stamp" not in payload["header"]:
                                    payload["header"]["stamp"] = ts
                                # frame_id is left missing rather than assumed
                            measurement = ros_pose_stamped_to_measurement(payload)
                            records.append(TelemetryRecord(
                                timestamp=measurement.timestamp,
                                ee_position=measurement.ee_position,
                                raw_metadata={"topic": channel.topic, "schema": schema.name if schema else None},
                            ))
                        else:
                            records.append(TelemetryRecord(
                                timestamp=ts,
                                raw_metadata={"topic": channel.topic, "data": payload},
                            ))

        except Exception as e:
            if isinstance(e, (TelemetryValidationError, TelemetryParseError, UnsupportedFormatError)):
                raise
            raise TelemetryParseError(f"Failed to parse MCAP file {file_path}: {e}")

        if not records:
            raise TelemetryValidationError(f"MCAP file contains zero telemetry records or supported channels: {file_path}")

        provenance = TelemetryProvenance(
            data_origin="recorded_offline",
            source_id=file_path.stem,
            recording_format="mcap",
            is_synthetic=True,
            time_basis="UTC_EPOCH_SECONDS",
            notes=f"Ingested from MCAP container. Channels: {channels_found}",
        )

        return RecordedTelemetry(
            session_id=file_path.stem,
            provenance=provenance,
            records=records,
            metadata=metadata,
        )


def write_mcap_telemetry(path: Union[str, Path], telemetry: RecordedTelemetry) -> str:
    """Write RecordedTelemetry instance to an MCAP container file using official mcap Writer."""
    try:
        from mcap.writer import Writer
    except ImportError:
        raise UnsupportedFormatError("Writing MCAP files requires the 'mcap' package ('pip install mcap').")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        writer = Writer(f)
        writer.start()
        schema_id = writer.register_schema(
            name="robot_ci.TelemetryRecord",
            encoding="jsonschema",
            data=b'{"type": "object"}',
        )
        channel_id = writer.register_channel(
            topic="/robot_ci/telemetry",
            message_encoding="json",
            schema_id=schema_id,
        )
        for rec in telemetry.records:
            log_time_ns = int(rec.timestamp * 1e9)
            data_bytes = json.dumps(rec.to_dict()).encode("utf-8")
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time_ns,
                data=data_bytes,
                publish_time=log_time_ns,
            )
        writer.finish()
    return str(out_path)


class Rosbag2TelemetryAdapter(TelemetryReaderAdapter):
    """Adapter for ROS 2 rosbag2 storage (SQLite3 / .db3).
    
    Does NOT require a live ROS 2 installation; uses the optional 'rosbags' library
    when installed. Fails closed with descriptive guidance if unavailable.
    """

    def can_read(self, path: Path) -> bool:
        return path.suffix.lower() in {".db3", ".bag"} or (path.is_dir() and (path / "metadata.yaml").exists())

    def read(self, path: Union[str, Path]) -> RecordedTelemetry:
        try:
            import rosbags  # type: ignore # noqa: F401
        except ImportError:
            raise UnsupportedFormatError(
                "rosbag2 (.db3/SQLite) parsing requires the offline 'rosbags' package ('pip install rosbags') "
                "or a native ROS 2 runtime. The current Windows environment does not support rosbag2 SQLite decoding. "
                "Use the MCAP (.mcap) format or JSON/JSONL format."
            )

        raise UnsupportedFormatError(
            "rosbags package detected but ROS 2 message definition decoding requires a compatible Linux/ROS 2 environment."
        )


class TelemetryIngestionRegistry:
    """Registry of format-specific telemetry adapters."""

    def __init__(self):
        self._adapters: List[TelemetryReaderAdapter] = [
            JsonTelemetryAdapter(),
            JsonlTelemetryAdapter(),
            McapTelemetryAdapter(),
            Rosbag2TelemetryAdapter(),
        ]

    def register(self, adapter: TelemetryReaderAdapter) -> None:
        """Register a new adapter at highest priority."""
        self._adapters.insert(0, adapter)

    def find_adapter(self, path: Path) -> TelemetryReaderAdapter:
        for adapter in self._adapters:
            if adapter.can_read(path):
                return adapter
        raise UnsupportedFormatError(
            f"No compatible telemetry adapter found for file '{path.name}'. "
            f"Supported extensions: .json, .jsonl, .ndjson (optional: .mcap, .db3 with extra dependencies)."
        )

    def load(self, path: Union[str, Path], validate: bool = True) -> RecordedTelemetry:
        file_path = Path(path)
        adapter = self.find_adapter(file_path)
        telemetry = adapter.read(file_path)
        if validate:
            telemetry.validate(raise_on_error=True)
        return telemetry


# Global default ingestion instance
_default_registry = TelemetryIngestionRegistry()


def load_telemetry(path: Union[str, Path], validate: bool = True) -> RecordedTelemetry:
    """Format-neutral entry point to load and validate offline robot telemetry.
    
    Args:
        path: Path to .json or .jsonl telemetry file.
        validate: Whether to run sanity and schema checks upon loading.
    
    Returns:
        RecordedTelemetry instance.
    """
    return _default_registry.load(path, validate=validate)
