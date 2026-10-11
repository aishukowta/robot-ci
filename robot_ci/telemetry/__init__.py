"""Telemetry logging and offline replay package for Robot CI."""

from .models import (
    TelemetryError,
    TelemetryValidationError,
    TelemetryParseError,
    UnsupportedFormatError,
    TelemetryProvenance,
    TelemetryRecord,
    TelemetryValidationReport,
    RecordedTelemetry,
)
from .reader import (
    TelemetryReaderAdapter,
    JsonTelemetryAdapter,
    JsonlTelemetryAdapter,
    McapTelemetryAdapter,
    Rosbag2TelemetryAdapter,
    TelemetryIngestionRegistry,
    load_telemetry,
    write_mcap_telemetry,
)
from .replay import OfflineTelemetrySession
from .workflow import (
    OfflineAnalysisReport,
    OfflineReportStore,
    OfflineTelemetryWorkflow,
)
from .comparator import (
    TrajectoryComparator,
    TrajectoryComparisonResult,
    compute_dtw,
    resample_trajectory,
    TRAJECTORY_SAFETY_DISCLAIMER,
)

__all__ = [
    "TelemetryError",
    "TelemetryValidationError",
    "TelemetryParseError",
    "UnsupportedFormatError",
    "TelemetryProvenance",
    "TelemetryRecord",
    "TelemetryValidationReport",
    "RecordedTelemetry",
    "TelemetryReaderAdapter",
    "JsonTelemetryAdapter",
    "JsonlTelemetryAdapter",
    "McapTelemetryAdapter",
    "Rosbag2TelemetryAdapter",
    "TelemetryIngestionRegistry",
    "load_telemetry",
    "OfflineTelemetrySession",
    "OfflineAnalysisReport",
    "OfflineReportStore",
    "OfflineTelemetryWorkflow",
    "TrajectoryComparator",
    "TrajectoryComparisonResult",
    "compute_dtw",
    "resample_trajectory",
    "TRAJECTORY_SAFETY_DISCLAIMER",
]
