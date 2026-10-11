"""Computer vision and video trajectory extraction package for Robot CI.

Extracts visible 2D end-effector trajectories from prerecorded robot videos (MP4, MOV),
recording image-space coordinates, normalized screen coordinates, and tracking confidence.
"""

from .models import (
    DetectionStatus,
    CoordinateSpace,
    UNCALIBRATED_CAMERA_DISCLAIMER,
    VisionError,
    OpenCVUnavailableError,
    InvalidVideoError,
    UnsupportedVideoFormatError,
    TrackingError,
    VideoFrameObservation,
    VideoMetadata,
    TrackerConfig,
    VideoTrajectoryResult,
)
from .tracker import (
    is_cv2_available,
    COLOR_PRESETS,
    MarkerTracker,
)
from .extractor import (
    SUPPORTED_VIDEO_FORMATS,
    extract_trajectory_from_video,
    create_synthetic_test_video,
)
from .projection import (
    CameraCalibration,
    CalibrationError,
    create_synthetic_camera_calibration,
)
from .sim_extractor import (
    SimulatedTrajectoryResult,
    extract_simulated_trajectory,
)
from .comparison import (
    ComparisonMode,
    VideoSimComparisonReport,
    compare_video_to_simulation,
    resample_2d_trajectory,
    compute_2d_path_length,
    compute_2d_dtw,
    procrustes_align_2d,
)

__all__ = [
    "DetectionStatus",
    "CoordinateSpace",
    "UNCALIBRATED_CAMERA_DISCLAIMER",
    "VisionError",
    "OpenCVUnavailableError",
    "InvalidVideoError",
    "UnsupportedVideoFormatError",
    "TrackingError",
    "VideoFrameObservation",
    "VideoMetadata",
    "TrackerConfig",
    "VideoTrajectoryResult",
    "is_cv2_available",
    "COLOR_PRESETS",
    "MarkerTracker",
    "SUPPORTED_VIDEO_FORMATS",
    "extract_trajectory_from_video",
    "create_synthetic_test_video",
    "CameraCalibration",
    "CalibrationError",
    "create_synthetic_camera_calibration",
    "SimulatedTrajectoryResult",
    "extract_simulated_trajectory",
    "ComparisonMode",
    "VideoSimComparisonReport",
    "compare_video_to_simulation",
    "resample_2d_trajectory",
    "compute_2d_path_length",
    "compute_2d_dtw",
    "procrustes_align_2d",
]
