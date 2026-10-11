"""Deterministic unit and integration tests for video-to-trajectory extraction (Phase 1).

Verifies:
1. Video loading, metadata inspection, and format validation.
2. High-contrast / color-marker segmentation across standard color presets.
3. Timestamp extraction and frame index ordering.
4. Handling of failed marker detection and occlusions (never silently interpolated).
5. Safe rejection of invalid video files and unsupported container formats.
6. Export to JSON, CSV, and Robot CI RecordedTelemetry adapter format.
7. Uncalibrated 2D camera coordinate labeling and explicit disclaimers.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

from robot_ci.vision.models import (
    DetectionStatus,
    CoordinateSpace,
    UNCALIBRATED_CAMERA_DISCLAIMER,
    TrackerConfig,
    VideoTrajectoryResult,
    VideoFrameObservation,
    UnsupportedVideoFormatError,
    InvalidVideoError,
    OpenCVUnavailableError,
)
from robot_ci.vision.tracker import (
    MarkerTracker,
    is_cv2_available,
    COLOR_PRESETS,
)
from robot_ci.vision.extractor import (
    extract_trajectory_from_video,
    create_synthetic_test_video,
    SUPPORTED_VIDEO_FORMATS,
)


@pytest.fixture(scope="module")
def synthetic_circular_video(tmp_path_factory):
    """Generate a deterministic synthetic circular trajectory test video."""
    tmp_dir = tmp_path_factory.mktemp("video_test")
    vid_path = tmp_dir / "synthetic_circular.mp4"
    actual_path = create_synthetic_test_video(
        output_path=vid_path,
        width=320,
        height=240,
        fps=30.0,
        num_frames=30,
        marker_color="red",
        trajectory_type="circular",
        occlude_frames=[10, 11, 12],  # Intentionally occlude 3 frames
    )
    return Path(actual_path)


# ==============================================================================
# 1. Tracker Unit Tests (Single Frame Segmentation)
# ==============================================================================

def test_tracker_presets_initialization():
    """Verify built-in color presets establish proper HSV boundaries."""
    assert "red" in COLOR_PRESETS
    assert "green" in COLOR_PRESETS
    assert "blue" in COLOR_PRESETS

    tracker_red = MarkerTracker(TrackerConfig(color_preset="red"))
    assert tracker_red.lower1 is not None
    assert tracker_red.lower2 is not None  # Red hue wrap-around

    tracker_green = MarkerTracker(TrackerConfig(color_preset="green"))
    assert tracker_green.lower1 is not None


def test_tracker_detects_clean_marker_in_frame():
    """Verify tracker detects a bright colored circle on dark background."""
    h, w = 240, 320
    # Dark gray background
    frame = np.full((h, w, 3), 30, dtype=np.uint8)

    # Draw bright red circle at (160, 120) with radius 10
    import cv2
    cv2.circle(frame, (160, 120), 10, (0, 0, 240), -1)

    tracker = MarkerTracker(TrackerConfig(color_preset="red", min_contour_area=10.0))
    obs = tracker.track_frame(frame, frame_index=0, timestamp_seconds=0.0)

    assert obs.detection_status == DetectionStatus.DETECTED
    assert obs.confidence > 0.5
    assert obs.image_coordinates is not None
    # Centroid should be within 1.5 pixels of (160, 120)
    assert abs(obs.image_coordinates[0] - 160.0) < 1.5
    assert abs(obs.image_coordinates[1] - 120.0) < 1.5

    # Check normalized coordinates
    assert obs.normalized_coordinates is not None
    assert abs(obs.normalized_coordinates[0] - 0.5) < 0.01
    assert abs(obs.normalized_coordinates[1] - 0.5) < 0.01


def test_tracker_reports_lost_when_marker_absent():
    """When no marker is present, tracker must report LOST with zero confidence."""
    h, w = 240, 320
    # Empty black frame
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    tracker = MarkerTracker(TrackerConfig(color_preset="red"))
    obs = tracker.track_frame(frame, frame_index=5, timestamp_seconds=0.166)

    assert obs.detection_status == DetectionStatus.LOST
    assert obs.confidence == 0.0
    assert obs.image_coordinates is None
    assert obs.normalized_coordinates is None


def test_tracker_manual_roi_constraint():
    """When an ROI is provided, marker outside the ROI must not be detected."""
    h, w = 240, 320
    frame = np.full((h, w, 3), 30, dtype=np.uint8)
    import cv2
    # Place marker at (50, 50)
    cv2.circle(frame, (50, 50), 10, (0, 0, 240), -1)

    # Restrict ROI to right half of the image: x in [150..300]
    config = TrackerConfig(color_preset="red", roi=(150, 0, 150, 240))
    tracker = MarkerTracker(config)
    obs = tracker.track_frame(frame, frame_index=0, timestamp_seconds=0.0)

    # Marker at (50, 50) is outside ROI, so must be reported LOST
    assert obs.detection_status == DetectionStatus.LOST


# ==============================================================================
# 2. Video Extraction Pipeline Integration Tests
# ==============================================================================

def test_extract_trajectory_from_synthetic_video(synthetic_circular_video):
    """Verify full video ingestion, tracking, and detection stats on synthetic video."""
    cfg = TrackerConfig(color_preset="red", min_contour_area=15.0)
    result = extract_trajectory_from_video(
        video_path=synthetic_circular_video,
        config=cfg,
        is_synthetic=True,
        notes="Synthetic test run",
    )

    assert isinstance(result, VideoTrajectoryResult)
    assert result.is_synthetic is True
    assert result.total_frames == 30
    assert result.video_metadata.width == 320
    assert result.video_metadata.height == 240
    assert result.video_metadata.fps == pytest.approx(30.0, rel=1e-2)

    # 3 frames were occluded ([10, 11, 12]), so exactly 27 detected and 3 lost
    assert result.lost_count == 3
    assert result.detected_count == 27
    assert result.detection_rate == pytest.approx(27 / 30, rel=1e-3)

    # Verify that occluded frames are explicitly LOST and NEVER silently filled
    for occluded_idx in [10, 11, 12]:
        obs = result.observations[occluded_idx]
        assert obs.frame_index == occluded_idx
        assert obs.detection_status == DetectionStatus.LOST
        assert obs.image_coordinates is None
        assert obs.normalized_coordinates is None
        assert obs.confidence == 0.0

    # Verify timestamps are monotonically increasing
    for i in range(1, len(result.observations)):
        assert result.observations[i].timestamp_seconds >= result.observations[i - 1].timestamp_seconds


def test_coordinate_space_and_uncalibrated_disclaimer(synthetic_circular_video):
    """Verify extracted results explicitly disclaim 3D coordinates and physical safety."""
    result = extract_trajectory_from_video(
        video_path=synthetic_circular_video,
        is_synthetic=True,
    )

    assert result.coordinate_space == CoordinateSpace.NORMALIZED_IMAGE.value
    assert result.camera_calibrated is False
    assert "uncalibrated camera" in result.disclaimer.lower()
    assert "do not represent metric 3d cartesian coordinates" in result.disclaimer.lower()


# ==============================================================================
# 3. Serialization & Format Exports
# ==============================================================================

def test_video_trajectory_json_export_and_roundtrip(synthetic_circular_video, tmp_path):
    """Verify JSON export and from_dict / load roundtrip behavior."""
    result = extract_trajectory_from_video(synthetic_circular_video, is_synthetic=True)
    out_json = tmp_path / "trajectory.json"

    result.to_json(out_json)
    assert out_json.exists()
    assert out_json.stat().st_size > 0

    loaded = VideoTrajectoryResult.load(out_json)
    assert loaded.total_frames == result.total_frames
    assert loaded.detected_count == result.detected_count
    assert loaded.lost_count == result.lost_count
    assert loaded.video_metadata.file_path == result.video_metadata.file_path
    assert len(loaded.observations) == len(result.observations)


def test_video_trajectory_csv_export(synthetic_circular_video, tmp_path):
    """Verify CSV export produces valid rows with pixel and normalized coordinates."""
    result = extract_trajectory_from_video(synthetic_circular_video, is_synthetic=True)
    out_csv = tmp_path / "trajectory.csv"

    csv_path = result.to_csv(out_csv)
    assert Path(csv_path).exists()

    lines = Path(csv_path).read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 31  # Header + 30 frame rows
    header = lines[0].split(",")
    assert "frame_index" in header
    assert "u_pixels" in header
    assert "u_normalized" in header
    assert "confidence" in header


def test_to_telemetry_adapter_conversion(synthetic_circular_video):
    """Verify conversion to Robot CI RecordedTelemetry marks ee_position as None."""
    result = extract_trajectory_from_video(synthetic_circular_video, is_synthetic=True)
    telemetry = result.to_telemetry()

    assert telemetry.provenance.data_origin == "video_extracted"
    assert telemetry.provenance.is_synthetic is True
    assert telemetry.provenance.time_basis == "VIDEO_RELATIVE_SECONDS"
    assert len(telemetry.records) == 30

    # ee_position MUST remain None to prevent pretending 2D pixels are 3D meters
    for rec in telemetry.records:
        assert rec.ee_position is None
        assert "image_coordinates" in rec.raw_metadata


# ==============================================================================
# 4. Error Handling and Negative Tests
# ==============================================================================

def test_nonexistent_video_raises_file_not_found(tmp_path):
    """Nonexistent video path must raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="not found"):
        extract_trajectory_from_video(tmp_path / "missing_video.mp4")


def test_unsupported_video_format_raises_error(tmp_path):
    """Unsupported file extension (e.g. .txt, .xyz) must raise UnsupportedVideoFormatError."""
    dummy_file = tmp_path / "recording.txt"
    dummy_file.write_text("not a video")

    with pytest.raises(UnsupportedVideoFormatError, match="Unsupported video container"):
        extract_trajectory_from_video(dummy_file)


def test_corrupt_video_raises_invalid_video_error(tmp_path):
    """Zero-byte or corrupted video file must raise InvalidVideoError."""
    corrupt_file = tmp_path / "corrupt.mp4"
    corrupt_file.write_bytes(b"CORRUPT_BYTES_NOT_VIDEO")

    with pytest.raises(InvalidVideoError, match="failed to open|zero decodable"):
        extract_trajectory_from_video(corrupt_file)


def test_opencv_unavailable_diagnostic(monkeypatch):
    """When cv2 is missing, system raises OpenCVUnavailableError with installation instructions."""
    from robot_ci.vision import tracker
    monkeypatch.setattr(tracker, "is_cv2_available", lambda: False)

    with pytest.raises(OpenCVUnavailableError, match="pip install opencv-python"):
        tracker.get_cv2()
