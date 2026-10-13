"""Video processing and end-effector trajectory extraction pipeline.

Provides:
- extract_trajectory_from_video: reads phone video container (MP4, MOV, etc.),
  extracts visual end-effector marker trajectory across frames, and computes
  detection statistics without fabricating missing data.
- create_synthetic_test_video: generates deterministic synthetic test videos
  with known ground-truth marker positions and configurable occlusions for unit testing.
"""

from pathlib import Path
from typing import Optional, Union, List, Tuple
import math
import numpy as np

from .models import (
    resolve_data_source,
    DATA_SOURCE_SYNTHETIC,
    DetectionStatus,
    VideoFrameObservation,
    VideoMetadata,
    TrackerConfig,
    VideoTrajectoryResult,
    InvalidVideoError,
    UnsupportedVideoFormatError,
    OpenCVUnavailableError,
)
from .tracker import MarkerTracker, get_cv2, is_cv2_available


SUPPORTED_VIDEO_FORMATS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def extract_trajectory_from_video(
    video_path: Union[str, Path],
    config: Optional[TrackerConfig] = None,
    is_synthetic: bool = False,
    notes: Optional[str] = None,
    data_source: Optional[str] = None,
) -> VideoTrajectoryResult:
    """Analyze a prerecorded video and extract end-effector 2D image coordinates.
    
    Args:
        video_path: Path to MP4, MOV, or other supported video recording.
        config: Tracker configuration settings (color presets, HSV bounds, ROI).
        is_synthetic: Flag indicating whether video is synthetic/simulated.
        notes: Optional user notes or recording description.
        data_source: Declared provenance (see vision.models.VALID_DATA_SOURCES).
            Fixtures and files named '*synthetic*' are always forced to
            'synthetic_fixture'. Undeclared sources become 'unspecified',
            which the diagnosis engine never treats as physical evidence.
    
    Returns:
        VideoTrajectoryResult with per-frame observations and detection statistics.
    
    Raises:
        FileNotFoundError: If video file path does not exist.
        UnsupportedVideoFormatError: If container extension is unsupported.
        InvalidVideoError: If video stream cannot be opened or contains zero frames.
        OpenCVUnavailableError: If opencv-python is not installed.
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    resolved_source = resolve_data_source(path, data_source, is_synthetic)
    is_synthetic = resolved_source == DATA_SOURCE_SYNTHETIC

    ext = path.suffix.lower()
    if ext not in SUPPORTED_VIDEO_FORMATS:
        raise UnsupportedVideoFormatError(
            f"Unsupported video container format '{ext}'. "
            f"Supported formats: {sorted(list(SUPPORTED_VIDEO_FORMATS))}"
        )

    cv2 = get_cv2()
    cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise InvalidVideoError(f"OpenCV failed to open video container: {path}")

    # Read container metadata
    total_reported = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)]).strip()

    # Guard against invalid or zero FPS reporting
    if fps <= 0.0 or not math.isfinite(fps):
        fps = 30.0

    tracker = MarkerTracker(config=config)
    observations: List[VideoFrameObservation] = []

    last_known_pos: Optional[Tuple[float, float]] = tracker.config.initial_point
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Obtain timestamp in seconds (use hardware PTS if valid, otherwise compute from frame index and fps)
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec > 0.0 and math.isfinite(pos_msec):
                ts = pos_msec / 1000.0
            else:
                ts = frame_idx / fps

            obs = tracker.track_frame(
                frame=frame,
                frame_index=frame_idx,
                timestamp_seconds=ts,
                last_known_pos=last_known_pos,
            )

            if obs.detection_status == DetectionStatus.DETECTED and obs.image_coordinates is not None:
                last_known_pos = obs.image_coordinates

            observations.append(obs)
            frame_idx += 1

    finally:
        cap.release()

    if not observations:
        raise InvalidVideoError(f"Video contained zero decodable frames: {path}")

    duration_sec = frame_idx / fps if fps > 0 else 0.0
    detected_count = sum(1 for o in observations if o.detection_status == DetectionStatus.DETECTED)
    lost_count = len(observations) - detected_count
    detection_rate = detected_count / float(len(observations)) if observations else 0.0

    video_meta = VideoMetadata(
        file_path=str(path),
        file_format=ext,
        total_frames=len(observations),
        fps=fps,
        duration_seconds=duration_sec,
        width=width,
        height=height,
        codec=codec_str,
        is_synthetic=is_synthetic,
    )

    return VideoTrajectoryResult(
        video_metadata=video_meta,
        tracker_config=tracker.config,
        observations=observations,
        total_frames=len(observations),
        detected_count=detected_count,
        lost_count=lost_count,
        detection_rate=detection_rate,
        is_synthetic=is_synthetic,
        notes=notes,
        data_source=resolved_source,
    )


def create_synthetic_test_video(
    output_path: Union[str, Path],
    width: int = 320,
    height: int = 240,
    fps: float = 30.0,
    num_frames: int = 30,
    marker_color: str = "red",
    trajectory_type: str = "circular",
    occlude_frames: Optional[List[int]] = None,
) -> str:
    """Generate a deterministic synthetic test video for unit testing.
    
    Renders a moving colored circle against a neutral background.
    Clearly designated as a synthetic testing fixture.
    
    Args:
        output_path: Output file path (.mp4 or .avi).
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Frames per second.
        num_frames: Total number of frames to generate.
        marker_color: 'red', 'green', or 'blue'.
        trajectory_type: 'circular', 'linear', or 'stationary'.
        occlude_frames: List of frame indices where marker is obscured/hidden.
    
    Returns:
        Path to generated video file.
    """
    cv2 = get_cv2()
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Color definitions in BGR
    bgr_map = {
        "red": (0, 0, 240),
        "green": (0, 240, 0),
        "blue": (240, 0, 0),
        "yellow": (0, 230, 230),
        "orange": (0, 140, 240),
    }
    bgr_color = bgr_map.get(marker_color.lower(), (0, 0, 240))
    occluded_set = set(occlude_frames or [])

    # Try MP4V codec for cross-platform compatibility
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    if not writer.isOpened():
        # Fallback to MJPG if mp4v is not supported on system
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_file = out_file.with_suffix(".avi")
        writer = cv2.VideoWriter(str(out_file), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise InvalidVideoError(f"Could not initialize OpenCV VideoWriter for {out_file}")

    center_x = width / 2.0
    center_y = height / 2.0
    radius = min(width, height) * 0.3
    marker_radius = 8

    try:
        for idx in range(num_frames):
            # Create neutral dark gray background (BGR 35, 35, 35)
            frame = np.full((height, width, 3), 35, dtype=np.uint8)

            # Draw background grid lines to simulate lab table
            cv2.line(frame, (0, int(center_y)), (width, int(center_y)), (50, 50, 50), 1)
            cv2.line(frame, (int(center_x), 0), (int(center_x), height), (50, 50, 50), 1)

            # Compute marker position along chosen trajectory
            if trajectory_type == "circular":
                angle = (2.0 * math.pi * idx) / float(num_frames)
                mx = int(center_x + radius * math.cos(angle))
                my = int(center_y + radius * math.sin(angle))
            elif trajectory_type == "linear":
                progress = idx / float(num_frames - 1) if num_frames > 1 else 0.0
                mx = int(width * 0.2 + progress * width * 0.6)
                my = int(height * 0.3 + progress * height * 0.4)
            else:  # stationary
                mx = int(center_x)
                my = int(center_y)

            # Draw marker unless occluded on this frame
            if idx not in occluded_set:
                cv2.circle(frame, (mx, my), marker_radius, bgr_color, -1)

            writer.write(frame)

    finally:
        writer.release()

    return str(out_file)
