"""Focused validation tests for trajectory alignment and comparison metrics (Phase 3).

Verifies mathematical and geometric integrity:
- Identical trajectories produce near-zero deviation.
- Known translated or perturbed paths produce expected deviations in an explicitly shared coordinate system.
- Missing observations are handled transparently.
- Geometric alignment does not conceal meaningful physical or shape differences.
- Camera calibration is rigorously validated rather than assumed.
- Dimensionless shape metrics are never presented as physical distances in meters.
"""

import math
import numpy as np
import pytest

from robot_ci.vision.models import (
    DetectionStatus,
    CoordinateSpace,
    VideoFrameObservation,
    VideoMetadata,
    TrackerConfig,
    VideoTrajectoryResult,
    UNCALIBRATED_CAMERA_DISCLAIMER,
)
from robot_ci.vision.projection import (
    CameraCalibration,
    CalibrationError,
    create_synthetic_camera_calibration,
)
from robot_ci.vision.sim_extractor import SimulatedTrajectoryResult
from robot_ci.vision.comparison import (
    ComparisonMode,
    VideoSimComparisonReport,
    compare_video_to_simulation,
    resample_2d_trajectory,
    compute_2d_path_length,
    compute_2d_dtw,
    procrustes_align_2d,
)


def _build_test_video_trajectory(
    pts_pixel: np.ndarray,
    width: int = 320,
    height: int = 240,
    fps: float = 30.0,
    lost_indices: list = None,
) -> VideoTrajectoryResult:
    """Helper to construct VideoTrajectoryResult from 2D pixel coordinates."""
    lost_set = set(lost_indices or [])
    n = len(pts_pixel)
    obs = []
    detected_count = 0
    lost_count = 0

    for i in range(n):
        t = i / fps
        if i in lost_set:
            obs.append(VideoFrameObservation(
                frame_index=i,
                timestamp_seconds=t,
                detection_status=DetectionStatus.LOST,
                confidence=0.0,
            ))
            lost_count += 1
        else:
            u, v = float(pts_pixel[i, 0]), float(pts_pixel[i, 1])
            obs.append(VideoFrameObservation(
                frame_index=i,
                timestamp_seconds=t,
                detection_status=DetectionStatus.DETECTED,
                image_coordinates=(u, v),
                normalized_coordinates=(u / float(width), v / float(height)),
                confidence=0.98,
            ))
            detected_count += 1

    meta = VideoMetadata(
        file_path="synthetic_test.mp4",
        file_format=".mp4",
        total_frames=n,
        fps=fps,
        duration_seconds=n / fps,
        width=width,
        height=height,
        is_synthetic=True,
    )

    return VideoTrajectoryResult(
        video_metadata=meta,
        tracker_config=TrackerConfig(),
        observations=obs,
        total_frames=n,
        detected_count=detected_count,
        lost_count=lost_count,
        detection_rate=detected_count / float(max(n, 1)),
        is_synthetic=True,
    )


def _build_test_sim_trajectory(
    pts_3d: np.ndarray,
    scenario_id: str = "reach_simple",
    policy_id: str = "baseline",
    dt: float = 0.1,
) -> SimulatedTrajectoryResult:
    """Helper to construct SimulatedTrajectoryResult from 3D points."""
    n = len(pts_3d)
    return SimulatedTrajectoryResult(
        run_id="test_run_validation",
        scenario_id=scenario_id,
        policy_id=policy_id,
        policy_version="1.0.0",
        simulator_backend="mock",
        seed=42,
        total_steps=n,
        duration_seconds=n * dt,
        success=True,
        collision_count=0,
        ee_positions=pts_3d,
        timesteps=list(range(n)),
        timestamps=[i * dt for i in range(n)],
        events=[],
    )


# ---------------------------------------------------------------------------
# Metric Validation Test Cases
# ---------------------------------------------------------------------------

def test_identical_trajectories_produce_near_zero_deviation():
    """Verify that identical trajectories yield near-zero RMS, Mean, Max, and DTW deviations."""
    # 30 points along a smooth 3D arc
    steps = 30
    t = np.linspace(0.0, 1.0, steps)
    pts_3d = np.column_stack([
        0.3 + 0.1 * t,
        -0.1 + 0.2 * t,
        0.2 + 0.05 * np.sin(np.pi * t),
    ])

    calib = create_synthetic_camera_calibration(width=320, height=240, camera_distance=1.2)
    sim_traj = _build_test_sim_trajectory(pts_3d)

    # Project to pixels to create an exactly matching video trajectory
    pixels, _, _ = calib.project_trajectory(pts_3d)
    video_traj = _build_test_video_trajectory(pixels, width=320, height=240)

    # Compare in Mode A (Calibrated Projection)
    report = compare_video_to_simulation(
        video_trajectory=video_traj,
        sim_trajectory=sim_traj,
        calibration=calib,
        mode=ComparisonMode.CALIBRATED_PROJECTION,
        num_resample_points=steps,
    )

    metrics = report.metrics
    assert metrics["rms_trajectory_deviation"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["mean_trajectory_deviation"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["max_trajectory_deviation"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["dtw_average_step_distance"] == pytest.approx(0.0, abs=1e-4)
    assert metrics["dtw_cost"] == pytest.approx(0.0, abs=1e-4)


def test_known_translated_path_produces_expected_deviation():
    """Verify that a known translation produces exact predictable deviation in shared pixel space."""
    steps = 20
    t = np.linspace(0.0, 1.0, steps)
    pts_3d = np.column_stack([
        0.3 + 0.1 * t,
        -0.1 + 0.1 * t,
        0.2 * np.ones(steps),
    ])

    calib = create_synthetic_camera_calibration(width=320, height=240, camera_distance=1.2)
    sim_traj = _build_test_sim_trajectory(pts_3d)
    pixels, _, _ = calib.project_trajectory(pts_3d)

    # Translate video path by exactly +15.0 pixels along U and +20.0 pixels along V
    # Expected Euclidean distance is sqrt(15^2 + 20^2) = 25.0 pixels everywhere
    shift_u = 15.0
    shift_v = 20.0
    expected_dist = math.sqrt(shift_u ** 2 + shift_v ** 2)

    translated_pixels = pixels.copy()
    translated_pixels[:, 0] += shift_u
    translated_pixels[:, 1] += shift_v

    video_traj = _build_test_video_trajectory(translated_pixels, width=320, height=240)

    report = compare_video_to_simulation(
        video_trajectory=video_traj,
        sim_trajectory=sim_traj,
        calibration=calib,
        mode=ComparisonMode.CALIBRATED_PROJECTION,
        num_resample_points=steps,
    )

    metrics = report.metrics
    assert metrics["rms_trajectory_deviation"] == pytest.approx(expected_dist, abs=1e-2)
    assert metrics["mean_trajectory_deviation"] == pytest.approx(expected_dist, abs=1e-2)
    assert metrics["max_trajectory_deviation"] == pytest.approx(expected_dist, abs=1e-2)
    assert report.units == "pixels (px)"


def test_missing_observations_handled_transparently():
    """Verify missing video frames are excluded from deviation without fabricated interpolation."""
    steps = 30
    t = np.linspace(0.0, 1.0, steps)
    pts_3d = np.column_stack([
        0.3 + 0.1 * t,
        -0.1 + 0.2 * t,
        0.2 * np.ones(steps),
    ])

    calib = create_synthetic_camera_calibration(width=320, height=240, camera_distance=1.2)
    sim_traj = _build_test_sim_trajectory(pts_3d)
    pixels, _, _ = calib.project_trajectory(pts_3d)

    # Drop 10 frames (33% lost)
    lost_frames = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    video_traj = _build_test_video_trajectory(pixels, width=320, height=240, lost_indices=lost_frames)

    report = compare_video_to_simulation(
        video_trajectory=video_traj,
        sim_trajectory=sim_traj,
        calibration=calib,
    )

    # 20 frames detected out of 30
    assert report.metrics["video_frames_detected"] == 20
    assert report.metrics["video_frames_total"] == 30
    assert report.metrics["tracking_coverage"] == pytest.approx(20.0 / 30.0, abs=1e-3)
    # Underlying path shape was identical for detected frames -> deviation remains very small (< 2.0 px)
    assert report.metrics["rms_trajectory_deviation"] < 2.0


def test_alignment_does_not_conceal_meaningful_differences():
    """Verify Procrustes alignment preserves distinct geometric shapes and reports large residual error."""
    steps = 40
    t = np.linspace(0.0, 1.0, steps)

    # Straight line in simulation
    pts_sim_3d = np.column_stack([
        0.2 + 0.2 * t,
        0.0 * np.ones(steps),
        0.2 * np.ones(steps),
    ])
    sim_traj = _build_test_sim_trajectory(pts_sim_3d)

    # S-curve with strong sinusoidal oscillation in video
    pts_video = np.column_stack([
        100.0 + 100.0 * t,
        120.0 + 50.0 * np.sin(2 * np.pi * t),
    ])
    video_traj = _build_test_video_trajectory(pts_video, width=320, height=240)

    # In Mode B with Procrustes alignment
    report = compare_video_to_simulation(
        video_trajectory=video_traj,
        sim_trajectory=sim_traj,
        calibration=None,
        shape_projection_plane="xy",
        apply_procrustes_alignment=True,
    )

    # The S-curve cannot be rigidly mapped to a straight line without substantial residual deviation
    metrics = report.metrics
    assert metrics["rms_trajectory_deviation"] > 0.1  # Significant normalized deviation
    assert metrics["max_trajectory_deviation"] > 0.2
    assert report.alignment_details["method"] == "uniform_normalization_progress_resampling_2d_procrustes_rotation_only"


def test_camera_calibration_strictly_validated():
    """Verify camera calibration parameters fail fast on non-physical matrices or depths."""
    # 1. Non-orthogonal rotation
    with pytest.raises(CalibrationError, match="not orthogonal"):
        CameraCalibration(
            fx=300.0, fy=300.0, cx=160.0, cy=120.0,
            image_width=320, image_height=240,
            rotation_matrix=np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
            translation_vector=np.zeros(3),
        )

    # 2. Inversion / reflection matrix (det(R) == -1)
    with pytest.raises(CalibrationError, match="determinant must be \\+1"):
        CameraCalibration(
            fx=300.0, fy=300.0, cx=160.0, cy=120.0,
            image_width=320, image_height=240,
            rotation_matrix=np.diag([1.0, 1.0, -1.0]),
            translation_vector=np.zeros(3),
        )

    # 3. Non-positive focal lengths
    with pytest.raises(CalibrationError, match="Focal lengths must be positive"):
        CameraCalibration(
            fx=0.0, fy=300.0, cx=160.0, cy=120.0,
            image_width=320, image_height=240,
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
        )


def test_dimensionless_shape_metrics_never_presented_as_physical_distances():
    """Verify Mode B (Shape-Only) enforces dimensionless units and disclaimers, never claiming meters."""
    pts_3d = np.array([[0.3, 0.0, 0.2], [0.4, 0.1, 0.2], [0.5, 0.2, 0.2]])
    sim_traj = _build_test_sim_trajectory(pts_3d)

    pts_video = np.array([[50.0, 100.0], [100.0, 120.0], [150.0, 140.0]])
    video_traj = _build_test_video_trajectory(pts_video, width=320, height=240)

    report = compare_video_to_simulation(
        video_trajectory=video_traj,
        sim_trajectory=sim_traj,
        calibration=None,
    )

    # Explicit unit validation
    assert report.comparison_mode == ComparisonMode.SHAPE_ONLY_2D
    assert report.units == "normalized_units (dimensionless)"
    assert report.coordinate_system == "normalized_screen_space"
    assert "meter" not in report.units.lower()
    assert "pixel" not in report.units.lower()

    # Disclaimers must explicitly state that metrics are not physical meters
    assert any("NOT physical meters" in d for d in report.disclaimers)
    assert any("UNCALIBRATED" in d.upper() for d in report.disclaimers)
    # Unavailable metrics explicitly documented
    assert any("3d_cartesian_error_meters" in u for u in report.unavailable_metrics)
    assert any("joint_angle_error" in u for u in report.unavailable_metrics)
