"""Focused tests for real video to simulation trajectory comparison (Phase 2).

Covers:
- Correct simulated trajectory extraction (Mock and PyBullet)
- Projection and coordinate validation (CameraCalibration)
- Shape-only alignment (Procrustes, normalized progress resampling)
- Calibrated projection mode (pixel reprojection error)
- Metric calculations (RMS, Mean, Max, DTW, path lengths)
- Missing and low-confidence video detections
- Invalid and mismatched scenarios/controllers
- Structured report persistence and round-trip loading
- Synthetic versus real data provenance labeling
"""

import json
from pathlib import Path
import numpy as np
import pytest

from robot_ci.vision.models import (
    DetectionStatus,
    CoordinateSpace,
    VideoFrameObservation,
    VideoMetadata,
    TrackerConfig,
    VideoTrajectoryResult,
)
from robot_ci.vision.projection import (
    CameraCalibration,
    CalibrationError,
    create_synthetic_camera_calibration,
)
from robot_ci.vision.sim_extractor import (
    SimulatedTrajectoryResult,
    extract_simulated_trajectory,
)
from robot_ci.vision.comparison import (
    ComparisonMode,
    VideoSimComparisonReport,
    compare_video_to_simulation,
    resample_2d_trajectory,
    compute_2d_path_length,
    compute_2d_dtw,
    procrustes_align_2d,
)


@pytest.fixture
def sample_video_trajectory() -> VideoTrajectoryResult:
    """Fixture providing a synthetic video trajectory with 30 frames."""
    meta = VideoMetadata(
        file_path="tests/fixtures/synthetic_phone_recording.mp4",
        file_format=".mp4",
        total_frames=30,
        fps=30.0,
        duration_seconds=1.0,
        width=320,
        height=240,
        is_synthetic=True,
    )
    cfg = TrackerConfig(color_preset="red")
    observations = []
    # Generate 30 frames with an arc trajectory and 1 lost frame
    for i in range(30):
        t = i / 29.0
        if i == 15:
            # Simulated lost detection
            obs = VideoFrameObservation(
                frame_index=i,
                timestamp_seconds=i / 30.0,
                detection_status=DetectionStatus.LOST,
                confidence=0.0,
            )
        else:
            u = 60.0 + 200.0 * t
            v = 120.0 + 40.0 * np.sin(np.pi * t)
            obs = VideoFrameObservation(
                frame_index=i,
                timestamp_seconds=i / 30.0,
                detection_status=DetectionStatus.DETECTED,
                image_coordinates=(float(u), float(v)),
                normalized_coordinates=(float(u / 320.0), float(v / 240.0)),
                confidence=0.95,
                bounding_box=(int(u - 5), int(v - 5), 10, 10),
                area_pixels=80.0,
            )
        observations.append(obs)

    return VideoTrajectoryResult(
        video_metadata=meta,
        tracker_config=cfg,
        observations=observations,
        total_frames=30,
        detected_count=29,
        lost_count=1,
        detection_rate=29.0 / 30.0,
        is_synthetic=True,
        notes="Synthetic test trajectory for reach_simple",
    )


@pytest.fixture
def sample_sim_trajectory() -> SimulatedTrajectoryResult:
    """Fixture providing a realistic 3D simulated end-effector trajectory."""
    steps = 25
    dt = 0.1
    ee_positions = []
    for i in range(steps):
        t = i / float(steps - 1)
        x = 0.4 + 0.2 * t
        y = -0.1 + 0.2 * t
        z = 0.3 + 0.05 * np.sin(np.pi * t)
        ee_positions.append([x, y, z])

    return SimulatedTrajectoryResult(
        run_id="sim_test_01",
        scenario_id="reach_simple",
        policy_id="baseline",
        policy_version="1.0.0",
        simulator_backend="mock",
        seed=42,
        total_steps=steps,
        duration_seconds=steps * dt,
        success=True,
        collision_count=0,
        ee_positions=np.array(ee_positions, dtype=np.float64),
        timesteps=list(range(steps)),
        timestamps=[i * dt for i in range(steps)],
        events=[{"step": i, "collision": False, "success": i == steps - 1} for i in range(steps)],
    )


# ---------------------------------------------------------------------------
# Step 2: Simulation Trajectory Extraction Tests
# ---------------------------------------------------------------------------

def test_extract_simulated_trajectory_mock():
    """Verify that extract_simulated_trajectory captures genuine EE positions with Mock backend."""
    result = extract_simulated_trajectory(
        policy_name="baseline",
        scenario_id="reach_simple",
        simulator_name="mock",
        seed=42,
    )
    assert isinstance(result, SimulatedTrajectoryResult)
    assert result.simulator_backend == "mock"
    assert result.scenario_id == "reach_simple"
    assert result.policy_id == "baseline"
    assert result.total_steps > 0
    assert len(result.ee_positions) == len(result.timesteps)
    assert len(result.ee_positions) >= result.total_steps
    assert result.ee_positions.shape == (len(result.ee_positions), 3)
    assert len(result.timesteps) == len(result.timestamps)
    assert isinstance(result.success, bool)
    assert isinstance(result.collision_count, int)
    assert np.all(np.isfinite(result.ee_positions))

    # Serialization roundtrip
    d = result.to_dict()
    restored = SimulatedTrajectoryResult.from_dict(d)
    assert restored.run_id == result.run_id
    assert np.allclose(restored.ee_positions, result.ee_positions)


def test_extract_simulated_trajectory_pybullet():
    """Verify trajectory extraction with the actual PyBullet simulator backend."""
    result = extract_simulated_trajectory(
        policy_name="baseline",
        scenario_id="reach_simple",
        simulator_name="pybullet",
        seed=42,
    )
    assert result.simulator_backend == "pybullet"
    assert result.total_steps > 0
    assert len(result.ee_positions) == len(result.timesteps)
    assert len(result.ee_positions) >= result.total_steps
    # Physical end-effector positions should be in realistic reach workspace
    assert np.all(result.ee_positions[:, 2] > 0.0)  # Z above ground
    assert np.all(np.isfinite(result.ee_positions))


def test_extract_simulated_trajectory_invalid_scenario():
    """Verify ValueError is raised for non-existent scenarios."""
    with pytest.raises(ValueError, match="Scenario 'non_existent_scenario' not found"):
        extract_simulated_trajectory(scenario_id="non_existent_scenario", simulator_name="mock")


# ---------------------------------------------------------------------------
# Step 3: Projection and Camera Calibration Tests
# ---------------------------------------------------------------------------

def test_camera_calibration_validation():
    """Verify that CameraCalibration strictly validates optical and geometric constraints."""
    # Invalid negative focal length
    with pytest.raises(CalibrationError, match="Focal lengths must be positive"):
        CameraCalibration(
            fx=-100.0, fy=100.0, cx=160.0, cy=120.0,
            image_width=320, image_height=240,
            rotation_matrix=np.eye(3),
            translation_vector=np.zeros(3),
        )

    # Invalid non-orthogonal rotation matrix
    bad_r = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(CalibrationError, match="Rotation matrix is not orthogonal"):
        CameraCalibration(
            fx=100.0, fy=100.0, cx=160.0, cy=120.0,
            image_width=320, image_height=240,
            rotation_matrix=bad_r,
            translation_vector=np.zeros(3),
        )

    # Invalid translation dimension
    with pytest.raises(CalibrationError, match="Translation vector must have length 3"):
        CameraCalibration(
            fx=100.0, fy=100.0, cx=160.0, cy=120.0,
            image_width=320, image_height=240,
            rotation_matrix=np.eye(3),
            translation_vector=np.array([1.0, 2.0]),
        )


def test_camera_calibration_projection_math():
    """Verify pinhole camera projection math and depth boundary handling."""
    calib = create_synthetic_camera_calibration(width=320, height=240, camera_distance=1.0)
    
    # Point directly in front of camera
    px, norm_px, visible = calib.project_point([0.0, 0.0, 0.0])
    assert px is not None
    assert norm_px is not None
    assert 0.0 <= norm_px[0] <= 1.0
    assert 0.0 <= norm_px[1] <= 1.0
    assert bool(visible) is True

    # Point behind camera (zc <= 0)
    # Using eye rotation looking along +Z, a point at z = -1.0 is behind camera
    calib_front = CameraCalibration(
        fx=200.0, fy=200.0, cx=160.0, cy=120.0,
        image_width=320, image_height=240,
        rotation_matrix=np.eye(3),
        translation_vector=np.array([0.0, 0.0, 0.0]),
    )
    behind_px, behind_norm, behind_vis = calib_front.project_point([0.0, 0.0, -1.0])
    assert behind_px is None
    assert behind_vis is False

    # Trajectory projection
    pts = np.array([[0.0, 0.0, 1.0], [0.1, 0.1, 1.5], [0.0, 0.0, -2.0]])
    pixels, normalized, valid_mask = calib_front.project_trajectory(pts)
    assert bool(valid_mask[0]) is True
    assert bool(valid_mask[1]) is True
    assert bool(valid_mask[2]) is False  # behind camera


# ---------------------------------------------------------------------------
# Step 4 & 5: Trajectory Comparison Modes and Metrics Tests
# ---------------------------------------------------------------------------

def test_calibrated_projection_mode(sample_video_trajectory, sample_sim_trajectory):
    """Verify Mode A (Calibrated Projection) correctly computes pixel errors and labels units."""
    calib = create_synthetic_camera_calibration(width=320, height=240, camera_distance=1.2)
    report = compare_video_to_simulation(
        video_trajectory=sample_video_trajectory,
        sim_trajectory=sample_sim_trajectory,
        calibration=calib,
        mode=ComparisonMode.CALIBRATED_PROJECTION,
        physical_controller_id="baseline",
        robot_model="ur5e",
    )

    assert report.comparison_mode == ComparisonMode.CALIBRATED_PROJECTION
    assert report.comparison_type == "matched_policy_verification"
    assert report.coordinate_system == "camera_image_pixel_space"
    assert report.units == "pixels (px)"
    assert report.calibration_metadata is not None
    assert report.calibration_metadata["image_width"] == 320

    # Metrics verification
    metrics = report.metrics
    assert 0.0 <= metrics["tracking_coverage"] <= 1.0
    assert metrics["video_frames_detected"] == 29
    assert metrics["video_frames_total"] == 30
    assert metrics["video_duration_seconds"] > 0.0
    assert metrics["sim_duration_seconds"] > 0.0
    assert metrics["rms_trajectory_deviation"] >= 0.0
    assert metrics["mean_trajectory_deviation"] >= 0.0
    assert metrics["max_trajectory_deviation"] >= metrics["mean_trajectory_deviation"]
    assert metrics["dtw_cost"] >= 0.0
    assert metrics["dtw_average_step_distance"] >= 0.0
    assert metrics["resampled_points_count"] == 50

    # Ensure disclaimers explain pixel projection
    assert any("CALIBRATED PROJECTION" in d for d in report.disclaimers)
    assert len(report.unavailable_metrics) >= 4


def test_shape_only_mode_comparison(sample_video_trajectory, sample_sim_trajectory):
    """Verify Mode B (Shape-Only) enforces dimensionless normalized units and Procrustes documentation."""
    report = compare_video_to_simulation(
        video_trajectory=sample_video_trajectory,
        sim_trajectory=sample_sim_trajectory,
        calibration=None,  # No calibration -> Mode B
        physical_controller_id="human_teleop",
        shape_projection_plane="xy",
    )

    assert report.comparison_mode == ComparisonMode.SHAPE_ONLY_2D
    assert report.comparison_type == "cross_controller_behavioural_comparison"
    assert report.coordinate_system == "normalized_screen_space"
    assert report.units == "normalized_units (dimensionless)"
    assert report.calibration_metadata is None

    # Check that Procrustes documentation is transparent
    details = report.alignment_details
    assert details["projection_plane"] == "xy"
    assert "procrustes_transform" in details
    assert "scale_factor" in details["procrustes_transform"]
    assert "rotation_degrees" in details["procrustes_transform"]
    assert "translation" in details["procrustes_transform"]

    # Metrics verification
    metrics = report.metrics
    assert metrics["rms_trajectory_deviation"] >= 0.0
    assert metrics["mean_trajectory_deviation"] >= 0.0
    assert metrics["max_trajectory_deviation"] >= 0.0
    # Values should be within normalized bounds [0..~1]
    assert metrics["rms_trajectory_deviation"] < 2.0

    # Cross controller warning should be present
    assert any("Cross-controller" in w for w in report.warnings)


def test_shape_only_projection_planes(sample_video_trajectory, sample_sim_trajectory):
    """Verify shape-only mode supports xz and yz projection planes."""
    for plane in ["xz", "yz"]:
        report = compare_video_to_simulation(
            video_trajectory=sample_video_trajectory,
            sim_trajectory=sample_sim_trajectory,
            calibration=None,
            shape_projection_plane=plane,
        )
        assert report.alignment_details["projection_plane"] == plane
        assert report.metrics["rms_trajectory_deviation"] >= 0.0


def test_procrustes_align_2d():
    """Verify 2D Procrustes alignment aligns translated and rotated shapes correctly."""
    # Square at origin
    pts_a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    # Square shifted by [2, 3] and rotated by 90 degrees
    pts_b = np.array([[2.0, 3.0], [2.0, 4.0], [1.0, 4.0], [1.0, 3.0]], dtype=np.float64)

    aligned, transform = procrustes_align_2d(pts_b, pts_a)
    assert np.allclose(aligned, pts_a, atol=1e-3)
    assert transform["scale_factor"] == pytest.approx(1.0, abs=1e-2)
    assert abs(transform["rotation_degrees"]) > 1.0


def test_resample_and_dtw_algorithms():
    """Verify 2D resampling and DTW algorithms on known curves."""
    pts = np.array([[0.0, 0.0], [10.0, 0.0]])
    resampled = resample_2d_trajectory(pts, 11)
    assert len(resampled) == 11
    assert np.isclose(resampled[5, 0], 5.0)
    assert np.isclose(resampled[10, 0], 10.0)

    # Identical curves should have zero DTW cost
    cost, avg_dist, _ = compute_2d_dtw(resampled, resampled)
    assert np.isclose(cost, 0.0, atol=1e-6)
    assert np.isclose(avg_dist, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Missing Detections, Edge Cases and Warnings Tests
# ---------------------------------------------------------------------------

def test_missing_video_detections_handling(sample_sim_trajectory):
    """Verify that dropped detections are filtered and low coverage triggers warnings."""
    meta = VideoMetadata(
        file_path="dummy.mp4", file_format=".mp4", total_frames=20,
        fps=10.0, duration_seconds=2.0, width=100, height=100,
    )
    cfg = TrackerConfig()
    observations = []
    # Only 4 detected frames out of 20 (20% coverage -> low coverage warning)
    for i in range(20):
        if i in [0, 5, 10, 15]:
            obs = VideoFrameObservation(
                frame_index=i, timestamp_seconds=i * 0.1,
                detection_status=DetectionStatus.DETECTED,
                image_coordinates=(float(i * 5), float(i * 5)),
                normalized_coordinates=(float(i * 0.05), float(i * 0.05)),
            )
        else:
            obs = VideoFrameObservation(
                frame_index=i, timestamp_seconds=i * 0.1,
                detection_status=DetectionStatus.LOST,
            )
        observations.append(obs)

    video_traj = VideoTrajectoryResult(
        video_metadata=meta, tracker_config=cfg,
        observations=observations, total_frames=20,
        detected_count=4, lost_count=16, detection_rate=0.2,
    )

    report = compare_video_to_simulation(video_traj, sample_sim_trajectory)
    assert any("Low visual tracking coverage" in w for w in report.warnings)
    assert report.metrics["video_frames_detected"] == 4


def test_insufficient_detections_raises_error(sample_sim_trajectory):
    """Verify that fewer than 2 detected frames raises a ValueError."""
    meta = VideoMetadata(
        file_path="empty.mp4", file_format=".mp4", total_frames=10,
        fps=10.0, duration_seconds=1.0, width=100, height=100,
    )
    obs = [VideoFrameObservation(frame_index=i, timestamp_seconds=0.1 * i, detection_status=DetectionStatus.LOST) for i in range(10)]
    video_traj = VideoTrajectoryResult(
        video_metadata=meta, tracker_config=TrackerConfig(),
        observations=obs, total_frames=10, detected_count=0, lost_count=10, detection_rate=0.0,
    )
    with pytest.raises(ValueError, match="insufficient detected points"):
        compare_video_to_simulation(video_traj, sample_sim_trajectory)


def test_calibrated_mode_behind_camera_raises(sample_video_trajectory):
    """Verify CalibrationError is raised if all simulated points are behind the camera."""
    # Camera facing away from robot workspace
    bad_calib = CameraCalibration(
        fx=200.0, fy=200.0, cx=160.0, cy=120.0,
        image_width=320, image_height=240,
        rotation_matrix=np.eye(3),
        translation_vector=np.array([0.0, 0.0, -100.0]),  # Pushes everything to negative Z
    )
    sim_traj = SimulatedTrajectoryResult(
        run_id="s1", scenario_id="sc1", policy_id="p1", policy_version="1",
        simulator_backend="mock", seed=1, total_steps=5, duration_seconds=0.5,
        success=True, collision_count=0,
        ee_positions=np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=np.float64),
        timesteps=[0, 1], timestamps=[0.0, 0.1], events=[],
    )
    with pytest.raises(CalibrationError, match="positive depth"):
        compare_video_to_simulation(
            sample_video_trajectory, sim_traj, calibration=bad_calib, mode=ComparisonMode.CALIBRATED_PROJECTION
        )


# ---------------------------------------------------------------------------
# Step 6: Report Persistence and Provenance Tests
# ---------------------------------------------------------------------------

def test_report_persistence_roundtrip(sample_video_trajectory, sample_sim_trajectory, tmp_path):
    """Verify that VideoSimComparisonReport serializes to and deserializes from JSON with high fidelity."""
    calib = create_synthetic_camera_calibration()
    report = compare_video_to_simulation(
        video_trajectory=sample_video_trajectory,
        sim_trajectory=sample_sim_trajectory,
        calibration=calib,
        physical_controller_id="baseline",
        robot_model="test_arm",
    )

    out_file = tmp_path / "comparison_report.json"
    saved = report.save(out_file)
    assert Path(saved).exists()

    # Load from disk
    loaded = VideoSimComparisonReport.load(out_file)
    assert loaded.report_id == report.report_id
    assert loaded.comparison_mode == report.comparison_mode
    assert loaded.scenario_id == report.scenario_id
    assert loaded.metrics["rms_trajectory_deviation"] == pytest.approx(report.metrics["rms_trajectory_deviation"])
    assert loaded.provenance["sim_backend"] == "mock"
    assert loaded.provenance["video_source_is_synthetic"] is True


def test_provenance_distinguishes_synthetic_vs_real(sample_sim_trajectory):
    """Verify that provenance correctly documents real vs synthetic video sources."""
    # Synthetic video
    meta_synth = VideoMetadata("synth.mp4", ".mp4", 5, 10.0, 0.5, 100, 100, is_synthetic=True)
    obs = [VideoFrameObservation(i, i * 0.1, DetectionStatus.DETECTED, (float(i), float(i)), (0.1 * i, 0.1 * i)) for i in range(5)]
    synth_res = VideoTrajectoryResult(meta_synth, TrackerConfig(), obs, 5, 5, 0, 1.0, is_synthetic=True)

    report_synth = compare_video_to_simulation(synth_res, sample_sim_trajectory)
    assert report_synth.provenance["video_source_is_synthetic"] is True

    # Real video
    meta_real = VideoMetadata("real_robot.mp4", ".mp4", 5, 10.0, 0.5, 100, 100, is_synthetic=False)
    real_res = VideoTrajectoryResult(meta_real, TrackerConfig(), obs, 5, 5, 0, 1.0, is_synthetic=False)

    report_real = compare_video_to_simulation(real_res, sample_sim_trajectory)
    assert report_real.provenance["video_source_is_synthetic"] is False
