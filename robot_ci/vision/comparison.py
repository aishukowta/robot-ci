"""Real video trajectory to simulation rollout comparison engine for Robot CI.

Implements rigorous real-to-simulation end-effector trajectory comparison with two
clearly distinguished operational modes:
1. Mode A (CALIBRATED_PROJECTION): Uses verified camera intrinsics (K) and extrinsics (R, t)
   to project simulated 3D robot coordinates into camera image-space (pixels).
2. Mode B (SHAPE_ONLY_2D): Exploratory uncalibrated 2D shape comparison using documented
   normalized projections, strictly labeled in dimensionless normalized units.

Never misrepresents pixel or normalized shape deviations as metric 3D physical errors in meters.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import uuid
import numpy as np

from robot_ci.vision.models import (
    DetectionStatus,
    CoordinateSpace,
    VideoTrajectoryResult,
    UNCALIBRATED_CAMERA_DISCLAIMER,
)
from robot_ci.vision.projection import CameraCalibration, CalibrationError
from robot_ci.vision.sim_extractor import (
    SimulatedTrajectoryResult,
    extract_simulated_trajectory,
)


class ComparisonMode(str, Enum):
    """Operational mode for video-to-simulation trajectory comparison."""
    CALIBRATED_PROJECTION = "CALIBRATED_PROJECTION"
    SHAPE_ONLY_2D = "SHAPE_ONLY_2D"


UNAVAILABLE_METRICS_LIST = [
    "joint_angle_error (requires internal robot encoder/joint-state telemetry)",
    "3d_cartesian_error_meters (monocular video lacks metric depth ground truth)",
    "physical_contact_forces (requires 6-axis F/T or tactile sensor hardware)",
    "physical_collision_count (requires contact switches or physical instrumentation)",
    "physical_task_success_confirmation (sub-millimeter tolerance unconfirmable via monocular video)",
]


@dataclass
class VideoSimComparisonReport:
    """Structured report documenting video vs simulation trajectory comparison."""
    report_id: str
    comparison_mode: ComparisonMode
    scenario_id: str
    sim_policy_id: str
    physical_controller_id: str
    comparison_type: str  # "matched_policy_verification" or "cross_controller_behavioural_comparison"
    robot_model: Optional[str]
    coordinate_system: str
    units: str
    metrics: Dict[str, Any]
    alignment_details: Dict[str, Any]
    provenance: Dict[str, Any]
    calibration_metadata: Optional[Dict[str, Any]]
    warnings: List[str]
    unavailable_metrics: List[str] = field(default_factory=lambda: list(UNAVAILABLE_METRICS_LIST))
    disclaimers: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "comparison_mode": self.comparison_mode.value if isinstance(self.comparison_mode, ComparisonMode) else str(self.comparison_mode),
            "scenario_id": self.scenario_id,
            "sim_policy_id": self.sim_policy_id,
            "physical_controller_id": self.physical_controller_id,
            "comparison_type": self.comparison_type,
            "robot_model": self.robot_model,
            "coordinate_system": self.coordinate_system,
            "units": self.units,
            "metrics": self.metrics,
            "alignment_details": self.alignment_details,
            "provenance": self.provenance,
            "calibration_metadata": self.calibration_metadata,
            "warnings": self.warnings,
            "unavailable_metrics": self.unavailable_metrics,
            "disclaimers": self.disclaimers,
            "created_at": self.created_at,
        }

    def to_json(self, filepath: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent)
        if filepath:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(payload, encoding="utf-8")
        return payload

    def save(self, filepath: Optional[Union[str, Path]] = None, directory: Union[str, Path] = "results/video_sim_comparisons") -> str:
        if filepath:
            target = Path(filepath)
        else:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            target = dir_path / f"video_sim_comparison_{self.report_id}.json"
        self.to_json(target)
        return str(target)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoSimComparisonReport":
        mode_val = data.get("comparison_mode", ComparisonMode.SHAPE_ONLY_2D.value)
        mode = ComparisonMode(mode_val) if mode_val in [m.value for m in ComparisonMode] else ComparisonMode.SHAPE_ONLY_2D
        return cls(
            report_id=data["report_id"],
            comparison_mode=mode,
            scenario_id=data["scenario_id"],
            sim_policy_id=data["sim_policy_id"],
            physical_controller_id=data["physical_controller_id"],
            comparison_type=data.get("comparison_type", "cross_controller_behavioural_comparison"),
            robot_model=data.get("robot_model"),
            coordinate_system=data["coordinate_system"],
            units=data["units"],
            metrics=data["metrics"],
            alignment_details=data["alignment_details"],
            provenance=data["provenance"],
            calibration_metadata=data.get("calibration_metadata"),
            warnings=list(data.get("warnings", [])),
            unavailable_metrics=list(data.get("unavailable_metrics", UNAVAILABLE_METRICS_LIST)),
            disclaimers=list(data.get("disclaimers", [])),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "VideoSimComparisonReport":
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Report file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


# ---------------------------------------------------------------------------
# Trajectory Resampling, DTW, and Geometric Alignment Algorithms
# ---------------------------------------------------------------------------

def resample_2d_trajectory(points: np.ndarray, num_points: int) -> np.ndarray:
    """Resample an Nx2 trajectory to a fixed number of points via linear interpolation.
    
    Interpolates uniformly over normalized progress parameter tau in [0, 1].
    """
    if len(points) == 0:
        return np.zeros((num_points, 2), dtype=np.float64)
    if len(points) == 1 or num_points == 1:
        return np.tile(points[0], (num_points, 1))

    old_t = np.linspace(0.0, 1.0, len(points))
    new_t = np.linspace(0.0, 1.0, num_points)

    resampled = np.zeros((num_points, 2), dtype=np.float64)
    for dim in range(2):
        resampled[:, dim] = np.interp(new_t, old_t, points[:, dim])
    return resampled


def compute_2d_path_length(points: np.ndarray) -> float:
    """Calculate cumulative 2D Euclidean path length."""
    if len(points) < 2:
        return 0.0
    diffs = np.diff(points, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    return float(np.sum(dists))


def compute_2d_dtw(traj_a: np.ndarray, traj_b: np.ndarray) -> Tuple[float, float, int]:
    """Compute Dynamic Time Warping (DTW) distance between two 2D trajectories.
    
    Returns:
        (total_cost, average_step_distance, warping_path_length)
    """
    n, m = len(traj_a), len(traj_b)
    if n == 0 or m == 0:
        return 0.0, 0.0, 0

    cost_matrix = np.full((n, m), np.inf, dtype=np.float64)
    cost_matrix[0, 0] = np.linalg.norm(traj_a[0] - traj_b[0])

    for i in range(1, n):
        cost_matrix[i, 0] = cost_matrix[i - 1, 0] + np.linalg.norm(traj_a[i] - traj_b[0])
    for j in range(1, m):
        cost_matrix[0, j] = cost_matrix[0, j - 1] + np.linalg.norm(traj_a[0] - traj_b[j])

    for i in range(1, n):
        for j in range(1, m):
            dist = np.linalg.norm(traj_a[i] - traj_b[j])
            cost_matrix[i, j] = dist + min(
                cost_matrix[i - 1, j],      # insertion
                cost_matrix[i, j - 1],      # deletion
                cost_matrix[i - 1, j - 1],  # match
            )

    total_cost = float(cost_matrix[n - 1, m - 1])

    # Backtrack to determine warping path length
    i, j = n - 1, m - 1
    path_len = 1
    while i > 0 or j > 0:
        path_len += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            prev_steps = [
                (cost_matrix[i - 1, j - 1], (i - 1, j - 1)),
                (cost_matrix[i - 1, j], (i - 1, j)),
                (cost_matrix[i, j - 1], (i, j - 1)),
            ]
            prev_steps.sort(key=lambda x: x[0])
            i, j = prev_steps[0][1]

    avg_dist = total_cost / float(path_len) if path_len > 0 else 0.0
    return total_cost, avg_dist, path_len


def procrustes_align_2d(
    source: np.ndarray,
    target: np.ndarray,
    allow_scaling: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Align source 2D trajectory onto target 2D trajectory using 2D Procrustes analysis.
    
    Computes optimal translation, rotation, and uniform scale minimizing squared deviations.
    Documents all applied transformation parameters so adjustments are transparent.
    """
    if len(source) != len(target):
        raise ValueError("Source and target must have the same number of points for Procrustes.")

    n = len(source)
    mean_src = np.mean(source, axis=0)
    mean_tgt = np.mean(target, axis=0)

    centered_src = source - mean_src
    centered_tgt = target - mean_tgt

    norm_src = np.linalg.norm(centered_src)
    norm_tgt = np.linalg.norm(centered_tgt)

    if norm_src < 1e-9:
        return target.copy(), {"translation": [0.0, 0.0], "rotation_degrees": 0.0, "scale": 1.0}

    # Cross-covariance matrix
    h = centered_src.T @ centered_tgt
    u, s, vt = np.linalg.svd(h)
    d = float(np.linalg.det(u @ vt))
    sign = 1.0 if d >= 0 else -1.0
    s_sign = np.diag([1.0, sign])
    r = u @ s_sign @ vt

    scale = float((s[0] + sign * s[1]) / (norm_src ** 2)) if allow_scaling and norm_src > 0 else 1.0
    aligned = (scale * (centered_src @ r)) + mean_tgt

    # Extract rotation angle in degrees
    angle_rad = math.atan2(r[1, 0], r[0, 0])
    angle_deg = math.degrees(angle_rad)

    transform_doc = {
        "source_centroid": [round(float(c), 4) for c in mean_src],
        "target_centroid": [round(float(c), 4) for c in mean_tgt],
        "translation": [round(float(t), 4) for t in (mean_tgt - mean_src)],
        "rotation_degrees": round(angle_deg, 2),
        "scale_factor": round(scale, 4),
    }

    return aligned, transform_doc


# ---------------------------------------------------------------------------
# Core Comparison Engine
# ---------------------------------------------------------------------------

def compare_video_to_simulation(
    video_trajectory: VideoTrajectoryResult,
    sim_trajectory: SimulatedTrajectoryResult,
    calibration: Optional[CameraCalibration] = None,
    mode: Optional[ComparisonMode] = None,
    physical_controller_id: str = "unknown_physical_controller",
    robot_model: Optional[str] = "unknown_robot",
    shape_projection_plane: str = "xy",
    apply_procrustes_alignment: bool = True,
    num_resample_points: int = 50,
) -> VideoSimComparisonReport:
    """Compare an extracted video trajectory to an actual simulation rollout.
    
    Args:
        video_trajectory: Extracted VideoTrajectoryResult from video recording.
        sim_trajectory: SimulatedTrajectoryResult from actual simulation rollout.
        calibration: Optional CameraCalibration parameters. If provided and valid,
            Mode A (CALIBRATED_PROJECTION) is active. Otherwise Mode B (SHAPE_ONLY_2D).
        mode: Explicit override for ComparisonMode.
        physical_controller_id: Name of physical controller (e.g. 'human_teleop', 'lead_through').
        robot_model: Robot hardware model name if known.
        shape_projection_plane: 2D projection plane for Mode B ('xy', 'xz', 'yz').
        apply_procrustes_alignment: Whether to optimize 2D rigid alignment in Mode B.
        num_resample_points: Number of uniformly resampled points for comparison.
    
    Returns:
        VideoSimComparisonReport containing metrics, coordinates, units, and disclaimers.
    
    Raises:
        ValueError: If trajectories are empty or invalid.
    """
    warnings: List[str] = []
    disclaimers: List[str] = []

    # 1. Determine comparison mode
    if mode is not None:
        effective_mode = mode
    else:
        effective_mode = ComparisonMode.CALIBRATED_PROJECTION if calibration is not None else ComparisonMode.SHAPE_ONLY_2D

    if effective_mode == ComparisonMode.CALIBRATED_PROJECTION and calibration is None:
        raise ValueError("Calibrated projection mode requested but no CameraCalibration provided.")

    # 2. Extract valid video trajectory points
    detected_obs = [
        obs for obs in video_trajectory.observations
        if obs.detection_status == DetectionStatus.DETECTED
    ]
    if len(detected_obs) < 2:
        raise ValueError(
            f"Video trajectory has insufficient detected points ({len(detected_obs)} detected). "
            "At least 2 detected frames are required for trajectory comparison."
        )

    tracking_coverage = float(len(detected_obs)) / float(max(video_trajectory.total_frames, 1))
    if tracking_coverage < 0.5:
        warnings.append(
            f"Low visual tracking coverage: {tracking_coverage * 100:.1f}% detected. "
            "Results may be skewed by tracking dropout."
        )

    # Check simulated trajectory
    if len(sim_trajectory.ee_positions) < 2:
        raise ValueError("Simulated trajectory has fewer than 2 end-effector positions.")

    # Durations
    video_duration = float(video_trajectory.video_metadata.duration_seconds)
    sim_duration = float(sim_trajectory.duration_seconds)
    timing_diff = abs(video_duration - sim_duration)

    # Check scenario compatibility
    if video_trajectory.notes and "scenario" in video_trajectory.notes.lower():
        if sim_trajectory.scenario_id.lower() not in video_trajectory.notes.lower():
            warnings.append(
                f"Possible scenario mismatch: Video notes reference '{video_trajectory.notes}' "
                f"while simulation ran '{sim_trajectory.scenario_id}'."
            )

    # Controller identity check
    if physical_controller_id.lower() == sim_trajectory.policy_id.lower():
        comp_type = "matched_policy_verification"
    else:
        comp_type = "cross_controller_behavioural_comparison"
        warnings.append(
            f"Cross-controller behavioural comparison: physical controller '{physical_controller_id}' "
            f"differs from simulation policy '{sim_trajectory.policy_id}'."
        )

    # 3. Branch by Comparison Mode
    if effective_mode == ComparisonMode.CALIBRATED_PROJECTION:
        # Mode A: Calibrated Projection
        coordinate_system = "camera_image_pixel_space"
        units = "pixels (px)"
        disclaimers.append(
            "CALIBRATED PROJECTION: Simulated 3D positions are projected into camera pixel space "
            "using verified pinhole camera calibration. Metric errors are reported in 2D image pixels (px), "
            "NOT 3D meters. Pixel reprojection error is sensitive to camera extrinsics and focal length."
        )

        # Video points in pixel space [u, v]
        video_pixels = np.array([
            obs.image_coordinates for obs in detected_obs
            if obs.image_coordinates is not None
        ], dtype=np.float64)

        # Project 3D simulation trajectory into 2D camera pixels
        sim_pixels, sim_norm, valid_mask = calibration.project_trajectory(sim_trajectory.ee_positions)
        valid_count = int(np.sum(valid_mask))
        if valid_count < 2:
            raise CalibrationError(
                f"Only {valid_count}/{len(sim_trajectory.ee_positions)} simulated points projected with positive depth. "
                "The robot trajectory appears to be behind the camera or outside the optical frustum."
            )

        sim_visible_pixels = sim_pixels[valid_mask]

        # Resample both curves uniformly over normalized progress
        resampled_video = resample_2d_trajectory(video_pixels, num_resample_points)
        resampled_sim = resample_2d_trajectory(sim_visible_pixels, num_resample_points)

        # Path lengths in pixels
        video_path_len = compute_2d_path_length(video_pixels)
        sim_path_len = compute_2d_path_length(sim_visible_pixels)

        # Euclidean deviations in pixels
        point_diffs = np.linalg.norm(resampled_video - resampled_sim, axis=1)
        mean_dev = float(np.mean(point_diffs))
        rms_dev = float(np.sqrt(np.mean(point_diffs ** 2)))
        max_dev = float(np.max(point_diffs))

        # DTW distance in pixels
        dtw_cost, dtw_avg_step, _ = compute_2d_dtw(video_pixels, sim_visible_pixels)

        alignment_details = {
            "method": "normalized_progress_resampling",
            "resampled_points_count": num_resample_points,
            "camera_model": calibration.camera_model_name,
            "focal_lengths": [calibration.fx, calibration.fy],
            "principal_point": [calibration.cx, calibration.cy],
            "image_dimensions": [calibration.image_width, calibration.image_height],
            "points_in_front_of_camera": f"{valid_count}/{len(sim_trajectory.ee_positions)}",
        }
        calibration_meta = calibration.to_dict()

    else:
        # Mode B: Shape-Only 2D Comparison
        coordinate_system = "normalized_screen_space"
        units = "normalized_units (dimensionless)"
        disclaimers.append(
            UNCALIBRATED_CAMERA_DISCLAIMER
        )
        disclaimers.append(
            "SHAPE-ONLY COMPARISON: Camera calibration is unavailable. Trajectories are normalized to [0, 1] "
            "in screen space and aligned by geometric shape. Deviations are strictly dimensionless normalized units, "
            "NOT physical meters and NOT calibrated camera pixels."
        )

        # Video points in normalized coordinates [u_norm, v_norm]
        video_pts = np.array([
            obs.normalized_coordinates for obs in detected_obs
            if obs.normalized_coordinates is not None
        ], dtype=np.float64)

        # Project 3D simulation trajectory onto selected 2D plane
        ee = sim_trajectory.ee_positions
        if shape_projection_plane.lower() == "xz":
            sim_2d_raw = ee[:, [0, 2]]
        elif shape_projection_plane.lower() == "yz":
            sim_2d_raw = ee[:, [1, 2]]
        else:  # default 'xy'
            sim_2d_raw = ee[:, [0, 1]]

        # Min-max normalize sim trajectory to [0, 1]
        sim_min = np.min(sim_2d_raw, axis=0)
        sim_max = np.max(sim_2d_raw, axis=0)
        sim_span = np.where((sim_max - sim_min) > 1e-6, sim_max - sim_min, 1.0)
        sim_2d_norm = (sim_2d_raw - sim_min) / sim_span

        # Resample both curves uniformly
        resampled_video = resample_2d_trajectory(video_pts, num_resample_points)
        resampled_sim = resample_2d_trajectory(sim_2d_norm, num_resample_points)

        # Calculate raw unaligned deviation
        raw_point_diffs = np.linalg.norm(resampled_video - resampled_sim, axis=1)
        raw_mean_dev = float(np.mean(raw_point_diffs))
        raw_rms_dev = float(np.sqrt(np.mean(raw_point_diffs ** 2)))
        raw_max_dev = float(np.max(raw_point_diffs))

        # Apply documented Procrustes alignment if requested
        if apply_procrustes_alignment:
            aligned_video, procrustes_doc = procrustes_align_2d(resampled_video, resampled_sim)
            compare_video = aligned_video
            compare_sim = resampled_sim
            alignment_doc = {
                "method": "normalized_progress_resampling_with_2d_procrustes",
                "projection_plane": shape_projection_plane.lower(),
                "sim_normalization_bounds": {
                    "min": [round(float(v), 4) for v in sim_min],
                    "max": [round(float(v), 4) for v in sim_max],
                },
                "procrustes_transform": procrustes_doc,
                "resampled_points_count": num_resample_points,
            }
        else:
            compare_video = resampled_video
            compare_sim = resampled_sim
            alignment_doc = {
                "method": "normalized_bounding_alignment",
                "projection_plane": shape_projection_plane.lower(),
                "sim_normalization_bounds": {
                    "min": [round(float(v), 4) for v in sim_min],
                    "max": [round(float(v), 4) for v in sim_max],
                },
                "resampled_points_count": num_resample_points,
            }

        video_path_len = compute_2d_path_length(compare_video)
        sim_path_len = compute_2d_path_length(compare_sim)

        point_diffs = np.linalg.norm(compare_video - compare_sim, axis=1)
        mean_dev = float(np.mean(point_diffs))
        rms_dev = float(np.sqrt(np.mean(point_diffs ** 2)))
        max_dev = float(np.max(point_diffs))

        if apply_procrustes_alignment and raw_rms_dev > 1.5 * rms_dev:
            reduction_pct = round((1.0 - (rms_dev / max(raw_rms_dev, 1e-9))) * 100, 1)
            warnings.append(
                f"Alignment impact notice: 2D Procrustes alignment reduced deviation from {raw_rms_dev:.4f} to {rms_dev:.4f} "
                f"({reduction_pct}% reduction via translation/rotation/scale adjustment). Check raw vs aligned deviations."
            )

        dtw_cost, dtw_avg_step, _ = compute_2d_dtw(compare_video, compare_sim)

        alignment_details = alignment_doc
        calibration_meta = None

    # In calibrated mode, raw deviation equals unadjusted reprojection deviation
    if effective_mode == ComparisonMode.CALIBRATED_PROJECTION:
        raw_mean_dev = mean_dev
        raw_rms_dev = rms_dev
        raw_max_dev = max_dev

    # Assemble metrics
    metrics = {
        "tracking_coverage": round(tracking_coverage, 4),
        "video_frames_total": video_trajectory.total_frames,
        "video_frames_detected": len(detected_obs),
        "video_duration_seconds": round(video_duration, 4),
        "sim_duration_seconds": round(sim_duration, 4),
        "timing_difference_seconds": round(timing_diff, 4),
        "video_path_length": round(video_path_len, 4),
        "sim_path_length": round(sim_path_len, 4),
        "raw_mean_trajectory_deviation": round(raw_mean_dev, 4),
        "raw_rms_trajectory_deviation": round(raw_rms_dev, 4),
        "raw_max_trajectory_deviation": round(raw_max_dev, 4),
        "mean_trajectory_deviation": round(mean_dev, 4),
        "rms_trajectory_deviation": round(rms_dev, 4),
        "max_trajectory_deviation": round(max_dev, 4),
        "dtw_cost": round(dtw_cost, 4),
        "dtw_average_step_distance": round(dtw_avg_step, 4),
        "resampled_points_count": num_resample_points,
    }

    provenance = {
        "video_file": video_trajectory.video_metadata.file_path,
        "video_source_is_synthetic": video_trajectory.is_synthetic or video_trajectory.video_metadata.is_synthetic,
        "sim_run_id": sim_trajectory.run_id,
        "sim_backend": sim_trajectory.simulator_backend,
        "sim_seed": sim_trajectory.seed,
        "sim_success": sim_trajectory.success,
        "sim_collision_count": sim_trajectory.collision_count,
    }

    report_id = str(uuid.uuid4())[:8]

    return VideoSimComparisonReport(
        report_id=report_id,
        comparison_mode=effective_mode,
        scenario_id=sim_trajectory.scenario_id,
        sim_policy_id=sim_trajectory.policy_id,
        physical_controller_id=physical_controller_id,
        comparison_type=comp_type,
        robot_model=robot_model,
        coordinate_system=coordinate_system,
        units=units,
        metrics=metrics,
        alignment_details=alignment_details,
        provenance=provenance,
        calibration_metadata=calibration_meta,
        warnings=warnings,
        unavailable_metrics=list(UNAVAILABLE_METRICS_LIST),
        disclaimers=disclaimers,
    )
