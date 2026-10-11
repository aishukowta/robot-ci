"""Simulation-to-recorded trajectory comparison utility for Robot CI.

Computes geometric trajectory deviations (RMS, mean, max Euclidean, Dynamic Time Warping).
Handles empty, mismatched, and variable-frequency trajectories gracefully.
Strictly disclaims physical safety or automated sim-to-real transfer claims.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, Tuple, List
import math
import numpy as np

from .models import RecordedTelemetry, TelemetryValidationError
from robot_ci.replay.rollout import Rollout


TRAJECTORY_SAFETY_DISCLAIMER = (
    "Trajectory comparison metrics quantify kinematic and geometric similarity only. "
    "This comparison does NOT prove physical robot safety, collision avoidance, "
    "dynamical feasibility, or successful sim-to-real transfer."
)


@dataclass
class TrajectoryComparisonResult:
    """Result of comparing simulated and recorded trajectories."""
    sim_points_count: int
    recorded_points_count: int
    rms_euclidean_distance: float
    mean_euclidean_distance: float
    max_euclidean_distance: float
    dtw_cost: float
    dtw_average_distance: float
    alignment_method: str
    resampled_points_count: int
    disclaimer: str = TRAJECTORY_SAFETY_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sim_points_count": self.sim_points_count,
            "recorded_points_count": self.recorded_points_count,
            "rms_euclidean_distance": round(self.rms_euclidean_distance, 6),
            "mean_euclidean_distance": round(self.mean_euclidean_distance, 6),
            "max_euclidean_distance": round(self.max_euclidean_distance, 6),
            "dtw_cost": round(self.dtw_cost, 6),
            "dtw_average_distance": round(self.dtw_average_distance, 6),
            "alignment_method": self.alignment_method,
            "resampled_points_count": self.resampled_points_count,
            "disclaimer": self.disclaimer,
        }


def _validate_3d_positions(name: str, arr: np.ndarray) -> np.ndarray:
    """Validate that input is a non-empty, finite Nx3 numpy array."""
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"{name} must have shape (N, 3), representing 3D Cartesian coordinates [x, y, z]. Got shape {arr.shape}."
        )

    if len(arr) == 0:
        raise ValueError(f"{name} trajectory contains zero points.")

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} trajectory contains NaN or infinite values.")

    return arr.astype(np.float64)


def resample_trajectory(positions: np.ndarray, num_points: int) -> np.ndarray:
    """Resample an Nx3 trajectory to a fixed number of points via linear interpolation.
    
    Semantic distinction:
    Points are interpolated uniformly across normalized trajectory progress parameter
    tau in [0, 1] based on sample index progress. This represents normalized path progress
    rather than physical elapsed time synchronization, as simulated rollouts operate on
    discrete environment steps without physical wall-clock timestamps.
    
    Formula:
    p(tau_k) = (1 - alpha) * p_i + alpha * p_{i+1}
    where alpha = (tau_k - tau_i) / (tau_{i+1} - tau_i).
    """
    n = len(positions)
    if n == 0:
        return np.zeros((num_points, 3), dtype=np.float64)
    if n == 1 or num_points == 1:
        return np.tile(positions[0], (num_points, 1))

    old_t = np.linspace(0.0, 1.0, n)
    new_t = np.linspace(0.0, 1.0, num_points)

    resampled = np.zeros((num_points, 3), dtype=np.float64)
    for dim in range(3):
        resampled[:, dim] = np.interp(new_t, old_t, positions[:, dim])
    return resampled


def compute_dtw(traj_a: np.ndarray, traj_b: np.ndarray) -> Tuple[float, float, int]:
    """Compute Dynamic Time Warping (DTW) distance between two 3D trajectories.
    
    Formula:
    Cost matrix: C(i, j) = || a_i - b_j ||_2
    Accumulation matrix:
        D(0, 0) = 0, D(i, 0) = inf, D(0, j) = inf
        D(i, j) = C(i, j) + min( D(i-1, j), D(i, j-1), D(i-1, j-1) )
    
    Total DTW cost = D(N, M).
    Average DTW distance = D(N, M) / path_length.
    
    Returns:
        (total_cost, average_distance_per_step, path_length)
    """
    n, m = len(traj_a), len(traj_b)
    if n == 0 or m == 0:
        raise ValueError("Cannot compute DTW on empty trajectories.")

    # Distance matrix
    # Compute pairwise Euclidean distances
    diff = traj_a[:, np.newaxis, :] - traj_b[np.newaxis, :, :]  # (N, M, 3)
    cost_matrix = np.linalg.norm(diff, axis=2)  # (N, M)

    # Dynamic programming accumulated cost matrix
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i - 1, j - 1]
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    total_cost = float(dp[n, m])

    # Traceback optimal path
    i, j = n, m
    path_len = 1
    while i > 1 or j > 1:
        path_len += 1
        steps = [
            (dp[i - 1, j - 1], i - 1, j - 1),
            (dp[i - 1, j], i - 1, j),
            (dp[i, j - 1], i, j - 1),
        ]
        _, next_i, next_j = min(steps, key=lambda s: s[0])
        i, j = next_i, next_j

    avg_distance = total_cost / path_len if path_len > 0 else total_cost
    return total_cost, avg_distance, path_len


class TrajectoryComparator:
    """Compares simulated end-effector trajectories with recorded telemetry trajectories."""

    @staticmethod
    def extract_sim_trajectory(sim_data: Union[Rollout, RecordedTelemetry, np.ndarray, List]) -> np.ndarray:
        """Extract Nx3 positions from Rollout, RecordedTelemetry, or array."""
        if isinstance(sim_data, Rollout):
            if not sim_data.observations:
                return np.zeros((0, 3), dtype=np.float64)
            # Rollout observation[0:3] contains EE position in meters
            positions = [obs[0:3] for obs in sim_data.observations]
            return np.array(positions, dtype=np.float64)
        if isinstance(sim_data, RecordedTelemetry):
            return sim_data.get_ee_trajectory().astype(np.float64)
        return np.asarray(sim_data, dtype=np.float64)

    @staticmethod
    def extract_recorded_trajectory(rec_data: Union[RecordedTelemetry, np.ndarray, List]) -> np.ndarray:
        """Extract Nx3 positions from RecordedTelemetry or array."""
        if isinstance(rec_data, RecordedTelemetry):
            return rec_data.get_ee_trajectory().astype(np.float64)
        return np.asarray(rec_data, dtype=np.float64)

    def compare(
        self,
        sim_trajectory: Union[Rollout, np.ndarray, List],
        recorded_telemetry: Union[RecordedTelemetry, np.ndarray, List],
        num_resample_points: Optional[int] = None,
    ) -> TrajectoryComparisonResult:
        """Compare simulated and recorded trajectories.
        
        Args:
            sim_trajectory: Simulated Rollout or (N, 3) array of positions.
            recorded_telemetry: RecordedTelemetry instance or (M, 3) array of positions.
            num_resample_points: Number of points for linear time-normalized alignment.
                                 Defaults to max(len(sim), len(rec), 100).
        
        Returns:
            TrajectoryComparisonResult containing RMS, mean, max, and DTW distances.
        """
        # 1. Extract and validate 3D positions
        sim_pts = _validate_3d_positions("Simulated", self.extract_sim_trajectory(sim_trajectory))
        rec_pts = _validate_3d_positions("Recorded", self.extract_recorded_trajectory(recorded_telemetry))

        n_sim = len(sim_pts)
        n_rec = len(rec_pts)

        # 2. Resampling for point-to-point comparison along normalized sample/path progress [0, 1]
        k = num_resample_points if num_resample_points is not None else max(n_sim, n_rec, 100)
        resampled_sim = resample_trajectory(sim_pts, k)
        resampled_rec = resample_trajectory(rec_pts, k)

        diff = resampled_sim - resampled_rec
        pointwise_dist = np.linalg.norm(diff, axis=1)

        rms_dist = float(np.sqrt(np.mean(pointwise_dist ** 2)))
        mean_dist = float(np.mean(pointwise_dist))
        max_dist = float(np.max(pointwise_dist))

        # 3. Dynamic Time Warping on raw trajectories (preserves temporal warping flexibility)
        dtw_cost, dtw_avg, _ = compute_dtw(sim_pts, rec_pts)

        return TrajectoryComparisonResult(
            sim_points_count=n_sim,
            recorded_points_count=n_rec,
            rms_euclidean_distance=rms_dist,
            mean_euclidean_distance=mean_dist,
            max_euclidean_distance=max_dist,
            dtw_cost=dtw_cost,
            dtw_average_distance=dtw_avg,
            alignment_method="normalized_path_progress_resampling",
            resampled_points_count=k,
        )
