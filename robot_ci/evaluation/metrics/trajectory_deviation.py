import numpy as np
from .base import Metric, MetricResult

def _resample_trajectory(positions: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Resample trajectory to fixed number of points via linear interpolation.
    
    Args:
        positions: (T, 3) array of 3D positions
        num_points: target number of points
    Returns:
        (num_points, 3) resampled positions
    """
    T = len(positions)
    if T == 0:
        return np.zeros((num_points, 3))
    if T == 1:
        return np.tile(positions[0], (num_points, 1))
    old_indices = np.linspace(0, 1, T)
    new_indices = np.linspace(0, 1, num_points)
    resampled = np.zeros((num_points, 3))
    for dim in range(3):
        resampled[:, dim] = np.interp(new_indices, old_indices, positions[:, dim])
    return resampled

def _rms_euclidean_deviation(traj_a: np.ndarray, traj_b: np.ndarray) -> float:
    """RMS Euclidean deviation between two same-length (N, 3) trajectories."""
    diff = traj_a - traj_b
    per_point_dist_sq = np.sum(diff ** 2, axis=1)
    return float(np.sqrt(np.mean(per_point_dist_sq)))

class TrajectoryDeviation(Metric):
    @property
    def name(self) -> str:
        return "trajectory_deviation"
        
    def compute(self, baseline_rollouts: dict, candidate_rollouts: dict) -> MetricResult:
        per_scenario = {}
        deviations = []
        for sid in baseline_rollouts:
            if sid not in candidate_rollouts:
                continue
                
            b_obs = baseline_rollouts[sid].observations
            c_obs = candidate_rollouts[sid].observations
            
            b_traj = np.array([obs[0:3] for obs in b_obs]) if b_obs else np.zeros((0, 3))
            c_traj = np.array([obs[0:3] for obs in c_obs]) if c_obs else np.zeros((0, 3))
            
            b_resampled = _resample_trajectory(b_traj)
            c_resampled = _resample_trajectory(c_traj)
            
            dev = _rms_euclidean_deviation(b_resampled, c_resampled)
            
            per_scenario[sid] = {"baseline": 0.0, "candidate": dev, "delta": dev}
            deviations.append(dev)
            
        agg = float(np.mean(deviations)) if deviations else 0.0
        
        return MetricResult(
            metric_name=self.name,
            baseline_value=0.0,
            candidate_value=agg,
            delta=agg,
            per_scenario=per_scenario
        )
