import numpy as np
from .base import Metric, MetricResult

def _resample_actions(actions: np.ndarray, num_points: int = 100) -> np.ndarray:
    """Resample actions to fixed number of points via linear interpolation."""
    T = len(actions)
    if T == 0:
        return np.zeros((num_points, 4))
    if T == 1:
        return np.tile(actions[0], (num_points, 1))
    old_indices = np.linspace(0, 1, T)
    new_indices = np.linspace(0, 1, num_points)
    resampled = np.zeros((num_points, 4))
    for dim in range(4):
        resampled[:, dim] = np.interp(new_indices, old_indices, actions[:, dim])
    return resampled

class ActionDifference(Metric):
    @property
    def name(self) -> str:
        return "action_difference"
        
    def compute(self, baseline_rollouts: dict, candidate_rollouts: dict) -> MetricResult:
        per_scenario = {}
        diffs = []
        for sid in baseline_rollouts:
            if sid not in candidate_rollouts:
                continue
                
            b_act = baseline_rollouts[sid].actions
            c_act = candidate_rollouts[sid].actions
            
            b_arr = np.array(b_act) if b_act else np.zeros((0, 4))
            c_arr = np.array(c_act) if c_act else np.zeros((0, 4))
            
            b_resampled = _resample_actions(b_arr)
            c_resampled = _resample_actions(c_arr)
            
            diff = b_resampled - c_resampled
            mean_l2 = float(np.mean(np.linalg.norm(diff, axis=1)))
            
            per_scenario[sid] = {"baseline": 0.0, "candidate": mean_l2, "delta": mean_l2}
            diffs.append(mean_l2)
            
        agg = float(np.mean(diffs)) if diffs else 0.0
        
        return MetricResult(
            metric_name=self.name,
            baseline_value=0.0,
            candidate_value=agg,
            delta=agg,
            per_scenario=per_scenario
        )
