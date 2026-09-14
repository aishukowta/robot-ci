from .base import Metric, MetricResult

class RewardDifference(Metric):
    @property
    def name(self) -> str:
        return "reward_difference"
        
    def compute(self, baseline_rollouts: dict, candidate_rollouts: dict) -> MetricResult:
        per_scenario = {}
        baseline_vals = []
        candidate_vals = []
        for sid in baseline_rollouts:
            if sid not in candidate_rollouts:
                continue
            b_val = float(sum(baseline_rollouts[sid].rewards))
            c_val = float(sum(candidate_rollouts[sid].rewards))
            delta = c_val - b_val
            per_scenario[sid] = {"baseline": b_val, "candidate": c_val, "delta": delta}
            baseline_vals.append(b_val)
            candidate_vals.append(c_val)
        
        baseline_agg = sum(baseline_vals) / len(baseline_vals) if baseline_vals else 0.0
        candidate_agg = sum(candidate_vals) / len(candidate_vals) if candidate_vals else 0.0
        delta = candidate_agg - baseline_agg
        
        return MetricResult(
            metric_name=self.name,
            baseline_value=baseline_agg,
            candidate_value=candidate_agg,
            delta=delta,
            per_scenario=per_scenario
        )
