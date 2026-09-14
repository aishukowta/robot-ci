class EvaluationEngine:
    def __init__(self, metrics=None):
        if metrics is None:
            from robot_ci.evaluation.metrics import DEFAULT_METRICS
            metrics = DEFAULT_METRICS
        self.metrics = metrics
    
    def evaluate(self, baseline_rollouts, candidate_rollouts) -> list:
        results = []
        for metric in self.metrics:
            result = metric.compute(baseline_rollouts, candidate_rollouts)
            results.append(result)
        return results
