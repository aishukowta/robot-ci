import yaml
from dataclasses import dataclass, asdict
from typing import List
from robot_ci.evaluation.metrics.base import MetricResult

@dataclass
class RegressionFinding:
    metric_name: str
    baseline_value: float
    candidate_value: float
    delta: float
    threshold: float
    regression_detected: bool
    severity: str  # "none", "minor", "major", "critical"
    reason: str
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data) -> 'RegressionFinding':
        return cls(**data)

class RegressionDetector:
    def __init__(self, config_path: str = "config/regression_thresholds.yaml"):
        with open(config_path) as f:
            self.thresholds = yaml.safe_load(f)
    
    def analyze(self, metric_results: List[MetricResult]) -> List[RegressionFinding]:
        findings = []
        for result in metric_results:
            config = self.thresholds.get(result.metric_name)
            if config is None:
                continue
            
            direction = config["direction"]
            threshold = config["threshold"]
            comparison = config["comparison"]
            severity_levels = config["severity_levels"]

            if direction == "monitor":
                reason = (
                    f"{result.metric_name}: baseline={result.baseline_value:.4f}, "
                    f"candidate={result.candidate_value:.4f}, delta={result.delta:.4f}; monitored only"
                )
                findings.append(RegressionFinding(
                    metric_name=result.metric_name,
                    baseline_value=result.baseline_value,
                    candidate_value=result.candidate_value,
                    delta=result.delta,
                    threshold=threshold,
                    regression_detected=False,
                    severity="none",
                    reason=reason,
                ))
                continue
            
            # Compute magnitude based on direction
            if direction == "decrease":
                raw_delta = result.baseline_value - result.candidate_value
            else:  # increase
                raw_delta = result.candidate_value - result.baseline_value
            
            # Apply comparison mode
            if comparison == "relative":
                magnitude = raw_delta / max(abs(result.baseline_value), 1e-6)
            else:
                magnitude = raw_delta
            
            # Determine regression and severity
            regression = magnitude >= threshold
            if magnitude >= severity_levels["critical"]:
                severity = "critical"
            elif magnitude >= severity_levels["major"]:
                severity = "major"
            elif magnitude >= severity_levels["minor"]:
                severity = "minor"
            else:
                severity = "none"
                regression = False
            
            reason = f"{result.metric_name}: baseline={result.baseline_value:.4f}, candidate={result.candidate_value:.4f}, delta={raw_delta:.4f}, threshold={threshold}"
            if regression:
                reason += f" -> {severity} regression detected"
            
            findings.append(RegressionFinding(
                metric_name=result.metric_name,
                baseline_value=result.baseline_value,
                candidate_value=result.candidate_value,
                delta=raw_delta,
                threshold=threshold,
                regression_detected=regression,
                severity=severity,
                reason=reason,
            ))
        return findings
