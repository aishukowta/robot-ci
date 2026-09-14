from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Dict

@dataclass
class MetricResult:
    """Result of computing a single metric across scenarios."""
    metric_name: str
    baseline_value: float    # aggregate baseline value
    candidate_value: float   # aggregate candidate value
    delta: float             # difference (sign depends on metric direction)
    per_scenario: Dict[str, dict]  # scenario_id -> {"baseline": float, "candidate": float, "delta": float}
    
    def to_dict(self) -> dict:
        return {"metric_name": self.metric_name, "baseline_value": self.baseline_value,
                "candidate_value": self.candidate_value, "delta": self.delta,
                "per_scenario": self.per_scenario}
    
    @classmethod
    def from_dict(cls, data: dict) -> 'MetricResult':
        return cls(**data)

class Metric(ABC):
    """Abstract base class for evaluation metrics."""
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @abstractmethod
    def compute(self, baseline_rollouts: dict, candidate_rollouts: dict) -> MetricResult:
        """Compute metric from matched rollouts. Both dicts map scenario_id -> Rollout."""
        ...
