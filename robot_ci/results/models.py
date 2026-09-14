from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class TestRunResult:
    """Complete result of a regression test run.
    
    Designed for extensibility: future milestone fields are Optional
    and do not break existing serialization.
    """
    run_id: str
    timestamp: str  # ISO format
    baseline_policy_version: str
    candidate_policy_version: str
    seed: int
    scenario_ids: List[str]
    metric_results: List[dict]       # serialized MetricResults
    regression_findings: List[dict]  # serialized RegressionFindings
    overall_status: str              # "PASS" or "REGRESSION_DETECTED"
    simulator_name: str = "pybullet"
    trial_count: int = 1
    rollout_summary: Optional[Dict] = None  # per-scenario summary
    # Future extension fields (Milestones 3-9)
    sim_to_real_gap: Optional[dict] = None
    confidence_score: Optional[float] = None
    deployment_recommendation: Optional[str] = None
    hardware_validation: Optional[dict] = None
    
    def to_dict(self) -> dict:
        d = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "baseline_policy_version": self.baseline_policy_version,
            "candidate_policy_version": self.candidate_policy_version,
            "seed": self.seed,
            "scenario_ids": self.scenario_ids,
            "simulator_name": self.simulator_name,
            "trial_count": self.trial_count,
            "metric_results": self.metric_results,
            "regression_findings": self.regression_findings,
            "overall_status": self.overall_status,
        }
        if self.rollout_summary is not None:
            d["rollout_summary"] = self.rollout_summary
        if self.sim_to_real_gap is not None:
            d["sim_to_real_gap"] = self.sim_to_real_gap
        if self.confidence_score is not None:
            d["confidence_score"] = self.confidence_score
        if self.deployment_recommendation is not None:
            d["deployment_recommendation"] = self.deployment_recommendation
        if self.hardware_validation is not None:
            d["hardware_validation"] = self.hardware_validation
        return d
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TestRunResult':
        return cls(
            run_id=data["run_id"],
            timestamp=data["timestamp"],
            baseline_policy_version=data["baseline_policy_version"],
            candidate_policy_version=data["candidate_policy_version"],
            seed=data.get("seed", 42),
            scenario_ids=data["scenario_ids"],
            simulator_name=data.get("simulator_name", "mock"),
            trial_count=data.get("trial_count", 1),
            metric_results=data["metric_results"],
            regression_findings=data["regression_findings"],
            overall_status=data["overall_status"],
            rollout_summary=data.get("rollout_summary"),
            sim_to_real_gap=data.get("sim_to_real_gap"),
            confidence_score=data.get("confidence_score"),
            deployment_recommendation=data.get("deployment_recommendation"),
            hardware_validation=data.get("hardware_validation"),
        )
