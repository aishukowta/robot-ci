"""Data models for evidence-based diagnosis in Robot CI.

Combines simulation policy regression findings with real-to-simulation trajectory
comparison reports to produce an evidence-based assessment of observed failures.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, List, Optional


class DiagnosisCategory(str, Enum):
    """Evidence-based diagnosis outcomes."""
    POLICY_REGRESSION_LIKELY = "POLICY_REGRESSION_LIKELY"
    SIM_TO_REAL_MISMATCH_LIKELY = "SIM_TO_REAL_MISMATCH_LIKELY"
    BOTH_ISSUES_LIKELY = "BOTH_ISSUES_LIKELY"
    NO_REGRESSION_DETECTED = "NO_REGRESSION_DETECTED"
    INCONCLUSIVE_INSUFFICIENT_EVIDENCE = "INCONCLUSIVE_INSUFFICIENT_EVIDENCE"


DIAGNOSIS_LIMITATIONS_DISCLAIMER = (
    "This assessment reflects an evidence-based evaluation of available test logs, "
    "simulation rollouts, and video tracking observations. It does NOT constitute definitive "
    "causal proof. Monocular video observations cannot measure 3D contact forces, motor torques, "
    "internal control loop timing, or physical collision dynamics."
)


@dataclass
class EvidenceItem:
    """Documented evidence item with source provenance and assessment."""
    source_type: str            # "SIMULATION_REGRESSION", "VIDEO_TRAJECTORY_COMPARISON", "OFFLINE_TELEMETRY"
    identifier: str             # e.g. run_id, report_id
    provenance: str             # "SYNTHETIC_FIXTURE", "PYBULLET_SIMULATION", "PHYSICAL_ROBOT_RECORDING", etc.
    is_physical_hardware: bool # Strictly False for synthetic fixtures and simulations
    is_valid: bool
    summary: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_type": self.source_type,
            "identifier": self.identifier,
            "provenance": self.provenance,
            "is_physical_hardware": self.is_physical_hardware,
            "is_valid": self.is_valid,
            "summary": self.summary,
            "details": self.details,
        }


@dataclass
class DiagnosisReport:
    """Structured report explaining evidence-based diagnosis."""
    diagnosis_id: str
    category: DiagnosisCategory
    summary_explanation: str
    recommended_next_step: str
    evidence_sources: List[EvidenceItem]
    supporting_findings: List[str]
    conflicting_findings: List[str]
    missing_evidence: List[str]
    data_quality_warnings: List[str]
    unsupported_conclusions: List[str]
    limitations: List[str] = field(default_factory=lambda: [DIAGNOSIS_LIMITATIONS_DISCLAIMER])
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagnosis_id": self.diagnosis_id,
            "category": self.category.value if isinstance(self.category, DiagnosisCategory) else str(self.category),
            "summary_explanation": self.summary_explanation,
            "recommended_next_step": self.recommended_next_step,
            "evidence_sources": [e.to_dict() for e in self.evidence_sources],
            "supporting_findings": self.supporting_findings,
            "conflicting_findings": self.conflicting_findings,
            "missing_evidence": self.missing_evidence,
            "data_quality_warnings": self.data_quality_warnings,
            "unsupported_conclusions": self.unsupported_conclusions,
            "limitations": self.limitations,
            "created_at": self.created_at,
        }
