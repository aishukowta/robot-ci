"""Evidence-based diagnosis package for Robot CI.

Evaluates multi-source evidence from simulation regression reports and real-to-simulation
video trajectory comparisons to diagnose policy regressions and sim-to-real domain gaps.
"""

from .models import (
    DiagnosisCategory,
    DiagnosisReport,
    EvidenceItem,
    DIAGNOSIS_LIMITATIONS_DISCLAIMER,
)
from .engine import evaluate_diagnosis

__all__ = [
    "DiagnosisCategory",
    "DiagnosisReport",
    "EvidenceItem",
    "DIAGNOSIS_LIMITATIONS_DISCLAIMER",
    "evaluate_diagnosis",
]
