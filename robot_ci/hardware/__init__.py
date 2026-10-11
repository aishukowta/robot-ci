"""Robot CI Hardware Integration & Readiness Package.

Contains hardware readiness assessment, driver preconditions audit,
and hardware safety boundaries verification.
"""

from .readiness import (
    HardwareReadinessStatus,
    HardwareReadinessReport,
    audit_hardware_readiness,
)

__all__ = [
    "HardwareReadinessStatus",
    "HardwareReadinessReport",
    "audit_hardware_readiness",
]
