"""Offline replay and session management for recorded telemetry.

Maintains strict separation between offline log playback and live execution backends.
Never pretends to be a live simulator or physical hardware backend.
"""

from typing import Dict, Any, List, Optional
import numpy as np

from .models import (
    RecordedTelemetry,
    TelemetryValidationReport,
    TelemetryValidationError,
)
from robot_ci.replay.rollout import Rollout


class OfflineTelemetrySession:
    """Offline telemetry inspector and playback view.
    
    Provides structured inspection of recorded robot logs.
    Deliberately does NOT implement SimulatorBackend:
    stepping, physical actuation, or live environment resets are forbidden.
    """

    def __init__(self, telemetry: RecordedTelemetry):
        self.telemetry = telemetry
        self._validation_report = self.telemetry.validate(raise_on_error=False)

    @property
    def session_id(self) -> str:
        return self.telemetry.session_id

    @property
    def provenance(self) -> Dict[str, Any]:
        return self.telemetry.provenance.to_dict()

    @property
    def sample_count(self) -> int:
        return self.telemetry.sample_count

    @property
    def validation_report(self) -> TelemetryValidationReport:
        return self._validation_report

    # Explicit guard against backend confusion
    def step(self, *args, **kwargs):
        raise TypeError(
            "OfflineTelemetrySession is an offline log playback view, not an execution backend. "
            "Physical or simulated actuation is impossible on recorded telemetry."
        )

    def reset(self, *args, **kwargs):
        raise TypeError(
            "OfflineTelemetrySession cannot reset a physical robot or simulator. "
            "Use MockSimulator or PyBulletSimulator for live execution."
        )

    def get_supported_metrics(self) -> Dict[str, Any]:
        """Determine which evaluation metrics can validly be computed from this recording.
        
        Never fabricates rewards, collision counts, or success flags.
        """
        supported = []
        unsupported = []
        reasons = {}

        # 1. Trajectory deviation
        if self._validation_report.has_ee_positions:
            supported.append("trajectory_deviation")
        else:
            unsupported.append("trajectory_deviation")
            reasons["trajectory_deviation"] = "Missing end-effector 3D position data."

        # 2. Completion time
        if self.sample_count > 1 and self.telemetry.duration_seconds >= 0:
            supported.append("completion_time")
        else:
            unsupported.append("completion_time")
            reasons["completion_time"] = "Insufficient timestamps or zero duration."

        # 3. Action difference
        if self._validation_report.has_actions:
            supported.append("action_difference")
        else:
            unsupported.append("action_difference")
            reasons["action_difference"] = "No commanded actions were recorded in this log."

        # 4. Rewards (strictly simulation artifact)
        unsupported.append("reward_difference")
        reasons["reward_difference"] = "Rewards are synthetic simulation artifacts and are not present in recorded telemetry."

        # 5. Collision count
        if self._validation_report.has_events:
            # Check if any event has collision or contact
            contact_events = [
                rec.events for rec in self.telemetry.records
                if rec.events and ("collision" in rec.events or "contact" in rec.events)
            ]
            if contact_events:
                supported.append("collision_count")
            else:
                unsupported.append("collision_count")
                reasons["collision_count"] = "No collision/contact events recorded in telemetry."
        else:
            unsupported.append("collision_count")
            reasons["collision_count"] = "Raw telemetry lacks collision ground-truth sensing."

        # 6. Success rate
        if self._validation_report.has_events:
            success_events = [
                rec.events for rec in self.telemetry.records
                if rec.events and "success" in rec.events
            ]
            if success_events:
                supported.append("success_rate")
            else:
                unsupported.append("success_rate")
                reasons["success_rate"] = "No explicit success events recorded in telemetry."
        else:
            unsupported.append("success_rate")
            reasons["success_rate"] = "Raw telemetry lacks ground-truth task completion events."

        return {
            "supported": supported,
            "unsupported": unsupported,
            "reasons": reasons,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return a structured summary of the offline telemetry session."""
        metrics_info = self.get_supported_metrics()
        return {
            "session_id": self.session_id,
            "session_type": "OFFLINE_RECORDED_TELEMETRY",
            "provenance": self.provenance,
            "sample_count": self.sample_count,
            "start_time": self.telemetry.start_time,
            "end_time": self.telemetry.end_time,
            "duration_seconds": self.telemetry.duration_seconds,
            "is_valid": self._validation_report.is_valid,
            "validation_errors": self._validation_report.errors,
            "validation_warnings": self._validation_report.warnings,
            "supported_metrics": metrics_info["supported"],
            "unsupported_metrics": metrics_info["unsupported"],
            "unsupported_reasons": metrics_info["reasons"],
        }

    def to_rollout_adapter(self, scenario_id: Optional[str] = None) -> Rollout:
        """Convert compatible telemetry into a Rollout representation with explicit provenance.
        
        Only populates fields that exist in telemetry; does NOT invent rewards
        or collision counts.
        """
        if not self._validation_report.has_ee_positions:
            raise TelemetryValidationError(
                "Cannot adapt telemetry to Rollout: missing 3D end-effector positions."
            )

        # Reconstruct 10D observation dummy vectors matching Rollout's expectations
        # [ee_x, ee_y, ee_z, grip, 0, 0, 0, 0, 0, 0]
        observations = []
        for rec in self.telemetry.records:
            ee = rec.ee_position if rec.ee_position else [0.0, 0.0, 0.0]
            grip = rec.gripper_state if rec.gripper_state is not None else 1.0
            # 10-dimensional observation where first 3 are EE position
            obs = np.array([ee[0], ee[1], ee[2], grip, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            observations.append(obs)

        # Actions
        actions = []
        if self._validation_report.has_actions:
            for rec in self.telemetry.records:
                if rec.action is not None:
                    actions.append(np.array(rec.action, dtype=np.float32))

        # Check for explicit events
        success = False
        collision_count = 0
        if self._validation_report.has_events:
            for rec in self.telemetry.records:
                if rec.events:
                    if isinstance(rec.events, dict):
                        if rec.events.get("success", False):
                            success = True
                        if rec.events.get("collision", False) or rec.events.get("contact", False):
                            collision_count += 1
                    elif isinstance(rec.events, (list, tuple, set)):
                        if "success" in rec.events:
                            success = True
                        if "collision" in rec.events or "contact" in rec.events:
                            collision_count += 1

        sid = scenario_id or self.session_id
        policy_ver = f"offline_record:{self.telemetry.provenance.source_id}"

        return Rollout(
            scenario_id=sid,
            policy_version=policy_ver,
            seed=-1,  # Negative seed explicitly denotes non-synthetic/offline playback
            observations=observations,
            actions=actions,
            states=[{"provenance": "OFFLINE_RECORDED_TELEMETRY", "source": self.telemetry.provenance.source_id}],
            rewards=[],  # Empty list: rewards are never fabricated
            success=success,
            collision_count=collision_count,
            total_steps=len(actions) if actions else len(observations) - 1 if len(observations) > 1 else 0,
        )
