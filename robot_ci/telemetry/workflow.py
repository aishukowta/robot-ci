"""Offline telemetry analysis report model and end-to-end integration workflow.

Produces structured, verifiable offline evaluation reports without pretending to be
live execution or altering CI regression schemas.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import uuid

from .models import (
    RecordedTelemetry,
    TelemetryValidationError,
    TelemetryParseError,
    UnsupportedFormatError,
)
from .reader import load_telemetry
from .replay import OfflineTelemetrySession
from .comparator import (
    TrajectoryComparator,
    TrajectoryComparisonResult,
    TRAJECTORY_SAFETY_DISCLAIMER,
)


@dataclass
class OfflineAnalysisReport:
    """Structured report produced by offline telemetry ingestion and analysis.
    
    Explicitly decoupled from TestRunResult to prevent recorded logs from
    being misrepresented as live CI test executions.
    """
    report_id: str
    created_at: str
    telemetry_source: str
    input_file: str
    provenance: Dict[str, Any]
    sample_count: int
    time_range: Dict[str, Any]
    available_fields: List[str]
    missing_fields: List[str]
    validation_summary: Dict[str, Any]
    supported_metrics: List[str]
    unsupported_metrics: List[str]
    unsupported_reasons: Dict[str, str]
    trajectory_comparison: Optional[Dict[str, Any]] = None
    reference_source: Optional[str] = None
    notes: Optional[str] = None
    is_synthetic: bool = True
    is_live_execution: bool = False  # Always False for offline analysis
    disclaimer: str = TRAJECTORY_SAFETY_DISCLAIMER

    def __post_init__(self):
        # Guarantee mutual exclusivity and semantic consistency between
        # supported_metrics, unsupported_metrics, and unsupported_reasons.
        self.supported_metrics = list(dict.fromkeys(self.supported_metrics))
        self.unsupported_metrics = list(dict.fromkeys(m for m in self.unsupported_metrics if m not in self.supported_metrics))
        
        # Remove any unsupported reason entries for metrics that are actually supported
        for m in self.supported_metrics:
            if m in self.unsupported_reasons:
                del self.unsupported_reasons[m]

        # Guarantee an explanatory reason for every unsupported metric
        for m in self.unsupported_metrics:
            if m not in self.unsupported_reasons:
                if m == "collision_count":
                    self.unsupported_reasons[m] = "Raw telemetry lacks collision ground-truth sensing."
                elif m == "success_rate":
                    self.unsupported_reasons[m] = "Raw telemetry lacks ground-truth task completion events."
                elif m == "reward_difference":
                    self.unsupported_reasons[m] = "Rewards are synthetic simulation artifacts and are not present in recorded telemetry."
                else:
                    self.unsupported_reasons[m] = f"Metric '{m}' cannot be computed from recorded telemetry data."


    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "created_at": self.created_at,
            "telemetry_source": self.telemetry_source,
            "input_file": self.input_file,
            "provenance": self.provenance,
            "sample_count": self.sample_count,
            "time_range": self.time_range,
            "available_fields": self.available_fields,
            "missing_fields": self.missing_fields,
            "validation_summary": self.validation_summary,
            "supported_metrics": self.supported_metrics,
            "unsupported_metrics": self.unsupported_metrics,
            "unsupported_reasons": self.unsupported_reasons,
            "trajectory_comparison": self.trajectory_comparison,
            "reference_source": self.reference_source,
            "notes": self.notes,
            "is_synthetic": self.is_synthetic,
            "is_live_execution": self.is_live_execution,
            "disclaimer": self.disclaimer,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, filepath: Union[str, Path]) -> str:
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(self.to_json(), encoding="utf-8")
        return str(out_path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OfflineAnalysisReport":
        return cls(
            report_id=data["report_id"],
            created_at=data["created_at"],
            telemetry_source=data.get("telemetry_source", "unknown"),
            input_file=data.get("input_file", ""),
            provenance=data.get("provenance", {}),
            sample_count=data.get("sample_count", 0),
            time_range=data.get("time_range", {}),
            available_fields=data.get("available_fields", []),
            missing_fields=data.get("missing_fields", []),
            validation_summary=data.get("validation_summary", {}),
            supported_metrics=data.get("supported_metrics", []),
            unsupported_metrics=data.get("unsupported_metrics", []),
            unsupported_reasons=data.get("unsupported_reasons", {}),
            trajectory_comparison=data.get("trajectory_comparison"),
            reference_source=data.get("reference_source"),
            notes=data.get("notes"),
            is_synthetic=data.get("is_synthetic", True),
            is_live_execution=False,
            disclaimer=data.get("disclaimer", TRAJECTORY_SAFETY_DISCLAIMER),
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "OfflineAnalysisReport":
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Offline report not found: {path}")
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        return cls.from_dict(data)


class OfflineReportStore:
    """Manages persistence and loading of offline telemetry analysis reports."""

    def __init__(self, directory: Union[str, Path] = "results/telemetry"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, report: OfflineAnalysisReport, filename: Optional[str] = None) -> str:
        name = filename or f"offline_report_{report.report_id}.json"
        path = self.directory / name
        return report.save(path)

    def load(self, filepath: Union[str, Path]) -> OfflineAnalysisReport:
        return OfflineAnalysisReport.load(filepath)

    def load_all(self) -> List[OfflineAnalysisReport]:
        files = sorted(self.directory.glob("offline_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        reports = []
        for f in files:
            try:
                reports.append(self.load(f))
            except Exception:
                continue
        return reports


class OfflineTelemetryWorkflow:
    """Orchestrates end-to-end ingestion, validation, comparison, and reporting."""

    def __init__(self, comparator: Optional[TrajectoryComparator] = None):
        self.comparator = comparator or TrajectoryComparator()

    def analyze(
        self,
        input_path: Union[str, Path],
        reference_path: Optional[Union[str, Path]] = None,
        sim_scenario: Optional[str] = None,
        sim_policy: str = "baseline",
        scenarios_dir: str = "scenarios",
        simulator_name: str = "pybullet",
        notes: Optional[str] = None,
    ) -> OfflineAnalysisReport:
        """Run offline ingestion and analysis on a telemetry file.
        
        Args:
            input_path: Path to .json or .jsonl telemetry recording.
            reference_path: Optional path to reference telemetry file for trajectory comparison.
            sim_scenario: Optional scenario ID to simulate as comparison baseline.
            sim_policy: Policy name to run if sim_scenario is specified.
            scenarios_dir: Directory containing scenarios if simulating.
            simulator_name: Simulator backend for reference simulation ('pybullet' or 'mock').
            notes: Optional descriptive notes.
        
        Returns:
            OfflineAnalysisReport instance.
        """
        in_file = Path(input_path)
        if not in_file.exists():
            raise FileNotFoundError(f"Telemetry input file not found: {in_file}")

        # 1. Ingest and validate telemetry
        telemetry = load_telemetry(in_file, validate=False)
        report_val = telemetry.validate(raise_on_error=False)

        session = OfflineTelemetrySession(telemetry)
        metrics_info = session.get_supported_metrics()

        # 2. Extract available vs missing fields
        available_fields = ["timestamp"]
        missing_fields = []
        if report_val.has_ee_positions:
            available_fields.append("ee_position")
        else:
            missing_fields.append("ee_position")

        if report_val.has_joint_positions:
            available_fields.append("joint_positions")
        else:
            missing_fields.append("joint_positions")

        if report_val.has_actions:
            available_fields.append("action")
        else:
            missing_fields.append("action")

        if report_val.has_events:
            available_fields.append("events")
        else:
            missing_fields.append("events")

        time_range = {
            "start_time": telemetry.start_time,
            "end_time": telemetry.end_time,
            "duration_seconds": round(telemetry.duration_seconds, 6),
            "time_basis": telemetry.provenance.time_basis,
        }

        # 3. Trajectory comparison if requested and compatible
        comparison_dict = None
        reference_source = None

        if reference_path is not None:
            ref_file = Path(reference_path)
            ref_telemetry = load_telemetry(ref_file, validate=True)
            reference_source = f"file:{ref_file.name}"
            if report_val.has_ee_positions:
                comp_result = self.comparator.compare(ref_telemetry, telemetry)
                comparison_dict = comp_result.to_dict()
            else:
                comparison_dict = {"error": "Recorded telemetry missing 3D end-effector positions for comparison."}

        elif sim_scenario is not None:
            # Run reference simulation
            from robot_ci.policies.registry import get_policy
            from robot_ci.scenarios.manager import ScenarioManager
            from robot_ci.replay.engine import ReplayEngine
            from robot_ci.simulation.mock import MockSimulator
            from robot_ci.simulation.pybullet_sim import PyBulletSimulator

            mgr = ScenarioManager()
            scenarios = mgr.load_scenarios(scenarios_dir)
            target_scenario = next((s for s in scenarios if s.scenario_id == sim_scenario), None)
            if target_scenario is None:
                raise ValueError(f"Scenario '{sim_scenario}' not found in '{scenarios_dir}'.")

            policy = get_policy(sim_policy)
            sim_cls = PyBulletSimulator if simulator_name == "pybullet" else MockSimulator
            sim = sim_cls()
            engine = ReplayEngine(sim)
            sim_rollout = engine.run(policy, target_scenario, seed=target_scenario.seed)

            reference_source = f"sim:{simulator_name}:{sim_scenario}:{sim_policy}"
            if report_val.has_ee_positions:
                comp_result = self.comparator.compare(sim_rollout, telemetry)
                comparison_dict = comp_result.to_dict()
            else:
                comparison_dict = {"error": "Recorded telemetry missing 3D end-effector positions for comparison."}

        report_id = str(uuid.uuid4())[:8]
        created_at = datetime.now(timezone.utc).isoformat()

        return OfflineAnalysisReport(
            report_id=report_id,
            created_at=created_at,
            telemetry_source=telemetry.provenance.source_id,
            input_file=str(in_file),
            provenance=telemetry.provenance.to_dict(),
            sample_count=telemetry.sample_count,
            time_range=time_range,
            available_fields=available_fields,
            missing_fields=missing_fields,
            validation_summary=report_val.to_dict(),
            supported_metrics=metrics_info["supported"],
            unsupported_metrics=metrics_info["unsupported"],
            unsupported_reasons=metrics_info["reasons"],
            trajectory_comparison=comparison_dict,
            reference_source=reference_source,
            notes=notes,
            is_synthetic=telemetry.provenance.is_synthetic,
            is_live_execution=False,
            disclaimer=TRAJECTORY_SAFETY_DISCLAIMER,
        )
