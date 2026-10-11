"""Simulation trajectory extraction utility for Robot CI.

Executes a specified policy and scenario on PyBullet or MockSimulator,
capturing actual 3D end-effector Cartesian positions, simulation timestamps,
and interaction events without fabricating data.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import uuid
import numpy as np

from robot_ci.policies.registry import get_policy
from robot_ci.scenarios.manager import ScenarioManager
from robot_ci.scenarios.models import Scenario
from robot_ci.replay.engine import ReplayEngine
from robot_ci.simulation.mock import MockSimulator
from robot_ci.simulation.pybullet_sim import PyBulletSimulator


@dataclass
class SimulatedTrajectoryResult:
    """Actual end-effector trajectory and execution metadata extracted from simulation."""
    run_id: str
    scenario_id: str
    policy_id: str
    policy_version: str
    simulator_backend: str
    seed: int
    total_steps: int
    duration_seconds: float
    success: bool
    collision_count: int
    ee_positions: np.ndarray      # Shape (N, 3) in meters [X, Y, Z]
    timesteps: List[int]
    timestamps: List[float]       # Seconds from rollout start
    events: List[Dict[str, Any]]  # Step-level events
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "scenario_id": self.scenario_id,
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "simulator_backend": self.simulator_backend,
            "seed": self.seed,
            "total_steps": self.total_steps,
            "duration_seconds": round(self.duration_seconds, 4),
            "success": self.success,
            "collision_count": self.collision_count,
            "created_at": self.created_at,
            "trajectory_points": len(self.ee_positions),
            "ee_positions": [list(p) for p in self.ee_positions],
            "timesteps": self.timesteps,
            "timestamps": [round(t, 4) for t in self.timestamps],
            "events": self.events,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SimulatedTrajectoryResult":
        ee_arr = np.array(data["ee_positions"], dtype=np.float64)
        return cls(
            run_id=data["run_id"],
            scenario_id=data["scenario_id"],
            policy_id=data["policy_id"],
            policy_version=data["policy_version"],
            simulator_backend=data["simulator_backend"],
            seed=int(data["seed"]),
            total_steps=int(data["total_steps"]),
            duration_seconds=float(data["duration_seconds"]),
            success=bool(data["success"]),
            collision_count=int(data["collision_count"]),
            ee_positions=ee_arr,
            timesteps=list(data["timesteps"]),
            timestamps=list(data["timestamps"]),
            events=list(data.get("events", [])),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )


def extract_simulated_trajectory(
    policy_name: str = "baseline",
    scenario_id: str = "reach_simple",
    simulator_name: str = "pybullet",
    seed: int = 42,
    scenarios_dir: Union[str, Path] = "scenarios",
) -> SimulatedTrajectoryResult:
    """Execute policy rollout in simulation and extract end-effector 3D trajectory.
    
    Args:
        policy_name: Name of policy in registry (e.g. 'baseline', 'candidate').
        scenario_id: Scenario identifier (e.g. 'reach_simple', 'pick_and_place_easy').
        simulator_name: Backend simulator ('pybullet' or 'mock').
        seed: Random seed for deterministic execution.
        scenarios_dir: Directory containing scenario YAML configurations.
    
    Returns:
        SimulatedTrajectoryResult containing Nx3 positions, timestamps, and status.
    
    Raises:
        FileNotFoundError: If scenarios directory is missing.
        ValueError: If scenario_id or policy_name is not found.
    """
    manager = ScenarioManager()
    scenarios = manager.load_scenarios(str(scenarios_dir))
    scenario = next((s for s in scenarios if s.scenario_id == scenario_id), None)
    if scenario is None:
        valid_ids = [s.scenario_id for s in scenarios]
        raise ValueError(
            f"Scenario '{scenario_id}' not found in '{scenarios_dir}'. Available: {valid_ids}"
        )

    policy = get_policy(policy_name)
    sim_cls = PyBulletSimulator if simulator_name.lower() == "pybullet" else MockSimulator
    sim = sim_cls()

    try:
        engine = ReplayEngine(sim)
        rollout = engine.run(policy=policy, scenario=scenario, seed=seed)

        # Extract end-effector positions from rollout states
        positions: List[List[float]] = []
        timesteps: List[int] = []
        timestamps: List[float] = []
        events: List[Dict[str, Any]] = []

        # In PyBullet/Mock: rollout.states contains step states
        # dt for PyBullet is (1/240) * 24 = 0.1s per step
        dt = 0.1 if simulator_name.lower() == "pybullet" else 0.1

        for step_idx, state in enumerate(rollout.states):
            ee = state.get("ee_position")
            if ee is not None and len(ee) == 3:
                positions.append([float(ee[0]), float(ee[1]), float(ee[2])])
                timesteps.append(step_idx)
                timestamps.append(step_idx * dt)

            events.append({
                "step": step_idx,
                "collision": state.get("collision_count", 0) > 0,
                "success": state.get("success", False),
            })

        ee_array = np.array(positions, dtype=np.float64)
        run_id = str(uuid.uuid4())[:8]
        duration = len(positions) * dt if positions else 0.0

        return SimulatedTrajectoryResult(
            run_id=run_id,
            scenario_id=scenario_id,
            policy_id=policy_name,
            policy_version=policy.version,
            simulator_backend=simulator_name.lower(),
            seed=seed,
            total_steps=rollout.total_steps,
            duration_seconds=duration,
            success=rollout.success,
            collision_count=rollout.collision_count,
            ee_positions=ee_array,
            timesteps=timesteps,
            timestamps=timestamps,
            events=events,
        )

    finally:
        sim.close()
