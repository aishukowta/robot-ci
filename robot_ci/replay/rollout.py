from dataclasses import dataclass
from typing import List, Dict, Any
import numpy as np

@dataclass
class Rollout:
    """Complete record of a single policy rollout on a scenario.
    
    Stores the full trajectory of observations, actions, states, and rewards.
    Designed for serialization to JSON for persistent storage.
    
    Seed is stored in metadata to support deterministic matched evaluation:
    the same seed is used for both baseline and candidate on the same scenario.
    """
    scenario_id: str
    policy_version: str
    seed: int
    observations: List[np.ndarray]  # len = total_steps + 1 (includes initial)
    actions: List[np.ndarray]       # len = total_steps
    states: List[Dict[str, Any]]    # len = total_steps + 1
    rewards: List[float]            # len = total_steps
    success: bool
    collision_count: int
    total_steps: int
    
    def to_dict(self) -> dict:
        return {
            "scenario_id": self.scenario_id,
            "policy_version": self.policy_version,
            "seed": self.seed,
            "observations": [obs.tolist() for obs in self.observations],
            "actions": [act.tolist() for act in self.actions],
            "states": self.states,
            "rewards": self.rewards,
            "success": self.success,
            "collision_count": self.collision_count,
            "total_steps": self.total_steps,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Rollout':
        return cls(
            scenario_id=data["scenario_id"],
            policy_version=data["policy_version"],
            seed=data["seed"],
            observations=[np.array(obs) for obs in data["observations"]],
            actions=[np.array(act) for act in data["actions"]],
            states=data["states"],
            rewards=data["rewards"],
            success=data["success"],
            collision_count=data["collision_count"],
            total_steps=data["total_steps"],
        )
