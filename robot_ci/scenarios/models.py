from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class Scenario:
    """A reusable test scenario for policy evaluation.
    
    Each scenario defines a complete test case with initial conditions,
    target state, environment parameters, and success/safety criteria.
    
    The seed field enables deterministic, matched evaluation: the SAME
    seed is used for both baseline and candidate rollouts on this scenario.
    """
    scenario_id: str
    task_name: str
    initial_state: Dict[str, Any]
    target_state: Dict[str, Any]
    environment_params: Dict[str, Any]
    max_episode_steps: int
    success_criteria: Dict[str, Any]
    safety_criteria: Dict[str, Any]
    seed: int = 42
