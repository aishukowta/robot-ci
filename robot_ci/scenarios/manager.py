import yaml
from pathlib import Path
from typing import List
from .models import Scenario

class ScenarioManager:
    """Manages loading and validating scenarios."""

    def load_scenarios(self, scenarios_dir: str) -> List[Scenario]:
        """Load and validate all .yaml and .yml scenarios in the directory."""
        path = Path(scenarios_dir)
        scenarios = []
        for file_path in path.glob("*.y*ml"):
            if file_path.suffix in [".yaml", ".yml"]:
                scenarios.append(self.load_scenario(str(file_path)))
        scenarios.sort(key=lambda s: s.scenario_id)
        return scenarios

    def load_scenario(self, filepath: str) -> Scenario:
        """Load a single scenario from a YAML file."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        scenario = Scenario(
            scenario_id=data.get("scenario_id", ""),
            task_name=data.get("task_name", ""),
            initial_state=data.get("initial_state", {}),
            target_state=data.get("target_state", {}),
            environment_params=data.get("environment_params", {}),
            max_episode_steps=data.get("max_episode_steps", 0),
            success_criteria=data.get("success_criteria", {}),
            safety_criteria=data.get("safety_criteria", {}),
            seed=data.get("seed", 42)
        )
        self.validate_scenario(scenario)
        return scenario

    def validate_scenario(self, scenario: Scenario) -> None:
        """Validate a scenario to ensure it meets requirements."""
        if not scenario.scenario_id:
            raise ValueError("scenario_id must be a non-empty string.")
        
        if scenario.task_name not in ["reach", "pick_and_place"]:
            raise ValueError(f"Invalid task_name: {scenario.task_name}. Must be 'reach' or 'pick_and_place'.")
            
        if "ee_position" not in scenario.initial_state or len(scenario.initial_state["ee_position"]) != 3:
            raise ValueError("initial_state must contain 'ee_position' as a list of 3 numbers.")
            
        if "position" not in scenario.target_state or len(scenario.target_state["position"]) != 3:
            raise ValueError("target_state must contain 'position' as a list of 3 numbers.")
            
        if scenario.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be > 0.")
            
        if "distance_threshold" not in scenario.success_criteria or scenario.success_criteria["distance_threshold"] <= 0:
            raise ValueError("success_criteria must contain a positive 'distance_threshold'.")
            
        if scenario.task_name == "pick_and_place":
            if "object_position" not in scenario.initial_state or len(scenario.initial_state["object_position"]) != 3:
                raise ValueError("initial_state must contain 'object_position' as a list of 3 numbers for pick_and_place task.")
