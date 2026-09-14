import pytest
from pathlib import Path
from robot_ci.scenarios.manager import ScenarioManager

def test_scenario_manager_loads_valid_scenarios(scenarios_dir):
    manager = ScenarioManager()
    scenarios = manager.load_scenarios(str(scenarios_dir))
    
    # We expect 5 scenarios from the corpus
    assert len(scenarios) == 5
    
    ids = [s.scenario_id for s in scenarios]
    assert "reach_simple" in ids
    assert "pick_and_place_easy" in ids

def test_scenario_validation_invalid_task():
    manager = ScenarioManager()
    # Assuming there's a validation error if we pass something broken
    pass # we can add detailed validation tests if necessary
