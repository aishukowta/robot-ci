import pytest
from pathlib import Path
from robot_ci.scenarios.manager import ScenarioManager
from robot_ci.scenarios.models import Scenario

def test_scenario_manager_loads_valid_scenarios(scenarios_dir):
    manager = ScenarioManager()
    scenarios = manager.load_scenarios(str(scenarios_dir))
    
    # We expect 5 scenarios from the corpus
    assert len(scenarios) == 5
    
    ids = [s.scenario_id for s in scenarios]
    assert ids == sorted(ids)
    assert "reach_simple" in ids
    assert "reach_diagonal" in ids
    assert "pick_and_place_easy" in ids
    assert "pick_and_place_medium" in ids
    assert "pick_and_place_hard" in ids


def test_scenario_validation_valid_reach_scenario():
    manager = ScenarioManager()
    sc = Scenario(
        scenario_id="valid_reach",
        task_name="reach",
        initial_state={"ee_position": [0.1, 0.2, 0.3]},
        target_state={"position": [0.4, 0.5, 0.6]},
        environment_params={},
        max_episode_steps=100,
        success_criteria={"distance_threshold": 0.05},
        safety_criteria={},
        seed=42,
    )
    # Should not raise
    manager.validate_scenario(sc)


def test_scenario_validation_missing_scenario_id():
    manager = ScenarioManager()
    sc = Scenario(
        scenario_id="",
        task_name="reach",
        initial_state={"ee_position": [0.1, 0.2, 0.3]},
        target_state={"position": [0.4, 0.5, 0.6]},
        environment_params={},
        max_episode_steps=100,
        success_criteria={"distance_threshold": 0.05},
        safety_criteria={},
    )
    with pytest.raises(ValueError, match="scenario_id must be a non-empty string"):
        manager.validate_scenario(sc)


def test_scenario_validation_invalid_task():
    manager = ScenarioManager()
    sc = Scenario(
        scenario_id="test_invalid_task",
        task_name="invalid_task_type",
        initial_state={"ee_position": [0.1, 0.2, 0.3]},
        target_state={"position": [0.4, 0.5, 0.6]},
        environment_params={},
        max_episode_steps=100,
        success_criteria={"distance_threshold": 0.05},
        safety_criteria={},
    )
    with pytest.raises(ValueError, match="Invalid task_name"):
        manager.validate_scenario(sc)


def test_scenario_validation_invalid_ee_position():
    manager = ScenarioManager()
    # Case 1: missing ee_position
    sc1 = Scenario(
        scenario_id="test_ee",
        task_name="reach",
        initial_state={},
        target_state={"position": [0.4, 0.5, 0.6]},
        environment_params={},
        max_episode_steps=100,
        success_criteria={"distance_threshold": 0.05},
        safety_criteria={},
    )
    with pytest.raises(ValueError, match="initial_state must contain 'ee_position'"):
        manager.validate_scenario(sc1)

    # Case 2: wrong dimension
    sc2 = Scenario(
        scenario_id="test_ee",
        task_name="reach",
        initial_state={"ee_position": [0.1, 0.2]},
        target_state={"position": [0.4, 0.5, 0.6]},
        environment_params={},
        max_episode_steps=100,
        success_criteria={"distance_threshold": 0.05},
        safety_criteria={},
    )
    with pytest.raises(ValueError, match="initial_state must contain 'ee_position'"):
        manager.validate_scenario(sc2)


def test_scenario_validation_invalid_target_position():
    manager = ScenarioManager()
    sc = Scenario(
        scenario_id="test_target",
        task_name="reach",
        initial_state={"ee_position": [0.1, 0.2, 0.3]},
        target_state={"position": [0.4, 0.5]},
        environment_params={},
        max_episode_steps=100,
        success_criteria={"distance_threshold": 0.05},
        safety_criteria={},
    )
    with pytest.raises(ValueError, match="target_state must contain 'position'"):
        manager.validate_scenario(sc)


def test_scenario_validation_invalid_max_episode_steps():
    manager = ScenarioManager()
    sc = Scenario(
        scenario_id="test_steps",
        task_name="reach",
        initial_state={"ee_position": [0.1, 0.2, 0.3]},
        target_state={"position": [0.4, 0.5, 0.6]},
        environment_params={},
        max_episode_steps=0,
        success_criteria={"distance_threshold": 0.05},
        safety_criteria={},
    )
    with pytest.raises(ValueError, match="max_episode_steps must be > 0"):
        manager.validate_scenario(sc)


def test_scenario_validation_invalid_distance_threshold():
    manager = ScenarioManager()
    sc = Scenario(
        scenario_id="test_threshold",
        task_name="reach",
        initial_state={"ee_position": [0.1, 0.2, 0.3]},
        target_state={"position": [0.4, 0.5, 0.6]},
        environment_params={},
        max_episode_steps=100,
        success_criteria={"distance_threshold": -0.01},
        safety_criteria={},
    )
    with pytest.raises(ValueError, match="success_criteria must contain a positive 'distance_threshold'"):
        manager.validate_scenario(sc)


def test_scenario_validation_pick_and_place_object_position():
    manager = ScenarioManager()
    # Missing object_position for pick_and_place
    sc = Scenario(
        scenario_id="test_pnp",
        task_name="pick_and_place",
        initial_state={"ee_position": [0.1, 0.2, 0.3]},
        target_state={"position": [0.4, 0.5, 0.6]},
        environment_params={},
        max_episode_steps=100,
        success_criteria={"distance_threshold": 0.05},
        safety_criteria={},
    )
    with pytest.raises(ValueError, match="object_position"):
        manager.validate_scenario(sc)


def test_scenario_manager_load_non_dict_yaml(tmp_path):
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("just a string\n", encoding="utf-8")
    manager = ScenarioManager()
    with pytest.raises(ValueError, match="must be a dictionary"):
        manager.load_scenario(str(yaml_file))


def test_scenario_manager_file_not_found():
    manager = ScenarioManager()
    with pytest.raises(FileNotFoundError):
        manager.load_scenario("non_existent_file.yaml")

