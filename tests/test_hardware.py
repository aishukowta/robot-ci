import pytest
import numpy as np
from robot_ci.simulation.hardware import HardwareBackend
from robot_ci.simulation.safety import (
    HardwareConfig,
    HardwareSafetyValidator,
    HardwareConnectionError,
    SafetyValidationError,
    SafetyInterlockError,
)
from robot_ci.scenarios.models import Scenario
from robot_ci.pipeline.runner import PipelineRunner


def test_hardware_backend_default_status():
    backend = HardwareBackend()
    assert backend.name == "hardware"
    assert backend.is_connected is False
    assert backend.safe_stop_active is False

    status = backend.get_status()
    assert status["simulator"] == "hardware"
    assert status["is_connected"] is False

    state = backend.get_state()
    assert state["is_simulation"] is False
    assert state["robot_name"] == "unconfigured_arm"
    assert "timestamp" in state
    assert "wall_time" in state


def test_hardware_backend_unconnected_connect_fails():
    backend = HardwareBackend()
    with pytest.raises(HardwareConnectionError, match="unconfigured"):
        backend.connect()

    with pytest.raises(HardwareConnectionError, match="unavailable"):
        backend.connect("192.168.1.100:50051")


def test_hardware_backend_reset_and_step_fail_closed():
    backend = HardwareBackend()
    sc = Scenario(
        scenario_id="s1",
        task_name="reach",
        initial_state={"ee_position": [0.1, 0.2, 0.3]},
        target_state={"position": [0.4, 0.5, 0.6]},
        environment_params={},
        max_episode_steps=10,
        success_criteria={"distance_threshold": 0.05},
        safety_criteria={},
    )

    with pytest.raises(HardwareConnectionError, match="Hardware transport disconnected"):
        backend.reset(sc)

    with pytest.raises(HardwareConnectionError, match="Hardware transport disconnected"):
        backend.step(np.array([0.01, 0.01, 0.0, 1.0]))


def test_hardware_safety_invalid_action_rejection():
    validator = HardwareSafetyValidator()

    # None action
    with pytest.raises(SafetyValidationError, match="cannot be None"):
        validator.validate_action(None)

    # Wrong shape (3D instead of 4D)
    with pytest.raises(SafetyValidationError, match="expected 4D array"):
        validator.validate_action([0.1, 0.2, 0.3])

    # NaN / Inf non-finite values
    with pytest.raises(SafetyValidationError, match="non-finite values"):
        validator.validate_action([np.nan, 0.0, 0.0, 1.0])

    with pytest.raises(SafetyValidationError, match="non-finite values"):
        validator.validate_action([0.0, np.inf, 0.0, 1.0])


def test_hardware_safety_action_magnitude_limit():
    config = HardwareConfig(max_step_size=0.05)
    validator = HardwareSafetyValidator(config)

    # 0.04m step size -> OK (under 2x safety ceiling of 0.10m)
    validator.validate_action([0.03, 0.02, 0.0, 1.0])

    # 0.25m step size -> Rejection
    with pytest.raises(SafetyValidationError, match="exceeds safety limit"):
        validator.validate_action([0.20, 0.15, 0.0, 1.0])


def test_hardware_safety_workspace_bounds():
    config = HardwareConfig(workspace_bounds=[[0.0, -0.5, 0.0], [1.0, 0.5, 1.0]])
    validator = HardwareSafetyValidator(config)

    # Position inside bounds -> OK
    validator.validate_cartesian_position(np.array([0.5, 0.0, 0.5]))

    # Position outside workspace bounds -> Rejection
    with pytest.raises(SafetyValidationError, match="violates configured workspace bounds"):
        validator.validate_cartesian_position(np.array([1.5, 0.0, 0.5]))


def test_hardware_safety_joint_limits():
    config = HardwareConfig(
        joint_names=["joint1", "joint2"],
        joint_limits={"joint1": (-1.57, 1.57), "joint2": (-0.5, 0.5)},
    )
    validator = HardwareSafetyValidator(config)

    # Within bounds -> OK
    validator.validate_joint_positions(np.array([0.0, 0.2]))

    # Exceeds joint2 limit (0.8 > 0.5) -> Rejection
    with pytest.raises(SafetyValidationError, match="Joint 'joint2' position 0.8000 rad violates limit"):
        validator.validate_joint_positions(np.array([0.0, 0.8]))


def test_hardware_safe_stop_interlock():
    backend = HardwareBackend()
    backend._connected = True  # Simulate connected state for interlock testing

    # Engage safe stop
    backend.safe_stop()
    assert backend.safe_stop_active is True

    sc = Scenario(
        scenario_id="s1",
        task_name="reach",
        initial_state={"ee_position": [0.1, 0.2, 0.3]},
        target_state={"position": [0.4, 0.5, 0.6]},
        environment_params={},
        max_episode_steps=10,
        success_criteria={"distance_threshold": 0.05},
        safety_criteria={},
    )

    with pytest.raises(SafetyInterlockError, match="Safe-stop interlock engaged"):
        backend.reset(sc)

    with pytest.raises(SafetyInterlockError, match="Safe-stop interlock engaged"):
        backend.step(np.array([0.01, 0.0, 0.0, 1.0]))

    # Reset safe stop
    backend.reset_safe_stop()
    assert backend.safe_stop_active is False


def test_hardware_pipeline_selection_fails_closed(config_dir, scenarios_dir, tmp_path):
    runner = PipelineRunner(config_dir=str(config_dir), results_dir=str(tmp_path))

    # Pipeline selection of --simulator hardware fails closed because physical hardware transport is unconnected
    with pytest.raises(HardwareConnectionError, match="Hardware transport disconnected"):
        runner.run(
            baseline_name="baseline",
            candidate_name="candidate",
            scenarios_dir=str(scenarios_dir),
            seed=42,
            simulator_name="hardware",
        )


# ---------------------------------------------------------------------------
# Audit hardening (B4.5)
# ---------------------------------------------------------------------------

def _connected_backend(cfg, execution_enabled=False):
    from robot_ci.simulation.hardware import HardwareBackend
    from robot_ci.transport.fake_transport import FakeRobotTransport
    t = FakeRobotTransport(execution_enabled=execution_enabled)
    t.connect()
    hb = HardwareBackend(config=cfg, transport=t)
    return hb


def _scenario():
    from robot_ci.scenarios.manager import ScenarioManager
    return ScenarioManager().load_scenario("scenarios/reach_simple.yaml")


def test_step_limit_is_exactly_max_step_size():
    import numpy as np
    import pytest
    from robot_ci.simulation.safety import HardwareConfig, SafetyValidationError
    cfg = HardwareConfig(workspace_bounds=[[0.2, -0.5, 0.0], [0.8, 0.5, 0.8]], max_step_size=0.03)
    hb = _connected_backend(cfg)
    hb.reset(_scenario())
    with pytest.raises(SafetyValidationError):
        hb.step(np.array([0.04, 0.0, 0.0, 1.0]))  # 0.04 > 0.03 (previously allowed up to 0.06)


def test_gripper_range_validated():
    import numpy as np
    import pytest
    from robot_ci.simulation.safety import HardwareConfig, SafetyValidationError
    hb = _connected_backend(HardwareConfig(workspace_bounds=[[0.2, -0.5, 0.0], [0.8, 0.5, 0.8]]))
    hb.reset(_scenario())
    with pytest.raises(SafetyValidationError):
        hb.step(np.array([0.0, 0.0, 0.0, 3.0]))


def test_hardware_config_bounds_are_authoritative():
    import numpy as np
    import pytest
    from robot_ci.simulation.safety import HardwareConfig, SafetyValidationError
    # Scenario allows x up to 1.0; hardware config only up to 0.32.
    hb = _connected_backend(HardwareConfig(workspace_bounds=[[0.2, -0.5, 0.0], [0.32, 0.5, 0.8]], max_step_size=0.05))
    hb.reset(_scenario())  # starts at x = 0.30
    with pytest.raises(SafetyValidationError):
        hb.step(np.array([0.04, 0.0, 0.0, 1.0]))


def test_success_never_inferred_from_command_completion():
    import numpy as np
    from robot_ci.simulation.safety import HardwareConfig
    hb = _connected_backend(HardwareConfig(workspace_bounds=[[0.2, -0.5, 0.0], [0.8, 0.5, 0.8]]), execution_enabled=True)
    hb.reset(_scenario())
    _, _, _, info = hb.step(np.array([0.01, 0.0, 0.0, 1.0]))
    assert info["success"] is False
    assert info["ee_position_source"] == "commanded_dead_reckoning_not_measured"
