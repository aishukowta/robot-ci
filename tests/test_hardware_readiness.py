"""Tests for Physical Hardware Integration Readiness Audit (Phase 5).

Verifies:
- ROS 2 availability check and behavior when ROS 2 runtime is unavailable.
- Fake transport produces READY_WITH_WARNINGS with explicit simulation disclaimers.
- Missing robot model or driver produces BLOCKED status.
- Missing critical safety configuration (workspace, joint limits, step size) produces BLOCKED.
- Fully configured readiness metadata yields READY / READY_WITH_WARNINGS.
- Readiness audit NEVER issues motion commands (send_command is never invoked).
- Existing simulation and regression workflows remain completely independent of ROS 2.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import numpy as np

from robot_ci.hardware.readiness import (
    HardwareReadinessStatus,
    HardwareReadinessReport,
    audit_hardware_readiness,
    check_python_packages,
    check_ros2_environment,
    format_readiness_report_text,
)
from robot_ci.simulation.safety import HardwareConfig
from robot_ci.transport.fake_transport import FakeRobotTransport
from robot_ci.transport.ros2_adapter import ROS2RobotTransport, is_ros2_available
from robot_ci.transport.models import RobotCommand, CommandType


@pytest.fixture
def valid_hardware_config():
    """Returns a fully-specified, safe hardware configuration."""
    return HardwareConfig(
        robot_name="franka_emika_panda",
        joint_names=[f"panda_joint{i+1}" for i in range(7)],
        joint_limits={
            f"panda_joint{i+1}": (-2.89, 2.89) for i in range(7)
        },
        workspace_bounds=[[0.2, -0.5, 0.0], [0.8, 0.5, 0.8]],
        max_step_size=0.03,
        enforce_workspace=True,
        enforce_joint_limits=True,
        hardware_transport_address="192.168.1.100:50051",
        driver_type="ros2_control/franka_hardware",
        driver_configured=True,
    )


class TestHardwareReadinessAudit:
    """Test suite for Phase 5 Hardware Readiness Audit."""

    def test_ros2_unavailable_detection(self):
        """Verify audit correctly identifies when ROS 2 (rclpy) is unavailable."""
        # On this Windows test host, is_ros2_available is False
        env_info = check_ros2_environment()
        assert "available" in env_info
        assert "os_platform" in env_info
        assert "python_version" in env_info
        assert isinstance(env_info["notes"], list)

        # When ROS 2 transport is configured without rclpy, audit must mark it BLOCKED
        transport = ROS2RobotTransport()
        report = audit_hardware_readiness(transport=transport)
        assert report.status == HardwareReadinessStatus.BLOCKED
        assert any("ROS 2" in b or "rclpy" in b for b in report.blockers)

    def test_fake_transport_selected_produces_ready_with_warnings(self, valid_hardware_config):
        """Verify fake transport with valid safety config produces READY_WITH_WARNINGS."""
        transport = FakeRobotTransport()
        transport.connect()

        report = audit_hardware_readiness(config=valid_hardware_config, transport=transport)

        assert report.status == HardwareReadinessStatus.READY_WITH_WARNINGS
        assert report.is_fake_transport is True
        assert len(report.blockers) == 0
        assert any("FakeRobotTransport" in w for w in report.warnings)
        assert any("simulated in software" in w for w in report.warnings)

    def test_missing_robot_model_blocks_readiness(self, valid_hardware_config):
        """Verify unconfigured robot model results in BLOCKED status."""
        transport = FakeRobotTransport()
        transport.connect()

        # Case A: unconfigured_arm
        cfg = HardwareConfig(
            robot_name="unconfigured_arm",
            joint_names=valid_hardware_config.joint_names,
            joint_limits=valid_hardware_config.joint_limits,
            workspace_bounds=valid_hardware_config.workspace_bounds,
            driver_configured=True,
            driver_type="mock_driver",
        )
        report = audit_hardware_readiness(config=cfg, transport=transport)
        assert report.status == HardwareReadinessStatus.BLOCKED
        assert any("Robot model is unconfigured" in b for b in report.blockers)

        # Case B: empty string
        cfg.robot_name = ""
        report2 = audit_hardware_readiness(config=cfg, transport=transport)
        assert report2.status == HardwareReadinessStatus.BLOCKED

    def test_missing_driver_blocks_readiness(self, valid_hardware_config):
        """Verify unconfigured driver results in BLOCKED status."""
        transport = FakeRobotTransport()
        transport.connect()

        cfg = HardwareConfig(
            robot_name="franka_emika_panda",
            joint_names=valid_hardware_config.joint_names,
            joint_limits=valid_hardware_config.joint_limits,
            workspace_bounds=valid_hardware_config.workspace_bounds,
            driver_configured=False,  # Unconfigured!
        )
        report = audit_hardware_readiness(config=cfg, transport=transport)
        assert report.status == HardwareReadinessStatus.BLOCKED
        assert any("Robot driver is unconfigured" in b for b in report.blockers)

    def test_missing_critical_safety_configuration_blocks_readiness(self, valid_hardware_config):
        """Verify missing workspace bounds, joint limits, or step sizes result in BLOCKED."""
        transport = FakeRobotTransport()
        transport.connect()

        # 1. Missing workspace bounds
        cfg_no_bounds = HardwareConfig(
            robot_name="franka_emika_panda",
            joint_names=valid_hardware_config.joint_names,
            joint_limits=valid_hardware_config.joint_limits,
            workspace_bounds=None,
            driver_configured=True,
        )
        rep1 = audit_hardware_readiness(config=cfg_no_bounds, transport=transport)
        assert rep1.status == HardwareReadinessStatus.BLOCKED
        assert any("workspace_bounds" in b for b in rep1.blockers)

        # 2. Inverted workspace bounds (min >= max)
        cfg_bad_bounds = HardwareConfig(
            robot_name="franka_emika_panda",
            joint_names=valid_hardware_config.joint_names,
            joint_limits=valid_hardware_config.joint_limits,
            workspace_bounds=[[1.0, 0.5, 0.8], [0.2, -0.5, 0.0]],  # inverted
            driver_configured=True,
        )
        rep2 = audit_hardware_readiness(config=cfg_bad_bounds, transport=transport)
        assert rep2.status == HardwareReadinessStatus.BLOCKED

        # 3. Missing joint limits
        cfg_no_joints = HardwareConfig(
            robot_name="franka_emika_panda",
            joint_names=[],
            joint_limits={},
            workspace_bounds=[[0.2, -0.5, 0.0], [0.8, 0.5, 0.8]],
            driver_configured=True,
        )
        rep3 = audit_hardware_readiness(config=cfg_no_joints, transport=transport)
        assert rep3.status == HardwareReadinessStatus.BLOCKED
        assert any("joint_limits" in b for b in rep3.blockers)

        # 4. Inverted joint limits (lower >= upper)
        cfg_bad_joint = HardwareConfig(
            robot_name="franka_emika_panda",
            joint_names=["joint1"],
            joint_limits={"joint1": (2.0, -2.0)},  # lower > upper
            workspace_bounds=[[0.2, -0.5, 0.0], [0.8, 0.5, 0.8]],
            driver_configured=True,
        )
        rep4 = audit_hardware_readiness(config=cfg_bad_joint, transport=transport)
        assert rep4.status == HardwareReadinessStatus.BLOCKED
        assert any("lower >= upper" in b for b in rep4.blockers)

        # 5. Non-positive max step size
        cfg_bad_step = HardwareConfig(
            robot_name="franka_emika_panda",
            joint_names=valid_hardware_config.joint_names,
            joint_limits=valid_hardware_config.joint_limits,
            workspace_bounds=valid_hardware_config.workspace_bounds,
            max_step_size=-0.01,
            driver_configured=True,
        )
        rep5 = audit_hardware_readiness(config=cfg_bad_step, transport=transport)
        assert rep5.status == HardwareReadinessStatus.BLOCKED
        assert any("max_step_size" in b for b in rep5.blockers)

        # 6. Safety enforcement flags disabled
        cfg_disabled_enforce = HardwareConfig(
            robot_name="franka_emika_panda",
            joint_names=valid_hardware_config.joint_names,
            joint_limits=valid_hardware_config.joint_limits,
            workspace_bounds=valid_hardware_config.workspace_bounds,
            enforce_workspace=False,
            driver_configured=True,
        )
        rep6 = audit_hardware_readiness(config=cfg_disabled_enforce, transport=transport)
        assert rep6.status == HardwareReadinessStatus.BLOCKED
        assert any("enforce_workspace" in b for b in rep6.blockers)

    def test_fully_configured_readiness_metadata(self, valid_hardware_config):
        """Verify fully configured real transport setup validates successfully."""
        mock_transport = MagicMock()
        mock_transport.transport_name = "MockRealRobotTransport"
        mock_transport.is_connected.return_value = True
        mock_transport.request_safe_stop.return_value = True
        mock_transport.is_safe_stop_confirmed.return_value = True

        with patch("robot_ci.hardware.readiness.is_ros2_available", return_value=True):
            report = audit_hardware_readiness(config=valid_hardware_config, transport=mock_transport)

            assert report.safety_configuration_verified is True
            assert report.joint_limits_verified is True
            assert report.workspace_bounds_verified is True
            assert report.max_step_size_verified is True
            assert report.is_robot_model_configured is True
            assert report.driver_configured is True
            assert report.stop_mechanisms["software_stop_available"] is True
            assert report.stop_mechanisms["hardware_estop_confirmed"] is True
            assert len(report.blockers) == 0
            # Since hardware E-stop warning is advised, status is READY_WITH_WARNINGS or READY
            assert report.status in (HardwareReadinessStatus.READY, HardwareReadinessStatus.READY_WITH_WARNINGS)

    def test_readiness_checks_never_issue_motion_commands(self, valid_hardware_config):
        """CRITICAL: Audit must NEVER call send_command or issue any physical motion."""
        mock_transport = MagicMock()
        mock_transport.transport_name = "MonitoredTransport"
        mock_transport.is_connected.return_value = True
        mock_transport.send_command = MagicMock()

        report = audit_hardware_readiness(config=valid_hardware_config, transport=mock_transport)

        # send_command MUST be called exactly ZERO times
        assert mock_transport.send_command.call_count == 0

    def test_format_readiness_report_text(self, valid_hardware_config):
        """Verify human-readable CLI output renders all sections properly."""
        transport = FakeRobotTransport()
        transport.connect()
        report = audit_hardware_readiness(config=valid_hardware_config, transport=transport)
        text = format_readiness_report_text(report)

        assert "ROBOT CI -- Physical Hardware Integration Readiness Audit" in text
        assert "ENVIRONMENT & PACKAGE AVAILABILITY:" in text
        assert "ROS 2 RUNTIME & PLATFORM:" in text
        assert "TRANSPORT & DRIVER CONFIGURATION:" in text
        assert "SAFETY CONFIGURATION & LIMITS:" in text
        assert "STOP MECHANISMS" in text
        assert "FINAL VERDICT:" in text

    def test_report_persistence(self, valid_hardware_config, tmp_path):
        """Verify report serialization and saving."""
        transport = FakeRobotTransport()
        transport.connect()
        report = audit_hardware_readiness(config=valid_hardware_config, transport=transport)

        out_file = tmp_path / "test_report.json"
        saved = report.save(out_file)
        assert saved.exists()

        with open(saved, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["status"] == HardwareReadinessStatus.READY_WITH_WARNINGS.value
        assert "python_packages" in data
        assert "disclaimers" in data

    def test_simulation_and_regression_workflows_independent_of_ros2(self):
        """Verify PyBullet and Mock simulators continue to run without ROS 2."""
        from robot_ci.simulation.mock import MockSimulator
        from robot_ci.simulation.pybullet_sim import PyBulletSimulator
        from robot_ci.scenarios.manager import ScenarioManager

        manager = ScenarioManager()
        scenarios = manager.load_scenarios("scenarios")
        scenario = next(s for s in scenarios if s.scenario_id == "reach_simple")

        # MockSimulator works 100% offline
        mock_sim = MockSimulator()
        obs = mock_sim.reset(scenario)
        assert obs is not None
        assert obs.shape == (10,)
        obs, r, done, info = mock_sim.step(np.array([0.01, 0.0, 0.0, 1.0]))
        assert "success" in info
        assert "collision" in info
        mock_sim.close()

        # PyBullet simulator works 100% offline
        pb_sim = PyBulletSimulator(gui=False, action_repeat=1)
        obs_pb = pb_sim.reset(scenario)
        assert obs_pb is not None
        obs_pb, r, done, info_pb = pb_sim.step(np.array([0.01, 0.0, 0.0, 1.0]))
        assert "success" in info_pb
        pb_sim.close()

    def test_check_hardware_cli(self, tmp_path, valid_hardware_config):
        """Test check-hardware CLI execution via subprocess."""
        import subprocess
        import sys

        # Save valid config to temp file
        config_path = tmp_path / "hw_config.json"
        config_dict = {
            "robot_name": valid_hardware_config.robot_name,
            "joint_names": valid_hardware_config.joint_names,
            "joint_limits": {k: list(v) for k, v in valid_hardware_config.joint_limits.items()},
            "workspace_bounds": valid_hardware_config.workspace_bounds,
            "max_step_size": valid_hardware_config.max_step_size,
            "enforce_workspace": True,
            "enforce_joint_limits": True,
            "driver_type": "test_driver",
            "driver_configured": True,
        }
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f)

        out_path = tmp_path / "cli_report.json"

        # Execute check-hardware with fake transport
        cmd = [
            sys.executable,
            "-m",
            "robot_ci",
            "check-hardware",
            "--config",
            str(config_path),
            "--transport",
            "fake",
            "--output",
            str(out_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "Readiness Status: [READY_WITH_WARNINGS]" in res.stdout
        assert out_path.exists()

        with open(out_path, "r", encoding="utf-8") as f:
            saved_data = json.load(f)
        assert saved_data["status"] == "READY_WITH_WARNINGS"
        assert saved_data["is_fake_transport"] is True

