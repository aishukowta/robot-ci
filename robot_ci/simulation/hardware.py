import time
from datetime import datetime, timezone
from typing import Tuple, Dict, Any, Optional
import numpy as np

from robot_ci.scenarios.models import Scenario
from .backend import SimulatorBackend
from .safety import (
    HardwareConfig,
    HardwareSafetyValidator,
    HardwareConnectionError,
    SafetyInterlockError,
    SafetyValidationError,
)

class HardwareBackend(SimulatorBackend):
    """Hardware execution backend scaffold.
    
    Provides a hardware-neutral contract for real robot arm integration while
    failing closed when physical transport is unconfigured or unavailable.
    
    Data Contracts & Units:
    - Cartesian position: Meters [x, y, z]
    - Actions: 4D delta-Cartesian pose + gripper command [dx, dy, dz, grip]
    - Joint state: Radians per joint
    - Timestamps: UTC ISO 8601 string & UNIX epoch float
    """

    def __init__(self, config: Optional[HardwareConfig] = None, transport: Optional[Any] = None):
        self.config = config or HardwareConfig()
        self.safety_validator = HardwareSafetyValidator(self.config)
        self._connected = False
        self._safe_stop_active = False
        self._transport = transport
        
        self.ee_position = np.array([0.5, 0.0, 0.3], dtype=np.float32)
        self.gripper_open = True
        self.timestep = 0
        self.collision_count = 0
        self._scenario = None

    @property
    def name(self) -> str:
        return "hardware"

    @property
    def is_connected(self) -> bool:
        if self._transport is not None:
            return self._transport.is_connected()
        return self._connected

    @property
    def safe_stop_active(self) -> bool:
        return self._safe_stop_active

    def connect(self, transport_address: Optional[str] = None) -> None:
        """Attempt connection to hardware transport (e.g. ROS 2 node or gRPC socket).
        
        Fails closed with HardwareConnectionError if physical robot transport
        is unconfigured or unreachable.
        """
        if self._transport is not None:
            addr = transport_address or self.config.hardware_transport_address
            self._transport.connect(addr)
            self._connected = self._transport.is_connected()
            return

        addr = transport_address or self.config.hardware_transport_address
        if not addr:
            raise HardwareConnectionError(
                "Physical robot hardware transport unconfigured. Cannot connect to hardware without transport configuration."
            )
        raise HardwareConnectionError(
            f"Physical robot transport at '{addr}' is unavailable. Physical hardware execution cannot proceed."
        )

    def seed(self, seed_value: int) -> None:
        """Record seed for tracking. Physical execution is non-deterministic."""
        pass

    def safe_stop(self) -> None:
        """Engage immediate safe-stop interlock."""
        self._safe_stop_active = True
        if self._transport is not None:
            self._transport.request_safe_stop()

    def reset_safe_stop(self) -> None:
        """Reset safe-stop interlock (requires manual safety clearance)."""
        self._safe_stop_active = False
        if self._transport is not None:
            self._transport.reset_safe_stop()

    def reset(self, scenario: Scenario) -> np.ndarray:
        """Prepare hardware episode. Fails closed if hardware transport is disconnected."""
        self.safety_validator.check_interlocks(self.is_connected, self.safe_stop_active)
        self._scenario = scenario
        self.ee_position = np.array(scenario.initial_state["ee_position"], dtype=np.float32)
        self.gripper_open = scenario.initial_state.get("gripper_open", True)
        self.timestep = 0
        self.collision_count = 0
        
        return self._build_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute physical action step. Fails closed if disconnected or unsafe."""
        # 1. Enforce safety interlocks (fails closed if disconnected or safe-stop engaged)
        self.safety_validator.check_interlocks(self.is_connected, self.safe_stop_active)

        # 2. Validate action command
        valid_action = self.safety_validator.validate_action(action)
        
        # 3. Validate target Cartesian workspace bounds
        target_pos = self.ee_position + valid_action[0:3]
        if self.config.enforce_workspace and self._scenario and "workspace_bounds" in self._scenario.environment_params:
            self.safety_validator.validate_cartesian_position(
                target_pos, self._scenario.environment_params["workspace_bounds"]
            )

        # 4. If transport is configured, dispatch command through validated transport
        receipt_ack = False
        phys_completed = False
        if self._transport is not None:
            from robot_ci.transport.models import RobotCommand, CommandType
            cmd = RobotCommand(
                command_id=f"cmd_{self.timestep}",
                command_type=CommandType.CARTESIAN_DELTA,
                target_values=valid_action.tolist(),
                timestamp=time.time(),
            )
            report = self._transport.send_command(cmd)
            receipt_ack = report.receipt_acknowledged
            phys_completed = report.is_physically_completed

        self.ee_position = target_pos
        self.gripper_open = valid_action[3] > 0.5
        self.timestep += 1

        info = {
            "simulator": self.name,
            "is_simulation": False,
            "connected": self.is_connected,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "wall_time": time.time(),
            "timestep": self.timestep,
            "receipt_acknowledged": receipt_ack,
            "success": phys_completed,  # Acknowledgment != physical completion!
            "collision": False,
        }
        return self._build_observation(), 0.0, False, info

    def get_state(self) -> Dict[str, Any]:
        return {
            "simulator": self.name,
            "is_simulation": False,
            "is_connected": self.is_connected,
            "safe_stop_active": self.safe_stop_active,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "wall_time": time.time(),
            "ee_position": self.ee_position.tolist(),
            "gripper_open": self.gripper_open,
            "joint_state": None,
            "robot_name": self.config.robot_name,
            "timestep": self.timestep,
        }

    def close(self) -> None:
        if self._transport is not None:
            self._transport.disconnect()
        self._connected = False
        self._safe_stop_active = False

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0:3] = self.ee_position
        obs[3] = 1.0 if self.gripper_open else 0.0
        if self._scenario and "position" in self._scenario.target_state:
            obs[7:10] = self._scenario.target_state["position"]
        return obs
