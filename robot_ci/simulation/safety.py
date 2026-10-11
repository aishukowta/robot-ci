from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

class HardwareConnectionError(Exception):
    """Raised when hardware operations are attempted without active hardware transport."""
    pass

class SafetyValidationError(ValueError):
    """Raised when command or state violates workspace, joint, or finite numeric limits."""
    pass

class SafetyInterlockError(RuntimeError):
    """Raised when safe-stop interlock is active or execution is disabled."""
    pass

@dataclass
class HardwareConfig:
    """Explicit configuration for physical robot hardware abstraction.
    
    Data units:
    - Joint positions: Radians (revolute) or Meters (prismatic)
    - Cartesian positions: Meters (x, y, z)
    - Action deltas: Meters per step
    - Timestamps: ISO 8601 UTC strings & UNIX epoch seconds
    """
    robot_name: str = "unconfigured_arm"
    joint_names: List[str] = field(default_factory=list)
    joint_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    workspace_bounds: Optional[List[List[float]]] = None  # [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    max_step_size: float = 0.05
    enforce_workspace: bool = True
    enforce_joint_limits: bool = True
    hardware_transport_address: Optional[str] = None
    telemetry_topic: Optional[str] = None
    command_topic: Optional[str] = None
    driver_type: Optional[str] = None
    driver_configured: bool = False

class HardwareSafetyValidator:
    """Software safety validator enforcing workspace limits, joint bounds, and numeric integrity."""

    def __init__(self, config: Optional[HardwareConfig] = None):
        self.config = config or HardwareConfig()

    def check_interlocks(self, is_connected: bool, safe_stop_active: bool) -> None:
        """Fail closed if hardware is disconnected or safe-stop interlock is engaged."""
        if safe_stop_active:
            raise SafetyInterlockError("Safe-stop interlock engaged. Physical hardware execution is halted.")
        if not is_connected:
            raise HardwareConnectionError(
                "Hardware transport disconnected. Cannot execute physical commands on an unconfigured or disconnected robot."
            )

    def validate_action(self, action: Any, workspace_bounds: Optional[List[List[float]]] = None) -> np.ndarray:
        """Validate action array dimensions, numeric finiteness, and step magnitude."""
        if action is None:
            raise SafetyValidationError("Action command cannot be None.")
        
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim != 1 or len(arr) < 4:
            raise SafetyValidationError(f"Invalid action dimension: expected 4D array [dx, dy, dz, gripper], got shape {arr.shape}.")

        if not np.all(np.isfinite(arr)):
            raise SafetyValidationError(f"Invalid non-finite values in action command: {arr}.")

        # Check maximum delta step size limit
        delta_mag = float(np.linalg.norm(arr[0:3]))
        max_allowed = self.config.max_step_size * 2.0  # 2x safety ceiling
        if delta_mag > max_allowed:
            raise SafetyValidationError(f"Action delta magnitude ({delta_mag:.4f}m) exceeds safety limit ({max_allowed:.4f}m).")

        return arr

    def validate_cartesian_position(self, position: np.ndarray, bounds: Optional[List[List[float]]] = None) -> None:
        """Check Cartesian 3D position against configured workspace bounds."""
        bounds = bounds or self.config.workspace_bounds
        if bounds is None or len(bounds) != 2:
            return

        low = np.array(bounds[0], dtype=np.float32)
        high = np.array(bounds[1], dtype=np.float32)

        if not (np.all(position >= low) and np.all(position <= high)):
            raise SafetyValidationError(
                f"Cartesian position {position.tolist()} violates configured workspace bounds: min={low.tolist()}, max={high.tolist()}."
            )

    def validate_joint_positions(self, positions: np.ndarray, joint_limits: Optional[Dict[str, Tuple[float, float]]] = None) -> None:
        """Check joint positions against configured lower/upper joint limits."""
        limits = joint_limits or self.config.joint_limits
        names = self.config.joint_names

        if not limits or not names:
            return

        for idx, name in enumerate(names):
            if idx >= len(positions):
                break
            if name in limits:
                lower, upper = limits[name]
                val = float(positions[idx])
                if val < lower or val > upper:
                    raise SafetyValidationError(
                        f"Joint '{name}' position {val:.4f} rad violates limit [{lower:.4f}, {upper:.4f}] rad."
                    )
            elif self.config.enforce_joint_limits:
                raise SafetyValidationError(
                    f"Joint '{name}' has no configured safety limits. Cannot assume default safe limits for physical execution."
                )
