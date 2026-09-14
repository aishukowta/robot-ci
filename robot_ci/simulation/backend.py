from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any
from robot_ci.scenarios.models import Scenario

class SimulatorBackend(ABC):
    """Abstract interface for simulation backends.
    
    Concrete implementations include MockSimulator (development) and
    future Isaac Sim integration.
    """
    
    @abstractmethod
    def reset(self, scenario: Scenario) -> np.ndarray:
        """Reset to scenario initial state. Returns initial observation (10D)."""
        ...
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute action. Returns (observation, reward, done, info)."""
        ...
    
    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """Return current internal state as dict."""
        ...
    
    @abstractmethod
    def seed(self, seed_value: int) -> None:
        """Set random seed for reproducibility."""
        ...
