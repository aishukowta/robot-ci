from abc import ABC, abstractmethod
import numpy as np

class PolicyInterface(ABC):
    """Abstract interface for robotic manipulation policies.
    
    Observation space (10D):
        [ee_x, ee_y, ee_z, gripper_state, obj_x, obj_y, obj_z, target_x, target_y, target_z]
        gripper_state: 1.0 = open, 0.0 = closed
        For reach tasks, obj_* = 0.0
    
    Action space (4D):
        [delta_x, delta_y, delta_z, gripper_command]
        gripper_command: > 0.5 = open, <= 0.5 = close
    """
    
    @property
    @abstractmethod
    def version(self) -> str: ...
    
    @property
    @abstractmethod
    def metadata(self) -> dict: ...
    
    @abstractmethod
    def load(self) -> None: ...
    
    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray: ...
    
    @abstractmethod
    def reset(self) -> None: ...
