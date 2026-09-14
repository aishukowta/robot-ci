import numpy as np
import torch
import torch.nn as nn
from .base import PolicyInterface

class BaselinePolicy(PolicyInterface):
    """Proportional control baseline with neural network refinement."""
    
    def __init__(self):
        super().__init__()
        self._step = 0
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 4)
        )
    
    @property
    def version(self) -> str:
        return "v1.0-baseline"
    
    @property
    def metadata(self) -> dict:
        return {
            "name": "BaselinePolicy",
            "description": "Proportional control baseline with neural network refinement",
            "training": "hand-initialized"
        }
    
    def _init_weights(self):
        torch.manual_seed(0)
        with torch.no_grad():
            for param in self.network.parameters():
                param.zero_()
            
            # Layer 1 (10->64)
            for i in range(3):
                self.network[0].weight[i, i] = -1.0
                self.network[0].weight[i, i+7] = 1.0
            
            for i in range(3):
                self.network[0].weight[i+3, i] = -1.0
                self.network[0].weight[i+3, i+4] = 1.0
            
            self.network[0].weight[6, 3] = 1.0
            self.network[0].weight[7:] = torch.randn(57, 10) * 0.01
            
            # Layer 2 (64->32)
            self.network[2].weight.copy_(torch.randn(32, 64) * 0.01)
            
            # Layer 3 (32->4)
            self.network[4].weight.copy_(torch.randn(4, 32) * 0.01)
            
    def load(self) -> None:
        self._init_weights()
        
    def reset(self) -> None:
        self._step = 0
        
    def act(self, observation: np.ndarray) -> np.ndarray:
        self._step += 1
        obs = observation
        ee_pos = obs[0:3]
        gripper_state = obs[3]
        obj_pos = obs[4:7]
        target_pos = obs[7:10]
        
        has_object = np.linalg.norm(obj_pos) > 0.01
        
        if has_object and gripper_state > 0.5:
            dist_to_obj = np.linalg.norm(ee_pos - obj_pos)
            if dist_to_obj > 0.06:
                direction = obj_pos - ee_pos
                gripper_cmd = 1.0
            else:
                direction = obj_pos - ee_pos
                gripper_cmd = 0.0
        elif has_object and gripper_state <= 0.5:
            direction = target_pos - ee_pos
            gripper_cmd = 0.0
        else:
            direction = target_pos - ee_pos
            gripper_cmd = 0.5
            
        with torch.no_grad():
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            network_output = self.network(obs_tensor).numpy()
            
        action = np.zeros(4, dtype=np.float32)
        action[0:3] = 0.3 * direction + 0.02 * network_output[0:3]
        action[0:3] = np.clip(action[0:3], -0.1, 0.1)
        action[3] = gripper_cmd
        
        return action
