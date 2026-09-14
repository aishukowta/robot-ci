import numpy as np
from .baseline import BaselinePolicy

class CandidatePolicy(BaselinePolicy):
    """Improved staged controller for PyBullet manipulation trials."""
    
    def __init__(self):
        super().__init__()
        
    @property
    def version(self) -> str:
        return "v2.0-candidate-pybullet"
        
    @property
    def metadata(self) -> dict:
        metadata = super().metadata
        metadata["name"] = "CandidatePolicy"
        metadata["description"] = "Improved staged pick-and-place controller with lift waypoints"
        metadata["training"] = "scripted-controller"
        return metadata
        
    def load(self) -> None:
        super().load()
                
    def act(self, observation: np.ndarray) -> np.ndarray:
        self._step += 1
        obs = observation
        ee_pos = obs[0:3]
        gripper_state = obs[3]
        obj_pos = obs[4:7]
        target_pos = obs[7:10]
        
        has_object = np.linalg.norm(obj_pos) > 0.01

        if not has_object:
            waypoint = target_pos
            gripper_cmd = 1.0
        elif gripper_state > 0.5:
            dist_to_obj = np.linalg.norm(ee_pos - obj_pos)
            hover_z = max(float(obj_pos[2]) + 0.16, 0.30)
            xy_error = np.linalg.norm(ee_pos[0:2] - obj_pos[0:2])
            if xy_error > 0.025:
                waypoint = np.array([obj_pos[0], obj_pos[1], hover_z], dtype=np.float32)
                gripper_cmd = 1.0
            elif ee_pos[2] > obj_pos[2] + 0.035:
                waypoint = np.array([obj_pos[0], obj_pos[1], obj_pos[2] + 0.015], dtype=np.float32)
                gripper_cmd = 1.0
            elif dist_to_obj <= 0.065:
                waypoint = obj_pos
                gripper_cmd = 0.0
            else:
                waypoint = obj_pos
                gripper_cmd = 1.0
        else:
            carry_z = max(float(target_pos[2]) + 0.20, 0.32)
            xy_error = np.linalg.norm(ee_pos[0:2] - target_pos[0:2])
            if ee_pos[2] < carry_z - 0.035:
                waypoint = np.array([ee_pos[0], ee_pos[1], carry_z], dtype=np.float32)
            elif xy_error > 0.030:
                waypoint = np.array([target_pos[0], target_pos[1], carry_z], dtype=np.float32)
            else:
                waypoint = target_pos
            gripper_cmd = 0.0
            
        action = np.zeros(4, dtype=np.float32)
        action[0:3] = 0.55 * (waypoint - ee_pos)
        action[0:3] = np.clip(action[0:3], -0.1, 0.1)
        action[3] = gripper_cmd
        
        return action
