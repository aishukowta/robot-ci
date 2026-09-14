import numpy as np
from typing import Tuple, Dict, Any
from .backend import SimulatorBackend
from robot_ci.scenarios.models import Scenario

class MockSimulator(SimulatorBackend):
    """Lightweight development simulator."""
    
    def __init__(self):
        self._rng = np.random.default_rng(42)
        self.ee_position = np.zeros(3)
        self.gripper_open = True
        self.object_position = None
        self.object_grasped = False
        self.timestep = 0
        self.collision_count = 0
        self._success = False
        self._scenario = None
        
    def seed(self, seed_value: int) -> None:
        self._rng = np.random.default_rng(seed_value)
        
    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0:3] = self.ee_position
        obs[3] = 1.0 if self.gripper_open else 0.0
        if self.object_position is not None:
            obs[4:7] = self.object_position
        else:
            obs[4:7] = [0.0, 0.0, 0.0]
        obs[7:10] = self._scenario.target_state["position"]
        return obs
        
    def reset(self, scenario: Scenario) -> np.ndarray:
        self._scenario = scenario
        self.ee_position = np.array(scenario.initial_state["ee_position"], dtype=np.float32)
        self.gripper_open = scenario.initial_state.get("gripper_open", True)
        
        if scenario.task_name == "pick_and_place":
            self.object_position = np.array(scenario.initial_state["object_position"], dtype=np.float32)
        else:
            self.object_position = None
            
        self.object_grasped = False
        self.timestep = 0
        self.collision_count = 0
        self._success = False
        
        return self._build_observation()
        
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        delta_pos = action[0:3].copy()
        gripper_cmd = action[3]
        
        params = self._scenario.environment_params
        max_step_size = params.get("max_step_size", 0.05)
        friction = params.get("friction", 0.98)
        noise_std = params.get("noise_std", 0.002)
        grasp_threshold = params.get("grasp_threshold", 0.05)
        
        delta_pos = np.clip(delta_pos, -max_step_size, max_step_size)
        delta_pos *= friction
        self.ee_position += delta_pos
        self.ee_position += self._rng.normal(0, noise_std, 3)
        
        if "workspace_bounds" in params:
            bounds = params["workspace_bounds"]
            low = np.array(bounds[0])
            high = np.array(bounds[1])
            self.ee_position = np.clip(self.ee_position, low, high)
            
        if gripper_cmd <= 0.5:
            if self.gripper_open and self.object_position is not None:
                if np.linalg.norm(self.ee_position - self.object_position) < grasp_threshold:
                    self.object_grasped = True
            self.gripper_open = False
        else:
            self.object_grasped = False
            self.gripper_open = True
            
        if self.object_grasped:
            self.object_position = self.ee_position.copy()
            
        collision_this_step = False
        collision_zones = self._scenario.safety_criteria.get("collision_zones", [])
        obstacles = params.get("obstacles", [])
        
        for zone in collision_zones:
            center = np.array(zone["center"])
            half_extent = np.array(zone["half_extents"])
            if np.all(np.abs(self.ee_position - center) < half_extent):
                collision_this_step = True
                break
                
        for obs in obstacles:
            center = np.array(obs["center"])
            half_extent = np.array(obs["half_extents"])
            if np.all(np.abs(self.ee_position - center) < half_extent):
                collision_this_step = True
                break
                
        if collision_this_step:
            self.collision_count += 1
            
        dist_threshold = self._scenario.success_criteria["distance_threshold"]
        target = np.array(self._scenario.target_state["position"])
        
        if self._scenario.task_name == "reach":
            dist_to_target = np.linalg.norm(self.ee_position - target)
            success = dist_to_target < dist_threshold
            dist_metric = dist_to_target
        else:
            if self.object_position is not None:
                dist_to_target = np.linalg.norm(self.object_position - target)
                success = self.object_grasped and dist_to_target < dist_threshold
                if not self.object_grasped:
                    dist_metric = np.linalg.norm(self.ee_position - self.object_position)
                else:
                    dist_metric = dist_to_target
            else:
                success = False
                dist_metric = 100.0
                dist_to_target = 100.0
                
        reward = -float(dist_metric)
        if success:
            reward += 10.0
        if collision_this_step:
            reward -= 1.0
        reward -= 0.01
        
        self._success = success
        done = success or (self.timestep + 1 >= self._scenario.max_episode_steps)
        
        info = {
            "success": success,
            "collision": collision_this_step,
            "timestep": self.timestep,
            "distance_to_target": float(dist_to_target)
        }
        
        self.timestep += 1
        
        return self._build_observation(), reward, done, info
        
    def get_state(self) -> Dict[str, Any]:
        return {
            "ee_position": self.ee_position.tolist(),
            "gripper_open": self.gripper_open,
            "object_position": self.object_position.tolist() if self.object_position is not None else None,
            "object_grasped": self.object_grasped,
            "timestep": self.timestep,
            "collision_count": self.collision_count,
            "success": self._success
        }
