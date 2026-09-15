import time
from typing import Any, Dict, List, Tuple

import numpy as np

from robot_ci.scenarios.models import Scenario
from .backend import SimulatorBackend


class PyBulletSimulator(SimulatorBackend):
    """PyBullet manipulation simulator using the existing Robot CI API.

    The backend keeps the project's 10D observation and 4D delta-action
    contract while replacing the development kinematics mock with a local
    physics scene: a Franka Panda arm, optional cube object, target marker,
    table, and scenario-defined obstacle boxes.

    Pick-and-place objects are kinematic until grasped so scenario YAML
    positions stay stable (the same semantics as MockSimulator). After a
    grasp, the cube is attached to the end-effector.
    """

    _PANDA_REST = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

    def __init__(self, gui: bool = False, action_repeat: int = 12, sleep_gui: bool = True):
        self.gui = gui
        self.action_repeat = action_repeat
        self.sleep_gui = sleep_gui
        self._rng = np.random.default_rng(42)
        self._client_id = None
        self._scenario = None
        self._p = None
        self._pybullet_data = None

        self.robot_id = None
        self.object_id = None
        self.target_id = None
        self.table_id = None
        self.obstacle_ids: List[int] = []
        self.ee_link_index = 11
        self.joint_indices = list(range(7))
        self.finger_indices = [9, 10]
        self._lower_limits = list(self._PANDA_REST)
        self._upper_limits = list(self._PANDA_REST)
        self._joint_ranges = [2.0] * 7

        self.ee_position = np.zeros(3, dtype=np.float32)
        self.gripper_open = True
        self.object_position = None
        self.object_grasped = False
        self.timestep = 0
        self.collision_count = 0
        self._success = False

    @property
    def name(self) -> str:
        return "pybullet"

    def _ensure_imported(self) -> None:
        if self._p is not None:
            return
        try:
            import pybullet as p
            import pybullet_data
        except ImportError as exc:
            raise RuntimeError(
                "PyBullet is required for the PyBullet simulator. "
                "Use the project's Python 3.11 virtual environment where pybullet is installed."
            ) from exc
        self._p = p
        self._pybullet_data = pybullet_data

    def _connect(self) -> None:
        self._ensure_imported()
        if self._client_id is not None and self._p.isConnected(self._client_id):
            return

        mode = self._p.GUI if self.gui else self._p.DIRECT
        self._client_id = self._p.connect(mode)
        self._p.setAdditionalSearchPath(self._pybullet_data.getDataPath(), physicsClientId=self._client_id)
        self._p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)
        self._p.setTimeStep(1.0 / 240.0, physicsClientId=self._client_id)
        self._p.setPhysicsEngineParameter(
            numSolverIterations=50,
            deterministicOverlappingPairs=1,
            physicsClientId=self._client_id,
        )
        if self.gui:
            self._p.resetDebugVisualizerCamera(
                cameraDistance=1.45,
                cameraYaw=50,
                cameraPitch=-32,
                cameraTargetPosition=[0.45, 0.0, 0.28],
                physicsClientId=self._client_id,
            )
            self._p.configureDebugVisualizer(self._p.COV_ENABLE_SHADOWS, 1, physicsClientId=self._client_id)

    def seed(self, seed_value: int) -> None:
        self._rng = np.random.default_rng(seed_value)

    def reset(self, scenario: Scenario) -> np.ndarray:
        self._connect()
        self._scenario = scenario
        self._p.resetSimulation(physicsClientId=self._client_id)
        self._p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)
        self._p.setTimeStep(1.0 / 240.0, physicsClientId=self._client_id)
        self._p.setPhysicsEngineParameter(
            numSolverIterations=50,
            deterministicOverlappingPairs=1,
            physicsClientId=self._client_id,
        )

        self._p.loadURDF("plane.urdf", physicsClientId=self._client_id)
        self.table_id = self._create_box(
            half_extents=[0.45, 0.40, 0.02],
            position=[0.50, 0.0, 0.02],
            color=[0.55, 0.42, 0.28, 1.0],
            mass=0.0,
        )
        self.robot_id = self._p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            flags=self._p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS,
            physicsClientId=self._client_id,
        )
        self._cache_joint_limits()
        self._reset_arm_home()

        self.ee_position = np.array(scenario.initial_state["ee_position"], dtype=np.float32)
        self.gripper_open = scenario.initial_state.get("gripper_open", True)
        self.object_grasped = False
        self.object_id = None
        self.object_position = None
        self.obstacle_ids = []
        self.timestep = 0
        self.collision_count = 0
        self._success = False

        self._reset_arm_to_position(self.ee_position)
        self._set_gripper(open_gripper=self.gripper_open)
        self._sync_ee_position()

        if scenario.task_name == "pick_and_place":
            self.object_position = np.array(scenario.initial_state["object_position"], dtype=np.float32)
            self.object_id = self._create_box(
                half_extents=[0.025, 0.025, 0.025],
                position=self.object_position,
                color=[0.12, 0.38, 0.88, 1.0],
                mass=0.0,
            )

        self._create_target_marker(np.array(scenario.target_state["position"], dtype=np.float32))
        for obstacle in scenario.environment_params.get("obstacles", []):
            self.obstacle_ids.append(
                self._create_box(
                    half_extents=obstacle["half_extents"],
                    position=obstacle["center"],
                    color=[0.80, 0.22, 0.16, 0.90],
                    mass=0.0,
                )
            )

        return self._build_observation()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        params = self._scenario.environment_params
        max_step_size = params.get("max_step_size", 0.05)
        noise_std = params.get("noise_std", 0.002)
        friction = params.get("friction", 0.98)
        grasp_threshold = params.get("grasp_threshold", 0.05)

        action = np.asarray(action, dtype=np.float32)
        delta_pos = np.clip(action[0:3], -max_step_size, max_step_size)
        target_position = self.ee_position + (delta_pos * friction)
        target_position += self._rng.normal(0.0, noise_std, 3)
        target_position = self._clip_to_workspace(target_position)

        self._set_arm_to_position(target_position)
        self._set_gripper(open_gripper=action[3] > 0.5)
        for _ in range(self.action_repeat):
            self._p.stepSimulation(physicsClientId=self._client_id)
            if self.gui and self.sleep_gui:
                time.sleep(1.0 / 240.0)

        self._sync_ee_position()
        self._update_gripper(action[3], grasp_threshold)
        if self.object_grasped and self.object_id is not None:
            self.object_position = self.ee_position.copy()
            self._p.resetBasePositionAndOrientation(
                self.object_id,
                self.object_position.tolist(),
                [0, 0, 0, 1],
                physicsClientId=self._client_id,
            )
        elif self.object_id is not None:
            pos, _ = self._p.getBasePositionAndOrientation(self.object_id, physicsClientId=self._client_id)
            self.object_position = np.array(pos, dtype=np.float32)

        collision_this_step = self._has_collision()
        if collision_this_step:
            self.collision_count += 1

        success, distance_metric, distance_to_target = self._measure_success()
        reward = -float(distance_metric) - 0.01
        if success:
            reward += 10.0
        if collision_this_step:
            reward -= 1.0

        self._success = success
        done = success or (self.timestep + 1 >= self._scenario.max_episode_steps)
        info = {
            "success": success,
            "collision": collision_this_step,
            "timestep": self.timestep,
            "distance_to_target": float(distance_to_target),
            "simulator": self.name,
        }
        self.timestep += 1
        return self._build_observation(), reward, done, info

    def get_state(self) -> Dict[str, Any]:
        return {
            "simulator": self.name,
            "ee_position": self.ee_position.tolist(),
            "gripper_open": self.gripper_open,
            "object_position": self.object_position.tolist() if self.object_position is not None else None,
            "object_grasped": self.object_grasped,
            "timestep": self.timestep,
            "collision_count": self.collision_count,
            "success": self._success,
        }

    def close(self) -> None:
        if self._p is not None and self._client_id is not None and self._p.isConnected(self._client_id):
            self._p.disconnect(self._client_id)
        self._client_id = None

    def _cache_joint_limits(self) -> None:
        lowers = []
        uppers = []
        ranges = []
        for joint_index in self.joint_indices:
            info = self._p.getJointInfo(self.robot_id, joint_index, physicsClientId=self._client_id)
            lower, upper = float(info[8]), float(info[9])
            lowers.append(lower)
            uppers.append(upper)
            ranges.append(max(upper - lower, 0.1))
        self._lower_limits = lowers
        self._upper_limits = uppers
        self._joint_ranges = ranges

    def _reset_arm_home(self) -> None:
        for joint_index, joint_position in zip(self.joint_indices, self._PANDA_REST):
            self._p.resetJointState(
                self.robot_id,
                joint_index,
                joint_position,
                targetVelocity=0.0,
                physicsClientId=self._client_id,
            )

    def _inverse_kinematics(self, position: np.ndarray, iterations: int, residual: float):
        return self._p.calculateInverseKinematics(
            self.robot_id,
            self.ee_link_index,
            position.tolist(),
            lowerLimits=self._lower_limits,
            upperLimits=self._upper_limits,
            jointRanges=self._joint_ranges,
            restPoses=self._PANDA_REST,
            maxNumIterations=iterations,
            residualThreshold=residual,
            physicsClientId=self._client_id,
        )

    def _set_arm_to_position(self, position: np.ndarray) -> None:
        joint_positions = self._inverse_kinematics(position, iterations=100, residual=1e-4)
        for joint_index, joint_position in zip(self.joint_indices, joint_positions[:7]):
            self._p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                self._p.POSITION_CONTROL,
                targetPosition=joint_position,
                force=200,
                positionGain=0.12,
                velocityGain=1.0,
                physicsClientId=self._client_id,
            )

    def _reset_arm_to_position(self, position: np.ndarray) -> None:
        joint_positions = self._inverse_kinematics(position, iterations=200, residual=1e-5)
        for joint_index, joint_position in zip(self.joint_indices, joint_positions[:7]):
            self._p.resetJointState(
                self.robot_id,
                joint_index,
                joint_position,
                targetVelocity=0.0,
                physicsClientId=self._client_id,
            )
            self._p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                self._p.POSITION_CONTROL,
                targetPosition=joint_position,
                force=200,
                physicsClientId=self._client_id,
            )

    def _set_gripper(self, open_gripper: bool) -> None:
        target = 0.04 if open_gripper else 0.0
        for joint_index in self.finger_indices:
            self._p.setJointMotorControl2(
                self.robot_id,
                joint_index,
                self._p.POSITION_CONTROL,
                targetPosition=target,
                force=20,
                physicsClientId=self._client_id,
            )

    def _sync_ee_position(self) -> None:
        link_state = self._p.getLinkState(
            self.robot_id,
            self.ee_link_index,
            computeForwardKinematics=True,
            physicsClientId=self._client_id,
        )
        self.ee_position = np.array(link_state[0], dtype=np.float32)

    def _build_observation(self) -> np.ndarray:
        obs = np.zeros(10, dtype=np.float32)
        obs[0:3] = self.ee_position
        obs[3] = 1.0 if self.gripper_open else 0.0
        if self.object_position is not None:
            obs[4:7] = self.object_position
        obs[7:10] = self._scenario.target_state["position"]
        return obs

    def _clip_to_workspace(self, position: np.ndarray) -> np.ndarray:
        bounds = self._scenario.environment_params.get("workspace_bounds")
        if not bounds:
            return position
        low = np.array(bounds[0], dtype=np.float32)
        high = np.array(bounds[1], dtype=np.float32)
        return np.clip(position, low, high)

    def _update_gripper(self, gripper_cmd: float, grasp_threshold: float) -> None:
        if gripper_cmd <= 0.5:
            if self.gripper_open and self.object_position is not None:
                if np.linalg.norm(self.ee_position - self.object_position) <= grasp_threshold:
                    self.object_grasped = True
            self.gripper_open = False
        else:
            self.object_grasped = False
            self.gripper_open = True

    def _measure_success(self) -> Tuple[bool, float, float]:
        dist_threshold = self._scenario.success_criteria["distance_threshold"]
        target = np.array(self._scenario.target_state["position"], dtype=np.float32)
        if self._scenario.task_name == "reach":
            dist_to_target = float(np.linalg.norm(self.ee_position - target))
            return dist_to_target < dist_threshold, dist_to_target, dist_to_target

        if self.object_position is None:
            return False, 100.0, 100.0

        dist_to_target = float(np.linalg.norm(self.object_position - target))
        success = self.object_grasped and dist_to_target < dist_threshold
        if self.object_grasped:
            distance_metric = dist_to_target
        else:
            distance_metric = float(np.linalg.norm(self.ee_position - self.object_position))
        return success, distance_metric, dist_to_target

    def _has_collision(self) -> bool:
        zones = list(self._scenario.safety_criteria.get("collision_zones", []))
        zones.extend(self._scenario.environment_params.get("obstacles", []))
        points = [self.ee_position]
        if self.object_position is not None:
            points.append(self.object_position)

        for zone in zones:
            center = np.array(zone["center"], dtype=np.float32)
            half_extent = np.array(zone["half_extents"], dtype=np.float32)
            for point in points:
                if np.all(np.abs(point - center) <= half_extent):
                    return True

        for obstacle_id in self.obstacle_ids:
            contacts = self._p.getContactPoints(
                bodyA=self.robot_id,
                bodyB=obstacle_id,
                physicsClientId=self._client_id,
            )
            if contacts:
                return True
            if self.object_id is not None:
                obj_contacts = self._p.getContactPoints(
                    bodyA=self.object_id,
                    bodyB=obstacle_id,
                    physicsClientId=self._client_id,
                )
                if obj_contacts:
                    return True
        return False

    def _create_box(self, half_extents, position, color, mass: float):
        collision_shape = self._p.createCollisionShape(
            self._p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self._client_id,
        )
        visual_shape = self._p.createVisualShape(
            self._p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self._client_id,
        )
        return self._p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=position,
            physicsClientId=self._client_id,
        )

    def _create_target_marker(self, position: np.ndarray) -> None:
        visual_shape = self._p.createVisualShape(
            self._p.GEOM_SPHERE,
            radius=0.03,
            rgbaColor=[0.12, 0.72, 0.28, 0.90],
            physicsClientId=self._client_id,
        )
        self.target_id = self._p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual_shape,
            basePosition=position.tolist(),
            physicsClientId=self._client_id,
        )
