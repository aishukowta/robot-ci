"""Execution backends for pen plans.

Two simulators execute the SAME PenPlan:

* ``KinematicPenSimulator`` — fast plotter-style model (first-order servo lag,
  speed and acceleration limits, latency, sensor noise). Used for quick
  regression runs and for the synthetic self-test.
* ``PyBulletPenSimulator`` — a Franka Panda in PyBullet holding a pen over an
  A4 sheet on the table. Joint position control + inverse kinematics; the pen
  tip position is read back from the physics engine each control step.
  With ``gui=True`` the ink is drawn live in the PyBullet window.

The Panda is a stand-in arm: the real handwriting robot's kinematics are not
known yet, so arm-specific numbers (joint limits, IK error) are simulation
properties, not predictions for the real robot.

Both return a PenExecution with timestamps, measured pen-tip positions (page
mm), pen-down flags and safety event counts.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .plan import PenPlan

PEN_UP_HEIGHT_MM = 12.0
PEN_CONTACT_TOLERANCE_MM = 0.8     # tip within this height of the paper counts as "down"


@dataclass
class CommandedTrajectory:
    t: np.ndarray            # (N,) seconds
    xy: np.ndarray           # (N, 2) page mm
    z: np.ndarray            # (N,) pen height above paper, mm
    pen_down: np.ndarray     # (N,) bool (commanded)


def build_commanded_trajectory(plan: PenPlan, dt: float) -> CommandedTrajectory:
    """Sample the plan at a fixed control period: straight segments at each waypoint's feed,
    plus a pen raise/lower phase of ``pen_settle_s`` whenever the pen state changes."""
    ts: List[float] = []
    xs: List[np.ndarray] = []
    zs: List[float] = []
    downs: List[bool] = []
    t = 0.0
    wps = plan.waypoints
    pos = np.array([wps[0].x_mm, wps[0].y_mm], dtype=float)
    pen = wps[0].pen_down
    z = 0.0 if pen else PEN_UP_HEIGHT_MM

    def emit(p, zz, d):
        nonlocal t
        ts.append(t)
        xs.append(p.copy())
        zs.append(zz)
        downs.append(d)
        t += dt

    emit(pos, z, pen)
    for w in wps[1:]:
        if w.pen_down != pen:
            n = max(1, int(round(plan.pen_settle_s / dt)))
            z_target = 0.0 if w.pen_down else PEN_UP_HEIGHT_MM
            for k in range(1, n + 1):
                emit(pos, z + (z_target - z) * k / n, w.pen_down if k == n else pen)
            pen, z = w.pen_down, z_target
        target = np.array([w.x_mm, w.y_mm], dtype=float)
        dist = float(np.linalg.norm(target - pos))
        if dist < 1e-9:
            continue
        step = max(w.feed_mm_s, 1e-6) * dt
        n = max(1, int(math.ceil(dist / step)))
        for k in range(1, n + 1):
            emit(pos + (target - pos) * (k / n), z, pen)
        pos = target
    return CommandedTrajectory(np.asarray(ts), np.asarray(xs), np.asarray(zs), np.asarray(downs, dtype=bool))


@dataclass
class PenExecution:
    backend: str
    plan_hash: str
    policy_name: str
    target_id: str
    seed: int
    control_dt: float
    t: np.ndarray                 # (N,) s
    xy: np.ndarray                # (N, 2) measured pen tip, page mm
    z: np.ndarray                 # (N,) measured height above paper, mm
    pen_down: np.ndarray          # (N,) measured contact (bool)
    commanded_xy: np.ndarray      # (N, 2)
    safety: Dict[str, Any] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    def ink_strokes(self) -> List[np.ndarray]:
        """Measured pen-down polylines (what ink would be on the paper)."""
        strokes, cur = [], []
        for p, d in zip(self.xy, self.pen_down):
            if d:
                cur.append(p)
            elif cur:
                if len(cur) >= 2:
                    strokes.append(np.asarray(cur))
                cur = []
        if len(cur) >= 2:
            strokes.append(np.asarray(cur))
        return strokes

    def summary(self) -> Dict[str, Any]:
        return {
            "backend": self.backend, "plan_hash": self.plan_hash, "policy_name": self.policy_name,
            "target_id": self.target_id, "seed": self.seed, "control_dt": self.control_dt,
            "duration_s": round(self.duration_s, 4), "samples": int(len(self.t)),
            "safety": self.safety, "notes": self.notes,
        }

    def to_dict(self, include_samples: bool = True) -> Dict[str, Any]:
        d = self.summary()
        if include_samples:
            d.update({
                "t": np.round(self.t, 5).tolist(),
                "xy": np.round(self.xy, 4).tolist(),
                "z": np.round(self.z, 4).tolist(),
                "pen_down": self.pen_down.astype(int).tolist(),
            })
        return d


# ---------------------------------------------------------------------------
# Kinematic plotter model
# ---------------------------------------------------------------------------

@dataclass
class KinematicParams:
    control_hz: float = 100.0
    servo_time_constant_s: float = 0.03
    max_speed_mm_s: float = 150.0
    max_accel_mm_s2: float = 3000.0
    latency_s: float = 0.0
    position_noise_mm: float = 0.03       # per-sample measurement/vibration noise
    offset_mm: tuple = (0.0, 0.0)         # systematic placement offset (used for perturbation studies)
    workspace_margin_mm: float = 3.0      # pen-down closer than this to the page edge = limit violation
    stall_error_mm: float = 3.0
    stall_duration_s: float = 0.25


class KinematicPenSimulator:
    name = "kinematic"

    def __init__(self, params: Optional[KinematicParams] = None):
        self.params = params or KinematicParams()

    @property
    def control_dt(self) -> float:
        return 1.0 / self.params.control_hz

    def execute(self, plan: PenPlan, seed: int = 0) -> PenExecution:
        p = self.params
        dt = self.control_dt
        cmd = build_commanded_trajectory(plan, dt)
        rng = np.random.default_rng(seed)
        n = len(cmd.t)
        lag = int(round(p.latency_s / dt))
        pos = cmd.xy[0].astype(float).copy()
        vel = np.zeros(2)
        z = float(cmd.z[0])
        xy_out = np.zeros((n, 2))
        z_out = np.zeros(n)
        alpha = dt / max(p.servo_time_constant_s, dt)
        for i in range(n):
            j = max(0, i - lag)
            desired = cmd.xy[j]
            v_des = (desired - pos) * (alpha / dt)
            sp = float(np.linalg.norm(v_des))
            if sp > p.max_speed_mm_s:
                v_des *= p.max_speed_mm_s / sp
            dv = v_des - vel
            dv_max = p.max_accel_mm_s2 * dt
            dvn = float(np.linalg.norm(dv))
            if dvn > dv_max:
                dv *= dv_max / dvn
            vel = vel + dv
            pos = pos + vel * dt
            z = z + (cmd.z[j] - z) * min(1.0, alpha * 1.5)
            xy_out[i] = pos + rng.normal(0.0, p.position_noise_mm, 2) + np.asarray(p.offset_mm)
            z_out[i] = z
        pen_down = z_out <= PEN_CONTACT_TOLERANCE_MM
        safety = _safety_from_tracking(xy_out, cmd.xy, pen_down, plan, dt, p.workspace_margin_mm,
                                       p.stall_error_mm, p.stall_duration_s)
        safety["joint_limit_violations"] = None
        safety["joint_limit_note"] = "not modelled (plotter kinematics have no joints in this model)"
        return PenExecution(self.name, plan.plan_hash, plan.policy_name, plan.target_id, seed, dt,
                            cmd.t.copy(), xy_out, z_out, pen_down, cmd.xy.copy(), safety,
                            ["Kinematic plotter model: first-order servo, speed/accel limits, Gaussian noise."])


def _safety_from_tracking(xy, cmd_xy, pen_down, plan, dt, margin, stall_err, stall_dur) -> Dict[str, Any]:
    """Shared safety proxies: page-edge violations, stall-like tracking failures."""
    w, h = plan.page_width_mm, plan.page_height_mm
    off_page = pen_down & ((xy[:, 0] < margin) | (xy[:, 1] < margin) | (xy[:, 0] > w - margin) | (xy[:, 1] > h - margin))
    err = np.linalg.norm(xy - cmd_xy, axis=1)
    stall_events = _count_runs(err > stall_err, int(math.ceil(stall_dur / dt)))
    return {
        "unexpected_contact_events": int(_count_runs(off_page, 1)),
        "unexpected_contact_note": "pen-down outside the page margin",
        "stall_events": int(stall_events),
        "stall_note": f"tracking error > {stall_err} mm for > {stall_dur} s (simulation proxy for a motor stall)",
        "max_tracking_error_mm": round(float(err.max()), 4) if len(err) else 0.0,
    }


def _count_runs(mask: np.ndarray, min_len: int) -> int:
    count, run = 0, 0
    for m in mask:
        if m:
            run += 1
            if run == min_len:
                count += 1
        else:
            run = 0
    return count


# ---------------------------------------------------------------------------
# PyBullet Franka Panda with a pen
# ---------------------------------------------------------------------------

@dataclass
class PyBulletPenParams:
    physics_hz: float = 240.0
    substeps: int = 4                     # physics steps per control step -> 60 Hz control
    page_center_world: tuple = (0.45, 0.0)
    table_top_z: float = 0.04
    paper_thickness: float = 0.0005
    # Joint servo: each control step the joints move toward the IK solution with a
    # first-order time constant and a per-joint speed limit (position-controlled arm).
    joint_time_constant_s: float = 0.035
    max_joint_speed_rad_s: float = 1.5
    pen_press_mm: float = 0.3             # spring-loaded holder: command slightly into the paper
    pen_length_m: float = 0.10            # pen protrudes this far beyond the gripper's grasp point
    tip_noise_mm: float = 0.03            # seeded measurement / vibration noise on the tip position
    ik_iterations: int = 100
    joint_limit_margin_rad: float = 0.02
    workspace_margin_mm: float = 3.0
    stall_error_mm: float = 3.0
    stall_duration_s: float = 0.25
    overpress_mm: float = 2.0             # tip this far below the paper = unexpected contact


class PyBulletPenSimulator:
    """Franka Panda (stand-in arm) writing on an A4 sheet.

    Arm model: inverse kinematics every control step (pen held vertical), and a
    velocity-limited first-order joint servo that moves the joints toward the IK
    solution. This captures the arm's kinematics, reach, joint limits and the
    tracking lag caused by limited joint speed. It does not model joint torques,
    link flexibility or pen-paper friction.

    Page -> world mapping (robot base at the origin, facing +x):
        world_x = cx - (y_mm - H/2) / 1000     (page top is far from the robot)
        world_y = cy - (x_mm - W/2) / 1000     (page right is on the robot's right)
    """

    name = "pybullet"
    _REST = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785]
    _EE_LINK = 11
    _ARM_JOINTS = list(range(7))
    _FINGERS = [9, 10]

    def __init__(self, gui: bool = False, params: Optional[PyBulletPenParams] = None, realtime: bool = True):
        self.gui = gui
        self.realtime = realtime
        self.params = params or PyBulletPenParams()
        self._p = None
        self._cid = None

    @property
    def control_dt(self) -> float:
        return self.params.substeps / self.params.physics_hz

    # -- coordinate mapping
    def page_to_world(self, x_mm: float, y_mm: float, z_mm: float, W: float, H: float) -> List[float]:
        cx, cy = self.params.page_center_world
        return [cx - (y_mm - H / 2.0) / 1000.0, cy - (x_mm - W / 2.0) / 1000.0, self._paper_z() + z_mm / 1000.0]

    def world_to_page(self, pos, W: float, H: float):
        cx, cy = self.params.page_center_world
        y_mm = (cx - pos[0]) * 1000.0 + H / 2.0
        x_mm = (cy - pos[1]) * 1000.0 + W / 2.0
        z_mm = (pos[2] - self._paper_z()) * 1000.0
        return x_mm, y_mm, z_mm

    def _paper_z(self) -> float:
        return self.params.table_top_z + self.params.paper_thickness

    # -- scene
    def _setup(self, W: float, H: float) -> None:
        try:
            import pybullet as p
            import pybullet_data
        except ImportError as exc:
            raise RuntimeError("PyBullet is required for the PyBullet writing simulator.") from exc
        self._p = p
        if self._cid is None or not p.isConnected(self._cid):
            self._cid = p.connect(p.GUI if self.gui else p.DIRECT)
        cid = self._cid
        p.resetSimulation(physicsClientId=cid)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(1.0 / self.params.physics_hz, physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=50, deterministicOverlappingPairs=1, physicsClientId=cid)
        if self.gui:
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=cid)
            p.resetDebugVisualizerCamera(cameraDistance=0.8, cameraYaw=90, cameraPitch=-60,
                                         cameraTargetPosition=[self.params.page_center_world[0], 0.0, 0.05],
                                         physicsClientId=cid)
        p.loadURDF("plane.urdf", physicsClientId=cid)
        tz = self.params.table_top_z
        self.table_id = self._box([0.35, 0.35, tz / 2], [0.45, 0.0, tz / 2], [0.55, 0.42, 0.28, 1])
        cx, cy = self.params.page_center_world
        self.paper_id = self._box([H / 2000.0, W / 2000.0, self.params.paper_thickness / 2],
                                  [cx, cy, tz + self.params.paper_thickness / 2], [0.97, 0.97, 0.95, 1])
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True,
                                   flags=p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS, physicsClientId=cid)
        self.lower, self.upper, self.ranges = [], [], []
        for j in self._ARM_JOINTS:
            info = p.getJointInfo(self.robot_id, j, physicsClientId=cid)
            self.lower.append(float(info[8]))
            self.upper.append(float(info[9]))
            self.ranges.append(float(info[9]) - float(info[8]))
        # Disable the default velocity motors so joints are driven only by the servo below
        for j in self._ARM_JOINTS + self._FINGERS:
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0, physicsClientId=cid)
        for j, q in zip(self._ARM_JOINTS, self._REST):
            p.resetJointState(self.robot_id, j, q, physicsClientId=cid)
        for j in self._FINGERS:  # gripper closed around the (virtual) pen
            p.resetJointState(self.robot_id, j, 0.0, physicsClientId=cid)
        self.down_orn = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])
        if self.gui:
            corners = [(0, 0), (W, 0), (W, H), (0, H), (0, 0)]
            for a, b in zip(corners[:-1], corners[1:]):
                p.addUserDebugLine(self.page_to_world(*a, 0.3, W, H), self.page_to_world(*b, 0.3, W, H),
                                   [0.6, 0.6, 0.6], 1.0, physicsClientId=cid)

    def _box(self, half, pos, rgba):
        p, cid = self._p, self._cid
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=cid)
        return p.createMultiBody(0.0, col, vis, basePosition=pos, physicsClientId=cid)

    def _ik(self, target):
        return self._p.calculateInverseKinematics(
            self.robot_id, self._EE_LINK, target, self.down_orn,
            lowerLimits=self.lower, upperLimits=self.upper, jointRanges=self.ranges, restPoses=self._REST,
            maxNumIterations=self.params.ik_iterations, residualThreshold=1e-6, physicsClientId=self._cid,
        )

    def _set_joints(self, q) -> None:
        for j, qq in zip(self._ARM_JOINTS, q):
            self._p.resetJointState(self.robot_id, j, float(qq), physicsClientId=self._cid)

    def _tip(self):
        """Pen tip = grasp point + pen_length along the gripper's approach (local +z) axis."""
        ls = self._p.getLinkState(self.robot_id, self._EE_LINK, computeForwardKinematics=True,
                                  physicsClientId=self._cid)
        R = np.asarray(self._p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
        return np.asarray(ls[4]) + R @ np.array([0.0, 0.0, self.params.pen_length_m])

    def _grasp_target_for_tip(self, tip_world):
        # Pen held vertically (tip pointing down): the grasp point sits pen_length above the tip
        return [tip_world[0], tip_world[1], tip_world[2] + self.params.pen_length_m]

    def execute(self, plan: PenPlan, seed: int = 0, ink_color=(0.05, 0.15, 0.85)) -> PenExecution:
        W, H = plan.page_width_mm, plan.page_height_mm
        self._setup(W, H)
        p, cid, prm = self._p, self._cid, self.params
        dt = self.control_dt
        cmd = build_commanded_trajectory(plan, dt)

        # Converge onto the start pose (not part of the measured execution)
        start = self.page_to_world(cmd.xy[0][0], cmd.xy[0][1], cmd.z[0], W, H)
        q = np.asarray(self._REST, dtype=float)
        for _ in range(20):
            q = np.asarray(self._ik(self._grasp_target_for_tip(start))[:7], dtype=float)
            self._set_joints(q)

        alpha = min(1.0, dt / max(prm.joint_time_constant_s, 1e-6))
        dq_max = prm.max_joint_speed_rad_s * dt
        lower = np.asarray(self.lower)
        upper = np.asarray(self.upper)

        rng = np.random.default_rng(seed)
        n = len(cmd.t)
        xy = np.zeros((n, 2))
        zz = np.zeros(n)
        joint_violations = 0
        contact_events = 0
        prev_in_violation = False
        prev_contact = False
        last_ink = None
        for i in range(n):
            z_cmd = cmd.z[i] - prm.pen_press_mm * max(0.0, 1.0 - cmd.z[i] / PEN_UP_HEIGHT_MM)
            target = self.page_to_world(cmd.xy[i][0], cmd.xy[i][1], z_cmd, W, H)
            q_goal = np.asarray(self._ik(self._grasp_target_for_tip(target))[:7], dtype=float)
            # IK above was solved from the CURRENT joint state; restore it, then servo toward the goal
            self._set_joints(q)
            step = np.clip(alpha * (q_goal - q), -dq_max, dq_max)
            q = q + step
            self._set_joints(q)
            for _ in range(prm.substeps):
                p.stepSimulation(physicsClientId=cid)
            self._set_joints(q)  # joints are servo-driven; physics stepping is used for contact checks

            tip = self._tip()
            x_mm, y_mm, z_mm = self.world_to_page(tip, W, H)
            if prm.tip_noise_mm > 0:
                x_mm += float(rng.normal(0.0, prm.tip_noise_mm))
                y_mm += float(rng.normal(0.0, prm.tip_noise_mm))
            xy[i] = (x_mm, y_mm)
            zz[i] = max(z_mm, 0.0)   # the paper stops a real tip; raw z is used for over-press detection

            in_violation = bool(np.any(q < lower + prm.joint_limit_margin_rad) or np.any(q > upper - prm.joint_limit_margin_rad))
            if in_violation and not prev_in_violation:
                joint_violations += 1
            prev_in_violation = in_violation

            touching = bool(p.getContactPoints(bodyA=self.robot_id, bodyB=self.table_id, physicsClientId=cid))
            contact = touching or (z_mm < -prm.overpress_mm)
            if contact and not prev_contact:
                contact_events += 1
            prev_contact = contact

            if self.gui:
                if z_mm <= PEN_CONTACT_TOLERANCE_MM:
                    here = self.page_to_world(x_mm, y_mm, 0.2, W, H)
                    if last_ink is not None:
                        p.addUserDebugLine(last_ink, here, ink_color, 2.5, physicsClientId=cid)
                    last_ink = here
                else:
                    last_ink = None
                if self.realtime:
                    time.sleep(dt)

        pen_down = zz <= PEN_CONTACT_TOLERANCE_MM
        safety = _safety_from_tracking(xy, cmd.xy, pen_down, plan, dt, prm.workspace_margin_mm,
                                       prm.stall_error_mm, prm.stall_duration_s)
        safety["unexpected_contact_events"] += contact_events
        safety["unexpected_contact_note"] = "pen-down outside page margin, robot-table contact, or tip > 2 mm below paper"
        safety["joint_limit_violations"] = int(joint_violations)
        safety["joint_limit_note"] = "Franka Panda stand-in model; entries into a 0.02 rad band at a joint limit"
        return PenExecution(self.name, plan.plan_hash, plan.policy_name, plan.target_id, seed, dt,
                            cmd.t.copy(), xy, zz, pen_down, cmd.xy.copy(), safety,
                            ["PyBullet Franka Panda stand-in arm, IK + velocity-limited joint servo; "
                             "virtual 10 cm pen held vertically in the gripper."])

    def close(self) -> None:
        if self._p is not None and self._cid is not None and self._p.isConnected(self._cid):
            self._p.disconnect(self._cid)
        self._cid = None


def make_writing_simulator(backend: str, gui: bool = False):
    backend = backend.lower()
    if backend == "kinematic":
        return KinematicPenSimulator()
    if backend == "pybullet":
        return PyBulletPenSimulator(gui=gui)
    raise ValueError(f"Unknown writing simulator '{backend}'. Use 'kinematic' or 'pybullet'.")
