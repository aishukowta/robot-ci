import numpy as np
from .baseline import BaselinePolicy

class CandidatePolicy(BaselinePolicy):
    """Same controller family as baseline, with realistic degradation.

    The candidate is not a different task strategy. It uses lower gain,
    no carry lift, tighter/later grasping, action noise, and a small
    systematic end-effector bias. Those changes come from the policy
    itself during rollouts, not from post-processing metrics.
    """

    def __init__(self):
        super().__init__()
        self.gain = 0.22
        self.grasp_distance = 0.032
        self.lift_height = 0.0
        self.action_noise = 0.014
        self.position_bias = np.array([0.018, -0.020, 0.012], dtype=np.float32)

    @property
    def version(self) -> str:
        return "v2.0-candidate"

    @property
    def metadata(self) -> dict:
        metadata = super().metadata
        metadata["name"] = "CandidatePolicy"
        metadata["description"] = "Degraded baseline controller (lower gain, noise, no lift, grasp error)"
        metadata["training"] = "noise-injected"
        return metadata

    def reset(self) -> None:
        self._step = 0
        self._rng = np.random.default_rng(99)
