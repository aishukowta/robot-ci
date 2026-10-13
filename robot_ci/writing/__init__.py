"""Handwriting robot extension for Robot CI.

Plan -> simulate -> measure -> compare baseline vs candidate -> export for the real
robot -> analyse the recording against simulation -> diagnose.
"""

from .font import GLYPHS, PATTERNS, SUPPORTED_CHARACTERS
from .targets import WritingTarget, Layout, load_targets, get_target
from .plan import PenPlan, PenWaypoint
from .policies import WritingPolicy, WritingPolicyParams, get_writing_policy, WRITING_POLICY_NAMES
from .simulate import KinematicPenSimulator, PyBulletPenSimulator, make_writing_simulator, PenExecution

__all__ = [
    "GLYPHS", "PATTERNS", "SUPPORTED_CHARACTERS", "WritingTarget", "Layout", "load_targets", "get_target",
    "PenPlan", "PenWaypoint", "WritingPolicy", "WritingPolicyParams", "get_writing_policy",
    "WRITING_POLICY_NAMES", "KinematicPenSimulator", "PyBulletPenSimulator", "make_writing_simulator",
    "PenExecution",
]
