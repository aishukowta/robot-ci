"""Writing policies for the handwriting robot.

A writing policy turns a target layout into a PenPlan (pen-up/pen-down
waypoints with feed rates). These are deterministic, rule-based planners —
they are not learned models.

* ``writing_baseline`` follows the font strokes faithfully.
* ``writing_candidate`` and the ``writing_candidate_*`` variants are the
  baseline with DELIBERATELY INJECTED defects (slower feed, corner cutting,
  tremor, offset, skipped pen lifts). They exist to check that the regression
  framework detects realistic handwriting regressions; they are not
  presented as genuinely new policy versions.

The existing manipulation policies (``baseline`` / ``candidate``) are
untouched; writing policies live in a separate registry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, List, Tuple

import numpy as np

from .font import densify
from .plan import PenPlan, PenWaypoint
from .targets import Layout


@dataclass(frozen=True)
class WritingPolicyParams:
    draw_feed_mm_s: float = 30.0          # pen-down speed
    travel_feed_mm_s: float = 60.0        # pen-up speed
    point_spacing_mm: float = 1.0         # waypoint spacing along strokes
    pen_settle_s: float = 0.15
    park_mm: Tuple[float, float] = (10.0, 10.0)   # start / end position, pen up
    # --- injected defects (all zero for the baseline) ---
    corner_smoothing_iterations: int = 0  # Chaikin corner cutting on raw strokes
    tremor_mm: float = 0.0                # seeded, low-pass filtered lateral noise
    offset_mm: Tuple[float, float] = (0.0, 0.0)
    scale: float = 1.0                    # about the layout centre
    rotation_deg: float = 0.0             # about the layout centre
    skip_pen_lift_below_mm: float = 0.0   # drag the pen across gaps shorter than this


class WritingPolicy:
    """Rule-based stroke planner. Subclass-free: behaviour comes from params."""

    def __init__(self, name: str, version: str, description: str, params: WritingPolicyParams):
        self.name = name
        self.version = version
        self.description = description
        self.params = params

    @property
    def metadata(self) -> Dict:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "type": "rule_based_stroke_planner",
            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(self.params).items()},
        }

    # ------------------------------------------------------------- planning
    def plan(self, layout: Layout, seed: int = 0) -> PenPlan:
        p = self.params
        rng = np.random.default_rng(seed)
        strokes = [np.asarray(s, dtype=float) for s in layout.strokes]

        if p.corner_smoothing_iterations > 0:
            strokes = [_chaikin(s, p.corner_smoothing_iterations) for s in strokes]
        strokes = [densify(s, p.point_spacing_mm) for s in strokes]

        if p.scale != 1.0 or p.rotation_deg != 0.0 or p.offset_mm != (0.0, 0.0):
            allp = np.vstack(strokes)
            centre = (allp.min(axis=0) + allp.max(axis=0)) / 2.0
            a = np.radians(p.rotation_deg)
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            off = np.asarray(p.offset_mm, dtype=float)
            strokes = [((s - centre) * p.scale) @ R.T + centre + off for s in strokes]

        if p.tremor_mm > 0.0:
            strokes = [_add_tremor(s, p.tremor_mm, rng) for s in strokes]

        waypoints: List[PenWaypoint] = [PenWaypoint(p.park_mm[0], p.park_mm[1], False, p.travel_feed_mm_s)]
        for i, s in enumerate(strokes):
            last = waypoints[-1]
            gap = float(np.hypot(s[0, 0] - last.x_mm, s[0, 1] - last.y_mm))
            drag = i > 0 and last.pen_down and gap < p.skip_pen_lift_below_mm
            if drag:
                # Defect: pen stays down across a short gap -> stray ink line
                waypoints.append(PenWaypoint(float(s[0, 0]), float(s[0, 1]), True, p.draw_feed_mm_s))
            else:
                if last.pen_down:
                    waypoints.append(PenWaypoint(last.x_mm, last.y_mm, False, p.travel_feed_mm_s))
                waypoints.append(PenWaypoint(float(s[0, 0]), float(s[0, 1]), False, p.travel_feed_mm_s))
            for pt in s[1:]:
                waypoints.append(PenWaypoint(float(pt[0]), float(pt[1]), True, p.draw_feed_mm_s))
        last = waypoints[-1]
        if last.pen_down:
            waypoints.append(PenWaypoint(last.x_mm, last.y_mm, False, p.travel_feed_mm_s))
        waypoints.append(PenWaypoint(p.park_mm[0], p.park_mm[1], False, p.travel_feed_mm_s))

        return PenPlan(
            policy_name=self.name,
            policy_version=self.version,
            target_id=layout.target_id,
            seed=int(seed),
            waypoints=waypoints,
            page_width_mm=layout.page_width_mm,
            page_height_mm=layout.page_height_mm,
            pen_settle_s=p.pen_settle_s,
            metadata={"policy": self.metadata},
        )


def _chaikin(stroke: np.ndarray, iterations: int) -> np.ndarray:
    """Chaikin corner cutting; keeps the end points (and closed strokes stay closed)."""
    pts = np.asarray(stroke, dtype=float)
    closed = len(pts) > 2 and np.linalg.norm(pts[0] - pts[-1]) < 1e-6
    for _ in range(iterations):
        if len(pts) < 3:
            break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        new = np.empty((2 * len(q), 2))
        new[0::2] = q
        new[1::2] = r
        if closed:
            pts = np.vstack([new, new[:1]])
        else:
            pts = np.vstack([pts[:1], new, pts[-1:]])
    return pts


def _add_tremor(stroke: np.ndarray, amplitude_mm: float, rng: np.random.Generator) -> np.ndarray:
    if len(stroke) < 3:
        return stroke
    noise = rng.normal(0.0, amplitude_mm, size=stroke.shape)
    k = 5  # moving-average low-pass so tremor is smooth, not per-point spikes
    kernel = np.ones(k) / k
    smooth = np.column_stack([np.convolve(noise[:, d], kernel, mode="same") for d in range(2)])
    smooth *= amplitude_mm / max(float(smooth.std()), 1e-9)
    return stroke + smooth


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BASE = WritingPolicyParams()

_WRITING_REGISTRY: Dict[str, Callable[[], WritingPolicy]] = {
    "writing_baseline": lambda: WritingPolicy(
        "writing_baseline", "w1.0-baseline",
        "Faithful stroke follower: 1 mm waypoints, 30 mm/s drawing, 60 mm/s travel, pen lifted between strokes.",
        _BASE,
    ),
    "writing_candidate": lambda: WritingPolicy(
        "writing_candidate", "w2.0-candidate",
        "Injected regression: 13% slower drawing, corner cutting, 0.25 mm tremor, 0.6/0.4 mm offset, "
        "skips pen lifts across gaps < 2.5 mm.",
        replace(_BASE, draw_feed_mm_s=26.0, corner_smoothing_iterations=1, tremor_mm=0.25,
                offset_mm=(0.6, 0.4), skip_pen_lift_below_mm=2.5),
    ),
    "writing_candidate_offset": lambda: WritingPolicy(
        "writing_candidate_offset", "w2.1-offset",
        "Injected regression: 1.5 mm / 1.0 mm offset and 1 degree rotation (calibration-style error).",
        replace(_BASE, offset_mm=(1.5, 1.0), rotation_deg=1.0),
    ),
    "writing_candidate_jitter": lambda: WritingPolicy(
        "writing_candidate_jitter", "w2.2-jitter",
        "Injected regression: 0.5 mm tremor.",
        replace(_BASE, tremor_mm=0.5),
    ),
    "writing_candidate_slow": lambda: WritingPolicy(
        "writing_candidate_slow", "w2.3-slow",
        "Injected regression: drawing feed 20 mm/s and travel feed 40 mm/s.",
        replace(_BASE, draw_feed_mm_s=20.0, travel_feed_mm_s=40.0),
    ),
    "writing_candidate_cornercut": lambda: WritingPolicy(
        "writing_candidate_cornercut", "w2.4-cornercut",
        "Injected regression: two Chaikin corner-cutting iterations (rounded, shrunken corners).",
        replace(_BASE, corner_smoothing_iterations=2),
    ),
    "writing_candidate_drag": lambda: WritingPolicy(
        "writing_candidate_drag", "w2.5-drag",
        "Injected regression: pen not lifted across gaps < 6 mm (stray connecting lines).",
        replace(_BASE, skip_pen_lift_below_mm=6.0),
    ),
}

WRITING_POLICY_NAMES = list(_WRITING_REGISTRY)


def get_writing_policy(name: str) -> WritingPolicy:
    if name not in _WRITING_REGISTRY:
        raise ValueError(f"Unknown writing policy '{name}'. Available: {WRITING_POLICY_NAMES}")
    return _WRITING_REGISTRY[name]()


def register_writing_policy(name: str, factory: Callable[[], WritingPolicy]) -> None:
    _WRITING_REGISTRY[name] = factory
    if name not in WRITING_POLICY_NAMES:
        WRITING_POLICY_NAMES.append(name)
