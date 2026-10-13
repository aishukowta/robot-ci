"""PenPlan: the exact motion a writing policy commands.

The SAME plan object is executed in simulation and exported for the real
robot. Its SHA-256 hash (over rounded waypoints) links an uploaded recording
back to the plan that produced it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class PenWaypoint:
    x_mm: float
    y_mm: float
    pen_down: bool
    feed_mm_s: float        # speed used to travel TO this waypoint

    def to_list(self) -> List[Any]:
        return [round(self.x_mm, 4), round(self.y_mm, 4), bool(self.pen_down), round(self.feed_mm_s, 3)]


@dataclass
class PenPlan:
    policy_name: str
    policy_version: str
    target_id: str
    seed: int
    waypoints: List[PenWaypoint]
    page_width_mm: float
    page_height_mm: float
    pen_settle_s: float = 0.15          # time to raise / lower the pen
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # ------------------------------------------------------------------ props
    @property
    def plan_hash(self) -> str:
        payload = json.dumps(
            {"w": [w.to_list() for w in self.waypoints], "settle": round(self.pen_settle_s, 4),
             "page": [self.page_width_mm, self.page_height_mm]},
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def xy(self) -> np.ndarray:
        return np.array([[w.x_mm, w.y_mm] for w in self.waypoints], dtype=float)

    def pen_down_strokes(self) -> List[np.ndarray]:
        """Polylines drawn with the pen down (consecutive pen-down waypoints)."""
        strokes: List[np.ndarray] = []
        cur: List[List[float]] = []
        prev: Optional[PenWaypoint] = None
        for w in self.waypoints:
            if w.pen_down:
                if not cur and prev is not None:
                    cur.append([prev.x_mm, prev.y_mm])  # pen goes down at previous location
                cur.append([w.x_mm, w.y_mm])
            else:
                if len(cur) >= 2:
                    strokes.append(np.asarray(cur))
                cur = []
            prev = w
        if len(cur) >= 2:
            strokes.append(np.asarray(cur))
        return strokes

    def planned_draw_length_mm(self) -> float:
        return float(sum(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1)) for s in self.pen_down_strokes()))

    def planned_travel_length_mm(self) -> float:
        xy = self.xy()
        if len(xy) < 2:
            return 0.0
        seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        down = np.array([w.pen_down for w in self.waypoints[1:]])
        return float(seg[~down].sum())

    def nominal_duration_s(self) -> float:
        xy = self.xy()
        t = 0.0
        for i in range(1, len(self.waypoints)):
            d = float(np.linalg.norm(xy[i] - xy[i - 1]))
            t += d / max(self.waypoints[i].feed_mm_s, 1e-6)
            if self.waypoints[i].pen_down != self.waypoints[i - 1].pen_down:
                t += self.pen_settle_s
        return t

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": "robot_ci_pen_plan_v1",
            "plan_hash": self.plan_hash,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "target_id": self.target_id,
            "seed": self.seed,
            "page_width_mm": self.page_width_mm,
            "page_height_mm": self.page_height_mm,
            "pen_settle_s": self.pen_settle_s,
            "coordinate_frame": "page_mm_origin_top_left_x_right_y_down",
            "waypoint_fields": ["x_mm", "y_mm", "pen_down", "feed_mm_s"],
            "waypoints": [w.to_list() for w in self.waypoints],
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PenPlan":
        if d.get("format") != "robot_ci_pen_plan_v1":
            raise ValueError("Not a Robot CI pen plan (format must be 'robot_ci_pen_plan_v1').")
        plan = cls(
            policy_name=d["policy_name"], policy_version=d["policy_version"], target_id=d["target_id"],
            seed=int(d["seed"]),
            waypoints=[PenWaypoint(float(x), float(y), bool(p), float(f)) for x, y, p, f in d["waypoints"]],
            page_width_mm=float(d["page_width_mm"]), page_height_mm=float(d["page_height_mm"]),
            pen_settle_s=float(d.get("pen_settle_s", 0.15)), metadata=dict(d.get("metadata", {})),
            created_at=d.get("created_at", ""),
        )
        if d.get("plan_hash") and d["plan_hash"] != plan.plan_hash:
            raise ValueError(
                f"Plan hash mismatch: file says {d['plan_hash']}, contents hash to {plan.plan_hash}. "
                "The plan file was modified after export."
            )
        return plan

    def save(self, path: Union[str, Path]) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "PenPlan":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
