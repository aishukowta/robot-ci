"""Handwriting quality metrics.

All distances are in millimetres on the page. Every metric works on plain
point sets / polylines, so the same code scores a simulated execution and the
ink extracted from a photo of the real page.

Metrics
-------
trajectory error   distance from each measured ink point to the nearest point of the
                   INTENDED target path (mean / RMS / 95th percentile / max), plus
                   coverage = share of the intended path that has ink within 1 mm,
                   and the symmetric Hausdorff distance.
writing accuracy   share of characters (or the pattern) recognised correctly by a
                   template recogniser built from the robot's own stroke font. This is
                   NOT general OCR: it answers "is this still legibly the intended symbol?"
execution time     seconds from first to last motion sample.
path efficiency    measured pen-down path length / planned pen-down length (1.0 = ideal);
                   regression uses |ratio - 1|.
repeatability      mean pairwise RMS distance between repeated executions of the same
                   policy and target (needs >= 2 trials).
safety             joint-limit entries + unexpected contacts + stall-like events (when detectable).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .font import GLYPHS, PATTERNS, RECOGNIZABLE_CHARACTERS, densify
from .targets import Cell, Layout

COVERAGE_TOLERANCE_MM = 1.0


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def nearest_distances(a: np.ndarray, b: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """For each point in a (N,2), Euclidean distance to the nearest point in b (M,2)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) == 0:
        return np.zeros(0)
    if len(b) == 0:
        return np.full(len(a), np.inf)
    out = np.empty(len(a))
    bb = (b ** 2).sum(1)
    for i in range(0, len(a), chunk):
        aa = a[i:i + chunk]
        d2 = (aa ** 2).sum(1)[:, None] + bb[None, :] - 2.0 * aa @ b.T
        out[i:i + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


def strokes_to_points(strokes: Sequence[np.ndarray], spacing_mm: float = 0.25) -> np.ndarray:
    pts = [densify(np.asarray(s, dtype=float), spacing_mm) for s in strokes if len(s) >= 2]
    singles = [np.asarray(s, dtype=float) for s in strokes if len(s) == 1]
    allp = pts + singles
    return np.vstack(allp) if allp else np.zeros((0, 2))


def _smoothed_length(stroke: np.ndarray) -> float:
    s = np.asarray(stroke, dtype=float)
    if len(s) < 2:
        return 0.0
    if len(s) >= 5:
        k = np.ones(3) / 3.0
        s = np.column_stack([np.r_[s[0, d], np.convolve(s[:, d], k, mode="valid"), s[-1, d]] for d in range(2)])
    return float(np.sum(np.linalg.norm(np.diff(s, axis=0), axis=1)))


# ---------------------------------------------------------------------------
# Trajectory error
# ---------------------------------------------------------------------------

def trajectory_error(measured_points: np.ndarray, intended_strokes: Sequence[np.ndarray]) -> Dict[str, Any]:
    ref = strokes_to_points(intended_strokes, 0.25)
    meas = np.asarray(measured_points, dtype=float)
    if len(meas) == 0:
        return {"mean_mm": None, "rms_mm": None, "p95_mm": None, "max_mm": None,
                "coverage": 0.0, "hausdorff_mm": None, "measured_points": 0}
    d_meas = nearest_distances(meas, ref)           # ink -> intended (precision side)
    d_ref = nearest_distances(ref, meas)            # intended -> ink (coverage side)
    return {
        "mean_mm": float(d_meas.mean()),
        "rms_mm": float(np.sqrt((d_meas ** 2).mean())),
        "p95_mm": float(np.quantile(d_meas, 0.95)),
        "max_mm": float(d_meas.max()),
        "coverage": float((d_ref <= COVERAGE_TOLERANCE_MM).mean()),
        "hausdorff_mm": float(max(d_meas.max(), d_ref.max())),
        "measured_points": int(len(meas)),
    }


# ---------------------------------------------------------------------------
# Template recogniser
# ---------------------------------------------------------------------------

class TemplateRecognizer:
    """Classifies a symbol by symmetric chamfer distance to the robot font's templates.

    Points are expressed in cell units (divide by character height, origin at the cell's
    top-left). A small translation search (+-0.1 cell units) tolerates minor offsets
    while still penalising shape distortion.
    """

    SHIFTS = np.array([[dx, dy] for dx in np.arange(-0.10, 0.101, 0.05) for dy in np.arange(-0.10, 0.101, 0.05)])

    def __init__(self):
        self.char_templates = {c: densify_all(GLYPHS[c][1], 0.04) for c in RECOGNIZABLE_CHARACTERS}
        self.pattern_templates = {n: _subsample(densify_all(s, 0.015), 240) for n, s in PATTERNS.items()}

    def score(self, pts: np.ndarray, template: np.ndarray) -> float:
        """Best symmetric mean chamfer distance over all translation shifts (vectorised)."""
        S, N, M = len(self.SHIFTS), len(pts), len(template)
        shifted = (pts[None, :, :] + self.SHIFTS[:, None, :]).reshape(S * N, 2)
        d2 = (shifted ** 2).sum(1)[:, None] + (template ** 2).sum(1)[None, :] - 2.0 * shifted @ template.T
        d = np.sqrt(np.maximum(d2, 0.0)).reshape(S, N, M)
        sym = 0.5 * (d.min(axis=2).mean(axis=1) + d.min(axis=1).mean(axis=1))
        return float(sym.min())

    def classify(self, pts: np.ndarray, kind: str) -> Dict[str, Any]:
        templates = self.char_templates if kind == "character" else self.pattern_templates
        if len(pts) < 3:
            return {"predicted": None, "score": None, "margin": None, "reason": "no ink in cell"}
        pts = _subsample(pts, 120)
        scores = sorted((self.score(pts, t), label) for label, t in templates.items())
        best, second = scores[0], scores[1]
        return {"predicted": best[1], "score": round(best[0], 4), "margin": round(second[0] - best[0], 4),
                "runner_up": second[1]}


def _subsample(pts: np.ndarray, n: int) -> np.ndarray:
    return pts if len(pts) <= n else pts[np.linspace(0, len(pts) - 1, n).astype(int)]


def densify_all(strokes: Sequence[np.ndarray], spacing: float) -> np.ndarray:
    return np.vstack([densify(np.asarray(s, dtype=float), spacing) for s in strokes])


_RECOGNIZER: Optional[TemplateRecognizer] = None


def get_recognizer() -> TemplateRecognizer:
    global _RECOGNIZER
    if _RECOGNIZER is None:
        _RECOGNIZER = TemplateRecognizer()
    return _RECOGNIZER


def points_in_cell(points: np.ndarray, cell: Cell) -> np.ndarray:
    h = cell.height_mm
    if cell.kind == "character":
        x0, x1 = cell.x_mm - 0.12 * h, cell.x_mm + cell.width_mm + 0.12 * h
        y0, y1 = cell.y_mm - 0.15 * h, cell.y_mm + 1.15 * h
    else:
        x0, x1 = cell.x_mm - 0.1 * h, cell.x_mm + 1.1 * h
        y0, y1 = cell.y_mm - 0.1 * h, cell.y_mm + 1.1 * h
    m = (points[:, 0] >= x0) & (points[:, 0] <= x1) & (points[:, 1] >= y0) & (points[:, 1] <= y1)
    sel = points[m]
    return np.column_stack([(sel[:, 0] - cell.x_mm) / h, (sel[:, 1] - cell.y_mm) / h])


def writing_accuracy(measured_points: np.ndarray, layout: Layout) -> Dict[str, Any]:
    rec = get_recognizer()
    pts = np.asarray(measured_points, dtype=float)
    results = []
    for cell in layout.cells:
        cell_pts = points_in_cell(pts, cell) if len(pts) else np.zeros((0, 2))
        r = rec.classify(cell_pts, cell.kind)
        r.update({"label": cell.label, "kind": cell.kind, "correct": r["predicted"] == cell.label})
        results.append(r)
    n = len(results)
    correct = sum(1 for r in results if r["correct"])
    return {
        "accuracy": correct / n if n else None,
        "correct": correct,
        "total": n,
        "cells": results,
        "method": "template_chamfer_recognizer_over_robot_font (not general OCR)",
    }


# ---------------------------------------------------------------------------
# Path efficiency, repeatability, safety
# ---------------------------------------------------------------------------

def path_efficiency(measured_strokes: Sequence[np.ndarray], planned_length_mm: float) -> Dict[str, Any]:
    measured = float(sum(_smoothed_length(s) for s in measured_strokes))
    ratio = measured / planned_length_mm if planned_length_mm > 0 else None
    return {
        "measured_draw_length_mm": measured,
        "planned_draw_length_mm": float(planned_length_mm),
        "ratio": ratio,
        "deviation": abs(ratio - 1.0) if ratio is not None else None,
    }


def resample_by_arclength(points: np.ndarray, n: int = 300) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return np.repeat(pts[:1], n, axis=0) if len(pts) else np.zeros((n, 2))
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.r_[0.0, np.cumsum(seg)]
    if s[-1] <= 0:
        return np.repeat(pts[:1], n, axis=0)
    u = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(u, s, pts[:, d]) for d in range(2)])


def repeatability(executions_strokes: List[Sequence[np.ndarray]], n: int = 300) -> Optional[Dict[str, Any]]:
    """Mean pairwise RMS distance (mm) between arc-length-resampled ink traces of repeated runs."""
    if len(executions_strokes) < 2:
        return None
    traces = []
    for strokes in executions_strokes:
        pts = np.vstack(strokes) if strokes else np.zeros((0, 2))
        traces.append(resample_by_arclength(pts, n))
    pair = [float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1)))) for a, b in combinations(traces, 2)]
    return {"mean_pairwise_rms_mm": float(np.mean(pair)), "max_pairwise_rms_mm": float(np.max(pair)),
            "pairs": len(pair), "runs": len(traces)}


def safety_total(safety: Dict[str, Any]) -> int:
    keys = ("joint_limit_violations", "unexpected_contact_events", "stall_events")
    return int(sum(int(safety.get(k) or 0) for k in keys))


# ---------------------------------------------------------------------------
# One execution -> metric dictionary
# ---------------------------------------------------------------------------

@dataclass
class TrialMetrics:
    target_id: str
    policy_name: str
    trial: int
    seed: int
    values: Dict[str, Optional[float]]
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"target_id": self.target_id, "policy_name": self.policy_name, "trial": self.trial,
                "seed": self.seed, "values": self.values, "details": self.details}


def score_execution(execution, layout: Layout, planned_length_mm: float, trial: int = 0) -> TrialMetrics:
    ink = execution.ink_strokes()
    pts = strokes_to_points(ink, 0.25)
    te = trajectory_error(pts, layout.strokes)
    acc = writing_accuracy(pts, layout)
    eff = path_efficiency(ink, planned_length_mm)
    values = {
        "trajectory_error_mean_mm": te["mean_mm"],
        "trajectory_error_max_mm": te["max_mm"],
        "path_coverage": te["coverage"],
        "writing_accuracy": acc["accuracy"],
        "execution_time_s": float(execution.duration_s),
        "path_efficiency_deviation": eff["deviation"],
        "safety_events": float(safety_total(execution.safety)),
    }
    details = {"trajectory_error": te, "writing_accuracy": acc, "path_efficiency": eff,
               "safety": execution.safety, "pen_down_strokes_measured": len(ink)}
    return TrialMetrics(layout.target_id, execution.policy_name, trial, execution.seed, values, details)
