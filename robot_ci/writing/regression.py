"""Baseline-vs-candidate regression testing for the handwriting robot.

Matched design: for every target and trial, both policies receive the same
plan seed and the simulator receives the same noise seed, so differences come
from the policies. Statistics are PAIRED over (target, trial).
"""

from __future__ import annotations

import json
import platform
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import yaml

from robot_ci.stats import bootstrap_mean_ci, summarize

from .font import polyline_length
from .metrics import TrialMetrics, repeatability, score_execution
from .policies import get_writing_policy
from .render import BLUE, RED, render_comparison
from .simulate import make_writing_simulator
from .targets import WritingTarget, load_targets

METRIC_ORDER = [
    "trajectory_error_mean_mm", "trajectory_error_max_mm", "path_coverage", "writing_accuracy",
    "execution_time_s", "path_efficiency_deviation", "repeatability_mm", "safety_events",
]

METRIC_LABELS = {
    "trajectory_error_mean_mm": ("Trajectory error (mean)", "mm"),
    "trajectory_error_max_mm": ("Trajectory error (max)", "mm"),
    "path_coverage": ("Path coverage within 1 mm", "fraction"),
    "writing_accuracy": ("Writing accuracy (recognition rate)", "fraction"),
    "execution_time_s": ("Execution time", "s"),
    "path_efficiency_deviation": ("Path efficiency |ratio - 1|", "fraction"),
    "repeatability_mm": ("Repeatability (pairwise RMS)", "mm"),
    "safety_events": ("Safety events", "count"),
}


def load_writing_thresholds(path: str) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    for name, cfg in data.items():
        for key in ("threshold", "direction", "comparison", "severity_levels"):
            if key not in cfg:
                raise ValueError(f"{path}: metric '{name}' is missing '{key}'.")
        if cfg["direction"] not in ("increase", "decrease", "monitor"):
            raise ValueError(f"{path}: metric '{name}' has invalid direction '{cfg['direction']}'.")
        if cfg["comparison"] not in ("absolute", "relative"):
            raise ValueError(f"{path}: metric '{name}' has invalid comparison '{cfg['comparison']}'.")
        lv = cfg["severity_levels"]
        if not (float(lv["minor"]) <= float(lv["major"]) <= float(lv["critical"])):
            raise ValueError(f"{path}: metric '{name}' severity levels must be minor <= major <= critical.")
    return data


def evaluate_metric(name: str, base_vals: List[float], cand_vals: List[float], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Paired comparison of one metric. base_vals[i] and cand_vals[i] are matched runs."""
    pairs = [(b, c) for b, c in zip(base_vals, cand_vals) if b is not None and c is not None
             and np.isfinite(b) and np.isfinite(c)]
    missing = len(base_vals) - len(pairs)
    label, unit = METRIC_LABELS.get(name, (name, ""))
    out: Dict[str, Any] = {"metric": name, "label": label, "unit": unit, "pairs": len(pairs), "missing_pairs": missing,
                           "direction": cfg["direction"], "comparison": cfg["comparison"],
                           "threshold": cfg["threshold"]}
    if not pairs:
        out.update({"regression_detected": False, "severity": "none",
                    "reason": "no valid paired values (metric unavailable)"})
        return out
    b = np.array([p[0] for p in pairs])
    c = np.array([p[1] for p in pairs])
    diffs = c - b
    mb, mc = float(b.mean()), float(c.mean())
    delta = float(diffs.mean())
    ci = bootstrap_mean_ci(diffs.tolist())
    out.update({"baseline": summarize(b), "candidate": summarize(c), "delta_mean": delta,
                "delta_ci95": ci})
    if cfg["direction"] == "monitor":
        out.update({"regression_detected": False, "severity": "none", "reason": "monitored only"})
        return out
    bad = delta if cfg["direction"] == "increase" else -delta
    scale = max(abs(mb), 1e-9) if cfg["comparison"] == "relative" else 1.0
    magnitude = bad / scale
    out["magnitude"] = magnitude
    if ci is not None:
        lo, hi = (ci[0], ci[1]) if cfg["direction"] == "increase" else (-ci[1], -ci[0])
        ci_excludes_zero = lo > 0.0
        out["bad_direction_ci95"] = [lo / scale, hi / scale]
    else:
        ci_excludes_zero = True  # fewer than 2 pairs: decide on the mean alone (reported)
    lv = cfg["severity_levels"]
    severity = "none"
    if magnitude >= float(lv["critical"]):
        severity = "critical"
    elif magnitude >= float(lv["major"]):
        severity = "major"
    elif magnitude >= float(lv["minor"]):
        severity = "minor"
    regression = magnitude >= float(cfg["threshold"]) and ci_excludes_zero and severity != "none"
    if not regression:
        severity = "none"
    unit_txt = "" if cfg["comparison"] == "relative" else f" {unit}"
    mag_txt = f"{magnitude * 100:.1f}%" if cfg["comparison"] == "relative" else f"{magnitude:.4f}{unit_txt}"
    reason = f"baseline {mb:.4f}, candidate {mc:.4f}; worse by {mag_txt} (threshold {cfg['threshold']})"
    if ci is not None:
        reason += f"; paired 95% CI of change [{ci[0]:.4f}, {ci[1]:.4f}]"
        if magnitude >= float(cfg["threshold"]) and not ci_excludes_zero:
            reason += " includes 0 -> not flagged (difference not consistent across runs)"
    out.update({"regression_detected": bool(regression), "severity": severity, "reason": reason})
    return out


@dataclass
class WritingRunConfig:
    baseline: str = "writing_baseline"
    candidate: str = "writing_candidate"
    targets: Optional[List[str]] = None
    trials: int = 3
    backend: str = "kinematic"
    targets_dir: str = "writing_targets"
    thresholds_path: str = "config/writing_thresholds.yaml"
    results_dir: str = "results/writing"
    render: bool = True


def run_writing_regression(cfg: WritingRunConfig, progress: Optional[Callable[[str], None]] = None) -> Dict[str, Any]:
    log = progress or (lambda msg: None)
    if cfg.trials < 1:
        raise ValueError("trials must be >= 1")
    thresholds = load_writing_thresholds(cfg.thresholds_path)
    base_pol = get_writing_policy(cfg.baseline)
    cand_pol = get_writing_policy(cfg.candidate)
    targets: List[WritingTarget] = load_targets(cfg.targets_dir, cfg.targets)
    sim = make_writing_simulator(cfg.backend)
    run_id = uuid.uuid4().hex[:8]
    run_dir = Path(cfg.results_dir) / f"run_{run_id}"

    per_trial: List[Dict[str, Any]] = []
    per_target: Dict[str, Any] = {}
    paired: Dict[str, Dict[str, List[Optional[float]]]] = {m: {"b": [], "c": []} for m in METRIC_ORDER}
    images: Dict[str, str] = {}
    plan_files: Dict[str, Dict[str, str]] = {}

    try:
        for t in targets:
            layout = t.layout()
            intended_len = float(sum(polyline_length(s) for s in layout.strokes))
            b_ink, c_ink = [], []
            b_trials: List[TrialMetrics] = []
            c_trials: List[TrialMetrics] = []
            for k in range(cfg.trials):
                seed = t.seed * 1000 + k
                pb = base_pol.plan(layout, seed)
                pc = cand_pol.plan(layout, seed)
                eb = sim.execute(pb, seed)
                ec = sim.execute(pc, seed)
                mb = score_execution(eb, layout, intended_len, k)
                mc = score_execution(ec, layout, intended_len, k)
                b_trials.append(mb)
                c_trials.append(mc)
                b_ink.append(eb.ink_strokes())
                c_ink.append(ec.ink_strokes())
                for m in METRIC_ORDER:
                    if m == "repeatability_mm":
                        continue
                    paired[m]["b"].append(mb.values.get(m))
                    paired[m]["c"].append(mc.values.get(m))
                for which, tm, ex, plan in (("baseline", mb, eb, pb), ("candidate", mc, ec, pc)):
                    per_trial.append({**tm.to_dict(), "role": which, "plan_hash": plan.plan_hash,
                                      "execution": ex.summary()})
                if k == 0:
                    plan_files[t.target_id] = {
                        "baseline": pb.save(run_dir / "plans" / f"{cfg.baseline}__{t.target_id}__seed{seed}.json"),
                        "candidate": pc.save(run_dir / "plans" / f"{cfg.candidate}__{t.target_id}__seed{seed}.json"),
                    }
                    if cfg.render:
                        images[t.target_id] = render_comparison(
                            layout,
                            [(f"baseline ({cfg.baseline})", eb.ink_strokes(), BLUE),
                             (f"candidate ({cfg.candidate})", ec.ink_strokes(), RED)],
                            run_dir / "images" / f"{t.target_id}.png",
                            title=f"{t.target_id} - {cfg.backend} simulation, trial 1",
                        )
                log(f"  {t.target_id} trial {k + 1}/{cfg.trials}: "
                    f"err {mb.values['trajectory_error_mean_mm']:.3f} vs {mc.values['trajectory_error_mean_mm']:.3f} mm, "
                    f"acc {mb.values['writing_accuracy']:.2f} vs {mc.values['writing_accuracy']:.2f}")
            rb = repeatability(b_ink)
            rc = repeatability(c_ink)
            paired["repeatability_mm"]["b"].append(rb["mean_pairwise_rms_mm"] if rb else None)
            paired["repeatability_mm"]["c"].append(rc["mean_pairwise_rms_mm"] if rc else None)

            def means(trials: List[TrialMetrics]) -> Dict[str, Optional[float]]:
                out = {}
                for m in METRIC_ORDER:
                    if m == "repeatability_mm":
                        continue
                    vals = [tm.values.get(m) for tm in trials if tm.values.get(m) is not None]
                    out[m] = float(np.mean(vals)) if vals else None
                return out

            per_target[t.target_id] = {
                "target": t.to_dict(),
                "baseline": {**means(b_trials), "repeatability_mm": rb["mean_pairwise_rms_mm"] if rb else None},
                "candidate": {**means(c_trials), "repeatability_mm": rc["mean_pairwise_rms_mm"] if rc else None},
                "misread_cells": {
                    "baseline": _misreads(b_trials),
                    "candidate": _misreads(c_trials),
                },
            }
    finally:
        if hasattr(sim, "close"):
            sim.close()

    findings = []
    for m in METRIC_ORDER:
        if m not in thresholds:
            continue
        findings.append(evaluate_metric(m, paired[m]["b"], paired[m]["c"], thresholds[m]))
    regressions = [f for f in findings if f.get("regression_detected")]
    if cfg.trials < 2:
        notes_rep = "Repeatability needs >= 2 trials; it is unavailable for this run."
    else:
        notes_rep = "Repeatability is computed per target across trials, then paired across targets."

    result = {
        "type": "writing_regression",
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline": base_pol.metadata,
        "candidate": cand_pol.metadata,
        "backend": cfg.backend,
        "evidence_level": "simulation",
        "trials": cfg.trials,
        "targets": [t.target_id for t in targets],
        "thresholds_file": cfg.thresholds_path,
        "thresholds": thresholds,
        "findings": findings,
        "overall_status": "REGRESSION_DETECTED" if regressions else "PASS",
        "per_target": per_target,
        "per_trial": per_trial,
        "images": images,
        "plan_files": plan_files,
        "environment": _environment(),
        "notes": [
            "Simulation results. Writing policies are rule-based planners; the candidate contains injected defects.",
            "Matched seeds: both policies use the same plan seed and simulator noise seed for each (target, trial).",
            notes_rep,
            "Writing accuracy uses a template recogniser over the robot's own font, not general OCR.",
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(cfg.results_dir) / f"run_{run_id}.json"
    out_path.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    result["saved_to"] = str(out_path)
    return result


def _misreads(trials: List[TrialMetrics]) -> List[Dict[str, Any]]:
    out = []
    for tm in trials:
        for c in tm.details["writing_accuracy"]["cells"]:
            if not c["correct"]:
                out.append({"trial": tm.trial, "label": c["label"], "predicted": c["predicted"]})
    return out


def _environment() -> Dict[str, str]:
    env = {"python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__}
    try:
        import pybullet  # noqa: F401
        env["pybullet"] = "installed"
    except ImportError:
        env["pybullet"] = "missing"
    try:
        import cv2
        env["opencv"] = cv2.__version__
    except ImportError:
        env["opencv"] = "missing"
    return env


def _json_default(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def load_writing_runs(results_dir: str = "results/writing") -> List[Dict[str, Any]]:
    runs = []
    for p in sorted(Path(results_dir).glob("run_*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("type") == "writing_regression":
                d["saved_to"] = str(p)
                runs.append(d)
        except (OSError, json.JSONDecodeError):
            continue
    return runs
