"""Export a PenPlan for the real handwriting robot (files only, never a live connection).

Outputs, in one export folder:
    plan.json        the exact plan (with its hash) - needed later to analyse the recording
    drawing.svg      page-sized SVG of the pen-down strokes (accepted by most robot apps)
    drawing.gcode    GRBL-style G-code (only when the robot profile is configured)
    waypoints.csv    page and machine coordinates of every waypoint
    manifest.json    what was exported, plan hash, policy, target, seed, profile, warnings
    LAB_INSTRUCTIONS.txt   checklist for drawing + recording
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from .plan import PenPlan


class ExportSafetyError(ValueError):
    pass


@dataclass
class RobotProfile:
    robot_name: str = "UNKNOWN"
    is_configured: bool = False
    execution_mode: str = "svg_import"
    machine_origin: str = "top_left"
    offset_mm: Tuple[float, float] = (0.0, 0.0)
    swap_xy: bool = False
    scale: float = 1.0
    workspace_mm: Optional[List[List[float]]] = None
    max_draw_feed_mm_s: float = 40.0
    max_travel_feed_mm_s: float = 80.0
    pen_mode: str = "servo"
    pen_down_command: str = "M3 S30"
    pen_up_command: str = "M3 S90"
    pen_dwell_s: float = 0.15
    gcode_header: List[str] = field(default_factory=lambda: ["G21", "G90"])
    gcode_footer: List[str] = field(default_factory=lambda: ["G0 X0 Y0"])
    source_file: Optional[str] = None

    @classmethod
    def load(cls, path: str) -> "RobotProfile":
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        ct = d.get("coordinate_transform", {}) or {}
        prof = cls(
            robot_name=str(d.get("robot_name", "UNKNOWN")),
            is_configured=bool(d.get("is_configured", False)),
            execution_mode=str(d.get("execution_mode", "svg_import")),
            machine_origin=str(ct.get("machine_origin", "top_left")),
            offset_mm=tuple(float(v) for v in ct.get("offset_mm", [0.0, 0.0])),
            swap_xy=bool(ct.get("swap_xy", False)),
            scale=float(ct.get("scale", 1.0)),
            workspace_mm=d.get("workspace_mm"),
            max_draw_feed_mm_s=float(d.get("max_draw_feed_mm_s", 40.0)),
            max_travel_feed_mm_s=float(d.get("max_travel_feed_mm_s", 80.0)),
            pen_mode=str(d.get("pen_mode", "servo")),
            pen_down_command=str(d.get("pen_down_command", "M3 S30")),
            pen_up_command=str(d.get("pen_up_command", "M3 S90")),
            pen_dwell_s=float(d.get("pen_dwell_s", 0.15)),
            gcode_header=list(d.get("gcode_header", ["G21", "G90"])),
            gcode_footer=list(d.get("gcode_footer", ["G0 X0 Y0"])),
            source_file=str(path),
        )
        if prof.machine_origin not in ("top_left", "bottom_left"):
            raise ValueError(f"{path}: machine_origin must be 'top_left' or 'bottom_left'.")
        if prof.execution_mode not in ("svg_import", "gcode", "other"):
            raise ValueError(f"{path}: execution_mode must be svg_import, gcode or other.")
        if not (0.1 <= prof.scale <= 10.0):
            raise ValueError(f"{path}: scale must be between 0.1 and 10.")
        return prof

    def page_to_machine(self, xy: np.ndarray, page_height_mm: float) -> np.ndarray:
        pts = np.asarray(xy, dtype=float).copy()
        if self.machine_origin == "bottom_left":
            pts[:, 1] = page_height_mm - pts[:, 1]
        pts *= self.scale
        if self.swap_xy:
            pts = pts[:, ::-1]
        return pts + np.asarray(self.offset_mm, dtype=float)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def _check_bounds(plan: PenPlan, machine_xy: np.ndarray, profile: RobotProfile) -> List[str]:
    xy = plan.xy()
    errors = []
    if (xy[:, 0] < 0).any() or (xy[:, 1] < 0).any() or (xy[:, 0] > plan.page_width_mm).any() \
            or (xy[:, 1] > plan.page_height_mm).any():
        errors.append("Plan contains points outside the page.")
    if profile.workspace_mm:
        lo = np.asarray(profile.workspace_mm[0], dtype=float)
        hi = np.asarray(profile.workspace_mm[1], dtype=float)
        bad = ~np.all((machine_xy >= lo) & (machine_xy <= hi), axis=1)
        if bad.any():
            errors.append(f"{int(bad.sum())} waypoint(s) fall outside the robot workspace {profile.workspace_mm} (machine mm).")
    return errors


def plan_to_svg(plan: PenPlan, stroke_width_mm: float = 0.5) -> str:
    w, h = plan.page_width_mm, plan.page_height_mm
    paths = []
    for s in plan.pen_down_strokes():
        d = "M " + " L ".join(f"{x:.3f} {y:.3f}" for x, y in s)
        paths.append(f'  <path d="{d}" />')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}">\n'
        f'  <!-- Robot CI plan {plan.plan_hash} | policy {plan.policy_name} ({plan.policy_version}) | '
        f'target {plan.target_id} | seed {plan.seed} -->\n'
        f'  <g fill="none" stroke="black" stroke-width="{stroke_width_mm}" stroke-linecap="round" stroke-linejoin="round">\n'
        + "\n".join(paths) + "\n  </g>\n</svg>\n"
    )


def plan_to_gcode(plan: PenPlan, profile: RobotProfile) -> Tuple[str, Dict[str, int]]:
    m = profile.page_to_machine(plan.xy(), plan.page_height_mm)
    lines = [f"; Robot CI plan {plan.plan_hash}", f"; policy {plan.policy_name} ({plan.policy_version})",
             f"; target {plan.target_id} seed {plan.seed}", f"; robot profile: {profile.robot_name}"]
    lines += list(profile.gcode_header)
    clamped = {"draw": 0, "travel": 0}
    pen = None
    for i, w in enumerate(plan.waypoints):
        if w.pen_down != pen:
            lines.append(profile.pen_down_command if w.pen_down else profile.pen_up_command)
            lines.append(f"G4 P{profile.pen_dwell_s:.3f}")
            pen = w.pen_down
        limit = profile.max_draw_feed_mm_s if w.pen_down else profile.max_travel_feed_mm_s
        feed = w.feed_mm_s
        if feed > limit:
            feed = limit
            clamped["draw" if w.pen_down else "travel"] += 1
        lines.append(f"G1 X{m[i, 0]:.3f} Y{m[i, 1]:.3f} F{feed * 60.0:.1f}")  # G-code feed is mm/min
    lines.append(profile.pen_up_command)
    lines += list(profile.gcode_footer)
    return "\n".join(lines) + "\n", clamped


LAB_TEXT = """ROBOT CI - LAB CHECKLIST FOR THIS EXPORT
=========================================
Plan hash : {hash}
Policy    : {policy} ({version})
Target    : {target}   seed {seed}
Page      : {w} x {h} mm, origin = TOP-LEFT corner of the paper, x right, y down

1. Paper: plain white {page_name} paper. Place it on a DARKER, non-glossy surface so all four
   page edges are visible. Tape the corners flat. The page must not move while writing.
2. Robot: load {file_hint} into the robot's own software. Align the robot's origin with the
   page as described in config/writing_robot_profile.yaml. Use the pen and settings you will
   use for every run. Do a dry run with the pen lifted (or removed) first if your robot allows.
3. Camera: phone on a stand or tripod, as close to directly ABOVE the page as possible, so the
   whole page is visible. Lock focus/exposure if you can. Do not move the phone during a run.
4. Record: start video -> start the robot -> let it finish and return away from the page ->
   keep recording 2 seconds with the page fully visible -> stop.
5. Photo (recommended): take one sharp, straight-down photo of the finished page with all four
   corners visible and no shadows across the writing.
6. Upload: dashboard > Handwriting Robot > Real-to-Sim, or:
   python -m robot_ci writing-analyze --plan {plan_rel} --video <video> --photo <photo> \\
       --source robot_phone_recording --executed-plan

Only tick / pass --executed-plan if the robot drew THIS exported file. If the robot used its
own built-in font or text tool instead, it is NOT the same policy (cross-controller comparison).
"""


def export_plan(plan: PenPlan, profile: RobotProfile, out_root: str = "results/writing/exports",
                formats: Optional[List[str]] = None) -> Dict[str, Any]:
    formats = formats or ["json", "svg", "csv", "gcode"]
    machine_xy = profile.page_to_machine(plan.xy(), plan.page_height_mm)
    errors = _check_bounds(plan, machine_xy, profile)
    if errors:
        raise ExportSafetyError("Export refused: " + " ".join(errors))

    folder = Path(out_root) / f"{plan.target_id}__{plan.policy_name}__seed{plan.seed}__{plan.plan_hash}"
    folder.mkdir(parents=True, exist_ok=True)
    files: Dict[str, str] = {}
    warnings: List[str] = []

    files["plan"] = plan.save(folder / "plan.json")
    if "svg" in formats:
        (folder / "drawing.svg").write_text(plan_to_svg(plan), encoding="utf-8")
        files["svg"] = str(folder / "drawing.svg")
        warnings.append("SVG import: the robot's app plans its own speeds, so execution time is not "
                        "comparable to the simulation; geometry is.")
    if "csv" in formats:
        with open(folder / "waypoints.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["index", "x_page_mm", "y_page_mm", "x_machine_mm", "y_machine_mm", "pen_down", "feed_mm_s"])
            for i, (w, mxy) in enumerate(zip(plan.waypoints, machine_xy)):
                wr.writerow([i, f"{w.x_mm:.4f}", f"{w.y_mm:.4f}", f"{mxy[0]:.4f}", f"{mxy[1]:.4f}", int(w.pen_down),
                             f"{w.feed_mm_s:.3f}"])
        files["csv"] = str(folder / "waypoints.csv")
    clamped = None
    if "gcode" in formats:
        if not profile.is_configured:
            warnings.append("G-code NOT written: config/writing_robot_profile.yaml has is_configured: false. "
                            "Fill in the robot's coordinate mapping and pen commands first.")
        else:
            text, clamped = plan_to_gcode(plan, profile)
            (folder / "drawing.gcode").write_text(text, encoding="utf-8")
            files["gcode"] = str(folder / "drawing.gcode")
            if clamped["draw"] or clamped["travel"]:
                warnings.append(f"Feeds clamped to the profile limits: {clamped}.")

    file_hint = "drawing.gcode" if "gcode" in files and profile.execution_mode == "gcode" else "drawing.svg"
    (folder / "LAB_INSTRUCTIONS.txt").write_text(LAB_TEXT.format(
        hash=plan.plan_hash, policy=plan.policy_name, version=plan.policy_version, target=plan.target_id,
        seed=plan.seed, w=plan.page_width_mm, h=plan.page_height_mm, page_name="A4" if plan.page_width_mm == 210 else "",
        file_hint=file_hint, plan_rel=Path(files["plan"]).as_posix()), encoding="utf-8")
    files["lab_instructions"] = str(folder / "LAB_INSTRUCTIONS.txt")

    manifest = {
        "format": "robot_ci_writing_export_v1",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "plan_hash": plan.plan_hash,
        "policy_name": plan.policy_name,
        "policy_version": plan.policy_version,
        "target_id": plan.target_id,
        "seed": plan.seed,
        "nominal_duration_s": round(plan.nominal_duration_s(), 3),
        "planned_draw_length_mm": round(plan.planned_draw_length_mm(), 3),
        "profile": profile.to_dict(),
        "files": files,
        "warnings": warnings,
        "feeds_clamped": clamped,
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files["manifest"] = str(folder / "manifest.json")
    manifest["folder"] = str(folder)
    return manifest
