"""End-to-end self-test of the real-to-sim pipeline using SYNTHETIC recordings.

Creates a fake "real robot" execution with KNOWN errors (placement offset and
extra noise), renders it as a phone photo and a phone video with perspective,
rotation, uneven lighting, sensor noise and compression, then runs the exact
upload pipeline. Because the injected offset is known, we can check that the
pipeline recovers it. Everything produced here is labelled synthetic_fixture
and can never count as physical evidence.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .analysis import AnalysisInputs, analyze_recording, _cv2
from .export import RobotProfile, export_plan
from .policies import get_writing_policy
from .simulate import KinematicParams, KinematicPenSimulator
from .targets import get_target

INJECTED_OFFSET_MM = (1.2, -0.8)


def _page_quad(canvas_w: int, canvas_h: int, page_w_mm: float, page_h_mm: float, fill: float,
               angle_deg: float, keystone: float) -> np.ndarray:
    """Image-space corners (page TL, TR, BR, BL) of a rotated, slightly keystoned page."""
    ph = canvas_h * fill
    pw = ph * page_w_mm / page_h_mm
    cx, cy = canvas_w / 2.0, canvas_h / 2.0
    base = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]])
    base[0, 0] += keystone * pw
    base[1, 0] -= keystone * pw          # top edge shorter: camera slightly tilted
    a = np.radians(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return base @ R.T + [cx, cy]


def _background(w: int, h: int, rng) -> np.ndarray:
    base = np.zeros((h, w, 3), np.float32)
    base[:] = (62, 70, 84)                      # dark table (BGR)
    base += rng.normal(0, 6, (h, w, 1)).astype(np.float32)
    return base


def _composite(page_bgr: np.ndarray, quad: np.ndarray, canvas: np.ndarray, rng, noise: float = 4.0) -> np.ndarray:
    cv2 = _cv2()
    h, w = canvas.shape[:2]
    ph, pw = page_bgr.shape[:2]
    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], np.float32)
    M = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    warped = cv2.warpPerspective(page_bgr, M, (w, h), flags=cv2.INTER_AREA)
    mask = cv2.warpPerspective(np.full((ph, pw), 255, np.uint8), M, (w, h))
    out = canvas.copy()
    m = mask > 127
    out[m] = warped[m]
    yy, xx = np.mgrid[0:h, 0:w]
    light = 0.86 + 0.16 * (xx / w) * (1 - 0.3 * yy / h)      # uneven lighting
    out = out * light[..., None]
    out += rng.normal(0, noise, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _draw_ink(page: np.ndarray, pts_mm: np.ndarray, px_per_mm: float, width_mm: float, color=(70, 30, 25)):
    cv2 = _cv2()
    p = np.round(pts_mm * px_per_mm).astype(np.int32)
    th = max(1, int(round(width_mm * px_per_mm)))
    for a, b in zip(p[:-1], p[1:]):
        cv2.line(page, tuple(a), tuple(b), color, th, cv2.LINE_AA)


def make_synthetic_recording(out_dir: Path, target_id: str = "word_hello", policy: str = "writing_baseline",
                             with_video: bool = True) -> Dict[str, Any]:
    cv2 = _cv2()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2024)
    target = get_target(target_id)
    plan = get_writing_policy(policy).plan(target.layout(), target.seed)
    export = export_plan(plan, RobotProfile(), str(out_dir / "exports"), formats=["json", "svg"])

    # "Real" robot = kinematic model with a known placement offset and more noise than the simulator
    fake_real = KinematicPenSimulator(replace(KinematicParams(), offset_mm=INJECTED_OFFSET_MM,
                                              position_noise_mm=0.06, servo_time_constant_s=0.05))
    ex = fake_real.execute(plan, seed=99)
    W, H = plan.page_width_mm, plan.page_height_mm

    # ---- photo ----
    ppm = 10.0
    page = np.full((int(H * ppm), int(W * ppm), 3), (236, 240, 242), np.uint8)
    for s in ex.ink_strokes():
        _draw_ink(page, s, ppm, 0.6)
    canvas = _background(3000, 2250, rng)
    quad = _page_quad(3000, 2250, W, H, 0.88, angle_deg=-6.0, keystone=0.04)
    photo = _composite(page, quad, canvas, rng)
    photo_path = out_dir / "synthetic_page_photo.jpg"
    cv2.imwrite(str(photo_path), photo, [cv2.IMWRITE_JPEG_QUALITY, 90])

    video_path = None
    if with_video:
        fps = 15.0
        vw, vh = 1280, 960
        vppm = 6.0
        vquad = _page_quad(vw, vh, W, H, 0.92, angle_deg=4.0, keystone=0.03)
        bg = _background(vw, vh, rng)
        vpage = np.full((int(H * vppm), int(W * vppm), 3), (236, 240, 242), np.uint8)
        src = np.array([[0, 0], [vpage.shape[1], 0], [vpage.shape[1], vpage.shape[0]], [0, vpage.shape[0]]], np.float32)
        Mp = cv2.getPerspectiveTransform(src, vquad.astype(np.float32))
        video_path = out_dir / "synthetic_robot_run.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh))
        base_px = np.array([vw * 0.5, vh * 1.15])
        t_ex = ex.t
        drawn = 0
        lead_in, park_out, tail = 1.0, 0.6, 2.0
        total = lead_in + ex.duration_s + park_out + tail
        for f in range(int(total * fps)):
            t = f / fps
            te = t - lead_in
            if 0 <= te <= ex.duration_s:
                upto = int(np.searchsorted(t_ex, te, side="right"))
                for i in range(max(drawn, 1), upto):
                    if ex.pen_down[i] and ex.pen_down[i - 1]:
                        _draw_ink(vpage, ex.xy[i - 1:i + 1], vppm, 0.8)
                drawn = max(drawn, upto)
                tip_mm = ex.xy[min(upto, len(ex.xy) - 1)]
            elif te < 0:
                tip_mm = ex.xy[0]
            else:
                tip_mm = None
            frame = _composite(vpage, vquad, bg, rng, noise=3.0).astype(np.uint8)
            if tip_mm is not None or te <= ex.duration_s + park_out:
                if tip_mm is None:  # arm leaving the page
                    k = (te - ex.duration_s) / park_out
                    tip_px = cv2.perspectiveTransform(np.array([[ex.xy[-1] * vppm]], np.float64), Mp)[0, 0]
                    tip_px = tip_px + (np.array([vw * 1.3, vh * 1.3]) - tip_px) * min(1.0, k)
                else:
                    tip_px = cv2.perspectiveTransform(np.array([[tip_mm * vppm]], np.float64), Mp)[0, 0]
                cv2.line(frame, tuple(np.int32(base_px)), tuple(np.int32(tip_px)), (45, 45, 50), 46, cv2.LINE_AA)
                cv2.circle(frame, tuple(np.int32(tip_px)), 16, (30, 30, 200), -1, cv2.LINE_AA)  # red marker
            writer.write(frame)
        writer.release()

    return {"plan_path": export["files"]["plan"], "photo_path": str(photo_path),
            "video_path": str(video_path) if video_path else None, "true_duration_s": ex.duration_s,
            "injected_offset_mm": list(INJECTED_OFFSET_MM)}


def run_selftest(out_dir: str = "results/writing/selftest", sim_backend: str = "kinematic",
                 target_id: str = "word_hello") -> Dict[str, Any]:
    out = Path(out_dir)
    rec = make_synthetic_recording(out, target_id)
    checks = []

    def check(name, ok, detail):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    rep_photo = analyze_recording(AnalysisInputs(
        plan_path=rec["plan_path"], data_source="synthetic_fixture", executed_plan=True,
        photo_path=rec["photo_path"], video_path=rec["video_path"], sim_backend=sim_backend,
        execution_mode="gcode", marker_color="red"), out / "analysis")
    m = rep_photo["metrics"]
    t_mm = rep_photo["placement_correction"]["translation_mm"]
    expected = [-INJECTED_OFFSET_MM[0], -INJECTED_OFFSET_MM[1]]
    err_offset = float(np.hypot(t_mm[0] - expected[0], t_mm[1] - expected[1]))
    check("page detected", rep_photo["page_detection"]["aspect_error"] < 0.05,
          f"aspect error {rep_photo['page_detection']['aspect_error']:.3f}")
    check("injected placement offset recovered (photo)", err_offset < 0.5,
          f"recovered correction {t_mm} mm, expected about {expected} mm (error {err_offset:.2f} mm)")
    check("aligned stroke error small (photo)", m["intended_vs_real_aligned"]["trajectory_error_mean_mm"] < 0.6,
          f"{m['intended_vs_real_aligned']['trajectory_error_mean_mm']:.3f} mm")
    check("all characters recognised (photo)", m["intended_vs_real_aligned"]["writing_accuracy"] == 1.0,
          f"accuracy {m['intended_vs_real_aligned']['writing_accuracy']}")
    td = m["timing"]["real_active_duration_s"]
    check("execution time measured from video", td is not None and abs(td - rec["true_duration_s"]) < 1.5,
          f"measured {td} s vs true {rec['true_duration_s']:.2f} s")
    check("synthetic data never counted as robot evidence", rep_photo["comparison_type"] == "not_robot_evidence",
          rep_photo["comparison_type"])

    rep_video = analyze_recording(AnalysisInputs(
        plan_path=rec["plan_path"], data_source="synthetic_fixture", executed_plan=True,
        video_path=rec["video_path"], sim_backend=sim_backend, execution_mode="gcode"), out / "analysis")
    mv = rep_video["metrics"]
    tv = rep_video["placement_correction"]["translation_mm"]
    err_v = float(np.hypot(tv[0] - expected[0], tv[1] - expected[1]))
    check("video-only: page found in final frame", rep_video["page_detection"]["aspect_error"] < 0.06,
          f"aspect error {rep_video['page_detection']['aspect_error']:.3f}")
    check("video-only: offset recovered", err_v < 0.8, f"recovered {tv} mm (error {err_v:.2f} mm)")
    check("video-only: characters recognised", mv["intended_vs_real_aligned"]["writing_accuracy"] >= 0.8,
          f"accuracy {mv['intended_vs_real_aligned']['writing_accuracy']}")

    summary = {"passed": all(c["passed"] for c in checks), "checks": checks,
               "photo_report": rep_photo["saved_to"], "video_report": rep_video["saved_to"],
               "synthetic_files": rec}
    (out / "selftest_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["saved_to"] = str(out / "selftest_summary.json")
    return summary
