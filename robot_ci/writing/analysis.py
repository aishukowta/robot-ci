"""Real-to-sim analysis of a handwriting robot recording.

Input: the exported plan (plan.json) plus a phone VIDEO and/or a PHOTO of the
finished page. Output: a report comparing three things on the same page
coordinates (mm):

    intended target path   <->   simulated execution of the SAME plan   <->   real ink on paper

Pipeline
--------
1. Page image: the photo, or the sharpest frame after motion stops in the video.
2. Page detection: the four paper corners (automatic, or supplied manually).
   A perspective transform (homography) maps the photo onto the page in mm.
   This is valid because writing happens on a flat sheet, so a single
   uncalibrated phone camera still gives metric (mm) measurements on the page.
3. Ink extraction: adaptive threshold -> remove specks -> skeleton -> points (mm).
4. Orientation: if the page might be rotated 180 deg in the photo, the orientation
   that best matches the intended path is chosen (and reported).
5. Registration: a rigid fit (rotation + translation, no scaling) separates
   "the paper was placed slightly off" from "the strokes have the wrong shape".
   Raw and aligned numbers are BOTH reported.
6. Video timing: motion energy between frames gives the active writing duration.
   Optional: track a coloured marker on the pen holder (parallax caveat applies).

Limits (always stated in the report): pen pressure, contact force, joint angles
and motor stalls cannot be measured from a photo or video.
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .font import polyline_length
from .metrics import nearest_distances, strokes_to_points, trajectory_error, writing_accuracy, score_execution
from .plan import PenPlan
from .render import BLUE, GREY, RED, add_legend, crop_to, draw_points, draw_strokes
from .simulate import make_writing_simulator
from .targets import Layout, get_target

PAGE_PX_PER_MM = 8.0

VALID_SOURCES = ("robot_phone_recording", "handheld_marker_recording", "synthetic_fixture",
                 "controlled_sim_perturbation")

UNMEASURABLE = [
    "pen pressure / contact force (no force sensor)",
    "joint angles, joint-limit margins and motor stalls (no robot telemetry)",
    "pen-lift height (monocular top view)",
]


class AnalysisError(RuntimeError):
    pass


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise AnalysisError("OpenCV is required: pip install -e .[vision]") from exc
    return cv2


# ---------------------------------------------------------------------------
# Image acquisition
# ---------------------------------------------------------------------------

@dataclass
class VideoMotion:
    fps: float
    frame_count: int
    duration_s: float
    active_start_s: Optional[float]
    active_end_s: Optional[float]
    active_duration_s: Optional[float]
    page_frame_index: int
    page_frame_time_s: float
    motion_threshold: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def load_image(path: Union[str, Path]) -> np.ndarray:
    cv2 = _cv2()
    p = Path(path)
    if not p.exists():
        raise AnalysisError(f"Image not found: {p}")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise AnalysisError(f"Could not read image: {p} (supported: jpg, png, bmp, tiff)")
    return img


def analyze_video_motion(path: Union[str, Path], max_width: int = 360) -> Tuple[VideoMotion, np.ndarray]:
    """Returns motion timing and the best frame of the finished page (BGR)."""
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise AnalysisError(f"OpenCV could not open the video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not (fps > 0 and math.isfinite(fps)):
        fps = 30.0
    energies: List[float] = []
    sharpness: List[float] = []
    frames_tail: List[Tuple[int, np.ndarray]] = []
    prev = None
    idx = 0
    tail_keep = int(max(fps * 4, 30))
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        h, w = frame.shape[:2]
        scale = max_width / float(w) if w > max_width else 1.0
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        g = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        energies.append(0.0 if prev is None else float(np.mean(cv2.absdiff(g, prev))))
        sharpness.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        prev = g
        frames_tail.append((idx, frame))
        if len(frames_tail) > tail_keep:
            frames_tail.pop(0)
        idx += 1
    cap.release()
    if idx == 0:
        raise AnalysisError(f"Video contains no decodable frames: {path}")

    e = np.asarray(energies)
    k = max(1, int(round(fps * 0.3)))
    smooth = np.convolve(e, np.ones(k) / k, mode="same")
    # Idle floor = quiet frames (camera noise); motion level = typical busy frames.
    # The threshold sits a little above the floor, so slow writing still counts as motion
    # while one large spike (e.g. the arm leaving the frame) cannot dominate it.
    floor = float(np.percentile(smooth, 10))
    busy = float(np.percentile(smooth, 90))
    thr = floor + max(0.05, 0.15 * (busy - floor))
    active = np.where(smooth > thr)[0]
    notes: List[str] = []
    if len(active) == 0 or busy - floor < 0.1:
        start = end = dur = None
        notes.append("No clear motion detected; execution time unavailable from this video.")
        last_active = -1
    else:
        start = float(active[0] / fps)
        end = float(active[-1] / fps)
        dur = end - start
        last_active = int(active[-1])
        if active[-1] >= idx - 2:
            notes.append("Motion continues until the end of the video; record ~2 s after the robot stops.")

    # Page frame: sharpest frame AFTER motion stops (falls back to the sharpest tail frame)
    candidates = [(i, f) for i, f in frames_tail if i > last_active + int(fps * 0.3)] or frames_tail
    best_i, best_frame = max(candidates, key=lambda it: sharpness[it[0]])
    motion = VideoMotion(fps, idx, idx / fps, start, end, dur, int(best_i), best_i / fps, thr, notes)
    return motion, best_frame


# ---------------------------------------------------------------------------
# Page detection and rectification
# ---------------------------------------------------------------------------

@dataclass
class PageDetection:
    corners_px: np.ndarray                 # (4,2) image points mapped to page TL, TR, BR, BL
    method: str                            # "auto" or "manual"
    area_fraction: float
    aspect_ratio: float                    # detected long side / short side
    expected_aspect: float
    warnings: List[str] = field(default_factory=list)
    orientation_note: Optional[str] = None

    @property
    def aspect_error(self) -> float:
        return abs(self.aspect_ratio - self.expected_aspect) / self.expected_aspect

    def to_dict(self) -> Dict[str, Any]:
        return {"corners_px": np.round(self.corners_px, 2).tolist(), "method": self.method,
                "area_fraction": round(self.area_fraction, 4), "aspect_ratio": round(self.aspect_ratio, 4),
                "expected_aspect": round(self.expected_aspect, 4), "aspect_error": round(self.aspect_error, 4),
                "warnings": self.warnings, "orientation_note": self.orientation_note}


def _order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise starting at the image top-left."""
    pts = np.asarray(pts, dtype=float).reshape(4, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    pts = pts[np.argsort(ang)]                     # clockwise in image coords (y down)
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def _quad_aspect(q: np.ndarray) -> Tuple[float, float, float]:
    side = lambda a, b: float(np.linalg.norm(q[a] - q[b]))
    top, right, bottom, left = side(0, 1), side(1, 2), side(2, 3), side(3, 0)
    horiz = (top + bottom) / 2.0
    vert = (left + right) / 2.0
    return max(horiz, vert) / max(min(horiz, vert), 1e-9), horiz, vert


def detect_page(img: np.ndarray, page_w_mm: float, page_h_mm: float,
                manual_corners: Optional[Sequence[Sequence[float]]] = None) -> PageDetection:
    cv2 = _cv2()
    expected = max(page_w_mm, page_h_mm) / min(page_w_mm, page_h_mm)
    H, W = img.shape[:2]
    if manual_corners is not None:
        q = np.asarray(manual_corners, dtype=float).reshape(4, 2)
        aspect, _, _ = _quad_aspect(q)
        area = abs(cv2.contourArea(q.astype(np.float32))) / float(W * H)
        det = PageDetection(q, "manual", area, aspect, expected)
        if det.aspect_error > 0.15:
            det.warnings.append(f"Manual corners give aspect {aspect:.3f}, expected {expected:.3f}; check the corner order "
                                "(page top-left, top-right, bottom-right, bottom-left).")
        return det

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 1000.0 / max(H, W)
    small = cv2.resize(gray, (int(W * scale), int(H * scale))) if scale < 1 else gray.copy()
    s = scale if scale < 1 else 1.0
    blur = cv2.GaussianBlur(small, (7, 7), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise AnalysisError("Could not find the paper. Put the page on a darker surface or pass manual corners.")
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt) / float(small.shape[0] * small.shape[1])
    warnings: List[str] = []
    peri = cv2.arcLength(cnt, True)
    quad = None
    for eps in (0.01, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(cnt, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quad = approx.reshape(4, 2).astype(float)
            break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(float)
        warnings.append("Page outline was not a clean quadrilateral; used the minimum-area rectangle (less accurate). "
                        "Consider passing manual corners.")
    # refine corners to sub-pixel on the full-resolution image
    quad = quad / s
    q32 = quad.astype(np.float32).reshape(-1, 1, 2)
    try:
        cv2.cornerSubPix(gray, q32, (7, 7), (-1, -1),
                         (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))
        quad = q32.reshape(4, 2).astype(float)
    except cv2.error:
        pass
    quad = _order_quad(quad)
    aspect, _, _ = _quad_aspect(quad)
    det = PageDetection(quad, "auto", float(area), aspect, expected, warnings)
    if area < 0.10:
        det.warnings.append(f"Paper covers only {area * 100:.0f}% of the image; move the camera closer for accuracy.")
    if det.aspect_error > 0.12:
        det.warnings.append(f"Detected page aspect {aspect:.3f} differs from expected {expected:.3f} by "
                            f"{det.aspect_error * 100:.0f}% (strong perspective, curled paper or wrong page size).")
    return det


def _page_orderings(quad: np.ndarray, portrait: bool) -> List[Tuple[np.ndarray, str]]:
    """Corner orderings (page TL, TR, BR, BL) consistent with the page's aspect ratio."""
    _, horiz, vert = _quad_aspect(quad)
    image_portrait = vert >= horiz
    rolls = [0, 2] if image_portrait == portrait else [1, 3]
    names = {0: "upright", 1: "rotated 90 deg", 2: "rotated 180 deg", 3: "rotated 270 deg"}
    return [(np.roll(quad, -r, axis=0), names[r]) for r in rolls]


def rectify(img: np.ndarray, corners_page_order: np.ndarray, page_w_mm: float, page_h_mm: float,
            px_per_mm: float = PAGE_PX_PER_MM) -> Tuple[np.ndarray, np.ndarray]:
    cv2 = _cv2()
    dst = np.array([[0, 0], [page_w_mm * px_per_mm, 0], [page_w_mm * px_per_mm, page_h_mm * px_per_mm],
                    [0, page_h_mm * px_per_mm]], dtype=np.float32)
    Hm = cv2.getPerspectiveTransform(corners_page_order.astype(np.float32), dst)
    warped = cv2.warpPerspective(img, Hm, (int(round(page_w_mm * px_per_mm)), int(round(page_h_mm * px_per_mm))),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return warped, Hm


# ---------------------------------------------------------------------------
# Ink extraction
# ---------------------------------------------------------------------------

def _skeleton(mask: np.ndarray) -> np.ndarray:
    cv2 = _cv2()
    img = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(200):
        eroded = cv2.erode(img, kernel)
        opened = cv2.dilate(eroded, kernel)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return skel


def extract_ink(page_bgr: np.ndarray, px_per_mm: float = PAGE_PX_PER_MM, ink: str = "dark",
                margin_mm: float = 4.0, sensitivity: float = 12.0) -> Dict[str, Any]:
    cv2 = _cv2()
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    block = int(round(6.0 * px_per_mm)) | 1
    mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, sensitivity)
    if ink != "dark":
        hsv = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2HSV)
        ranges = {
            "blue": [((90, 60, 30), (135, 255, 255))],
            "red": [((0, 70, 40), (10, 255, 255)), ((160, 70, 40), (179, 255, 255))],
            "black": [((0, 0, 0), (179, 255, 110))],
        }
        if ink not in ranges:
            raise AnalysisError(f"Unknown ink colour '{ink}'. Use dark, black, blue or red.")
        cm = np.zeros_like(mask)
        for lo, hi in ranges[ink]:
            cm = cv2.bitwise_or(cm, cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))
        mask = cv2.bitwise_and(mask, cm)
    m = int(round(margin_mm * px_per_mm))
    mask[:m, :] = 0
    mask[-m:, :] = 0
    mask[:, :m] = 0
    mask[:, -m:] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    min_area = max(4, int(0.15 * px_per_mm * px_per_mm))
    page_area = mask.shape[0] * mask.shape[1]
    keep = np.zeros(n, bool)
    warnings: List[str] = []
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        if a > 0.05 * page_area:
            warnings.append("A very large dark region was ignored (shadow, fold or object on the page?).")
            continue
        keep[i] = True
    clean = (keep[labels] * 255).astype(np.uint8)
    skel = _skeleton(clean)
    ys, xs = np.nonzero(skel)
    pts = np.column_stack([xs / px_per_mm, ys / px_per_mm]).astype(float)
    ink_area_mm2 = float(np.count_nonzero(clean)) / (px_per_mm ** 2)
    skel_len_mm = float(len(pts)) / px_per_mm
    if len(pts) == 0:
        warnings.append("No ink detected on the page.")
    return {
        "points_mm": pts,
        "mask": clean,
        "ink_area_mm2": ink_area_mm2,
        "skeleton_length_mm_approx": skel_len_mm,
        "stroke_width_mm_approx": (ink_area_mm2 / skel_len_mm) if skel_len_mm > 0 else None,
        "components": int(keep.sum()),
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Registration (rigid, trimmed ICP)
# ---------------------------------------------------------------------------

def rigid_register(source: np.ndarray, target: np.ndarray, iterations: int = 40, trim_mm: float = 3.0,
                   max_points: int = 1500) -> Dict[str, Any]:
    """Find rotation R and translation t minimising |R*source + t - nearest(target)| (no scaling)."""
    src = np.asarray(source, dtype=float)
    tgt = np.asarray(target, dtype=float)
    if len(src) < 5 or len(tgt) < 5:
        return {"ok": False, "rotation_deg": 0.0, "translation_mm": [0.0, 0.0], "R": np.eye(2), "t": np.zeros(2)}
    rs = np.random.default_rng(0)
    if len(src) > max_points:
        src = src[rs.choice(len(src), max_points, replace=False)]
    if len(tgt) > 3 * max_points:
        tgt = tgt[rs.choice(len(tgt), 3 * max_points, replace=False)]
    R = np.eye(2)
    t = np.zeros(2)
    for it in range(iterations):
        cur = src @ R.T + t
        d2 = (cur ** 2).sum(1)[:, None] + (tgt ** 2).sum(1)[None, :] - 2 * cur @ tgt.T
        j = np.argmin(d2, axis=1)
        d = np.sqrt(np.maximum(d2[np.arange(len(cur)), j], 0))
        thr = np.inf if it < 3 else max(trim_mm, float(np.quantile(d, 0.8)))
        keep = d <= thr
        if keep.sum() < 5:
            break
        A, B = cur[keep], tgt[j[keep]]
        ca, cb = A.mean(0), B.mean(0)
        U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
        Ri = (U @ Vt).T
        if np.linalg.det(Ri) < 0:
            Vt[-1] *= -1
            Ri = (U @ Vt).T
        ti = cb - ca @ Ri.T
        R = Ri @ R
        t = t @ Ri.T + ti
    return {"ok": True, "rotation_deg": float(np.degrees(np.arctan2(R[1, 0], R[0, 0]))),
            "translation_mm": [float(t[0]), float(t[1])], "R": R, "t": t}


def apply_rigid(points: np.ndarray, reg: Dict[str, Any]) -> np.ndarray:
    return np.asarray(points, dtype=float) @ np.asarray(reg["R"]).T + np.asarray(reg["t"])


def symmetric_distance(a: np.ndarray, b: np.ndarray) -> Dict[str, Optional[float]]:
    if len(a) == 0 or len(b) == 0:
        return {"a_to_b_mean_mm": None, "b_to_a_mean_mm": None, "symmetric_mean_mm": None, "hausdorff_mm": None}
    dab = nearest_distances(a, b)
    dba = nearest_distances(b, a)
    return {"a_to_b_mean_mm": float(dab.mean()), "b_to_a_mean_mm": float(dba.mean()),
            "symmetric_mean_mm": float(0.5 * (dab.mean() + dba.mean())),
            "hausdorff_mm": float(max(dab.max(), dba.max()))}


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------

@dataclass
class AnalysisInputs:
    plan_path: str
    data_source: str
    executed_plan: bool = False            # user confirms the robot drew THIS exported plan
    photo_path: Optional[str] = None
    video_path: Optional[str] = None
    manual_corners: Optional[List[List[float]]] = None
    ink: str = "dark"
    sim_backend: str = "pybullet"
    targets_dir: str = "writing_targets"
    execution_mode: str = "svg_import"     # from the robot profile: svg_import | gcode | other
    marker_color: Optional[str] = None     # optional coloured marker on the pen holder
    notes: str = ""


def analyze_recording(inp: AnalysisInputs, out_dir: Union[str, Path] = "results/writing/real_to_sim") -> Dict[str, Any]:
    cv2 = _cv2()
    if inp.data_source not in VALID_SOURCES:
        raise AnalysisError(f"data_source must be one of {VALID_SOURCES}")
    if not inp.photo_path and not inp.video_path:
        raise AnalysisError("Provide a photo of the finished page, a video of the run, or both.")
    plan = PenPlan.load(inp.plan_path)
    target = get_target(plan.target_id, inp.targets_dir)
    layout = target.layout()
    W, H = plan.page_width_mm, plan.page_height_mm
    warnings: List[str] = []
    for path in (inp.photo_path, inp.video_path):
        if path and "synthetic" in Path(path).name.lower() and inp.data_source != "synthetic_fixture":
            warnings.append("File name contains 'synthetic'; data source forced to synthetic_fixture.")
            inp.data_source = "synthetic_fixture"

    # 1. images
    motion = None
    video_frame = None
    if inp.video_path:
        motion, video_frame = analyze_video_motion(inp.video_path)
        warnings += motion.notes
    if inp.photo_path:
        page_img = load_image(inp.photo_path)
        page_source = "photo"
    else:
        page_img = video_frame
        page_source = f"video frame {motion.page_frame_index} ({motion.page_frame_time_s:.2f} s)"

    # 2. page detection + orientation by content match
    det = detect_page(page_img, W, H, inp.manual_corners)
    warnings += det.warnings
    portrait = H >= W
    orderings = [(det.corners_px, "as supplied")] if det.method == "manual" else _page_orderings(det.corners_px, portrait)
    intended_pts = strokes_to_points(layout.strokes, 0.25)
    best = None
    for corners, name in orderings:
        warped, Hm = rectify(page_img, corners, W, H)
        ink = extract_ink(warped, PAGE_PX_PER_MM, inp.ink)
        pts = ink["points_mm"]
        score = float(nearest_distances(pts, intended_pts).mean()) if len(pts) else np.inf
        if best is None or score < best[0]:
            best = (score, corners, name, warped, Hm, ink)
    _, corners, orient_name, warped, Hm, ink = best
    det.corners_px = corners
    if det.method == "auto" and len(orderings) > 1:
        det.orientation_note = f"Page orientation chosen by best match to the intended path: {orient_name}."
    warnings += ink["warnings"]
    real_pts = ink["points_mm"]
    if len(real_pts) < 20:
        raise AnalysisError("Too little ink detected on the page. Check lighting, ink colour (--ink) and the corners.")

    # 3. simulation of the SAME plan
    sim = make_writing_simulator(inp.sim_backend)
    try:
        sim_exec = sim.execute(plan, plan.seed)
    finally:
        if hasattr(sim, "close"):
            sim.close()
    sim_pts = strokes_to_points(sim_exec.ink_strokes(), 0.25)
    intended_len = float(sum(polyline_length(s) for s in layout.strokes))
    sim_scores = score_execution(sim_exec, layout, intended_len)

    # 4. real metrics (raw page coordinates, then after rigid placement correction)
    real_te_raw = trajectory_error(real_pts, layout.strokes)
    real_acc_raw = writing_accuracy(real_pts, layout)
    reg = rigid_register(real_pts, intended_pts)
    real_aligned = apply_rigid(real_pts, reg)
    real_te_aligned = trajectory_error(real_aligned, layout.strokes)
    real_acc_aligned = writing_accuracy(real_aligned, layout)
    placement = {"rotation_deg": round(reg["rotation_deg"], 3),
                 "translation_mm": [round(v, 3) for v in reg["translation_mm"]],
                 "translation_norm_mm": round(float(np.hypot(*reg["translation_mm"])), 3)}
    if placement["translation_norm_mm"] > 15 or abs(placement["rotation_deg"]) > 8:
        warnings.append(f"Large placement correction ({placement}). Wrong target, wrong page orientation, or the "
                        "paper was far from the expected position.")
    real_vs_sim_raw = symmetric_distance(real_pts, sim_pts)
    real_vs_sim_aligned = symmetric_distance(real_aligned, sim_pts)
    masking = None
    if real_vs_sim_raw["symmetric_mean_mm"] and real_vs_sim_aligned["symmetric_mean_mm"]:
        if real_vs_sim_raw["symmetric_mean_mm"] > 1.5 * real_vs_sim_aligned["symmetric_mean_mm"] + 0.2:
            masking = ("Placement correction reduced the real-vs-sim distance from "
                       f"{real_vs_sim_raw['symmetric_mean_mm']:.2f} to {real_vs_sim_aligned['symmetric_mean_mm']:.2f} mm. "
                       "Most of the raw difference is where the paper/robot origin was, not stroke shape.")
            warnings.append(masking)

    real_eff = (ink["skeleton_length_mm_approx"] / intended_len) if intended_len > 0 else None

    # 5. timing
    timing = {"real_active_duration_s": motion.active_duration_s if motion else None,
              "sim_duration_s": round(sim_exec.duration_s, 4),
              "nominal_plan_duration_s": round(plan.nominal_duration_s(), 4),
              "comparable": bool(motion and motion.active_duration_s and inp.execution_mode == "gcode")}
    if motion and motion.active_duration_s and inp.execution_mode != "gcode":
        timing["note"] = ("Robot executed an SVG/vendor-app job that plans its own speeds; execution time is reported "
                          "but not compared with the simulation.")
    elif motion and motion.active_duration_s:
        timing["difference_s"] = round(motion.active_duration_s - sim_exec.duration_s, 4)
        timing["relative_difference"] = round(timing["difference_s"] / max(sim_exec.duration_s, 1e-9), 4)

    # 6. optional marker track
    marker = None
    if inp.video_path and inp.marker_color:
        marker = _track_marker_on_page(inp.video_path, Hm, inp.marker_color, layout)
        warnings += marker.get("warnings", [])

    # 7. evidence classification
    if inp.data_source == "robot_phone_recording" and inp.executed_plan:
        comparison_type = "matched_plan_execution"
    elif inp.data_source == "robot_phone_recording":
        comparison_type = "cross_controller_behavioural_comparison"
        warnings.append("Robot recording without --executed-plan: treated as cross-controller (robot may have used "
                        "its own font or settings).")
    else:
        comparison_type = "not_robot_evidence"

    report_id = uuid.uuid4().hex[:8]
    out = Path(out_dir) / f"analysis_{report_id}"
    out.mkdir(parents=True, exist_ok=True)
    overlay_path = _render_overlay(warped, layout, sim_exec.ink_strokes(), real_pts, out / "overlay_page.png",
                                   title=f"{plan.target_id} | {plan.policy_name} | real vs sim")
    aligned_path = _render_overlay(None, layout, sim_exec.ink_strokes(), real_aligned, out / "overlay_aligned.png",
                                   title="after placement correction")
    cv2.imwrite(str(out / "rectified_page.png"), warped)
    cv2.imwrite(str(out / "ink_mask.png"), ink["mask"])

    report = {
        "type": "writing_real_to_sim_analysis",
        "report_id": report_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_source": inp.data_source,
        "comparison_type": comparison_type,
        "executed_plan_declared": bool(inp.executed_plan),
        "execution_mode": inp.execution_mode,
        "plan": {"path": inp.plan_path, "plan_hash": plan.plan_hash, "policy_name": plan.policy_name,
                 "policy_version": plan.policy_version, "target_id": plan.target_id, "seed": plan.seed},
        "inputs": {"photo": inp.photo_path, "video": inp.video_path, "page_image_source": page_source,
                   "ink": inp.ink, "sim_backend": inp.sim_backend, "notes": inp.notes},
        "page_detection": det.to_dict(),
        "video_motion": motion.to_dict() if motion else None,
        "metrics": {
            "intended_vs_sim": {k: sim_scores.values[k] for k in (
                "trajectory_error_mean_mm", "trajectory_error_max_mm", "path_coverage", "writing_accuracy",
                "execution_time_s", "path_efficiency_deviation")},
            "intended_vs_real_raw": {
                "trajectory_error_mean_mm": real_te_raw["mean_mm"], "trajectory_error_max_mm": real_te_raw["max_mm"],
                "trajectory_error_p95_mm": real_te_raw["p95_mm"], "path_coverage": real_te_raw["coverage"],
                "writing_accuracy": real_acc_raw["accuracy"]},
            "intended_vs_real_aligned": {
                "trajectory_error_mean_mm": real_te_aligned["mean_mm"], "trajectory_error_max_mm": real_te_aligned["max_mm"],
                "trajectory_error_p95_mm": real_te_aligned["p95_mm"], "path_coverage": real_te_aligned["coverage"],
                "writing_accuracy": real_acc_aligned["accuracy"]},
            "real_vs_sim_raw": real_vs_sim_raw,
            "real_vs_sim_aligned": real_vs_sim_aligned,
            "real_path_efficiency_ratio_approx": real_eff,
            "real_ink": {"area_mm2": round(ink["ink_area_mm2"], 2), "components": ink["components"],
                         "stroke_width_mm_approx": ink["stroke_width_mm_approx"]},
            "timing": timing,
        },
        "placement_correction": placement,
        "character_results_real": real_acc_aligned["cells"],
        "sim_safety": sim_exec.safety,
        "marker_track": marker,
        "unmeasurable_from_video": UNMEASURABLE,
        "warnings": warnings,
        "files": {"overlay_page": overlay_path, "overlay_aligned": aligned_path,
                  "rectified_page": str(out / "rectified_page.png"), "ink_mask": str(out / "ink_mask.png")},
        "method_notes": [
            "Page-plane homography from the four paper corners: measurements are in mm on the paper.",
            "Ink skeleton points are compared with the intended path and with the simulated execution of the same plan.",
            "'aligned' values remove a rigid placement offset (rotation + translation, no scaling).",
            "Writing accuracy uses a template recogniser over the robot's own font (not general OCR).",
        ],
    }
    path = out / "report.json"
    report["saved_to"] = str(path)
    path.write_text(json.dumps(report, indent=2, default=_jd), encoding="utf-8")
    return report


def _jd(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o).__name__)


def _render_overlay(page_bgr, layout: Layout, sim_strokes, real_pts, path: Path, title: str) -> str:
    cv2 = _cv2()
    from .render import PX_PER_MM, blank_page
    if page_bgr is not None:
        img = cv2.resize(page_bgr, (int(layout.page_width_mm * PX_PER_MM), int(layout.page_height_mm * PX_PER_MM)))
        img = cv2.addWeighted(img, 0.45, np.full_like(img, 255), 0.55, 0)
    else:
        img = blank_page(layout.page_width_mm, layout.page_height_mm)
    draw_strokes(img, layout.strokes, GREY, 5)
    draw_strokes(img, sim_strokes, BLUE, 1)
    draw_points(img, real_pts, RED, 1)
    img = crop_to(img, layout.bbox, 10)
    img = add_legend(img, [("intended path", GREY), ("simulation (same plan)", BLUE), ("real ink (detected)", RED)], title)
    cv2.imwrite(str(path), img)
    return str(path)


def _track_marker_on_page(video_path: str, Hm: np.ndarray, color: str, layout: Layout) -> Dict[str, Any]:
    """Track a coloured marker on the pen holder and map it onto the page plane (mm)."""
    cv2 = _cv2()
    from robot_ci.vision.models import TrackerConfig, DetectionStatus
    from robot_ci.vision.tracker import MarkerTracker
    tracker = MarkerTracker(TrackerConfig(color_preset=color, min_contour_area=15))
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    ts, pix = [], []
    last = None
    i = 0
    total = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        total += 1
        obs = tracker.track_frame(frame, i, i / fps, last)
        if obs.detection_status == DetectionStatus.DETECTED and obs.image_coordinates is not None:
            last = obs.image_coordinates
            ts.append(i / fps)
            pix.append(obs.image_coordinates)
        i += 1
    cap.release()
    if len(pix) < 5:
        return {"available": False, "warnings": [f"Marker '{color}' detected in only {len(pix)} frames."]}
    p = np.asarray(pix, dtype=np.float64).reshape(-1, 1, 2)
    mm = cv2.perspectiveTransform(p, Hm).reshape(-1, 2) / PAGE_PX_PER_MM
    te = trajectory_error(mm, layout.strokes)
    return {
        "available": True, "color": color, "frames_detected": len(pix), "frames_total": total,
        "coverage": round(len(pix) / max(total, 1), 4),
        "path_mm": np.round(mm, 3).tolist(), "t_s": np.round(ts, 4).tolist(),
        "trajectory_error_vs_intended_mm": te["mean_mm"],
        "warnings": ["Marker is above the paper: page-plane mapping has parallax error unless the camera is directly "
                     "overhead. Treat the marker path as approximate motion timing, not ink accuracy."],
    }
