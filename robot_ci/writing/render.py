"""Render pages (intended path, simulated ink, real ink) to PNG images."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

PX_PER_MM = 6.0

# BGR colours
GREY = (170, 170, 170)
BLUE = (200, 90, 20)
RED = (40, 40, 220)
GREEN = (60, 160, 40)
BLACK = (20, 20, 20)


def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Rendering requires opencv-python ('pip install -e .[vision]').") from exc
    return cv2


def blank_page(width_mm: float, height_mm: float, px_per_mm: float = PX_PER_MM) -> np.ndarray:
    return np.full((int(round(height_mm * px_per_mm)), int(round(width_mm * px_per_mm)), 3), 255, np.uint8)


def draw_strokes(img: np.ndarray, strokes: Sequence[np.ndarray], color, thickness: int = 2,
                 px_per_mm: float = PX_PER_MM) -> np.ndarray:
    cv2 = _cv2()
    for s in strokes:
        s = np.asarray(s, dtype=float)
        if len(s) < 2:
            continue
        pts = np.round(s * px_per_mm).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts], False, color, thickness, lineType=cv2.LINE_AA)
    return img


def draw_points(img: np.ndarray, points: np.ndarray, color, radius: int = 1, px_per_mm: float = PX_PER_MM):
    cv2 = _cv2()
    for x, y in np.asarray(points, dtype=float):
        cv2.circle(img, (int(round(x * px_per_mm)), int(round(y * px_per_mm))), radius, color, -1)
    return img


def crop_to(img: np.ndarray, bbox_mm: Sequence[float], margin_mm: float = 8.0, px_per_mm: float = PX_PER_MM):
    x0, y0, x1, y1 = bbox_mm
    h, w = img.shape[:2]
    a = max(0, int((x0 - margin_mm) * px_per_mm))
    b = max(0, int((y0 - margin_mm) * px_per_mm))
    c = min(w, int((x1 + margin_mm) * px_per_mm))
    d = min(h, int((y1 + margin_mm) * px_per_mm))
    return img[b:d, a:c].copy()


def add_legend(img: np.ndarray, items: List[Tuple[str, tuple]], title: Optional[str] = None) -> np.ndarray:
    cv2 = _cv2()
    pad = 24 * (len(items) + (1 if title else 0)) + 10
    out = np.full((img.shape[0] + pad, max(img.shape[1], 360), 3), 255, np.uint8)
    out[:img.shape[0], :img.shape[1]] = img
    y = img.shape[0] + 20
    if title:
        cv2.putText(out, title, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
        y += 24
    for label, color in items:
        cv2.line(out, (10, y - 5), (40, y - 5), color, 3)
        cv2.putText(out, label, (48, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1, cv2.LINE_AA)
        y += 24
    return out


def render_comparison(layout, layers: List[Tuple[str, Sequence[np.ndarray], tuple]], path: Union[str, Path],
                      title: Optional[str] = None, crop: bool = True) -> str:
    """layers: (label, strokes, colour). The intended target is always drawn first in grey."""
    cv2 = _cv2()
    img = blank_page(layout.page_width_mm, layout.page_height_mm)
    draw_strokes(img, layout.strokes, GREY, 5)
    for _, strokes, color in layers:
        draw_strokes(img, strokes, color, 1)
    if crop:
        img = crop_to(img, layout.bbox)
    img = add_legend(img, [("intended path", GREY)] + [(lbl, c) for lbl, _, c in layers], title)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), img)
    return str(p)
