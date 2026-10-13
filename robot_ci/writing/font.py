"""Single-stroke font and geometric patterns for the handwriting robot.

Glyph coordinate system (a "glyph unit" = cap height):
    x grows to the right, y grows DOWN (0 = top of capital, 1 = baseline).
    This matches image / page coordinates, so a straightened photo of the page
    can be compared directly without any axis flipping.

Every glyph is a list of strokes; every stroke is a polyline (N x 2 array).
The pen is lowered at the start of a stroke and lifted at its end.

The same definitions are used for (1) planning what the robot writes and
(2) the template recogniser that measures writing accuracy.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

Stroke = np.ndarray            # (N, 2)
GlyphStrokes = List[Stroke]


def _pl(*pts: Tuple[float, float]) -> Stroke:
    return np.asarray(pts, dtype=float)


def _arc(cx: float, cy: float, rx: float, ry: float, a0: float, a1: float, n: int = 0) -> Stroke:
    """Elliptical arc from angle a0 to a1 (degrees). With y down, +90 deg is the bottom."""
    if n <= 0:
        n = max(8, int(abs(a1 - a0) / 7.5) + 1)
    a = np.radians(np.linspace(a0, a1, n))
    return np.column_stack([cx + rx * np.cos(a), cy + ry * np.sin(a)])


def _cat(*parts: Stroke) -> Stroke:
    """Concatenate polyline pieces into one continuous stroke (drops duplicate joints)."""
    out = [parts[0]]
    for p in parts[1:]:
        if np.linalg.norm(out[-1][-1] - p[0]) < 1e-9:
            out.append(p[1:])
        else:
            out.append(p)
    return np.vstack(out)


# ---------------------------------------------------------------------------
# Glyph definitions: (advance width in glyph units, strokes)
# ---------------------------------------------------------------------------

def _build_glyphs() -> Dict[str, Tuple[float, GlyphStrokes]]:
    g: Dict[str, Tuple[float, GlyphStrokes]] = {}

    g["A"] = (0.6, [_pl((0, 1), (0.3, 0), (0.6, 1)), _pl((0.11, 0.63), (0.49, 0.63))])
    g["B"] = (0.62, [
        _cat(_pl((0, 1), (0, 0), (0.36, 0)), _arc(0.36, 0.25, 0.22, 0.25, -90, 90), _pl((0.36, 0.5), (0, 0.5))),
        _cat(_pl((0, 0.5), (0.38, 0.5)), _arc(0.38, 0.75, 0.24, 0.25, -90, 90), _pl((0.38, 1.0), (0, 1))),
    ])
    g["C"] = (0.62, [_arc(0.35, 0.5, 0.35, 0.5, -40, -320)])
    g["D"] = (0.62, [_cat(_pl((0, 1), (0, 0), (0.25, 0)), _arc(0.25, 0.5, 0.37, 0.5, -90, 90), _pl((0.25, 1.0), (0, 1)))])
    g["E"] = (0.55, [_pl((0.55, 0), (0, 0), (0, 1), (0.55, 1)), _pl((0, 0.5), (0.42, 0.5))])
    g["F"] = (0.55, [_pl((0.55, 0), (0, 0), (0, 1)), _pl((0, 0.5), (0.42, 0.5))])
    g["G"] = (0.7, [_cat(_arc(0.35, 0.5, 0.35, 0.5, -40, -360), _pl((0.70, 0.5), (0.42, 0.5)))])
    g["H"] = (0.55, [_pl((0, 0), (0, 1)), _pl((0.55, 0), (0.55, 1)), _pl((0, 0.5), (0.55, 0.5))])
    g["I"] = (0.3, [_pl((0.15, 0), (0.15, 1)), _pl((0, 0), (0.3, 0)), _pl((0, 1), (0.3, 1))])
    g["J"] = (0.5, [_cat(_pl((0.5, 0), (0.5, 0.75)), _arc(0.25, 0.75, 0.25, 0.25, 0, 180))])
    g["K"] = (0.55, [_pl((0, 0), (0, 1)), _pl((0.55, 0), (0, 0.6)), _pl((0.18, 0.45), (0.55, 1))])
    g["L"] = (0.5, [_pl((0, 0), (0, 1), (0.5, 1))])
    g["M"] = (0.7, [_pl((0, 1), (0, 0), (0.35, 0.6), (0.7, 0), (0.7, 1))])
    g["N"] = (0.55, [_pl((0, 1), (0, 0), (0.55, 1), (0.55, 0))])
    g["O"] = (0.75, [_arc(0.375, 0.5, 0.375, 0.5, -90, 270)])
    g["P"] = (0.58, [_cat(_pl((0, 1), (0, 0), (0.33, 0)), _arc(0.33, 0.25, 0.25, 0.25, -90, 90), _pl((0.33, 0.5), (0, 0.5)))])
    g["Q"] = (0.78, [_arc(0.375, 0.5, 0.375, 0.5, -90, 270), _pl((0.45, 0.7), (0.78, 1.05))])
    g["R"] = (0.58, [
        _cat(_pl((0, 1), (0, 0), (0.33, 0)), _arc(0.33, 0.25, 0.25, 0.25, -90, 90), _pl((0.33, 0.5), (0, 0.5))),
        _pl((0.25, 0.5), (0.58, 1)),
    ])
    g["S"] = (0.6, [_cat(_arc(0.3, 0.25, 0.3, 0.25, -20, -270), _arc(0.3, 0.75, 0.3, 0.25, -90, 160))])
    g["T"] = (0.6, [_pl((0, 0), (0.6, 0)), _pl((0.3, 0), (0.3, 1))])
    g["U"] = (0.56, [_cat(_pl((0, 0), (0, 0.7)), _arc(0.28, 0.7, 0.28, 0.3, 180, 0), _pl((0.56, 0.7), (0.56, 0)))])
    g["V"] = (0.6, [_pl((0, 0), (0.3, 1), (0.6, 0))])
    g["W"] = (0.8, [_pl((0, 0), (0.2, 1), (0.4, 0.35), (0.6, 1), (0.8, 0))])
    g["X"] = (0.55, [_pl((0, 0), (0.55, 1)), _pl((0.55, 0), (0, 1))])
    g["Y"] = (0.6, [_pl((0, 0), (0.3, 0.5), (0.6, 0)), _pl((0.3, 0.5), (0.3, 1))])
    g["Z"] = (0.55, [_pl((0, 0), (0.55, 0), (0, 1), (0.55, 1))])

    g["0"] = (0.5, [_arc(0.25, 0.5, 0.25, 0.5, -90, 270), _pl((0.42, 0.12), (0.08, 0.88))])
    g["1"] = (0.45, [_pl((0.05, 0.2), (0.25, 0), (0.25, 1)), _pl((0.05, 1), (0.45, 1))])
    g["2"] = (0.5, [_cat(_arc(0.25, 0.27, 0.25, 0.27, -160, 20), _pl((0.485, 0.36), (0, 1), (0.5, 1)))])
    g["3"] = (0.5, [_cat(_arc(0.25, 0.25, 0.24, 0.25, -150, 90), _arc(0.25, 0.75, 0.26, 0.25, -90, 150))])
    g["4"] = (0.55, [_pl((0.4, 1), (0.4, 0), (0, 0.7), (0.55, 0.7))])
    g["5"] = (0.52, [_cat(_pl((0.5, 0), (0.05, 0), (0.07, 0.45)), _arc(0.25, 0.7, 0.27, 0.3, -125, 140))])
    g["6"] = (0.52, [_cat(_arc(0.45, 0.65, 0.43, 0.65, -90, -180), _arc(0.27, 0.72, 0.25, 0.28, 180, -180))])
    g["7"] = (0.5, [_pl((0, 0), (0.5, 0), (0.15, 1))])
    g["8"] = (0.5, [_arc(0.25, 0.25, 0.21, 0.25, 90, 450), _arc(0.25, 0.73, 0.25, 0.27, -90, 270)])
    g["9"] = (0.5, [_cat(_arc(0.25, 0.28, 0.25, 0.28, 0, 360), _pl((0.5, 0.28), (0.42, 1)))])

    g["-"] = (0.45, [_pl((0.05, 0.55), (0.4, 0.55))])
    g[" "] = (0.5, [])
    return g


GLYPHS: Dict[str, Tuple[float, GlyphStrokes]] = _build_glyphs()
SUPPORTED_CHARACTERS = "".join(sorted(k for k in GLYPHS if k != " "))
RECOGNIZABLE_CHARACTERS = [k for k in GLYPHS if k not in (" ", "-")]


# ---------------------------------------------------------------------------
# Geometric patterns, defined in a unit box (0..1, y down)
# ---------------------------------------------------------------------------

def _star(points: int = 5) -> Stroke:
    outer = []
    for k in range(points + 1):
        a = -90 + k * (720.0 / points)  # pentagram: skip every other vertex
        outer.append((0.5 + 0.5 * math.cos(math.radians(a)), 0.5 + 0.5 * math.sin(math.radians(a))))
    return np.asarray(outer)


def _spiral(turns: float = 3.0, n: int = 400) -> Stroke:
    t = np.linspace(0, 1, n)
    r = 0.04 + 0.46 * t
    a = 2 * np.pi * turns * t
    return np.column_stack([0.5 + r * np.cos(a), 0.5 + r * np.sin(a)])


def _sine(periods: float = 2.0, n: int = 200) -> Stroke:
    x = np.linspace(0, 1, n)
    return np.column_stack([x, 0.5 - 0.35 * np.sin(2 * np.pi * periods * x)])


PATTERNS: Dict[str, GlyphStrokes] = {
    "square": [_pl((0, 0), (1, 0), (1, 1), (0, 1), (0, 0))],
    "triangle": [_pl((0.5, 0), (1, 1), (0, 1), (0.5, 0))],
    "circle": [_arc(0.5, 0.5, 0.5, 0.5, 0, 360, 145)],
    "star": [_star(5)],
    "spiral": [_spiral()],
    "zigzag": [_pl(*[(i / 8.0, 0.15 if i % 2 == 0 else 0.85) for i in range(9)])],
    "sine_wave": [_sine()],
    "cross_grid": [_pl((0, 0), (1, 0), (1, 1), (0, 1), (0, 0)), _pl((0.5, 0), (0.5, 1)), _pl((0, 0.5), (1, 0.5))],
}


def glyph(char: str) -> Tuple[float, GlyphStrokes]:
    """Return (advance width, strokes) for a character, case-insensitive."""
    key = char.upper()
    if key not in GLYPHS:
        raise ValueError(
            f"Character '{char}' is not in the robot font. Supported: {SUPPORTED_CHARACTERS!r} and space."
        )
    w, strokes = GLYPHS[key]
    return w, [s.copy() for s in strokes]


def pattern(name: str) -> GlyphStrokes:
    if name not in PATTERNS:
        raise ValueError(f"Unknown pattern '{name}'. Available: {sorted(PATTERNS)}")
    return [s.copy() for s in PATTERNS[name]]


def densify(stroke: Stroke, spacing: float) -> Stroke:
    """Resample a polyline so consecutive points are at most `spacing` apart (keeps corners)."""
    stroke = np.asarray(stroke, dtype=float)
    if len(stroke) < 2:
        return stroke.copy()
    out = [stroke[0]]
    for a, b in zip(stroke[:-1], stroke[1:]):
        seg = float(np.linalg.norm(b - a))
        n = max(1, int(math.ceil(seg / spacing)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return np.asarray(out)


def polyline_length(stroke: Stroke) -> float:
    stroke = np.asarray(stroke, dtype=float)
    if len(stroke) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(stroke, axis=0), axis=1)))
