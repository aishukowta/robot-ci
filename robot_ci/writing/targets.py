"""Writing targets: what the robot must write, and where on the page.

A target is a YAML file in ``writing_targets/``. It is laid out into page
coordinates (millimetres, origin = top-left corner of the paper, x right,
y down). The layout also records one "cell" per character (or one for the
whole pattern), which the recogniser uses to measure writing accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import yaml

from .font import glyph, pattern, PATTERNS

PAGE_SIZES_MM = {"A4": (210.0, 297.0), "A5": (148.0, 210.0), "LETTER": (215.9, 279.4)}


@dataclass
class Cell:
    """Region of the page that should contain one recognisable item."""
    label: str                 # character, or pattern name
    kind: str                  # "character" or "pattern"
    x_mm: float
    y_mm: float
    width_mm: float
    height_mm: float           # = character height (glyph unit) or pattern size

    def to_dict(self) -> Dict[str, Any]:
        return dict(label=self.label, kind=self.kind, x_mm=self.x_mm, y_mm=self.y_mm,
                    width_mm=self.width_mm, height_mm=self.height_mm)


@dataclass
class Layout:
    target_id: str
    strokes: List[np.ndarray]          # page mm, pen down along each
    cells: List[Cell]
    page_width_mm: float
    page_height_mm: float

    @property
    def bbox(self) -> List[float]:
        pts = np.vstack(self.strokes)
        return [float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 0].max()), float(pts[:, 1].max())]


@dataclass
class WritingTarget:
    target_id: str
    kind: str                           # "text" or "pattern"
    content: str                        # text to write, or pattern name
    origin_mm: List[float]              # top-left of the first character / pattern box
    height_mm: float                    # character cap height or pattern size
    char_spacing: float = 0.25          # gap between characters, in glyph units
    word_spacing: float = 0.6
    page: str = "A4"
    seed: int = 7
    description: str = ""
    tags: List[str] = field(default_factory=list)

    @property
    def page_size_mm(self):
        return PAGE_SIZES_MM[self.page.upper()]

    def validate(self) -> None:
        if self.kind not in ("text", "pattern"):
            raise ValueError(f"Target '{self.target_id}': kind must be 'text' or 'pattern'.")
        if self.page.upper() not in PAGE_SIZES_MM:
            raise ValueError(f"Target '{self.target_id}': unknown page '{self.page}'. Use {list(PAGE_SIZES_MM)}.")
        if not (2.0 <= float(self.height_mm) <= 150.0):
            raise ValueError(f"Target '{self.target_id}': height_mm must be between 2 and 150.")
        if self.kind == "pattern" and self.content not in PATTERNS:
            raise ValueError(f"Target '{self.target_id}': unknown pattern '{self.content}'.")
        if self.kind == "text":
            for ch in self.content:
                glyph(ch)  # raises a clear error for unsupported characters
        lay = self.layout()
        x0, y0, x1, y1 = lay.bbox
        w, h = self.page_size_mm
        margin = 5.0
        if x0 < margin or y0 < margin or x1 > w - margin or y1 > h - margin:
            raise ValueError(
                f"Target '{self.target_id}' does not fit on the {self.page} page with a {margin} mm margin "
                f"(bbox {[round(v, 1) for v in lay.bbox]} mm)."
            )

    def layout(self) -> Layout:
        h = float(self.height_mm)
        ox, oy = float(self.origin_mm[0]), float(self.origin_mm[1])
        strokes: List[np.ndarray] = []
        cells: List[Cell] = []
        if self.kind == "pattern":
            for s in pattern(self.content):
                strokes.append(np.column_stack([ox + s[:, 0] * h, oy + s[:, 1] * h]))
            cells.append(Cell(self.content, "pattern", ox, oy, h, h))
        else:
            cursor = 0.0
            for ch in self.content:
                w, glyph_strokes = glyph(ch)
                if ch == " ":
                    cursor += self.word_spacing
                    continue
                for s in glyph_strokes:
                    strokes.append(np.column_stack([ox + (cursor + s[:, 0]) * h, oy + s[:, 1] * h]))
                if ch.upper() != "-":
                    cells.append(Cell(ch.upper(), "character", ox + cursor * h, oy, w * h, h))
                cursor += w + self.char_spacing
        pw, ph = self.page_size_mm
        return Layout(self.target_id, strokes, cells, pw, ph)

    def to_dict(self) -> Dict[str, Any]:
        return dict(target_id=self.target_id, kind=self.kind, content=self.content,
                    origin_mm=list(self.origin_mm), height_mm=self.height_mm,
                    char_spacing=self.char_spacing, word_spacing=self.word_spacing,
                    page=self.page, seed=self.seed, description=self.description, tags=list(self.tags))

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WritingTarget":
        t = cls(
            target_id=str(d["target_id"]),
            kind=str(d["kind"]),
            content=str(d["content"]),
            origin_mm=[float(v) for v in d["origin_mm"]],
            height_mm=float(d["height_mm"]),
            char_spacing=float(d.get("char_spacing", 0.25)),
            word_spacing=float(d.get("word_spacing", 0.6)),
            page=str(d.get("page", "A4")),
            seed=int(d.get("seed", 7)),
            description=str(d.get("description", "")),
            tags=list(d.get("tags", [])),
        )
        t.validate()
        return t


def load_target(path: Union[str, Path]) -> WritingTarget:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: target file must contain a mapping.")
    return WritingTarget.from_dict(data)


def load_targets(directory: Union[str, Path] = "writing_targets", ids: Optional[List[str]] = None) -> List[WritingTarget]:
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"Writing targets directory not found: {d}")
    targets = sorted((load_target(p) for p in d.glob("*.y*ml")), key=lambda t: t.target_id)
    if ids:
        wanted = set(ids)
        missing = wanted - {t.target_id for t in targets}
        if missing:
            raise ValueError(f"Unknown writing targets {sorted(missing)}. Available: {[t.target_id for t in targets]}")
        targets = [t for t in targets if t.target_id in wanted]
    if not targets:
        raise ValueError(f"No writing targets found in {d}")
    return targets


def get_target(target_id: str, directory: Union[str, Path] = "writing_targets") -> WritingTarget:
    return load_targets(directory, [target_id])[0]
