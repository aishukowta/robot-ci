"""Color-marker and high-contrast end-effector visual tracking engine.

Provides modular visual segmentation for end-effector markers in phone videos,
supporting HSV color thresholding, red hue wrap-around, circularity checking,
spatial windowing, and manual ROI constraints.
"""

from typing import Tuple, Optional, Dict, Any, List
import math
import numpy as np

from .models import (
    DetectionStatus,
    VideoFrameObservation,
    TrackerConfig,
    OpenCVUnavailableError,
)

# Optional OpenCV import checking
_CV2_AVAILABLE: Optional[bool] = None


def is_cv2_available() -> bool:
    """Check whether opencv-python (cv2) is available at runtime."""
    global _CV2_AVAILABLE
    if _CV2_AVAILABLE is None:
        try:
            import cv2  # noqa: F401
            _CV2_AVAILABLE = True
        except ImportError:
            _CV2_AVAILABLE = False
    return _CV2_AVAILABLE


def get_cv2():
    """Return cv2 module or raise OpenCVUnavailableError."""
    if not is_cv2_available():
        raise OpenCVUnavailableError(
            "Video tracking requires the 'opencv-python' package. "
            "Install with 'pip install opencv-python' or 'pip install -e .[vision]'."
        )
    import cv2
    return cv2


# Standard HSV Presets (OpenCV H range: 0..179, S: 0..255, V: 0..255)
COLOR_PRESETS: Dict[str, Dict[str, Any]] = {
    "red": {
        "lower1": (0, 100, 100),
        "upper1": (10, 255, 255),
        "lower2": (160, 100, 100),
        "upper2": (179, 255, 255),
    },
    "green": {
        "lower1": (35, 80, 80),
        "upper1": (85, 255, 255),
        "lower2": None,
        "upper2": None,
    },
    "blue": {
        "lower1": (95, 100, 100),
        "upper1": (135, 255, 255),
        "lower2": None,
        "upper2": None,
    },
    "yellow": {
        "lower1": (20, 100, 100),
        "upper1": (35, 255, 255),
        "lower2": None,
        "upper2": None,
    },
    "orange": {
        "lower1": (10, 120, 120),
        "upper1": (22, 255, 255),
        "lower2": None,
        "upper2": None,
    },
    "cyan": {
        "lower1": (80, 100, 100),
        "upper1": (95, 255, 255),
        "lower2": None,
        "upper2": None,
    },
    "magenta": {
        "lower1": (140, 100, 100),
        "upper1": (165, 255, 255),
        "lower2": None,
        "upper2": None,
    },
}


class MarkerTracker:
    """Configurable color and high-contrast end-effector marker tracker."""

    def __init__(self, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        self._cv2 = get_cv2()
        self._setup_bounds()

    def _setup_bounds(self) -> None:
        """Establish lower/upper HSV boundaries based on presets or explicit config."""
        preset_name = (self.config.color_preset or "").lower()

        if self.config.hsv_lower is not None and self.config.hsv_upper is not None:
            self.lower1 = np.array(self.config.hsv_lower, dtype=np.uint8)
            self.upper1 = np.array(self.config.hsv_upper, dtype=np.uint8)
            self.lower2 = np.array(self.config.hsv_lower_2, dtype=np.uint8) if self.config.hsv_lower_2 else None
            self.upper2 = np.array(self.config.hsv_upper_2, dtype=np.uint8) if self.config.hsv_upper_2 else None
        elif preset_name in COLOR_PRESETS:
            preset = COLOR_PRESETS[preset_name]
            self.lower1 = np.array(preset["lower1"], dtype=np.uint8)
            self.upper1 = np.array(preset["upper1"], dtype=np.uint8)
            self.lower2 = np.array(preset["lower2"], dtype=np.uint8) if preset["lower2"] else None
            self.upper2 = np.array(preset["upper2"], dtype=np.uint8) if preset["upper2"] else None
        else:
            # Default to red marker if unconfigured
            preset = COLOR_PRESETS["red"]
            self.lower1 = np.array(preset["lower1"], dtype=np.uint8)
            self.upper1 = np.array(preset["upper1"], dtype=np.uint8)
            self.lower2 = np.array(preset["lower2"], dtype=np.uint8)
            self.upper2 = np.array(preset["upper2"], dtype=np.uint8)

    def track_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_seconds: float,
        last_known_pos: Optional[Tuple[float, float]] = None,
    ) -> VideoFrameObservation:
        """Segment marker in a single video frame and extract 2D image coordinates.
        
        Args:
            frame: BGR numpy image frame from cv2.VideoCapture.
            frame_index: Current 0-based frame sequence number.
            timestamp_seconds: Estimated or exact timestamp in video timeline.
            last_known_pos: (u, v) pixel coordinates from previous frame for temporal gating.
        
        Returns:
            VideoFrameObservation with 2D image coordinates, normalized coordinates, and confidence.
        """
        cv2 = self._cv2

        if frame is None or frame.size == 0:
            return VideoFrameObservation(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                detection_status=DetectionStatus.INVALID_FRAME,
                confidence=0.0,
            )

        h, w = frame.shape[:2]

        # 1. Apply manual Region of Interest (ROI) if configured
        roi_offset_x, roi_offset_y = 0, 0
        search_image = frame
        if self.config.roi is not None:
            rx, ry, rw, rh = self.config.roi
            rx = max(0, min(w - 1, int(rx)))
            ry = max(0, min(h - 1, int(ry)))
            rw = max(1, min(w - rx, int(rw)))
            rh = max(1, min(h - ry, int(rh)))
            search_image = frame[ry : ry + rh, rx : rx + rw]
            roi_offset_x, roi_offset_y = rx, ry

        # 2. Convert to HSV color space
        hsv = cv2.cvtColor(search_image, cv2.COLOR_BGR2HSV)

        # 3. Create threshold mask
        mask = cv2.inRange(hsv, self.lower1, self.upper1)
        if self.lower2 is not None and self.upper2 is not None:
            mask2 = cv2.inRange(hsv, self.lower2, self.upper2)
            mask = cv2.bitwise_or(mask, mask2)

        # 4. Morphological noise cleanup (opening removes dust, closing fills holes)
        if self.config.apply_morphology:
            k_size = max(1, int(self.config.morph_kernel_size))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 5. Extract contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return VideoFrameObservation(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                detection_status=DetectionStatus.LOST,
                confidence=0.0,
                metadata={"reason": "No contours found in threshold mask"},
            )

        # 6. Filter and score candidate contours
        valid_candidates = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.config.min_contour_area:
                continue
            if self.config.max_contour_area is not None and area > self.config.max_contour_area:
                continue

            # Compute centroid moments
            m = cv2.moments(cnt)
            if m["m00"] <= 0:
                continue

            cx = (m["m10"] / m["m00"]) + roi_offset_x
            cy = (m["m01"] / m["m00"]) + roi_offset_y

            # Compute circularity: 4 * pi * Area / Perimeter^2
            perimeter = float(cv2.arcLength(cnt, True))
            circularity = (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0
            circularity = max(0.0, min(1.0, circularity))

            if self.config.min_circularity > 0 and circularity < self.config.min_circularity:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            bbox = (bx + roi_offset_x, by + roi_offset_y, bw, bh)

            # Spatial consistency penalty if previous position is known
            dist_to_prev = 0.0
            if last_known_pos is not None:
                dist_to_prev = math.hypot(cx - last_known_pos[0], cy - last_known_pos[1])

                if self.config.max_jump_pixels is not None and dist_to_prev > self.config.max_jump_pixels:
                    continue
                if self.config.search_window_radius is not None and dist_to_prev > self.config.search_window_radius:
                    continue

            # Confidence calculation: blend circularity and area consistency
            confidence = 0.6 * circularity + 0.4 * min(1.0, area / 100.0)
            if last_known_pos is not None:
                proximity_score = max(0.0, 1.0 - (dist_to_prev / 200.0))
                confidence = 0.7 * confidence + 0.3 * proximity_score

            valid_candidates.append({
                "centroid": (float(cx), float(cy)),
                "area": area,
                "circularity": circularity,
                "bbox": bbox,
                "confidence": max(0.01, min(1.0, confidence)),
                "dist_to_prev": dist_to_prev,
            })

        if not valid_candidates:
            return VideoFrameObservation(
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                detection_status=DetectionStatus.LOST,
                confidence=0.0,
                metadata={"reason": "Contours filtered out by area, circularity, or distance thresholds"},
            )

        # Select candidate: if last position is known, prefer closest; otherwise prefer largest confidence
        if last_known_pos is not None:
            best = min(valid_candidates, key=lambda c: c["dist_to_prev"])
        else:
            best = max(valid_candidates, key=lambda c: c["confidence"])

        u, v = best["centroid"]
        norm_u = u / float(w)
        norm_v = v / float(h)

        status = DetectionStatus.DETECTED if best["confidence"] >= 0.25 else DetectionStatus.LOW_CONFIDENCE

        return VideoFrameObservation(
            frame_index=frame_index,
            timestamp_seconds=timestamp_seconds,
            detection_status=status,
            image_coordinates=(u, v),
            normalized_coordinates=(norm_u, norm_v),
            confidence=best["confidence"],
            bounding_box=best["bbox"],
            area_pixels=best["area"],
            metadata={
                "circularity": round(best["circularity"], 4),
                "frame_size": [w, h],
            },
        )
