"""Data models and schemas for video-to-trajectory extraction.

Defines:
- VideoFrameObservation: per-frame detection result with 2D image coordinates and confidence.
- VideoMetadata: resolution, frame rate, duration, and container info.
- TrackerConfig: color ranges, thresholds, ROI, and fallback settings.
- VideoTrajectoryResult: full extracted trajectory container with export and telemetry adapters.
- Explicit uncalibrated camera disclaimers to prevent mistaking pixel coordinates for meters.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timezone
import json
import csv


UNCALIBRATED_CAMERA_DISCLAIMER = (
    "Visual tracking measurements are extracted in 2D image coordinates (pixel [u, v] "
    "and normalized screen-space [u_norm, v_norm]) from an uncalibrated camera recording. "
    "These measurements DO NOT represent metric 3D Cartesian coordinates (meters), joint angles, "
    "contact forces, or physical robot safety, and are subject to perspective distortion, "
    "lens distortion, monocular parallax, camera motion, and visual occlusion."
)


class DetectionStatus(str, Enum):
    """Detection outcome for an individual video frame."""
    DETECTED = "DETECTED"
    LOST = "LOST"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
    INVALID_FRAME = "INVALID_FRAME"


class CoordinateSpace(str, Enum):
    """Explicit coordinate space classification."""
    IMAGE_PIXELS = "IMAGE_PIXELS"          # [u, v] in pixel space (0..W, 0..H)
    NORMALIZED_IMAGE = "NORMALIZED_IMAGE"  # [u_norm, v_norm] in [0.0, 1.0] relative to frame dimensions


class VisionError(Exception):
    """Base exception for video analysis and tracking errors."""
    pass


class OpenCVUnavailableError(VisionError):
    """Raised when OpenCV (cv2) is not installed in the environment."""
    pass


class InvalidVideoError(VisionError):
    """Raised when a video file cannot be opened or contains zero decodable frames."""
    pass


class UnsupportedVideoFormatError(VisionError):
    """Raised when a video format or container extension is unsupported."""
    pass


class TrackingError(VisionError):
    """Raised when marker tracking encounters unrecoverable configuration errors."""
    pass


@dataclass
class VideoFrameObservation:
    """A single observation extracted from a video frame.
    
    Coordinates are strictly 2D image-space coordinates. Never meters.
    """
    frame_index: int
    timestamp_seconds: float
    detection_status: DetectionStatus
    image_coordinates: Optional[Tuple[float, float]] = None      # (u, v) in pixels
    normalized_coordinates: Optional[Tuple[float, float]] = None # (u_norm, v_norm) in [0.0, 1.0]
    confidence: float = 0.0                                     # [0.0, 1.0]
    bounding_box: Optional[Tuple[int, int, int, int]] = None    # (x, y, w, h) in pixels
    area_pixels: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_seconds": round(self.timestamp_seconds, 6),
            "detection_status": self.detection_status.value if isinstance(self.detection_status, DetectionStatus) else str(self.detection_status),
            "image_coordinates": [round(c, 2) for c in self.image_coordinates] if self.image_coordinates else None,
            "normalized_coordinates": [round(c, 5) for c in self.normalized_coordinates] if self.normalized_coordinates else None,
            "confidence": round(self.confidence, 4),
            "bounding_box": list(self.bounding_box) if self.bounding_box else None,
            "area_pixels": round(self.area_pixels, 2),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoFrameObservation":
        status_val = data.get("detection_status", "LOST")
        try:
            status = DetectionStatus(status_val)
        except ValueError:
            status = DetectionStatus.LOST

        img_coords = tuple(data["image_coordinates"]) if data.get("image_coordinates") else None
        norm_coords = tuple(data["normalized_coordinates"]) if data.get("normalized_coordinates") else None
        bbox = tuple(data["bounding_box"]) if data.get("bounding_box") else None

        return cls(
            frame_index=int(data["frame_index"]),
            timestamp_seconds=float(data["timestamp_seconds"]),
            detection_status=status,
            image_coordinates=img_coords,
            normalized_coordinates=norm_coords,
            confidence=float(data.get("confidence", 0.0)),
            bounding_box=bbox,
            area_pixels=float(data.get("area_pixels", 0.0)),
            metadata=data.get("metadata", {}),
        )


@dataclass
class VideoMetadata:
    """Technical metadata for an analyzed video stream."""
    file_path: str
    file_format: str
    total_frames: int
    fps: float
    duration_seconds: float
    width: int
    height: int
    codec: Optional[str] = None
    is_synthetic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "file_format": self.file_format,
            "total_frames": self.total_frames,
            "fps": round(self.fps, 3),
            "duration_seconds": round(self.duration_seconds, 3),
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "is_synthetic": self.is_synthetic,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoMetadata":
        return cls(
            file_path=data["file_path"],
            file_format=data["file_format"],
            total_frames=int(data["total_frames"]),
            fps=float(data["fps"]),
            duration_seconds=float(data["duration_seconds"]),
            width=int(data["width"]),
            height=int(data["height"]),
            codec=data.get("codec"),
            is_synthetic=bool(data.get("is_synthetic", False)),
        )


@dataclass
class TrackerConfig:
    """Configurable parameters for marker tracking and visual segmentation."""
    color_preset: Optional[str] = "red"  # "red", "green", "blue", "yellow", "orange"
    hsv_lower: Optional[Tuple[int, int, int]] = None
    hsv_upper: Optional[Tuple[int, int, int]] = None
    hsv_lower_2: Optional[Tuple[int, int, int]] = None  # Secondary range for red hue wrap-around
    hsv_upper_2: Optional[Tuple[int, int, int]] = None
    min_contour_area: float = 10.0                      # Minimum blob size in pixels
    max_contour_area: Optional[float] = None            # Maximum blob size to filter large backgrounds
    min_circularity: float = 0.0                        # Optional circularity filter [0..1]
    roi: Optional[Tuple[int, int, int, int]] = None     # Optional (x, y, w, h) bounding box
    initial_point: Optional[Tuple[float, float]] = None # Optional manual seed point (u, v)
    search_window_radius: Optional[float] = None        # Radius around last position to constrain search
    max_jump_pixels: Optional[float] = None             # Max displacement between consecutive frames
    apply_morphology: bool = True                       # Clean binary mask with open/close ops
    morph_kernel_size: int = 3                          # Morphology kernel size

    def to_dict(self) -> Dict[str, Any]:
        return {
            "color_preset": self.color_preset,
            "hsv_lower": list(self.hsv_lower) if self.hsv_lower else None,
            "hsv_upper": list(self.hsv_upper) if self.hsv_upper else None,
            "hsv_lower_2": list(self.hsv_lower_2) if self.hsv_lower_2 else None,
            "hsv_upper_2": list(self.hsv_upper_2) if self.hsv_upper_2 else None,
            "min_contour_area": self.min_contour_area,
            "max_contour_area": self.max_contour_area,
            "min_circularity": self.min_circularity,
            "roi": list(self.roi) if self.roi else None,
            "initial_point": list(self.initial_point) if self.initial_point else None,
            "search_window_radius": self.search_window_radius,
            "max_jump_pixels": self.max_jump_pixels,
            "apply_morphology": self.apply_morphology,
            "morph_kernel_size": self.morph_kernel_size,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrackerConfig":
        hsv_l = tuple(data["hsv_lower"]) if data.get("hsv_lower") else None
        hsv_u = tuple(data["hsv_upper"]) if data.get("hsv_upper") else None
        hsv_l2 = tuple(data["hsv_lower_2"]) if data.get("hsv_lower_2") else None
        hsv_u2 = tuple(data["hsv_upper_2"]) if data.get("hsv_upper_2") else None
        roi = tuple(data["roi"]) if data.get("roi") else None
        init_pt = tuple(data["initial_point"]) if data.get("initial_point") else None

        return cls(
            color_preset=data.get("color_preset"),
            hsv_lower=hsv_l,
            hsv_upper=hsv_u,
            hsv_lower_2=hsv_l2,
            hsv_upper_2=hsv_u2,
            min_contour_area=float(data.get("min_contour_area", 10.0)),
            max_contour_area=float(data["max_contour_area"]) if data.get("max_contour_area") is not None else None,
            min_circularity=float(data.get("min_circularity", 0.0)),
            roi=roi,
            initial_point=init_pt,
            search_window_radius=float(data["search_window_radius"]) if data.get("search_window_radius") is not None else None,
            max_jump_pixels=float(data["max_jump_pixels"]) if data.get("max_jump_pixels") is not None else None,
            apply_morphology=bool(data.get("apply_morphology", True)),
            morph_kernel_size=int(data.get("morph_kernel_size", 3)),
        )


@dataclass
class VideoTrajectoryResult:
    """Full result container for an extracted video trajectory."""
    video_metadata: VideoMetadata
    tracker_config: TrackerConfig
    observations: List[VideoFrameObservation]
    total_frames: int
    detected_count: int
    lost_count: int
    detection_rate: float
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_synthetic: bool = False
    camera_calibrated: bool = False
    coordinate_space: str = CoordinateSpace.NORMALIZED_IMAGE.value
    disclaimer: str = UNCALIBRATED_CAMERA_DISCLAIMER
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_metadata": self.video_metadata.to_dict(),
            "tracker_config": self.tracker_config.to_dict(),
            "summary": {
                "total_frames": self.total_frames,
                "detected_count": self.detected_count,
                "lost_count": self.lost_count,
                "detection_rate": round(self.detection_rate, 4),
            },
            "coordinate_space": self.coordinate_space,
            "camera_calibrated": self.camera_calibrated,
            "is_synthetic": self.is_synthetic,
            "created_at": self.created_at,
            "disclaimer": self.disclaimer,
            "notes": self.notes,
            "observations": [obs.to_dict() for obs in self.observations],
        }

    def to_json(self, filepath: Optional[Union[str, Path]] = None, indent: int = 2) -> str:
        payload = json.dumps(self.to_dict(), indent=indent)
        if filepath:
            p = Path(filepath)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(payload, encoding="utf-8")
        return payload

    def to_csv(self, filepath: Union[str, Path]) -> str:
        """Export extracted observations to CSV format."""
        out_path = Path(filepath)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "frame_index",
            "timestamp_seconds",
            "detection_status",
            "u_pixels",
            "v_pixels",
            "u_normalized",
            "v_normalized",
            "confidence",
            "area_pixels",
            "bbox_x",
            "bbox_y",
            "bbox_w",
            "bbox_h",
        ]

        with open(out_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for obs in self.observations:
                row = {
                    "frame_index": obs.frame_index,
                    "timestamp_seconds": round(obs.timestamp_seconds, 6),
                    "detection_status": obs.detection_status.value if isinstance(obs.detection_status, DetectionStatus) else str(obs.detection_status),
                    "u_pixels": round(obs.image_coordinates[0], 2) if obs.image_coordinates else None,
                    "v_pixels": round(obs.image_coordinates[1], 2) if obs.image_coordinates else None,
                    "u_normalized": round(obs.normalized_coordinates[0], 5) if obs.normalized_coordinates else None,
                    "v_normalized": round(obs.normalized_coordinates[1], 5) if obs.normalized_coordinates else None,
                    "confidence": round(obs.confidence, 4),
                    "area_pixels": round(obs.area_pixels, 2),
                    "bbox_x": obs.bounding_box[0] if obs.bounding_box else None,
                    "bbox_y": obs.bounding_box[1] if obs.bounding_box else None,
                    "bbox_w": obs.bounding_box[2] if obs.bounding_box else None,
                    "bbox_h": obs.bounding_box[3] if obs.bounding_box else None,
                }
                writer.writerow(row)
        return str(out_path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoTrajectoryResult":
        meta = VideoMetadata.from_dict(data["video_metadata"])
        cfg = TrackerConfig.from_dict(data["tracker_config"])
        obs_list = [VideoFrameObservation.from_dict(o) for o in data.get("observations", [])]
        summary = data.get("summary", {})

        return cls(
            video_metadata=meta,
            tracker_config=cfg,
            observations=obs_list,
            total_frames=int(summary.get("total_frames", len(obs_list))),
            detected_count=int(summary.get("detected_count", sum(1 for o in obs_list if o.detection_status == DetectionStatus.DETECTED))),
            lost_count=int(summary.get("lost_count", sum(1 for o in obs_list if o.detection_status != DetectionStatus.DETECTED))),
            detection_rate=float(summary.get("detection_rate", 0.0)),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            is_synthetic=bool(data.get("is_synthetic", False)),
            camera_calibrated=bool(data.get("camera_calibrated", False)),
            coordinate_space=data.get("coordinate_space", CoordinateSpace.NORMALIZED_IMAGE.value),
            disclaimer=data.get("disclaimer", UNCALIBRATED_CAMERA_DISCLAIMER),
            notes=data.get("notes"),
        )

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "VideoTrajectoryResult":
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Trajectory file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)

    def to_telemetry(self) -> Any:
        """Convert detected visual observations to Robot CI RecordedTelemetry representation.
        
        Note: ee_position is deliberately left None because monocular tracking
        without calibration does NOT yield metric 3D Cartesian coordinates.
        Image coordinates are placed in raw_metadata.
        """
        from robot_ci.telemetry.models import RecordedTelemetry, TelemetryRecord, TelemetryProvenance

        records: List[TelemetryRecord] = []
        for obs in self.observations:
            records.append(
                TelemetryRecord(
                    timestamp=obs.timestamp_seconds,
                    ee_position=None,  # NEVER fabricate 3D coordinates from uncalibrated 2D image pixels
                    raw_metadata={
                        "frame_index": obs.frame_index,
                        "detection_status": obs.detection_status.value if isinstance(obs.detection_status, DetectionStatus) else str(obs.detection_status),
                        "image_coordinates": obs.image_coordinates,
                        "normalized_coordinates": obs.normalized_coordinates,
                        "tracking_confidence": obs.confidence,
                        "area_pixels": obs.area_pixels,
                    },
                )
            )

        provenance = TelemetryProvenance(
            data_origin="video_extracted",
            source_id=Path(self.video_metadata.file_path).stem,
            recording_format="video_json",
            is_synthetic=self.is_synthetic,
            time_basis="VIDEO_RELATIVE_SECONDS",
            notes=f"Uncalibrated 2D video trajectory. Detections: {self.detected_count}/{self.total_frames}",
        )

        return RecordedTelemetry(
            session_id=Path(self.video_metadata.file_path).stem,
            provenance=provenance,
            records=records,
            metadata={
                "video_metadata": self.video_metadata.to_dict(),
                "camera_calibrated": self.camera_calibrated,
                "disclaimer": self.disclaimer,
            },
        )
