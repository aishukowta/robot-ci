"""Camera calibration models and 3D-to-2D projection utilities.

Implements the pinhole camera projection model to map simulated 3D robot base
coordinates (meters) into camera image pixel coordinates [u, v] and normalized
screen-space coordinates [u_norm, v_norm].
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import json
import math
import numpy as np


class CalibrationError(ValueError):
    """Raised when camera calibration parameters fail geometric or numerical validation."""
    pass


@dataclass
class CameraCalibration:
    """Pinhole camera intrinsic and extrinsic calibration parameters."""
    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    rotation_matrix: np.ndarray        # Shape (3, 3): world/robot to camera rotation
    translation_vector: np.ndarray     # Shape (3,): world/robot to camera translation in meters
    distortion_coefficients: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    camera_model_name: str = "pinhole_calibrated"
    calibration_date: Optional[str] = None
    notes: Optional[str] = None
    # Provenance: "measured" (real checkerboard/known-geometry calibration of the
    # camera used for the recording), "synthetic_fixture", or "unspecified".
    # Only "measured" calibrations are accepted as physical evidence.
    source: str = "unspecified"

    def __post_init__(self):
        text = f"{self.camera_model_name} {self.notes or ''}".lower()
        if "synthetic" in text:
            self.source = "synthetic_fixture"
        if self.source not in ("measured", "synthetic_fixture", "unspecified"):
            raise CalibrationError(f"Invalid calibration source '{self.source}'")
        self.validate()

    def validate(self) -> None:
        """Validate calibration parameters against optical and geometric constraints."""
        if self.fx <= 0 or self.fy <= 0:
            raise CalibrationError(f"Focal lengths must be positive: fx={self.fx}, fy={self.fy}")
        if self.image_width <= 0 or self.image_height <= 0:
            raise CalibrationError(f"Image dimensions must be positive: {self.image_width}x{self.image_height}")
        if not (math.isfinite(self.cx) and math.isfinite(self.cy)):
            raise CalibrationError(f"Principal point must be finite: ({self.cx}, {self.cy})")

        # Convert and validate rotation matrix
        r_arr = np.asarray(self.rotation_matrix, dtype=np.float64)
        if r_arr.shape != (3, 3):
            raise CalibrationError(f"Rotation matrix must have shape (3, 3), got {r_arr.shape}")
        if not np.all(np.isfinite(r_arr)):
            raise CalibrationError("Rotation matrix contains non-finite values.")

        # Check orthonormality: R @ R.T ~= I and det(R) ~= 1
        r_rt = r_arr @ r_arr.T
        if not np.allclose(r_rt, np.eye(3), atol=1e-3):
            raise CalibrationError(f"Rotation matrix is not orthogonal: R @ R.T =\n{r_rt}")
        det_r = float(np.linalg.det(r_arr))
        if not np.isclose(det_r, 1.0, atol=1e-2):
            raise CalibrationError(f"Rotation matrix determinant must be +1 (got {det_r:.4f})")

        # Convert and validate translation vector
        t_arr = np.asarray(self.translation_vector, dtype=np.float64).flatten()
        if t_arr.shape != (3,):
            raise CalibrationError(f"Translation vector must have length 3, got shape {t_arr.shape}")
        if not np.all(np.isfinite(t_arr)):
            raise CalibrationError("Translation vector contains non-finite values.")

        self.rotation_matrix = r_arr
        self.translation_vector = t_arr

    def get_intrinsic_matrix(self) -> np.ndarray:
        """Return 3x3 camera intrinsic matrix K."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def project_point(self, point_robot_3d: Union[List[float], np.ndarray]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]], bool]:
        """Project a single 3D point in robot coordinates into 2D camera pixels.
        
        Args:
            point_robot_3d: 3D point [X, Y, Z] in meters relative to robot base.
        
        Returns:
            Tuple of:
            - pixel_coords: (u, v) in pixels or None if behind camera
            - norm_coords: (u_norm, v_norm) in [0..1] or None if behind camera
            - is_visible: True if point is in front of camera and inside frame bounds
        """
        pt = np.asarray(point_robot_3d, dtype=np.float64).flatten()
        if len(pt) != 3 or not np.all(np.isfinite(pt)):
            raise CalibrationError(f"Point must be a finite 3D vector, got {pt}")

        # Transform point into camera reference frame: P_cam = R @ P_robot + t
        p_cam = self.rotation_matrix @ pt + self.translation_vector
        xc, yc, zc = p_cam[0], p_cam[1], p_cam[2]

        # Check positive depth (point must be in front of optical center)
        if zc <= 1e-4:
            return None, None, False

        u = (self.fx * xc / zc) + self.cx
        v = (self.fy * yc / zc) + self.cy

        norm_u = u / float(self.image_width)
        norm_v = v / float(self.image_height)

        is_visible = bool((0.0 <= u < self.image_width) and (0.0 <= v < self.image_height))

        return (float(u), float(v)), (float(norm_u), float(norm_v)), is_visible

    def project_trajectory(self, trajectory_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project an Nx3 trajectory of 3D points into 2D camera coordinates.
        
        Args:
            trajectory_3d: Nx3 numpy array of Cartesian coordinates [X, Y, Z] in meters.
        
        Returns:
            Tuple of:
            - pixel_coords: Nx2 numpy array of [u, v] coordinates in pixels
            - normalized_coords: Nx2 numpy array of [u_norm, v_norm]
            - valid_mask: 1D boolean array indicating which points have positive depth
        """
        arr = np.asarray(trajectory_3d, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise CalibrationError(f"Trajectory must have shape (N, 3), got {arr.shape}")

        n = len(arr)
        pixels = np.zeros((n, 2), dtype=np.float64)
        normalized = np.zeros((n, 2), dtype=np.float64)
        valid_mask = np.zeros(n, dtype=bool)

        for i in range(n):
            px, norm_px, visible = self.project_point(arr[i])
            if px is not None:
                pixels[i] = px
                normalized[i] = norm_px
                valid_mask[i] = True

        return pixels, normalized, valid_mask

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "rotation_matrix": self.rotation_matrix.tolist(),
            "translation_vector": self.translation_vector.tolist(),
            "distortion_coefficients": self.distortion_coefficients,
            "camera_model_name": self.camera_model_name,
            "calibration_date": self.calibration_date,
            "notes": self.notes,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CameraCalibration":
        return cls(
            fx=float(data["fx"]),
            fy=float(data["fy"]),
            cx=float(data["cx"]),
            cy=float(data["cy"]),
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            rotation_matrix=np.array(data["rotation_matrix"], dtype=np.float64),
            translation_vector=np.array(data["translation_vector"], dtype=np.float64),
            distortion_coefficients=list(data.get("distortion_coefficients", [0.0] * 5)),
            camera_model_name=data.get("camera_model_name", "pinhole_calibrated"),
            calibration_date=data.get("calibration_date"),
            notes=data.get("notes"),
            source=data.get("source", "unspecified"),
        )

    def save(self, filepath: Union[str, Path]) -> str:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return str(p)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "CameraCalibration":
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Calibration file not found: {p}")
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


def create_synthetic_camera_calibration(
    width: int = 320,
    height: int = 240,
    camera_distance: float = 1.2,
) -> CameraCalibration:
    """Create a calibrated camera model matching standard synthetic side-view setup.
    
    Camera positioned along +X / -Y viewing the robot workspace center.
    """
    fx = float(width) * 0.9
    fy = float(width) * 0.9
    cx = float(width) / 2.0
    cy = float(height) / 2.0

    # Camera looking along +Z_cam towards workspace origin
    # Optical frame: X right, Y down, Z forward
    R = np.array([
        [0.0,  1.0,  0.0],
        [0.0,  0.0, -1.0],
        [-1.0, 0.0,  0.0],
    ], dtype=np.float64)

    t = np.array([0.0, 0.25, camera_distance], dtype=np.float64)

    return CameraCalibration(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        image_width=width,
        image_height=height,
        rotation_matrix=R,
        translation_vector=t,
        camera_model_name="synthetic_side_view",
        notes="Synthetic test calibration fixture",
        source="synthetic_fixture",
    )
