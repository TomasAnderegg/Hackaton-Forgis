"""
Camera-to-robot coordinate transform using a perspective homography.

Usage:
    cal = CameraCalibration()
    cal.set_points([(px1,py1,rx1,ry1), (px2,py2,rx2,ry2), ...])  # >= 4 points
    rx, ry = cal.pixel_to_robot(cx_px, cy_px)

Calibration is persisted to /app/camera_calibration.json inside the container.
"""

import json
import logging
import os
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

CALIB_FILE = os.environ.get("CALIB_FILE", "/app/camera_calibration.json")


class CameraCalibration:
    """Holds a pixel→robot homography transform."""

    def __init__(self):
        self._H: Optional[np.ndarray] = None  # 3x3 homography matrix
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(CALIB_FILE):
            return
        try:
            with open(CALIB_FILE) as f:
                data = json.load(f)
            self._H = np.array(data["H"], dtype=np.float64)
            logger.info("Camera calibration loaded from %s", CALIB_FILE)
        except Exception as e:
            logger.warning("Could not load calibration: %s", e)

    def _save(self) -> None:
        if self._H is None:
            return
        try:
            with open(CALIB_FILE, "w") as f:
                json.dump({"H": self._H.tolist()}, f)
            logger.info("Camera calibration saved to %s", CALIB_FILE)
        except Exception as e:
            logger.warning("Could not save calibration: %s", e)

    # ── Calibration ──────────────────────────────────────────────────────────

    def set_points(self, points: list[tuple[float, float, float, float]]) -> None:
        """
        Compute homography from >=4 (pixel_x, pixel_y, robot_x, robot_y) pairs.
        Robot coordinates are in the same unit as your robot (mm or m).
        """
        if len(points) < 4:
            raise ValueError("Need at least 4 calibration points")

        src = np.array([(p[0], p[1]) for p in points], dtype=np.float32)
        dst = np.array([(p[2], p[3]) for p in points], dtype=np.float32)

        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            raise ValueError("Homography computation failed — check calibration points")

        self._H = H
        self._save()
        inliers = int(mask.sum()) if mask is not None else len(points)
        logger.info("Homography computed — %d/%d inliers", inliers, len(points))

    # ── Transform ────────────────────────────────────────────────────────────

    def is_calibrated(self) -> bool:
        return self._H is not None

    def pixel_to_robot(self, px: float, py: float) -> tuple[float, float]:
        """Transform a pixel (px, py) to robot (rx, ry) using the homography."""
        if self._H is None:
            raise RuntimeError("Camera not calibrated — call set_points() first")

        pt = np.array([[[px, py]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, self._H)
        rx, ry = float(result[0, 0, 0]), float(result[0, 0, 1])
        return rx, ry

    def robot_to_pixel(self, rx: float, ry: float) -> tuple[float, float]:
        """Inverse transform: robot (rx, ry) → pixel (px, py)."""
        if self._H is None:
            raise RuntimeError("Camera not calibrated")

        H_inv = np.linalg.inv(self._H)
        pt = np.array([[[rx, ry]]], dtype=np.float32)
        result = cv2.perspectiveTransform(pt, H_inv)
        return float(result[0, 0, 0]), float(result[0, 0, 1])

    def get_matrix(self) -> Optional[list]:
        return self._H.tolist() if self._H is not None else None
