"""
ScanAndHoverSkill — detects all boxes with YOLO, numbers them,
and moves the robot TCP above each one sequentially.

Requires camera calibration (pixel → robot mm) to be set.
"""

import asyncio
import logging
import math
from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill

logger = logging.getLogger(__name__)


class ScanAndHoverParams(BaseModel):
    object_class: str = Field(
        default="cardboard box",
        description="YOLO class to detect",
    )
    confidence_threshold: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence",
    )
    z_height_m: float = Field(
        default=0.20,
        description="Height above workspace to hover (meters, in robot frame)",
    )
    orientation: list[float] = Field(
        default=[math.pi, 0.0, 0.0],
        min_length=3,
        max_length=3,
        description="TCP orientation [rx, ry, rz] in radians. [π,0,0] = pointing down for UR.",
    )
    hover_time_s: float = Field(
        default=1.0,
        ge=0.0,
        description="Seconds to pause above each box",
    )
    velocity: float = Field(default=0.2, ge=0.01, le=1.0)
    acceleration: float = Field(default=0.5, ge=0.01, le=3.0)


@register_skill
class ScanAndHoverSkill(Skill[ScanAndHoverParams]):
    """
    Scan for all boxes, number them, and hover above each one in sequence.
    Uses camera calibration to convert pixel positions to robot mm coordinates.
    """

    name = "scan_and_hover"
    executor_type = "robot"
    description = (
        "Detect all boxes via YOLO, number them 1..N, "
        "then move TCP above each one sequentially."
    )

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return ScanAndHoverParams

    async def validate(self, params: ScanAndHoverParams) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(self, params: ScanAndHoverParams, context: ExecutionContext) -> SkillResult:
        robot_executor = context.get_executor("robot")
        camera_executor = context.get_executor("camera")

        # ── 1. Detect all boxes ───────────────────────────────────────────────
        detections = await camera_executor.detect_objects(
            class_name=params.object_class,
            confidence_threshold=params.confidence_threshold,
        )

        if not detections:
            logger.info("scan_and_hover: no objects detected")
            return SkillResult.ok({"found": False, "count": 0, "boxes": []})

        # ── 2. Filter to calibrated detections (have robot coords) ────────────
        calibrated = [d for d in detections if d.robot_x is not None and d.robot_y is not None]

        if not calibrated:
            return SkillResult.fail(
                "Detections found but camera not calibrated — "
                "run POST /api/camera/calibrate first"
            )

        # ── 3. Sort left-to-right (by robot X) ───────────────────────────────
        calibrated.sort(key=lambda d: d.robot_x)

        logger.info(f"scan_and_hover: found {len(calibrated)} box(es)")
        boxes_visited = []

        for i, det in enumerate(calibrated, start=1):
            rx_m = det.robot_x / 1000.0  # mm → meters
            ry_m = det.robot_y / 1000.0
            rz_m = params.z_height_m
            pose = [rx_m, ry_m, rz_m] + list(params.orientation)

            logger.info(
                f"scan_and_hover: box #{i}/{len(calibrated)} "
                f"robot=({det.robot_x:.1f}, {det.robot_y:.1f}) mm  "
                f"conf={det.confidence:.0%}"
            )

            success = await robot_executor.move_linear(
                pose=pose,
                acceleration=params.acceleration,
                velocity=params.velocity,
            )

            if not success:
                logger.warning(f"scan_and_hover: move to box #{i} failed, skipping")
                boxes_visited.append({
                    "index": i,
                    "robot_x_mm": det.robot_x,
                    "robot_y_mm": det.robot_y,
                    "confidence": det.confidence,
                    "reached": False,
                })
                continue

            logger.info(f"scan_and_hover: hovering above box #{i} for {params.hover_time_s}s")
            await asyncio.sleep(params.hover_time_s)

            boxes_visited.append({
                "index": i,
                "robot_x_mm": det.robot_x,
                "robot_y_mm": det.robot_y,
                "confidence": det.confidence,
                "reached": True,
            })

        return SkillResult.ok({
            "found": True,
            "count": len(calibrated),
            "boxes": boxes_visited,
        })
