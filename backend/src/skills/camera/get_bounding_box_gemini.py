"""GetBoundingBoxGemini skill — zero-shot object detection via Gemini Vision."""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill


class GetBoundingBoxGeminiParams(BaseModel):
    """Parameters for the get_bounding_box_gemini skill."""

    object_description: str = Field(
        ...,
        min_length=1,
        description=(
            "Natural language description of the object to detect. "
            "Be specific for better results: e.g. 'brown cardboard box', "
            "'red plastic bottle', 'white package with label'."
        ),
    )
    confidence_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum Gemini-reported confidence to accept a detection",
    )


@register_skill
class GetBoundingBoxGeminiSkill(Skill[GetBoundingBoxGeminiParams]):
    """
    Detect objects using Google Gemini Vision (zero-shot).

    Unlike YOLO, no pre-trained class is required — describe the target
    object in plain English. Gemini returns normalised bounding boxes
    (0-1000 scale) which are converted to absolute pixel coordinates
    using the live Zivid frame dimensions.

    The result format is identical to get_bounding_box so both skills
    are interchangeable in flow definitions.
    """

    name = "get_bounding_box_gemini"
    executor_type = "camera"
    description = (
        "Zero-shot object detection via Gemini Vision — describe the object "
        "in natural language, no YOLO class name needed"
    )

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return GetBoundingBoxGeminiParams

    async def validate(
        self, params: GetBoundingBoxGeminiParams
    ) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(
        self, params: GetBoundingBoxGeminiParams, context: ExecutionContext
    ) -> SkillResult:
        camera_executor = context.get_executor("camera")

        if not camera_executor.is_ready():
            return SkillResult.fail("Camera not connected")

        detections = await camera_executor.detect_with_gemini(
            object_description=params.object_description,
            confidence_threshold=params.confidence_threshold,
        )

        if not detections:
            return SkillResult.ok({
                "found": False,
                "bbox": None,
                "confidence": 0.0,
                "object_description": params.object_description,
            })

        best = detections[0]
        return SkillResult.ok({
            "found": True,
            "bbox": {
                "x": best.x,
                "y": best.y,
                "width": best.width,
                "height": best.height,
            },
            "confidence": best.confidence,
            "object_description": best.class_name,
            "total_detections": len(detections),
        })
