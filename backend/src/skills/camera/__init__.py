"""Camera skills for vision and streaming operations."""

from .start_streaming import StartStreamingSkill
from .stop_streaming import StopStreamingSkill
from .get_bounding_box import GetBoundingBoxSkill
from .get_bounding_box_gemini import GetBoundingBoxGeminiSkill
from .get_label import GetLabelSkill

__all__ = [
    "StartStreamingSkill",
    "StopStreamingSkill",
    "GetBoundingBoxSkill",
    "GetBoundingBoxGeminiSkill",
    "GetLabelSkill",
]
