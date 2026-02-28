"""
RLUpdateSkill — close the RL loop by providing a reward signal.

Place this step AFTER the gripper close + a short delay.
The skill checks whether the object was actually grasped (via camera),
computes the reward, and updates the Q-table.

Two detection strategies are supported:
  - "camera"  : run get_bounding_box again; if the object is GONE → pick succeeded
  - "manual"  : always reward +1 (useful for first tests without feedback loop)
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill

import logging

logger = logging.getLogger(__name__)


class RLUpdateParams(BaseModel):
    """Parameters for the rl_update skill."""

    strategy: Literal["camera", "manual"] = Field(
        default="camera",
        description=(
            "'camera': re-detect; object absent → success reward. "
            "'manual': always +1 (skip feedback, good for initial testing)."
        ),
    )
    object_class: str = Field(
        default="box",
        description="Object class used to verify pick success (strategy='camera' only)",
    )
    confidence_threshold: float = Field(
        default=0.4,
        description="Detection confidence threshold for re-check",
    )
    reward_success: float = Field(default=1.0, description="Reward when pick succeeds")
    reward_failure: float = Field(default=-1.0, description="Reward when pick fails")


@register_skill
class RLUpdateSkill(Skill[RLUpdateParams]):
    """
    Update the Q-learning agent after a pick attempt.

    With strategy='camera': takes a new frame and checks if the object
    disappeared (= picked up successfully).
    With strategy='manual': always gives a success reward (useful for
    debugging without the full feedback loop).
    """

    name = "rl_update"
    executor_type = "camera"
    description = (
        "Update the Q-learning pick agent with success/failure reward. "
        "Must follow an rl_pick step."
    )

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return RLUpdateParams

    async def validate(self, params: RLUpdateParams) -> tuple[bool, Optional[str]]:
        return True, None

    async def execute(self, params: RLUpdateParams, context: ExecutionContext) -> SkillResult:
        from skills.robot.rl_pick import RLPickSkill

        learner = RLPickSkill._learner
        if learner is None:
            return SkillResult.fail("rl_update: no RL learner found — run rl_pick first")

        # ── Determine reward ────────────────────────────────────────────────
        if params.strategy == "manual":
            reward = params.reward_success
            success = True
            logger.info("rl_update: manual strategy → reward=%.1f", reward)

        else:  # strategy == "camera"
            camera_executor = context.get_executor("camera")
            detections = await camera_executor.detect_objects(
                class_name=params.object_class,
                confidence_threshold=params.confidence_threshold,
            )
            # Object gone from the scene → it's in the gripper → success
            success = len(detections) == 0
            reward = params.reward_success if success else params.reward_failure
            logger.info(
                "rl_update: camera strategy → %d detections → %s → reward=%.1f",
                len(detections), "SUCCESS" if success else "FAILURE", reward,
            )

        # ── Update Q-table ──────────────────────────────────────────────────
        learner.update(reward)
        learner.save()

        return SkillResult.ok({
            "success": success,
            "reward": reward,
            "episode_count": learner.episode_count,
            "epsilon": learner.epsilon,
            "q_summary": learner.get_summary(),
        })
