"""
RLPickSkill — vision-guided pick with online Q-learning adaptation.

This skill combines three operations in one step:
  1. Read the latest bounding box (set by a preceding get_bounding_box step)
  2. Ask the RL agent for the best pick-joint offset for that image zone
  3. Execute the move and report the joints used

The reward is provided by the *next* step (RLUpdateSkill) after the flow
checks whether the pick succeeded.
"""

from typing import Optional

from pydantic import BaseModel, Field

from ..base import ExecutionContext, Skill, SkillResult
from ..registry import register_skill

import math
import logging

logger = logging.getLogger(__name__)


class RLPickParams(BaseModel):
    """Parameters for the rl_pick skill."""

    base_pick_joints: list[float] = Field(
        ...,
        min_length=6,
        max_length=6,
        description="Base pick joint positions in degrees — RL offsets are added on top",
    )
    frame_width: float = Field(
        default=1920.0,
        description="Camera frame width in pixels (used to discretise bbox position)",
    )
    frame_height: float = Field(
        default=1200.0,
        description="Camera frame height in pixels",
    )


@register_skill
class RLPickSkill(Skill[RLPickParams]):
    """
    Move to the pick position with online Q-learning correction.

    Requires a preceding get_bounding_box step with store_result="detection"
    so the bounding box is available in flow variables.
    """

    name = "rl_pick"
    executor_type = "robot"
    description = (
        "Vision-guided pick: reads last bbox, selects best joint offset via Q-learning, "
        "executes the move. Pair with rl_update to close the RL loop."
    )

    # Shared learner instance (one per process, persists across flow runs)
    _learner = None

    @classmethod
    def params_schema(cls) -> type[BaseModel]:
        return RLPickParams

    async def validate(self, params: RLPickParams) -> tuple[bool, Optional[str]]:
        return True, None

    def _get_learner(self, base_joints: list[float]):
        """Lazy-init or return the shared learner (base joints may change after calibration)."""
        from rl.pick_position_learner import PickPositionLearner
        if RLPickSkill._learner is None or RLPickSkill._learner.base_pick_joints != base_joints:
            RLPickSkill._learner = PickPositionLearner(base_pick_joints=base_joints)
        return RLPickSkill._learner

    async def execute(self, params: RLPickParams, context: ExecutionContext) -> SkillResult:
        robot_executor = context.get_executor("robot")
        learner = self._get_learner(params.base_pick_joints)

        # ── Retrieve last bbox from flow variables ──────────────────────────
        detection = context.variables.get("detection") or {}
        bbox = detection.get("bbox") if isinstance(detection, dict) else None

        if bbox and detection.get("found"):
            cx = bbox["x"] + bbox["width"] / 2
            cy = bbox["y"] + bbox["height"] / 2
            state = learner.bbox_to_state(cx, cy, params.frame_width, params.frame_height)
            logger.info("rl_pick: object found at (%.0f, %.0f) → state %d", cx, cy, state)
        else:
            # No detection — use centre state, no offset
            state = 4  # centre of 3×3 grid
            logger.warning("rl_pick: no detection available, using centre state")

        # ── Select action (epsilon-greedy) ──────────────────────────────────
        joints_deg = learner.select_action(state)
        logger.info("rl_pick: moving to joints %s", [round(j, 2) for j in joints_deg])

        # ── Execute the move ────────────────────────────────────────────────
        target_rad = [math.radians(d) for d in joints_deg]
        success = await robot_executor.move_joint(
            target_rad=target_rad,
            tolerance_rad=math.radians(1.0),
        )

        if success:
            return SkillResult.ok({
                "joints_used": joints_deg,
                "state": state,
                "episode": learner.episode_count,
            })
        else:
            return SkillResult.fail(
                "rl_pick: move_joint failed",
                {"joints_used": joints_deg, "state": state},
            )
