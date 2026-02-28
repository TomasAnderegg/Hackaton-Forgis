"""Safe, deterministic baseline policy for pick-and-place."""

from __future__ import annotations

import math
from typing import Any, Optional

from ..types import ActionType, DiscreteAction, Observation


class BaselinePickPlacePolicy:
    """Always-safe nearest-target baseline."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.max_missing_before_rescan = int(config.get("baseline_missing_before_rescan", 3))
        self.last_target_id: Optional[int] = None

    def reset(self) -> None:
        self.last_target_id = None

    def observe(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def _choose_axis_move(self, obs: Observation, target_xy: tuple[float, float]) -> DiscreteAction:
        tcp = obs.raw["tcp"]
        dx = target_xy[0] - tcp[0]
        dy = target_xy[1] - tcp[1]
        primary = DiscreteAction(ActionType.MOVE_POS_X if dx >= 0 else ActionType.MOVE_NEG_X) if abs(dx) >= abs(dy) else DiscreteAction(ActionType.MOVE_POS_Y if dy >= 0 else ActionType.MOVE_NEG_Y)
        secondary = DiscreteAction(ActionType.MOVE_POS_Y if dy >= 0 else ActionType.MOVE_NEG_Y) if abs(dx) >= abs(dy) else DiscreteAction(ActionType.MOVE_POS_X if dx >= 0 else ActionType.MOVE_NEG_X)
        if obs.raw.get("last_info", {}).get("unsafe_action"):
            return secondary
        return primary

    def act(self, obs: Observation) -> DiscreteAction:
        if obs.raw.get("mode") != "SAFE":
            return DiscreteAction(ActionType.SET_MODE_SAFE)
        if obs.raw.get("holding_object_id") is not None:
            box = next(d["xyz"] for d in obs.raw["detections"] if d["class"] == "box")
            dx = box[0] - obs.raw["tcp"][0]
            dy = box[1] - obs.raw["tcp"][1]
            if math.hypot(dx, dy) <= self.config.get("drop_radius", 0.04):
                return DiscreteAction(ActionType.RELEASE)
            return self._choose_axis_move(obs, (box[0], box[1]))

        visible_objects = [d for d in obs.raw["detections"] if d["class"] == "object"]
        if not visible_objects:
            return DiscreteAction(ActionType.RESCAN)
        if obs.raw.get("target_missing_steps", 0) >= self.max_missing_before_rescan:
            return DiscreteAction(ActionType.RESCAN)

        selected_id = obs.raw.get("selected_target_id")
        if selected_id is None:
            candidates = []
            tcp = obs.raw["tcp"]
            for idx, detection in enumerate(visible_objects):
                distance = abs(detection["xyz"][0] - tcp[0]) + abs(detection["xyz"][1] - tcp[1])
                candidates.append((distance, idx))
            _, best_index = min(candidates, key=lambda item: item[0])
            return DiscreteAction(ActionType.SELECT_TARGET, best_index)

        target = obs.raw.get("selected_target_xyz")
        if target is None:
            candidates = []
            tcp = obs.raw["tcp"]
            for idx, detection in enumerate(visible_objects):
                distance = abs(detection["xyz"][0] - tcp[0]) + abs(detection["xyz"][1] - tcp[1])
                candidates.append((distance, idx))
            _, best_index = min(candidates, key=lambda item: item[0])
            return DiscreteAction(ActionType.SELECT_TARGET, best_index)
        dx = target[0] - obs.raw["tcp"][0]
        dy = target[1] - obs.raw["tcp"][1]
        if math.hypot(dx, dy) <= self.config.get("grasp_radius", 0.03):
            return DiscreteAction(ActionType.GRASP)
        return self._choose_axis_move(obs, (target[0], target[1]))
