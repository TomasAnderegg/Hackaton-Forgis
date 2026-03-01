"""Single deterministic controller for obstacle-aware pick-and-place."""

from __future__ import annotations

import math
from typing import Any

from .types import AABB, ActionType, DiscreteAction, Observation
from .utils import choose_nearest_target_index, choose_shortest_path_move


class PickPlaceController:
    """Simple obstacle-aware controller with fixed rescans and nearest-target selection."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.max_missing_before_rescan = int(config.get("missing_before_rescan", 3))
        self.rescan_interval_steps = int(config.get("rescan_interval_steps", 20))
        self.steps_since_rescan = 0

    def reset(self) -> None:
        self.steps_since_rescan = 0

    def observe(
        self,
        _action: DiscreteAction,
        _obs: Observation,
        _reward: float,
        next_obs: Observation,
        _done: bool,
        _info: dict[str, Any],
    ) -> None:
        if next_obs.raw.get("last_info", {}).get("rescanned"):
            self.steps_since_rescan = 0
        else:
            self.steps_since_rescan += 1

    def _choose_move(self, obs: Observation, target_xy: tuple[float, float]) -> DiscreteAction:
        obstacles = [
            AABB.from_detection(detection)
            for detection in obs.raw["detections"]
            if detection["class"] == "obstacle"
        ]
        return choose_shortest_path_move(
            current_xy=(obs.raw["tcp"][0], obs.raw["tcp"][1]),
            target_xy=target_xy,
            obstacles=obstacles,
            workspace=self.config["workspace"],
            delta=float(self.config["delta"]),
            margin=float(self.config["obstacle_margin"]),
        )

    def act(self, obs: Observation) -> DiscreteAction:
        visible_objects = [d for d in obs.raw["detections"] if d["class"] == "object"]

        if obs.raw.get("holding_object_id") is not None:
            box = next(d["xyz"] for d in obs.raw["detections"] if d["class"] == "box")
            dx = box[0] - obs.raw["tcp"][0]
            dy = box[1] - obs.raw["tcp"][1]
            if math.hypot(dx, dy) <= self.config.get("drop_radius", 0.04):
                return DiscreteAction(ActionType.RELEASE)
            return self._choose_move(obs, (box[0], box[1]))

        if not visible_objects:
            return DiscreteAction(ActionType.RESCAN)
        if obs.raw.get("target_missing_steps", 0) >= self.max_missing_before_rescan:
            return DiscreteAction(ActionType.RESCAN)
        if (
            self.steps_since_rescan >= self.rescan_interval_steps
            and obs.raw.get("selected_target_id") is not None
            and obs.raw.get("phase") not in {"GRASPING", "DROPPING"}
        ):
            return DiscreteAction(ActionType.RESCAN)

        if obs.raw.get("selected_target_id") is None or obs.raw.get("selected_target_xyz") is None:
            best_index = choose_nearest_target_index((obs.raw["tcp"][0], obs.raw["tcp"][1]), visible_objects)
            return DiscreteAction(ActionType.SELECT_TARGET, best_index)

        target = obs.raw["selected_target_xyz"]
        dx = target[0] - obs.raw["tcp"][0]
        dy = target[1] - obs.raw["tcp"][1]
        if math.hypot(dx, dy) <= self.config.get("grasp_radius", 0.03):
            return DiscreteAction(ActionType.GRASP)
        return self._choose_move(obs, (target[0], target[1]))
