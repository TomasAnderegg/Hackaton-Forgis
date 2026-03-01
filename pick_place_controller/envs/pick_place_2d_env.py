"""Observation building and backend for pick-and-place."""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

from ..types import AABB, DiscreteAction, Observation, Phase
from ..utils import clamp, distance_xy, nearest_items


class Backend:
    """Backend using live detections and the safe UR5e wrapper."""

    def __init__(
        self,
        config: dict[str, Any],
        safe_wrapper: Any,
        perception_fn: Callable[[], dict[str, Any]],
    ) -> None:
        self.config = config
        self.safe_wrapper = safe_wrapper
        self.perception_fn = perception_fn
        self.prev_payload: Optional[dict[str, Any]] = None

    def reset(self) -> dict[str, Any]:
        self.prev_payload = self.perception_fn()
        return self.prev_payload

    async def step_async(self, action: DiscreteAction) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        payload = self.prev_payload or self.reset()
        obstacles = [AABB.from_detection(d) for d in payload["detections"] if d["class"] == "obstacle"]
        result = await self.safe_wrapper.execute(
            action=action,
            tcp=list(payload["tcp"]),
            obstacles=obstacles,
        )
        next_payload = self.perception_fn()
        next_payload.setdefault("placed_count", payload.get("placed_count", 0))
        next_payload.setdefault("remaining_count", payload.get("remaining_count", 0))
        self.prev_payload = next_payload
        done = next_payload.get("phase") == Phase.FINISHED.value or bool(result.get("terminated"))
        return next_payload, done, result


def build_observation(payload: dict[str, Any], config: dict[str, Any], k_nearest_obstacles: int = 3) -> Observation:
    workspace = config["workspace"]
    x_center = (workspace["x_min"] + workspace["x_max"]) / 2.0
    y_center = (workspace["y_min"] + workspace["y_max"]) / 2.0
    x_scale = max(1e-6, (workspace["x_max"] - workspace["x_min"]) / 2.0)
    y_scale = max(1e-6, (workspace["y_max"] - workspace["y_min"]) / 2.0)
    tcp = payload["tcp"]
    selected = payload.get("selected_target_xyz") or [tcp[0], tcp[1], 0.0]
    box = next((d["xyz"] for d in payload["detections"] if d["class"] == "box"), [tcp[0], tcp[1], 0.0])
    phase = payload.get("phase", Phase.SEARCHING.value)
    phase_vec = [1.0 if phase == item.value else 0.0 for item in Phase]
    diag = math.hypot(workspace["x_max"] - workspace["x_min"], workspace["y_max"] - workspace["y_min"])
    dist_to_target = distance_xy((tcp[0], tcp[1]), (selected[0], selected[1])) / max(diag, 1e-6)
    dist_to_box = distance_xy((tcp[0], tcp[1]), (box[0], box[1])) / max(diag, 1e-6)
    obstacles = [AABB.from_detection(d) for d in payload["detections"] if d["class"] == "obstacle"]
    ranked = nearest_items(
        ((distance_xy((tcp[0], tcp[1]), (((o.min_x + o.max_x) / 2.0), ((o.min_y + o.max_y) / 2.0))), o) for o in obstacles),
        k_nearest_obstacles,
    )
    obstacle_features: list[float] = []
    for obstacle in ranked:
        obstacle_features.extend(
            [
                clamp((tcp[0] - obstacle.min_x) / x_scale, -1.0, 1.0),
                clamp((obstacle.max_x - tcp[0]) / x_scale, -1.0, 1.0),
                clamp((tcp[1] - obstacle.min_y) / y_scale, -1.0, 1.0),
                clamp((obstacle.max_y - tcp[1]) / y_scale, -1.0, 1.0),
            ]
        )
    while len(obstacle_features) < k_nearest_obstacles * 4:
        obstacle_features.extend([1.0, 1.0, 1.0, 1.0])
    vector = [
        clamp((tcp[0] - x_center) / x_scale, -1.0, 1.0),
        clamp((tcp[1] - y_center) / y_scale, -1.0, 1.0),
        clamp((selected[0] - x_center) / x_scale, -1.0, 1.0),
        clamp((selected[1] - y_center) / y_scale, -1.0, 1.0),
        clamp((box[0] - x_center) / x_scale, -1.0, 1.0),
        clamp((box[1] - y_center) / y_scale, -1.0, 1.0),
        *phase_vec,
        payload.get("remaining_count", 0) / max(1.0, float(config.get("num_objects", 5))),
        payload.get("placed_count", 0) / max(1.0, float(config.get("num_objects", 5))),
        dist_to_target,
        dist_to_box,
        *obstacle_features,
    ]
    return Observation(vector=vector, raw=payload)
