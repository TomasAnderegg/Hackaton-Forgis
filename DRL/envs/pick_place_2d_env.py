"""Gym-like pick-and-place environment with sim and real backends."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..types import AABB, ActionType, DiscreteAction, Mode, MOVE_ACTIONS, Observation, Phase
from ..utils import clamp, distance_xy, line_intersects_aabb_xy, nearest_items


@dataclass
class TargetView:
    object_id: int
    x: float
    y: float
    z: float


class SimBackend2D:
    """Fast 2D simulator for adaptive pick-and-place."""

    def __init__(self, config: dict[str, Any], seed: Optional[int] = None) -> None:
        self.config = config
        self.random = random.Random(seed)
        self.delta = float(config["delta"])
        self.workspace = config["workspace"]
        self.max_steps = int(config.get("max_steps", 300))
        self.k_nearest_obstacles = int(config.get("k_nearest_obstacles", 3))
        self.reset()

    def reset(self) -> dict[str, Any]:
        self.step_count = 0
        self.mode = Mode.SAFE
        self.phase = Phase.SEARCHING
        self.tcp = [self.workspace["x_min"] + 0.04, self.workspace["y_min"] + 0.04, float(self.config["travel_height"])]
        self.box = [self.workspace["x_max"] - 0.10, self.workspace["y_max"] - 0.10, float(self.config["drop_height"])]
        self.n_objects = int(self.config.get("num_objects", 5))
        self.objects = []
        self.obstacles = []
        self.selected_target_id: Optional[int] = None
        self.holding_object_id: Optional[int] = None
        self.placed_count = 0
        self.remaining_count = self.n_objects
        self.last_visible_target_ids: list[int] = []
        self.target_missing_steps = 0
        self.rescan_boost = 0
        self._spawn_world()
        self.current_payload = self._observation_payload()
        return self.current_payload

    def _spawn_world(self) -> None:
        for obstacle_id in range(int(self.config.get("num_obstacles", 2))):
            cx = self.random.uniform(self.workspace["x_min"] + 0.12, self.workspace["x_max"] - 0.12)
            cy = self.random.uniform(self.workspace["y_min"] + 0.12, self.workspace["y_max"] - 0.12)
            sx = self.random.uniform(0.05, 0.10)
            sy = self.random.uniform(0.05, 0.10)
            self.obstacles.append(
                {
                    "id": obstacle_id,
                    "base": AABB(cx - sx / 2.0, cy - sy / 2.0, 0.0, cx + sx / 2.0, cy + sy / 2.0, 0.40),
                    "active": True,
                    "dynamic": obstacle_id == 0 and bool(self.config.get("dynamic_obstacles", True)),
                }
            )
        for object_id in range(self.n_objects):
            while True:
                x = self.random.uniform(self.workspace["x_min"] + 0.08, self.workspace["x_max"] - 0.18)
                y = self.random.uniform(self.workspace["y_min"] + 0.08, self.workspace["y_max"] - 0.18)
                point_ok = True
                for obstacle in self.active_obstacles():
                    expanded = obstacle.dilated(0.04)
                    if expanded.contains_xy(x, y):
                        point_ok = False
                        break
                if point_ok:
                    self.objects.append(
                        {
                            "id": object_id,
                            "x": x,
                            "y": y,
                            "z": float(self.config["pick_height"]),
                            "placed": False,
                            "lost": False,
                        }
                    )
                    break

    def active_obstacles(self) -> list[AABB]:
        return [obstacle["base"] for obstacle in self.obstacles if obstacle["active"]]

    def _toggle_dynamic_obstacles(self) -> None:
        for obstacle in self.obstacles:
            if obstacle["dynamic"] and self.random.random() < float(self.config.get("dynamic_obstacle_flip_prob", 0.10)):
                obstacle["active"] = not obstacle["active"]

    def _visible_targets(self) -> list[TargetView]:
        views: list[TargetView] = []
        miss_prob = float(self.config["miss_detection_prob"])
        noise_std = float(self.config["detection_noise_std"])
        force_visible = self.rescan_boost > 0
        for obj in self.objects:
            if obj["placed"] or obj["lost"]:
                continue
            if self.holding_object_id == obj["id"]:
                continue
            visible = force_visible or self.random.random() > miss_prob
            if not visible:
                continue
            nx = self.random.gauss(0.0, noise_std / (2.0 if force_visible else 1.0))
            ny = self.random.gauss(0.0, noise_std / (2.0 if force_visible else 1.0))
            views.append(TargetView(obj["id"], obj["x"] + nx, obj["y"] + ny, obj["z"]))
        self.last_visible_target_ids = [view.object_id for view in views]
        self.rescan_boost = max(0, self.rescan_boost - 1)
        return views

    def _target_by_id(self, object_id: Optional[int]) -> Optional[dict[str, Any]]:
        if object_id is None:
            return None
        for obj in self.objects:
            if obj["id"] == object_id and not obj["placed"] and not obj["lost"]:
                return obj
        return None

    def _move_safe(self, next_xy: tuple[float, float]) -> bool:
        if not (self.workspace["x_min"] <= next_xy[0] <= self.workspace["x_max"]):
            return False
        if not (self.workspace["y_min"] <= next_xy[1] <= self.workspace["y_max"]):
            return False
        margin_cfg = self.config["obstacle_margin"]
        margin = float(margin_cfg["safe"] if self.mode == Mode.SAFE else margin_cfg["fast"])
        for obstacle in self.active_obstacles():
            expanded = obstacle.dilated(margin)
            if line_intersects_aabb_xy((self.tcp[0], self.tcp[1]), next_xy, expanded):
                return False
        return True

    def _step_move(self, action: DiscreteAction, info: dict[str, Any]) -> None:
        dx, dy = MOVE_ACTIONS[action.kind]
        next_xy = (self.tcp[0] + dx * self.delta, self.tcp[1] + dy * self.delta)
        if not self._move_safe(next_xy):
            info["unsafe_action"] = True
            if str(self.config.get("unsafe_action_behavior", "NOOP")).upper() == "TERMINATE":
                info["terminated"] = True
            return
        self.tcp[0], self.tcp[1] = next_xy

    def _select_target(self, action: DiscreteAction, visible_targets: list[TargetView], info: dict[str, Any]) -> None:
        if action.index is None or action.index < 0 or action.index >= len(visible_targets):
            info["invalid_target_selection"] = True
            return
        self.selected_target_id = visible_targets[action.index].object_id
        self.target_missing_steps = 0

    def _grasp(self, info: dict[str, Any]) -> None:
        target = self._target_by_id(self.selected_target_id)
        if target is None:
            info["grasp_failed"] = True
            return
        dist = distance_xy((self.tcp[0], self.tcp[1]), (target["x"], target["y"]))
        if dist > float(self.config.get("grasp_radius", 0.03)):
            info["grasp_failed"] = True
            return
        fail_prob_cfg = self.config["grasp_fail_prob"]
        fail_prob = float(fail_prob_cfg["safe"] if self.mode == Mode.SAFE else fail_prob_cfg["fast"])
        if self.random.random() < fail_prob:
            info["grasp_failed"] = True
            return
        self.holding_object_id = target["id"]
        info["grasp_succeeded"] = True

    def _release(self, info: dict[str, Any]) -> None:
        if self.holding_object_id is None:
            return
        obj = self._target_by_id(self.holding_object_id)
        if obj is None:
            self.holding_object_id = None
            return
        dist = distance_xy((self.tcp[0], self.tcp[1]), (self.box[0], self.box[1]))
        if dist <= float(self.config.get("drop_radius", 0.04)):
            obj["placed"] = True
            obj["x"], obj["y"] = self.box[0], self.box[1]
            self.placed_count += 1
            self.remaining_count -= 1
            self.selected_target_id = None
            info["placed"] = True
        else:
            obj["lost"] = True
            self.remaining_count -= 1
            self.selected_target_id = None
            info["dropped"] = True
        self.holding_object_id = None

    def _update_phase(self, visible_targets: list[TargetView]) -> None:
        if self.placed_count >= self.n_objects:
            self.phase = Phase.FINISHED
            return
        if self.holding_object_id is not None:
            dist_to_box = distance_xy((self.tcp[0], self.tcp[1]), (self.box[0], self.box[1]))
            self.phase = Phase.DROPPING if dist_to_box <= float(self.config.get("drop_radius", 0.04)) else Phase.CARRYING
            return
        target = self._target_by_id(self.selected_target_id)
        if target is None:
            self.phase = Phase.SEARCHING
            return
        is_visible = any(view.object_id == target["id"] for view in visible_targets)
        if not is_visible:
            self.target_missing_steps += 1
        else:
            self.target_missing_steps = 0
        dist = distance_xy((self.tcp[0], self.tcp[1]), (target["x"], target["y"]))
        self.phase = Phase.GRASPING if dist <= float(self.config.get("grasp_radius", 0.03)) else Phase.APPROACHING

    def step(self, action: DiscreteAction) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        self.step_count += 1
        self._toggle_dynamic_obstacles()
        visible_targets = self._visible_targets()
        info: dict[str, Any] = {
            "mode": self.mode.value,
            "selected_target_id": self.selected_target_id,
        }

        if action.kind in MOVE_ACTIONS:
            self._step_move(action, info)
        elif action.kind == ActionType.SELECT_TARGET:
            self._select_target(action, visible_targets, info)
        elif action.kind == ActionType.SET_MODE_FAST:
            self.mode = Mode.FAST
            info["mode_switch"] = True
        elif action.kind == ActionType.SET_MODE_SAFE:
            self.mode = Mode.SAFE
            info["mode_switch"] = True
        elif action.kind == ActionType.RESCAN:
            self.rescan_boost = 2
            visible_targets = self._visible_targets()
            info["rescanned"] = True
        elif action.kind == ActionType.GRASP:
            self._grasp(info)
        elif action.kind == ActionType.RELEASE:
            self._release(info)

        self._update_phase(visible_targets)
        done = bool(info.get("terminated")) or self.phase == Phase.FINISHED or self.step_count >= self.max_steps or self.remaining_count <= 0
        info["done_reason"] = "finished" if self.phase == Phase.FINISHED else ("terminated" if info.get("terminated") else None)
        self.current_payload = self._observation_payload(visible_targets)
        return self.current_payload, done, info

    def _observation_payload(self, visible_targets: Optional[list[TargetView]] = None) -> dict[str, Any]:
        visible_targets = visible_targets if visible_targets is not None else self._visible_targets()
        selected_target = None
        true_target = self._target_by_id(self.selected_target_id)
        if true_target is not None:
            selected_target = TargetView(true_target["id"], true_target["x"], true_target["y"], true_target["z"])
        detections: list[dict[str, Any]] = []
        for view in visible_targets:
            detections.append({"class": "object", "id": view.object_id, "xyz": [view.x, view.y, view.z]})
        detections.append({"class": "box", "id": "box", "xyz": list(self.box), "size": [0.14, 0.14, 0.10]})
        for idx, obstacle in enumerate(self.active_obstacles()):
            detections.append({"class": "obstacle", "id": idx, "min_xyz": [obstacle.min_x, obstacle.min_y, obstacle.min_z], "max_xyz": [obstacle.max_x, obstacle.max_y, obstacle.max_z]})
        return {
            "detections": detections,
            "tcp": list(self.tcp),
            "phase": self.phase.value,
            "placed_count": self.placed_count,
            "remaining_count": self.remaining_count,
            "mode": self.mode.value,
            "selected_target_id": self.selected_target_id,
            "selected_target_xyz": [selected_target.x, selected_target.y, selected_target.z] if selected_target else None,
            "visible_target_ids": list(self.last_visible_target_ids),
            "holding_object_id": self.holding_object_id,
            "target_missing_steps": self.target_missing_steps,
            "step_count": self.step_count,
        }

    def get_observation(self) -> Observation:
        payload = getattr(self, "current_payload", None) or self._observation_payload()
        return build_observation(payload, self.config, self.k_nearest_obstacles)


class RealBackend:
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
        self.mode = Mode.SAFE
        self.prev_payload: Optional[dict[str, Any]] = None

    def reset(self) -> dict[str, Any]:
        self.mode = Mode.SAFE
        self.prev_payload = self.perception_fn()
        self.prev_payload.setdefault("mode", self.mode.value)
        return self.prev_payload

    async def step_async(self, action: DiscreteAction) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        payload = self.prev_payload or self.reset()
        obstacles = [AABB.from_detection(d) for d in payload["detections"] if d["class"] == "obstacle"]
        result = await self.safe_wrapper.execute(
            action=action,
            tcp=list(payload["tcp"]),
            obstacles=obstacles,
            mode=self.mode,
            carrying=payload.get("phase") in {Phase.CARRYING.value, Phase.DROPPING.value},
        )
        if "mode" in result:
            self.mode = result["mode"]
        next_payload = self.perception_fn()
        next_payload["mode"] = self.mode.value
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
    mode = payload.get("mode", Mode.SAFE.value)
    phase_vec = [1.0 if phase == item.value else 0.0 for item in Phase]
    mode_vec = [1.0 if mode == Mode.FAST.value else 0.0, 1.0 if mode == Mode.SAFE.value else 0.0]
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
        *mode_vec,
        payload.get("remaining_count", 0) / max(1.0, float(config.get("num_objects", 5))),
        payload.get("placed_count", 0) / max(1.0, float(config.get("num_objects", 5))),
        dist_to_target,
        dist_to_box,
        *obstacle_features,
    ]
    return Observation(vector=vector, raw=payload)


class PickPlace2DEnv:
    """Gym-like wrapper providing reset() and step()."""

    def __init__(self, backend: Any, config: dict[str, Any]) -> None:
        self.backend = backend
        self.config = config
        self.k_nearest_obstacles = int(config.get("k_nearest_obstacles", 3))
        self.cumulative_reward = 0.0

    def reset(self) -> Observation:
        payload = self.backend.reset()
        self.cumulative_reward = 0.0
        return build_observation(payload, self.config, self.k_nearest_obstacles)

    def _reward(self, prev_obs: Observation, next_obs: Observation, action: DiscreteAction, info: dict[str, Any], done: bool) -> float:
        reward = float(self.config.get("step_cost", -0.1))
        if info.get("grasp_succeeded"):
            reward += float(self.config.get("grasp_success_reward", 1.0))
        if info.get("dropped"):
            reward += float(self.config.get("dropped_penalty", -10.0))
        if info.get("unsafe_action") or info.get("terminated"):
            reward += float(self.config.get("error_penalty", -10.0))
        if action.kind == ActionType.RESCAN:
            reward += float(self.config.get("rescan_cost", -0.5))
        if info.get("mode_switch"):
            reward += float(self.config.get("mode_switch_penalty", -0.05))
        placed_delta = next_obs.raw["placed_count"] - prev_obs.raw["placed_count"]
        if placed_delta > 0:
            reward += placed_delta * float(self.config.get("placed_reward", 10.0))
        if done and next_obs.raw.get("phase") == Phase.FINISHED.value:
            reward += float(self.config.get("all_placed_bonus", 0.0))

        selected = prev_obs.raw.get("selected_target_xyz") or prev_obs.raw["tcp"]
        box = next((d["xyz"] for d in prev_obs.raw["detections"] if d["class"] == "box"), prev_obs.raw["tcp"])
        dist_to_target = distance_xy((prev_obs.raw["tcp"][0], prev_obs.raw["tcp"][1]), (selected[0], selected[1]))
        dist_to_box = distance_xy((prev_obs.raw["tcp"][0], prev_obs.raw["tcp"][1]), (box[0], box[1]))
        if prev_obs.raw.get("holding_object_id") is None:
            reward -= float(self.config["shaping"]["k1"]) * dist_to_target
        else:
            reward -= float(self.config["shaping"]["k2"]) * dist_to_box
        return reward

    def step(self, action: DiscreteAction) -> tuple[Observation, float, bool, dict[str, Any]]:
        prev_payload = getattr(self.backend, "current_payload", None)
        prev_obs = build_observation(prev_payload, self.config, self.k_nearest_obstacles) if prev_payload is not None else self.reset()
        payload, done, info = self.backend.step(action)
        next_obs = build_observation(payload, self.config, self.k_nearest_obstacles)
        reward = self._reward(prev_obs, next_obs, action, info, done)
        self.cumulative_reward += reward
        info["cumulative_reward"] = self.cumulative_reward
        next_obs.raw["last_info"] = dict(info)
        return next_obs, reward, done, info
