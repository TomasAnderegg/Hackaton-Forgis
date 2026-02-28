"""Contextual-bandit policy with a low-level pick-and-place executor."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .baseline_pick_place import BaselinePickPlacePolicy
from ..types import ActionType, DiscreteAction, Observation, ObservationLayout, Strategy


@dataclass
class ArmState:
    a_inv: list[list[float]]
    b: list[float]


def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mat_vec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(m * v for m, v in zip(row, vector)) for row in matrix]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _rank_one_update(a_inv: list[list[float]], x: list[float]) -> list[list[float]]:
    ax = _mat_vec(a_inv, x)
    denom = 1.0 + _dot(x, ax)
    outer = [[ax_i * ax_j / denom for ax_j in ax] for ax_i in ax]
    return [[a_inv[i][j] - outer[i][j] for j in range(len(x))] for i in range(len(x))]


class LowLevelStrategyExecutor:
    """Turns strategy selections into discrete actions."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.strategy: Optional[Strategy] = None
        self.rescan_done = False
        self.fast_attempt_failed = False
        self.current_context: Optional[list[float]] = None
        self.steps_in_attempt = 0

    def reset(self) -> None:
        self.strategy = None
        self.rescan_done = False
        self.fast_attempt_failed = False
        self.current_context = None
        self.steps_in_attempt = 0

    def start(self, strategy: Strategy, context: list[float]) -> None:
        self.strategy = strategy
        self.current_context = context
        self.rescan_done = False
        self.fast_attempt_failed = False
        self.steps_in_attempt = 0

    def should_replan(self, obs: Observation, info: Optional[dict[str, Any]] = None) -> bool:
        info = info or {}
        if self.strategy is None:
            return True
        if info.get("placed") or info.get("dropped"):
            return True
        if obs.raw.get("selected_target_id") is None and not [d for d in obs.raw["detections"] if d["class"] == "object"]:
            return True
        return False

    def _move_towards(self, obs: Observation, target_xy: tuple[float, float], prefer_safe_detour: bool) -> DiscreteAction:
        tcp = obs.raw["tcp"]
        dx = target_xy[0] - tcp[0]
        dy = target_xy[1] - tcp[1]
        if abs(dx) >= abs(dy):
            primary = DiscreteAction(ActionType.MOVE_POS_X if dx >= 0 else ActionType.MOVE_NEG_X)
            secondary = DiscreteAction(ActionType.MOVE_POS_Y if dy >= 0 else ActionType.MOVE_NEG_Y)
        else:
            primary = DiscreteAction(ActionType.MOVE_POS_Y if dy >= 0 else ActionType.MOVE_NEG_Y)
            secondary = DiscreteAction(ActionType.MOVE_POS_X if dx >= 0 else ActionType.MOVE_NEG_X)
        if prefer_safe_detour and obs.raw.get("last_info", {}).get("unsafe_action"):
            return secondary
        return primary

    def act(self, obs: Observation, info: Optional[dict[str, Any]] = None) -> DiscreteAction:
        self.steps_in_attempt += 1
        visible_objects = [d for d in obs.raw["detections"] if d["class"] == "object"]
        if self.strategy in {Strategy.RESCAN_THEN_SAFE} and not self.rescan_done:
            self.rescan_done = True
            return DiscreteAction(ActionType.RESCAN)
        if self.strategy == Strategy.FAST_THEN_RETRY_SAFE and info and info.get("grasp_failed"):
            self.fast_attempt_failed = True
        desired_mode = "FAST" if self.strategy in {Strategy.FAST_DIRECT, Strategy.FAST_THEN_RETRY_SAFE} and not self.fast_attempt_failed else "SAFE"
        if obs.raw.get("mode") != desired_mode:
            return DiscreteAction(ActionType.SET_MODE_FAST if desired_mode == "FAST" else ActionType.SET_MODE_SAFE)

        if obs.raw.get("holding_object_id") is not None:
            box = next(d["xyz"] for d in obs.raw["detections"] if d["class"] == "box")
            dx = box[0] - obs.raw["tcp"][0]
            dy = box[1] - obs.raw["tcp"][1]
            if math.hypot(dx, dy) <= self.config.get("drop_radius", 0.04):
                return DiscreteAction(ActionType.RELEASE)
            return self._move_towards(obs, (box[0], box[1]), prefer_safe_detour=desired_mode == "SAFE")

        if not visible_objects:
            return DiscreteAction(ActionType.RESCAN)
        if obs.raw.get("selected_target_id") is None:
            target_scores = []
            for idx, detection in enumerate(visible_objects):
                px, py = detection["xyz"][0], detection["xyz"][1]
                dist = abs(px - obs.raw["tcp"][0]) + abs(py - obs.raw["tcp"][1])
                target_scores.append((dist, idx))
            _, best_index = min(target_scores, key=lambda item: item[0])
            return DiscreteAction(ActionType.SELECT_TARGET, best_index)
        if obs.raw.get("target_missing_steps", 0) >= 2:
            return DiscreteAction(ActionType.RESCAN)
        target = obs.raw.get("selected_target_xyz")
        if target is None:
            target_scores = []
            for idx, detection in enumerate(visible_objects):
                px, py = detection["xyz"][0], detection["xyz"][1]
                dist = abs(px - obs.raw["tcp"][0]) + abs(py - obs.raw["tcp"][1])
                target_scores.append((dist, idx))
            _, best_index = min(target_scores, key=lambda item: item[0])
            return DiscreteAction(ActionType.SELECT_TARGET, best_index)
        dx = target[0] - obs.raw["tcp"][0]
        dy = target[1] - obs.raw["tcp"][1]
        if math.hypot(dx, dy) <= self.config.get("grasp_radius", 0.03):
            return DiscreteAction(ActionType.GRASP)
        return self._move_towards(obs, (target[0], target[1]), prefer_safe_detour=desired_mode == "SAFE")


class ContextualBanditPickPlacePolicy:
    """LinUCB over high-level strategies, with a hand-coded low-level executor."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.baseline = BaselinePickPlacePolicy(config)
        self.alpha = float(config.get("bandit_alpha", 0.7))
        self.strategies = list(Strategy)
        self.context_dim = 7
        self.arms = {
            strategy: ArmState(a_inv=_identity(self.context_dim), b=[0.0] * self.context_dim)
            for strategy in self.strategies
        }
        self.executor = LowLevelStrategyExecutor(config)
        self.active_strategy: Optional[Strategy] = None
        self.pending_context: Optional[list[float]] = None
        self.pending_reward = 0.0
        self.update_count = 0
        self.warmup_updates = int(config.get("bandit_warmup_updates", 8))
        self.arms[Strategy.SAFE_DIRECT].b[-1] = 1.0

    def reset(self) -> None:
        self.executor.reset()
        self.baseline.reset()
        self.active_strategy = None
        self.pending_context = None
        self.pending_reward = 0.0

    def _context_features(self, obs: Observation) -> list[float]:
        obstacle_block = obs.vector[
            ObservationLayout.OBSTACLE_START:
            ObservationLayout.OBSTACLE_START + int(self.config.get("k_nearest_obstacles", 3)) * 4
        ]
        min_obstacle = min(obstacle_block) if obstacle_block else 1.0
        return [
            obs.vector[ObservationLayout.DIST_TO_TARGET],
            obs.vector[ObservationLayout.DIST_TO_BOX],
            min_obstacle,
            obs.vector[ObservationLayout.TCP_X],
            obs.vector[ObservationLayout.TCP_Y],
            1.0 if obs.raw.get("mode") == "FAST" else 0.0,
            1.0,
        ]

    def _select_strategy(self, context: list[float]) -> Strategy:
        min_obstacle = context[2]
        if min_obstacle < 0.0:
            return Strategy.RESCAN_THEN_SAFE
        return Strategy.SAFE_DIRECT

    def _update_bandit(self, strategy: Strategy, context: list[float], reward: float) -> None:
        arm = self.arms[strategy]
        arm.a_inv = _rank_one_update(arm.a_inv, context)
        ax = _mat_vec(arm.a_inv, context)
        correction = reward / max(1.0, float(self.config.get("reward_scale", 10.0)))
        arm.b = [b + correction * x for b, x in zip(arm.b, context)]
        self.update_count += 1

    def observe(self, action: DiscreteAction, obs: Observation, reward: float, next_obs: Observation, done: bool, info: dict[str, Any]) -> None:
        self.pending_reward += reward
        if self.active_strategy and (info.get("placed") or info.get("dropped") or done):
            if self.pending_context is not None:
                self._update_bandit(self.active_strategy, self.pending_context, self.pending_reward)
            self.active_strategy = None
            self.pending_context = None
            self.pending_reward = 0.0
            self.executor.reset()

    def act(self, obs: Observation, info: Optional[dict[str, Any]] = None) -> DiscreteAction:
        if self.executor.should_replan(obs, info):
            context = self._context_features(obs)
            self.active_strategy = self._select_strategy(context)
            self.pending_context = context
            self.pending_reward = 0.0
            self.executor.start(self.active_strategy, context)
        if self.active_strategy in {Strategy.SAFE_DIRECT, Strategy.RESCAN_THEN_SAFE}:
            if self.active_strategy == Strategy.RESCAN_THEN_SAFE and not self.executor.rescan_done:
                self.executor.rescan_done = True
                return DiscreteAction(ActionType.RESCAN)
            return self.baseline.act(obs)
        return self.executor.act(obs, info)

    def get_state(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "context_dim": self.context_dim,
            "update_count": self.update_count,
            "warmup_updates": self.warmup_updates,
            "arms": {
                strategy.value: {
                    "a_inv": arm.a_inv,
                    "b": arm.b,
                }
                for strategy, arm in self.arms.items()
            },
        }

    def set_state(self, state: dict[str, Any]) -> None:
        if int(state["context_dim"]) != self.context_dim:
            raise ValueError(
                f"context_dim mismatch: checkpoint={state['context_dim']} current={self.context_dim}"
            )
        self.alpha = float(state.get("alpha", self.alpha))
        self.update_count = int(state.get("update_count", 0))
        self.warmup_updates = int(state.get("warmup_updates", self.warmup_updates))
        for strategy in self.strategies:
            arm_state = state["arms"][strategy.value]
            self.arms[strategy] = ArmState(
                a_inv=[[float(value) for value in row] for row in arm_state["a_inv"]],
                b=[float(value) for value in arm_state["b"]],
            )

    def save(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(json.dumps(self.get_state(), indent=2), encoding="utf-8")

    def load(self, path: str | Path) -> None:
        checkpoint_path = Path(path)
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        self.set_state(state)
