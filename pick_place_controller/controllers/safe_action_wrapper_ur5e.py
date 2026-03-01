"""Safe discrete-action wrapper over the existing URR UR robot control path."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

from ..types import AABB, ActionType, DiscreteAction, MOVE_ACTIONS
from ..utils import line_intersects_aabb_xy

URR_SRC = Path(__file__).resolve().parents[2] / "URR" / "flows" / "backend" / "src"
if str(URR_SRC) not in sys.path:
    sys.path.insert(0, str(URR_SRC))

from executors.io_executor import IOExecutor  # type: ignore  # noqa: E402
from executors.robot_executor import RobotExecutor  # type: ignore  # noqa: E402
from nodes.ur_node import RobotNode  # type: ignore  # noqa: E402


class SafeActionWrapperUR5e:
    """
    Converts discrete controller actions into URR move-linear calls.

    This wrapper does not modify URR. It imports and uses the existing
    `RobotExecutor.move_linear()` / `RobotNode.send_movel()` path directly.
    """

    def __init__(
        self,
        config: dict[str, Any],
        robot_executor: Optional[RobotExecutor] = None,
        io_executor: Optional[IOExecutor] = None,
        robot_node: Optional[RobotNode] = None,
        grasp_fn: Optional[Callable[[], Awaitable[Any] | Any]] = None,
        release_fn: Optional[Callable[[], Awaitable[Any] | Any]] = None,
    ) -> None:
        self.config = config
        self.delta = float(config["delta"])
        self.orientation = list(config.get("tool_orientation", [3.14159, 0.0, 0.0]))
        self.workspace = config["workspace"]
        self.motion_profile = config["motion_profile"]
        self.obstacle_margin = float(config["obstacle_margin"])
        self.unsafe_behavior = str(config.get("unsafe_action_behavior", "NOOP")).upper()
        self.robot_node = robot_node or RobotNode()
        self.robot_executor = robot_executor or RobotExecutor(self.robot_node)
        self.io_executor = io_executor or IOExecutor(self.robot_node, executor_type="io_robot")
        self._grasp_fn = grasp_fn
        self._release_fn = release_fn

    async def initialize(self) -> None:
        await self.robot_executor.initialize()
        await self.io_executor.initialize()

    async def _maybe_await(self, value: Awaitable[Any] | Any) -> Any:
        if asyncio.iscoroutine(value):
            return await value
        return value

    async def _move_to_pose(self, pose: list[float]) -> bool:
        speed_cfg = self.motion_profile
        return await self.robot_executor.move_linear(
            pose=pose,
            acceleration=float(speed_cfg["acceleration"]),
            velocity=float(speed_cfg["velocity"]),
        )

    def _expanded_obstacles(self, obstacles: list[AABB]) -> list[AABB]:
        return [obstacle.dilated(self.obstacle_margin) for obstacle in obstacles]

    def _within_workspace(self, x: float, y: float, z: float) -> bool:
        bounds = self.workspace
        return (
            bounds["x_min"] <= x <= bounds["x_max"]
            and bounds["y_min"] <= y <= bounds["y_max"]
            and bounds["z_min"] <= z <= bounds["z_max"]
        )

    def _is_move_safe(
        self,
        start_xyz: list[float],
        end_xyz: list[float],
        obstacles: list[AABB],
    ) -> bool:
        if not self._within_workspace(end_xyz[0], end_xyz[1], end_xyz[2]):
            return False
        for obstacle in self._expanded_obstacles(obstacles):
            if not obstacle.overlaps_z(end_xyz[2]):
                continue
            if line_intersects_aabb_xy((start_xyz[0], start_xyz[1]), (end_xyz[0], end_xyz[1]), obstacle):
                return False
        return True

    async def _ensure_height(self, tcp: list[float], target_z: float) -> tuple[bool, list[float]]:
        current = list(tcp)
        if abs(current[2] - target_z) < 1e-6:
            return True, current
        pose = [current[0], current[1], target_z, *self.orientation]
        success = await self._move_to_pose(pose)
        current[2] = target_z
        return success, current

    async def _default_grasp(self) -> None:
        await self.io_executor.set_digital_output(int(self.config["gripper"]["pin"]), True)

    async def _default_release(self) -> None:
        await self.io_executor.set_digital_output(int(self.config["gripper"]["pin"]), False)

    async def execute(
        self,
        action: DiscreteAction,
        tcp: list[float],
        obstacles: list[AABB],
    ) -> dict[str, Any]:
        current = list(tcp)
        travel_height = float(self.config["travel_height"])
        pick_height = float(self.config["pick_height"])
        drop_height = float(self.config["drop_height"])

        if action.kind in MOVE_ACTIONS:
            ok, current = await self._ensure_height(current, travel_height)
            if not ok:
                return {"success": False, "error": "failed_to_reach_travel_height"}
            dx, dy = MOVE_ACTIONS[action.kind]
            target = [current[0] + dx * self.delta, current[1] + dy * self.delta, travel_height]
            if not self._is_move_safe(current, target, obstacles):
                if self.unsafe_behavior == "TERMINATE":
                    return {"success": False, "error": "unsafe_move_terminated", "terminated": True}
                return {"success": False, "error": "unsafe_move_noop", "noop": True}
            pose = [target[0], target[1], target[2], *self.orientation]
            success = await self._move_to_pose(pose)
            return {"success": success, "tcp": target}

        if action.kind == ActionType.GRASP:
            ok, current = await self._ensure_height(current, pick_height)
            if not ok:
                return {"success": False, "error": "failed_to_reach_pick_height"}
            await self._maybe_await(self._grasp_fn() if self._grasp_fn else self._default_grasp())
            return {"success": True, "tcp": current, "carrying": True}

        if action.kind == ActionType.RELEASE:
            ok, current = await self._ensure_height(current, drop_height)
            if not ok:
                return {"success": False, "error": "failed_to_reach_drop_height"}
            await self._maybe_await(self._release_fn() if self._release_fn else self._default_release())
            return {"success": True, "tcp": current, "carrying": False}

        if action.kind in {ActionType.RESCAN, ActionType.NOOP, ActionType.SELECT_TARGET}:
            return {"success": True, "tcp": current}
        return {"success": False, "error": f"unsupported_action:{action.kind.value}"}
