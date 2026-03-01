"""Geometry and utility helpers."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

from .types import AABB, ActionType, DiscreteAction


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def distance_xy(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def line_intersects_aabb_xy(start: tuple[float, float], end: tuple[float, float], aabb: AABB) -> bool:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    t0 = 0.0
    t1 = 1.0
    for p, q in (
        (-dx, start[0] - aabb.min_x),
        (dx, aabb.max_x - start[0]),
        (-dy, start[1] - aabb.min_y),
        (dy, aabb.max_y - start[1]),
    ):
        if p == 0:
            if q < 0:
                return False
            continue
        t = q / p
        if p < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return False
    return True


def nearest_items(items: Iterable[tuple[float, object]], k: int) -> list[object]:
    return [item for _, item in sorted(items, key=lambda pair: pair[0])[:k]]


def choose_nearest_target_index(
    tcp_xy: tuple[float, float],
    visible_objects: list[dict[str, object]],
) -> int:
    candidates: list[tuple[float, int]] = []
    for idx, detection in enumerate(visible_objects):
        xyz = detection["xyz"]
        if not isinstance(xyz, (list, tuple)) or len(xyz) < 2:
            continue
        distance = abs(float(xyz[0]) - tcp_xy[0]) + abs(float(xyz[1]) - tcp_xy[1])
        candidates.append((distance, idx))
    if not candidates:
        raise ValueError("choose_nearest_target_index requires at least one visible object")
    _, best_index = min(candidates, key=lambda item: item[0])
    return best_index


def choose_shortest_path_move(
    current_xy: tuple[float, float],
    target_xy: tuple[float, float],
    obstacles: list[AABB],
    workspace: dict[str, float],
    delta: float,
    margin: float = 0.0,
) -> DiscreteAction:
    """
    Choose the next discrete XY move along the shortest obstacle-free grid path.

    The grid resolution is `delta`, matching the controller's move step.
    """

    x_min = float(workspace["x_min"])
    x_max = float(workspace["x_max"])
    y_min = float(workspace["y_min"])
    y_max = float(workspace["y_max"])

    expanded_obstacles = [obstacle.dilated(margin) for obstacle in obstacles]

    def to_cell(point: tuple[float, float]) -> tuple[int, int]:
        return (
            int(round((point[0] - x_min) / delta)),
            int(round((point[1] - y_min) / delta)),
        )

    def to_point(cell: tuple[int, int]) -> tuple[float, float]:
        return (
            x_min + cell[0] * delta,
            y_min + cell[1] * delta,
        )

    def is_blocked(cell: tuple[int, int]) -> bool:
        px, py = to_point(cell)
        if px < x_min or px > x_max or py < y_min or py > y_max:
            return True
        return any(obstacle.contains_xy(px, py) for obstacle in expanded_obstacles)

    start = to_cell(current_xy)
    goal = to_cell(target_xy)
    if start == goal:
        dx = target_xy[0] - current_xy[0]
        dy = target_xy[1] - current_xy[1]
        if abs(dx) >= abs(dy):
            return DiscreteAction(ActionType.MOVE_POS_X if dx >= 0 else ActionType.MOVE_NEG_X)
        return DiscreteAction(ActionType.MOVE_POS_Y if dy >= 0 else ActionType.MOVE_NEG_Y)

    neighbors = [
        ((1, 0), DiscreteAction(ActionType.MOVE_POS_X)),
        ((-1, 0), DiscreteAction(ActionType.MOVE_NEG_X)),
        ((0, 1), DiscreteAction(ActionType.MOVE_POS_Y)),
        ((0, -1), DiscreteAction(ActionType.MOVE_NEG_Y)),
    ]

    frontier: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(frontier, (0.0, start))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    cost_so_far: dict[tuple[int, int], float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal:
            break

        for (dx, dy), _action in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if is_blocked(nxt):
                continue
            new_cost = cost_so_far[current] + 1.0
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + abs(goal[0] - nxt[0]) + abs(goal[1] - nxt[1])
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal not in came_from:
        dx = target_xy[0] - current_xy[0]
        dy = target_xy[1] - current_xy[1]
        if abs(dx) >= abs(dy):
            return DiscreteAction(ActionType.MOVE_POS_X if dx >= 0 else ActionType.MOVE_NEG_X)
        return DiscreteAction(ActionType.MOVE_POS_Y if dy >= 0 else ActionType.MOVE_NEG_Y)

    step = goal
    while came_from[step] is not None and came_from[step] != start:
        step = came_from[step]

    move_delta = (step[0] - start[0], step[1] - start[1])
    for neighbor_delta, action in neighbors:
        if neighbor_delta == move_delta:
            return action

    dx = target_xy[0] - current_xy[0]
    dy = target_xy[1] - current_xy[1]
    if abs(dx) >= abs(dy):
        return DiscreteAction(ActionType.MOVE_POS_X if dx >= 0 else ActionType.MOVE_NEG_X)
    return DiscreteAction(ActionType.MOVE_POS_Y if dy >= 0 else ActionType.MOVE_NEG_Y)
