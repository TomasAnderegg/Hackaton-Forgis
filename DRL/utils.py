"""Geometry and utility helpers."""

from __future__ import annotations

import math
from typing import Iterable

from .types import AABB


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

