"""Shared types for the adaptive pick-and-place controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Phase(str, Enum):
    SEARCHING = "SEARCHING"
    APPROACHING = "APPROACHING"
    GRASPING = "GRASPING"
    CARRYING = "CARRYING"
    DROPPING = "DROPPING"
    FINISHED = "FINISHED"


class ActionType(str, Enum):
    MOVE_POS_X = "MOVE_POS_X"
    MOVE_NEG_X = "MOVE_NEG_X"
    MOVE_POS_Y = "MOVE_POS_Y"
    MOVE_NEG_Y = "MOVE_NEG_Y"
    GRASP = "GRASP"
    RELEASE = "RELEASE"
    RESCAN = "RESCAN"
    SELECT_TARGET = "SELECT_TARGET"
    NOOP = "NOOP"


@dataclass(frozen=True)
class DiscreteAction:
    kind: ActionType
    index: Optional[int] = None


@dataclass
class AABB:
    min_x: float
    min_y: float
    min_z: float
    max_x: float
    max_y: float
    max_z: float

    @classmethod
    def from_detection(cls, detection: dict[str, Any]) -> "AABB":
        if "min_xyz" in detection and "max_xyz" in detection:
            min_x, min_y, min_z = detection["min_xyz"]
            max_x, max_y, max_z = detection["max_xyz"]
            return cls(min_x, min_y, min_z, max_x, max_y, max_z)
        center = detection["xyz"]
        size = detection.get("size", detection.get("extents", [0.05, 0.05, 0.10]))
        hx, hy, hz = size[0] / 2.0, size[1] / 2.0, size[2] / 2.0
        return cls(center[0] - hx, center[1] - hy, center[2] - hz, center[0] + hx, center[1] + hy, center[2] + hz)

    def dilated(self, margin: float) -> "AABB":
        return AABB(
            self.min_x - margin,
            self.min_y - margin,
            self.min_z - margin,
            self.max_x + margin,
            self.max_y + margin,
            self.max_z + margin,
        )

    def contains_xy(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def overlaps_z(self, z: float) -> bool:
        return self.min_z <= z <= self.max_z


@dataclass
class Observation:
    vector: list[float]
    raw: dict[str, Any]


class ObservationLayout:
    TCP_X = 0
    TCP_Y = 1
    TARGET_X = 2
    TARGET_Y = 3
    BOX_X = 4
    BOX_Y = 5
    PHASE_START = 6
    PHASE_LEN = 6
    REMAINING = PHASE_START + PHASE_LEN
    PLACED = REMAINING + 1
    DIST_TO_TARGET = PLACED + 1
    DIST_TO_BOX = DIST_TO_TARGET + 1
    OBSTACLE_START = DIST_TO_BOX + 1


@dataclass
class PolicyStep:
    action: DiscreteAction
    metadata: dict[str, Any] = field(default_factory=dict)


MOVE_ACTIONS = {
    ActionType.MOVE_POS_X: (1.0, 0.0),
    ActionType.MOVE_NEG_X: (-1.0, 0.0),
    ActionType.MOVE_POS_Y: (0.0, 1.0),
    ActionType.MOVE_NEG_Y: (0.0, -1.0),
}
