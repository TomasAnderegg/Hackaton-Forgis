"""
Q-learning agent for online pick position adaptation.

The robot divides the camera field-of-view into a 3×3 grid.
Each grid cell is a *state*. For each state the agent learns which
joint-space offset (action) maximises pick success.

Training happens live on the real robot:
  - reward = +1  if the gripper closes on the object (successful pick)
  - reward = -1  if the gripper closes on air (missed pick)

With epsilon-greedy exploration the agent converges in ~20 episodes.
"""

import json
import logging
import math
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Tuneable constants ──────────────────────────────────────────────────────

GRID_ROWS = 3          # rows in the image grid
GRID_COLS = 3          # cols in the image grid
N_STATES  = GRID_ROWS * GRID_COLS   # 9 states

# Each action is a (Δj0, Δj1, Δj2) offset in degrees applied to pick_joints.
# Keep offsets small to stay safe on real hardware.
ACTIONS: list[tuple[float, float, float]] = [
    ( 0.0,  0.0,  0.0),   # 0 – no correction
    ( 2.0,  0.0,  0.0),   # 1 – base +2°
    (-2.0,  0.0,  0.0),   # 2 – base -2°
    ( 0.0,  2.0, -2.0),   # 3 – shoulder forward
    ( 0.0, -2.0,  2.0),   # 4 – shoulder back
    ( 2.0,  2.0,  0.0),   # 5 – diagonal ++
    (-2.0, -2.0,  0.0),   # 6 – diagonal --
    ( 2.0, -2.0,  0.0),   # 7 – diagonal +-
    (-2.0,  2.0,  0.0),   # 8 – diagonal -+
]
N_ACTIONS = len(ACTIONS)

ALPHA   = 0.5    # learning rate
GAMMA   = 0.9    # discount factor (single-step task → less important)
EPSILON = 0.3    # exploration rate (drops over time)
EPSILON_MIN   = 0.05
EPSILON_DECAY = 0.95   # multiplied after each episode

QTABLE_PATH = Path("/app/rl_qtable.json")   # persisted inside the container


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class Episode:
    """Stores the context of one pick attempt."""
    state:  int
    action: int
    joints_used: list[float]


@dataclass
class PickPositionLearner:
    """
    Online Q-learning agent that improves pick-joint offsets over time.

    Usage
    -----
    learner = PickPositionLearner(base_pick_joints=[...])

    # Before the pick move
    state   = learner.bbox_to_state(bbox_cx, bbox_cy, frame_w, frame_h)
    joints  = learner.select_action(state)

    # After checking whether the pick succeeded
    learner.update(reward=+1)   # or -1
    learner.save()
    """

    base_pick_joints: list[float]          # 6-DOF base position in degrees
    q: list[list[float]] = field(init=False)
    epsilon: float = field(init=False, default=EPSILON)
    _last_episode: Optional[Episode] = field(init=False, default=None)
    episode_count: int = field(init=False, default=0)

    def __post_init__(self):
        self.q = [[0.0] * N_ACTIONS for _ in range(N_STATES)]
        self.epsilon = EPSILON
        self._last_episode = None
        self.episode_count = 0
        self._try_load()

    # ── State mapping ───────────────────────────────────────────────────────

    def bbox_to_state(
        self,
        cx: float, cy: float,
        frame_w: float, frame_h: float,
    ) -> int:
        """Map object centre pixel to a grid-cell state index (0-8)."""
        col = min(int(cx / frame_w * GRID_COLS), GRID_COLS - 1)
        row = min(int(cy / frame_h * GRID_ROWS), GRID_ROWS - 1)
        state = row * GRID_COLS + col
        logger.debug("RL state: bbox_centre=(%.0f,%.0f) → grid(%d,%d) → state %d",
                     cx, cy, col, row, state)
        return state

    # ── Action selection ────────────────────────────────────────────────────

    def select_action(self, state: int) -> list[float]:
        """
        Epsilon-greedy policy.

        Returns the 6-DOF joint target (base_pick_joints + offset).
        Stores episode context for the next update() call.
        """
        if random.random() < self.epsilon:
            action = random.randint(0, N_ACTIONS - 1)
            logger.info("RL explore: state=%d action=%d ε=%.2f", state, action, self.epsilon)
        else:
            action = int(max(range(N_ACTIONS), key=lambda a: self.q[state][a]))
            logger.info("RL exploit: state=%d best_action=%d Q=%.3f",
                        state, action, self.q[state][action])

        offset = ACTIONS[action]
        joints = list(self.base_pick_joints)
        joints[0] += offset[0]   # base joint
        joints[1] += offset[1]   # shoulder
        joints[2] += offset[2]   # elbow

        self._last_episode = Episode(state=state, action=action, joints_used=joints)
        return joints

    # ── Online update ───────────────────────────────────────────────────────

    def update(self, reward: float) -> None:
        """
        Update Q-table after observing the reward for the last action.

        Call this right after checking pick success:
            learner.update(+1.0)   # object grasped
            learner.update(-1.0)   # missed / dropped
        """
        if self._last_episode is None:
            logger.warning("RL update called before any episode — skipped")
            return

        s = self._last_episode.state
        a = self._last_episode.action
        old_q = self.q[s][a]

        # Q-learning update (no next state — single-step task)
        self.q[s][a] = old_q + ALPHA * (reward + GAMMA * max(self.q[s]) - old_q)

        self.episode_count += 1
        self.epsilon = max(EPSILON_MIN, self.epsilon * EPSILON_DECAY)

        logger.info(
            "RL update: state=%d action=%d reward=%.1f  Q: %.3f → %.3f  "
            "(episode %d, ε=%.3f)",
            s, a, reward, old_q, self.q[s][a], self.episode_count, self.epsilon,
        )
        self._last_episode = None

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist Q-table and metadata to disk."""
        try:
            data = {
                "q": self.q,
                "epsilon": self.epsilon,
                "episode_count": self.episode_count,
                "base_pick_joints": self.base_pick_joints,
            }
            QTABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
            QTABLE_PATH.write_text(json.dumps(data, indent=2))
            logger.info("RL Q-table saved (%d episodes)", self.episode_count)
        except Exception as e:
            logger.warning("RL save failed: %s", e)

    def _try_load(self) -> None:
        """Load Q-table from disk if it exists (resume across restarts)."""
        try:
            if QTABLE_PATH.exists():
                data = json.loads(QTABLE_PATH.read_text())
                self.q = data["q"]
                self.epsilon = data.get("epsilon", EPSILON)
                self.episode_count = data.get("episode_count", 0)
                logger.info(
                    "RL Q-table loaded from %s (%d prior episodes)",
                    QTABLE_PATH, self.episode_count,
                )
        except Exception as e:
            logger.warning("RL load failed (%s) — starting fresh", e)

    # ── Diagnostics ─────────────────────────────────────────────────────────

    def get_summary(self) -> dict:
        """Return a JSON-serialisable summary for the REST API."""
        best_actions = [int(max(range(N_ACTIONS), key=lambda a: self.q[s][a]))
                        for s in range(N_STATES)]
        best_offsets = [ACTIONS[a] for a in best_actions]
        return {
            "episode_count": self.episode_count,
            "epsilon": round(self.epsilon, 4),
            "grid": f"{GRID_ROWS}x{GRID_COLS}",
            "n_actions": N_ACTIONS,
            "best_actions_per_state": best_actions,
            "best_offsets_per_state": best_offsets,
            "q_table": [[round(v, 3) for v in row] for row in self.q],
        }
