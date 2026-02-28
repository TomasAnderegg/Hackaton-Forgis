"""Smoke tests for the adaptive pick-and-place simulation stack."""

from __future__ import annotations

import unittest

from DRL.config import load_config
from DRL.envs.pick_place_2d_env import PickPlace2DEnv, SimBackend2D
from DRL.policies.baseline_pick_place import BaselinePickPlacePolicy
from DRL.policies.contextual_bandit import ContextualBanditPickPlacePolicy


def _run_episode(env, policy, max_steps):
    obs = env.reset()
    policy.reset()
    done = False
    info = {}
    while not done and obs.raw["step_count"] < max_steps:
        action = policy.act(obs, info) if isinstance(policy, ContextualBanditPickPlacePolicy) else policy.act(obs)
        next_obs, reward, done, info = env.step(action)
        if hasattr(policy, "observe"):
            policy.observe(action, obs, reward, next_obs, done, info)
        obs = next_obs
    return obs.raw["placed_count"] == env.config.get("num_objects", 5)


class SmokeTests(unittest.TestCase):
    def test_baseline_and_bandit(self) -> None:
        config = load_config()
        max_steps = int(config.get("max_steps", 300))

        baseline_successes = 0
        baseline_policy = BaselinePickPlacePolicy(config)
        eval_seeds = list(range(500, 510))
        for seed in range(10):
            env = PickPlace2DEnv(SimBackend2D(config, seed=seed), config)
            baseline_successes += 1 if _run_episode(env, baseline_policy, max_steps) else 0

        bandit_policy = ContextualBanditPickPlacePolicy(config)
        for seed in range(15):
            env = PickPlace2DEnv(SimBackend2D(config, seed=200 + seed), config)
            _run_episode(env, bandit_policy, max_steps)

        baseline_eval_successes = 0
        for seed in eval_seeds:
            env = PickPlace2DEnv(SimBackend2D(config, seed=seed), config)
            baseline_eval_successes += 1 if _run_episode(env, baseline_policy, max_steps) else 0

        bandit_successes = 0
        for seed in eval_seeds:
            env = PickPlace2DEnv(SimBackend2D(config, seed=seed), config)
            bandit_successes += 1 if _run_episode(env, bandit_policy, max_steps) else 0

        self.assertGreater(baseline_successes / 10.0, 0.70)
        self.assertGreaterEqual(bandit_successes, baseline_eval_successes)


if __name__ == "__main__":
    unittest.main()
