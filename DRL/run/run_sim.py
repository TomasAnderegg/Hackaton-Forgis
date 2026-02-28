"""Run baseline and bandit controllers in simulation."""

from __future__ import annotations

from statistics import mean

from ..config import load_config
from ..envs.pick_place_2d_env import PickPlace2DEnv, SimBackend2D
from ..policies.baseline_pick_place import BaselinePickPlacePolicy
from ..policies.contextual_bandit import ContextualBanditPickPlacePolicy


def run_episode(env: PickPlace2DEnv, policy, max_steps: int) -> dict:
    obs = env.reset()
    policy.reset()
    done = False
    total_reward = 0.0
    info = {}
    steps = 0
    while not done and steps < max_steps:
        action = policy.act(obs, info) if hasattr(policy, "act") and policy.__class__.__name__.startswith("Contextual") else policy.act(obs)
        next_obs, reward, done, info = env.step(action)
        if hasattr(policy, "observe"):
            policy.observe(action, obs, reward, next_obs, done, info)
        obs = next_obs
        total_reward += reward
        steps += 1
    placed = obs.raw["placed_count"]
    success = obs.raw["placed_count"] == env.config.get("num_objects", 5)
    return {"reward": total_reward, "steps": steps, "placed": placed, "success": success}


def summarize(name: str, episodes: list[dict]) -> None:
    print(
        f"{name}: success_rate={mean(1.0 if ep['success'] else 0.0 for ep in episodes):.2f} "
        f"avg_steps={mean(ep['steps'] for ep in episodes):.1f} "
        f"avg_reward={mean(ep['reward'] for ep in episodes):.2f} "
        f"avg_placed={mean(ep['placed'] for ep in episodes):.2f}"
    )


def main() -> None:
    config = load_config()
    episodes = int(config.get("sim_eval_episodes", 50))
    max_steps = int(config.get("max_steps", 300))

    baseline_results = []
    baseline_policy = BaselinePickPlacePolicy(config)
    for seed in range(episodes):
        env = PickPlace2DEnv(SimBackend2D(config, seed=seed), config)
        baseline_results.append(run_episode(env, baseline_policy, max_steps))

    bandit_results = []
    bandit_policy = ContextualBanditPickPlacePolicy(config)
    for seed in range(min(15, episodes)):
        env = PickPlace2DEnv(SimBackend2D(config, seed=1000 + seed), config)
        run_episode(env, bandit_policy, max_steps)
    for seed in range(episodes):
        env = PickPlace2DEnv(SimBackend2D(config, seed=seed), config)
        result = run_episode(env, bandit_policy, max_steps)
        bandit_results.append(result)
        if (seed + 1) % 10 == 0:
            window = bandit_results[-10:]
            print(f"bandit episodes {seed - 8}-{seed + 1}: avg_reward={mean(ep['reward'] for ep in window):.2f}")

    summarize("baseline", baseline_results)
    summarize("bandit", bandit_results)


if __name__ == "__main__":
    main()
