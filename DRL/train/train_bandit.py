"""Train the contextual bandit in simulation and persist a checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from ..config import load_config
from ..envs.pick_place_2d_env import PickPlace2DEnv, SimBackend2D
from ..policies.contextual_bandit import ContextualBanditPickPlacePolicy


def run_episode(env: PickPlace2DEnv, policy: ContextualBanditPickPlacePolicy, max_steps: int) -> dict[str, float]:
    obs = env.reset()
    policy.reset()
    done = False
    reward_total = 0.0
    info: dict[str, object] = {}
    steps = 0
    while not done and steps < max_steps:
        action = policy.act(obs, info)
        next_obs, reward, done, info = env.step(action)
        policy.observe(action, obs, reward, next_obs, done, info)
        obs = next_obs
        reward_total += reward
        steps += 1
    return {
        "reward": reward_total,
        "steps": float(steps),
        "placed": float(obs.raw["placed_count"]),
        "success": 1.0 if obs.raw["placed_count"] == env.config.get("num_objects", 5) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=250)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--output",
        default=str(Path("DRL") / "outputs" / "contextual_bandit.json"),
    )
    args = parser.parse_args()

    config = load_config(args.config)
    max_steps = int(config.get("max_steps", 300))
    policy = ContextualBanditPickPlacePolicy(config)

    training_rewards: list[float] = []
    for episode in range(args.episodes):
        env = PickPlace2DEnv(SimBackend2D(config, seed=episode), config)
        result = run_episode(env, policy, max_steps)
        training_rewards.append(result["reward"])
        if (episode + 1) % 25 == 0:
            recent = training_rewards[-25:]
            print(
                f"train episodes {episode - 23}-{episode + 1}: "
                f"avg_reward={mean(recent):.2f} updates={policy.update_count}"
            )

    policy.save(args.output)
    print(f"saved checkpoint: {args.output}")

    eval_results = []
    eval_policy = ContextualBanditPickPlacePolicy(config)
    eval_policy.load(args.output)
    for episode in range(args.eval_episodes):
        env = PickPlace2DEnv(SimBackend2D(config, seed=10000 + episode), config)
        eval_results.append(run_episode(env, eval_policy, max_steps))

    print(
        "eval: "
        f"success_rate={mean(item['success'] for item in eval_results):.2f} "
        f"avg_steps={mean(item['steps'] for item in eval_results):.1f} "
        f"avg_reward={mean(item['reward'] for item in eval_results):.2f}"
    )


if __name__ == "__main__":
    main()
