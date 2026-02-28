"""Run the adaptive pick-and-place policy against the real URR-backed wrapper."""

from __future__ import annotations

import argparse
import asyncio
import importlib
from typing import Any

from ..config import load_config
from ..controllers.safe_action_wrapper_ur5e import SafeActionWrapperUR5e
from ..envs.pick_place_2d_env import RealBackend, build_observation
from ..policies.baseline_pick_place import BaselinePickPlacePolicy
from ..policies.contextual_bandit import ContextualBanditPickPlacePolicy


def _load_callable(spec: str):
    module_name, _, attr = spec.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=["baseline", "bandit"], default="baseline")
    parser.add_argument("--config", default=None)
    parser.add_argument("--perception", required=True, help="Import path module:function returning current payload dict")
    args = parser.parse_args()

    config = load_config(args.config)
    wrapper = SafeActionWrapperUR5e(config)
    await wrapper.initialize()
    perception_fn = _load_callable(args.perception)
    backend = RealBackend(config=config, safe_wrapper=wrapper, perception_fn=perception_fn)
    policy: Any = BaselinePickPlacePolicy(config) if args.policy == "baseline" else ContextualBanditPickPlacePolicy(config)
    payload = backend.reset()
    obs = build_observation(payload, config, int(config.get("k_nearest_obstacles", 3)))
    policy.reset()
    info = {}
    while True:
        action = policy.act(obs, info) if args.policy == "bandit" else policy.act(obs)
        payload, done, info = await backend.step_async(action)
        next_obs = build_observation(payload, config, int(config.get("k_nearest_obstacles", 3)))
        if hasattr(policy, "observe"):
            policy.observe(action, obs, 0.0, next_obs, done, info)
        obs = next_obs
        print({"action": action.kind.value, "info": info, "phase": obs.raw.get("phase"), "placed": obs.raw.get("placed_count")})
        if done:
            break


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
