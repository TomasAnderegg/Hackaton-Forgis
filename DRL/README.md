# Adaptive Pick-and-Place

This package adds an ML control layer for adaptive pick-and-place without editing the existing URR control stack. The problem is to place `N` objects into a box quickly while handling noisy detections, temporary misses, grasp failures, and optional dynamic obstacles. A fixed rule set is brittle here because the best behavior changes with scene risk; the ML layer selects strategies that trade off speed, rescans, and safe-vs-fast execution.

Architecture:
- `policy -> low-level executor -> safe_action_wrapper_ur5e -> URR`
- `SimBackend2D` provides fast training and evaluation under noise/failure dynamics.
- `RealBackend` consumes live detections and executes discrete actions through the safe wrapper.

The real wrapper imports URR from its existing location and uses the existing URR motion path (`RobotExecutor.move_linear()` / `RobotNode.send_movel()`). URR is not edited.

Files:
- `DRL/envs/pick_place_2d_env.py`: gym-like environment, `SimBackend2D`, `RealBackend`
- `DRL/controllers/safe_action_wrapper_ur5e.py`: safety, height management, URR integration
- `DRL/policies/baseline_pick_place.py`: always-safe baseline
- `DRL/policies/contextual_bandit.py`: contextual bandit over strategy choices
- `DRL/run/run_sim.py`: sim comparison runner
- `DRL/run/run_real.py`: real runner
- `DRL/tests/test_smoke.py`: smoke test

Run sim:

```bash
python -m DRL.run.run_sim
```

Train the contextual bandit in simulation and save a checkpoint:

```bash
python -m DRL.train.train_bandit --episodes 250 --output DRL/outputs/contextual_bandit.json
```

Run real:

```bash
python -m DRL.run.run_real --policy baseline --perception your_module:your_function
python -m DRL.run.run_real --policy bandit --perception your_module:your_function
```

The perception callable must return a payload shaped like:

```python
{
    "detections": [...],
    "tcp": [x, y, z],
    "phase": "SEARCHING",
    "placed_count": 0,
    "remaining_count": 5,
}
```

Tune behavior in `DRL/config/default.yaml`:
- motion delta, heights, workspace bounds
- detection noise and miss probability
- safe/fast grasp failure probabilities
- obstacle margins and speed profiles
- shaping rewards and unsafe-action behavior

The contextual bandit chooses among:
- `FAST_DIRECT`
- `SAFE_DIRECT`
- `RESCAN_THEN_SAFE`
- `FAST_THEN_RETRY_SAFE`

The low-level executor then emits discrete actions such as move, select target, rescan, grasp, and release.
