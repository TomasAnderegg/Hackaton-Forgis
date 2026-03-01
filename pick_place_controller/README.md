# Pick-and-Place Controller

This package adds a deterministic control layer for pick-and-place without editing the existing URR control stack. The controller is intentionally simple: fixed-period rescans, nearest-target selection, and shortest-path XY moves around obstacles.

Architecture:
- `controller -> safe_action_wrapper_ur5e -> URR`
- `Backend` consumes live detections and executes discrete actions through the safe wrapper.

The real wrapper imports URR from its existing location and uses the existing URR motion path (`RobotExecutor.move_linear()` / `RobotNode.send_movel()`). URR is not edited.

Files:
- `pick_place_controller/controller.py`: single deterministic controller
- `pick_place_controller/envs/pick_place_2d_env.py`: observation builder and `Backend`
- `pick_place_controller/controllers/safe_action_wrapper_ur5e.py`: safety, height management, URR integration
- `pick_place_controller/run/run_real.py`: real runner

Run real:

```bash
python -m pick_place_controller.run.run_real --perception your_module:your_function
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

Tune behavior in `pick_place_controller/config/default.yaml`:
- motion delta, heights, workspace bounds
- obstacle margins and motion profile
- fixed rescan interval
- missing-target rescan threshold
- unsafe-action behavior

The controller does three simple things:
- rescans every fixed number of steps
- selects the nearest visible object
- moves one 2 cm step along the shortest grid path that avoids the obstacle AABBs
