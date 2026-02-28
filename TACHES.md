# Full Pipeline — Pick & Place with Vision + RL

## Overview

```
[Zivid Camera] → YOLO detects object → RL selects pick offset
      → DOBOT grasps object → places in box → Q-table updated
```

**Active flow:** `dobot_rl_pick_place` (auto-loop, learns after each pick)

---

---

# PERSON 1 — Robot & Calibration

## Prerequisites
- Docker running: `docker compose up`
- Zivid server running: `python assets/zivid-server/zivid_server.py --port 8766`

## Step 1 — Read current robot position

```bash
curl http://localhost:8000/api/robot/state
```
Returns the 6 joint angles in degrees. **Write these down.**

## Step 2 — Calibrate the 3 key positions

Jog the robot manually (via DOBOT interface or teach pendant) to each position, then read the joints each time.

| Position | Description | Variable in the flow |
|----------|-------------|----------------------|
| **midair** | Safe position in the air, between pick and place | `midair_joints` |
| **pick** | Just above the grasping zone | `base_pick_joints` |
| **place** | Above the destination box | `place_joints` |

## Step 3 — Update the flow

Open `backend/src/data/flow/dobot_rl_pick_place.json`, replace the values in `"variables"`:

```json
"variables": {
  "midair_joints":    [J0, J1, J2, J3, J4, J5],
  "base_pick_joints": [J0, J1, J2, J3, J4, J5],
  "place_joints":     [J0, J1, J2, J3, J4, J5]
}
```

## Step 4 — Test the gripper alone

```bash
curl -X POST http://localhost:8000/api/flows/dobot_test_pick/start
docker logs forgis-backend --tail 20
```

Verify that the gripper opens and closes correctly.

## Step 5 — Test movement without vision

```bash
curl -X POST http://localhost:8000/api/flows/dobot_test_pick/start
```

Verify that the arm moves to `midair` then to `pick` without collision.

## Deliverable for P3
- `dobot_rl_pick_place.json` with real joint values
- Confirmation that the gripper works (open index=2, close index=1)

---

---

# PERSON 2 — Vision & YOLO Detection

## Prerequisites
- Zivid server running: `python assets/zivid-server/zivid_server.py --port 8766`
- Docker running: `docker compose up`

## Step 1 — Verify the video stream

Open `http://localhost` in the browser.
The camera feed should appear in the interface.

If no image:
```bash
docker logs forgis-backend -f | grep zivid
```

## Step 2 — Identify the object class

The YOLO model used is `roboflow_logistics.pt` (trained on logistics objects).

Test detection by launching a flow containing `get_bounding_box`:
```bash
curl -X POST http://localhost:8000/api/flows/dobot_pick_place_vision/start
docker logs forgis-backend -f | grep "detect_objects"
```

Logs show all detected classes:
```
detect_objects: raw box class='carton' conf=0.823
detect_objects: raw box class='bottle' conf=0.412
```

**Identify the class that matches the real object.**

## Step 3 — Adjust detection parameters

In `backend/src/data/flow/dobot_rl_pick_place.json`, step `detect_object`:

```json
"params": {
  "object_class": "<class_found_in_step_2>",
  "confidence_threshold": 0.4
}
```

Same thing in the `rl_feedback` step (skill `rl_update`):
```json
"params": {
  "object_class": "<same_class>",
  "confidence_threshold": 0.4
}
```

## Step 4 — Verify detection returns `found: true`

```bash
docker logs forgis-backend -f | grep "found"
```

Should see: `detect_objects: ACCEPTED bbox x=... y=... w=... h=...`

## Step 5 — Check Zivid frame dimensions

```bash
curl http://localhost:8000/api/camera/state
```

Note `frame_size.width` and `frame_size.height`.
Update in `dobot_rl_pick_place.json`, step `rl_move_to_pick`:

```json
"params": {
  "base_pick_joints": "{{base_pick_joints}}",
  "frame_width": <width>,
  "frame_height": <height>
}
```

## Deliverable for P3
- Correct `object_class` + adjusted `confidence_threshold` in `dobot_rl_pick_place.json`
- Confirmation that `found: true` appears in logs when the object is placed under the camera
- Correct frame dimensions in the flow

---

---

# PERSON 3 — Flow, Integration & Demo

## Prerequisites
- Wait for deliverables from P1 (joints) and P2 (object_class)
- Have Azure OpenAI credentials (optional, for LLM generation)

## Step 1 — Validate Docker is running correctly

```bash
docker logs forgis-backend --tail 30
docker logs dobot-driver --tail 30
```

No critical errors = OK.

## Step 2 — Verify RL skills are loaded

```bash
curl http://localhost:8000/api/skills | python -m json.tool | grep "rl_"
```

Should show `rl_pick` and `rl_update`.

## Step 3 — (Optional) Configure Azure OpenAI for LLM generation

In `.env`:
```
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
```

Test:
```bash
curl -X POST http://localhost:8000/api/flows/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Pick a box with vision and place it in the bin using RL"}'
```

## Step 4 — Integrate values from P1 and P2

Once P1 and P2 have their values, update `dobot_rl_pick_place.json`:
- `variables.midair_joints`, `variables.base_pick_joints`, `variables.place_joints` (P1 values)
- `object_class` in `detect_object` and `rl_feedback` (P2 value)
- `frame_width`, `frame_height` in `rl_move_to_pick` (P2 value)

## Step 5 — Rebuild Docker with new values

```bash
docker compose down
docker compose up --build
```

## Step 6 — First RL flow run

```bash
curl -X POST http://localhost:8000/api/flows/dobot_rl_pick_place/start
docker logs forgis-backend -f
```

Expected logs:
```
RL explore: state=4 action=0 ε=0.30
rl_pick: object found at (960, 600) → state 4
DobotNova5Executor: Joint target reached
rl_update: camera strategy → 0 detections → SUCCESS → reward=1.0
RL update: state=4 action=0 reward=1.0  Q: 0.000 → 0.500
```

## Step 7 — Monitor RL learning

```bash
# In another terminal, every 5 seconds:
watch -n 5 "curl -s http://localhost:8000/api/skills/rl/qtable | python -m json.tool"
```

Watch: `epsilon` decreasing, `episode_count` increasing, `q_table` filling up.

## Step 8 — Stop after convergence (~20 picks)

```bash
curl -X POST http://localhost:8000/api/flows/finish
```

The Q-table is saved automatically. On next launch it is reloaded.

## Final deliverable
- End-to-end loop demo: detect → adaptive pick → place → RL update
- Q-table visible via `curl http://localhost:8000/api/skills/rl/qtable`
- Success rate visible in logs

---

---

# Summary of modified files

| File | Who | What |
|------|-----|-------|
| `backend/src/data/flow/dobot_rl_pick_place.json` | P1 + P2 | Calibrated joints + object_class + frame_size |
| `.env` | P3 | Azure OpenAI credentials (optional) |

# Quick monitoring commands

```bash
# Robot state
curl http://localhost:8000/api/robot/state

# Camera state
curl http://localhost:8000/api/camera/state

# RL Q-table
curl http://localhost:8000/api/skills/rl/qtable

# Current flow status
curl http://localhost:8000/api/flows/status

# Live logs
docker logs forgis-backend -f

# Stop the flow
curl -X POST http://localhost:8000/api/flows/abort
```
