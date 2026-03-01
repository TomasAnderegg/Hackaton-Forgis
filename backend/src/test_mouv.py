"""
Test script: detect boxes with camera, move UR robot above each one.

Usage (inside Docker container):
    source /opt/ros/humble/setup.bash
    uv run python src/test_mouv.py

Adjust Z_HOVER_M to the height (in meters) above the table in the robot frame.
"""

import json
import logging
import math
import threading
import time
import urllib.request

import rclpy
from rclpy.executors import MultiThreadedExecutor

from nodes.ur_node import RobotNode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

# Height above the workspace in meters (Z in robot frame).
# Measure: place TCP just above a box, read Z on teach pendant (meters).
Z_HOVER_M = 0.10

# TCP orientation for pointing down [rx, ry, rz] in radians.
ORIENTATION = [math.pi, 0.0, 0.0]

# Safe joint position to go first (degrees) — avoids collisions while moving.
SAFE_JOINTS_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]

VELOCITY   = 0.15  # m/s
ACCEL      = 0.3   # m/s²
HOVER_TIME = 1.5   # seconds above each box

DETECT_URL = "http://localhost:8000/api/camera/detect"
DETECT_BODY = json.dumps({
    "object_class": "cardboard box",
    "confidence_threshold": 0.3,
}).encode()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_detections() -> list[dict]:
    """Call the backend detection API and return all detected boxes."""
    req = urllib.request.Request(
        DETECT_URL,
        data=DETECT_BODY,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())
    return data.get("detections", [])


def wait_for_stable(robot: RobotNode, timeout: float = 30.0) -> bool:
    """Wait until joint velocities settle (robot stopped moving)."""
    prev = None
    stable = 0
    t0 = time.time()
    while time.time() - t0 < timeout:
        curr = robot.get_joint_positions()
        if curr and prev:
            if max(abs(c - p) for c, p in zip(curr, prev)) < 0.002:
                stable += 1
                if stable >= 5:
                    return True
            else:
                stable = 0
        prev = curr
        time.sleep(0.1)
    return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    robot = RobotNode()

    ros_executor = MultiThreadedExecutor()
    ros_executor.add_node(robot)
    ros_thread = threading.Thread(target=ros_executor.spin, daemon=True)
    ros_thread.start()
    logger.info("ROS 2 executor started")

    # Wait for joint state feedback
    logger.info("Waiting for joint states...")
    t0 = time.time()
    while robot.get_joint_positions() is None and time.time() - t0 < 10:
        time.sleep(0.1)

    if robot.get_joint_positions() is None:
        logger.error("No joint states — is the UR driver running?")
        rclpy.shutdown()
        return

    logger.info(f"Joints: {[round(j, 1) for j in robot.get_joint_positions_deg()]} deg")

    # ── 1. Go to safe position ────────────────────────────────────────────────
    logger.info("Moving to safe position...")
    safe_rad = [math.radians(d) for d in SAFE_JOINTS_DEG]
    robot.send_movej(safe_rad, accel=0.5, vel=0.5)
    wait_for_stable(robot)
    robot.resend_robot_program()
    time.sleep(0.5)

    # ── 2. Detect boxes ───────────────────────────────────────────────────────
    logger.info("Detecting boxes...")
    try:
        detections = get_detections()
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        rclpy.shutdown()
        return

    # Filter to calibrated detections (have robot X, Y)
    boxes = [d for d in detections if d.get("robot_x") is not None]

    if not boxes:
        logger.warning("No calibrated box detections found")
        rclpy.shutdown()
        return

    # Sort left-to-right by robot X
    boxes.sort(key=lambda d: d["robot_x"])
    logger.info(f"Found {len(boxes)} box(es)")

    # ── 3. Move above each box ────────────────────────────────────────────────
    for i, box in enumerate(boxes, start=1):
        rx_m = box["robot_x"] / 1000.0  # mm → meters
        ry_m = box["robot_y"] / 1000.0
        pose = [rx_m, ry_m, Z_HOVER_M] + ORIENTATION

        logger.info(
            f"Box #{i}/{len(boxes)}: "
            f"robot=({box['robot_x']:.1f}, {box['robot_y']:.1f}) mm  "
            f"conf={box['confidence']:.0%}  "
            f"→ movel({rx_m:.3f}, {ry_m:.3f}, {Z_HOVER_M}) m"
        )

        robot.send_movel(pose, accel=ACCEL, vel=VELOCITY)
        wait_for_stable(robot)
        robot.resend_robot_program()

        logger.info(f"Above box #{i} — hovering {HOVER_TIME}s")
        time.sleep(HOVER_TIME)

    # ── 4. Return to safe position ────────────────────────────────────────────
    logger.info("Returning to safe position...")
    robot.send_movej(safe_rad, accel=0.5, vel=0.5)
    wait_for_stable(robot)
    robot.resend_robot_program()

    logger.info("Done.")
    ros_executor.shutdown()
    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
