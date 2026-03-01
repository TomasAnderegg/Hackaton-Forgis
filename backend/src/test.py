"""
Test script for UR robot node — sends a movej command and waits for completion.

Usage (inside Docker container):
    source /opt/ros/humble/setup.bash
    uv run python src/test.py
"""

import math
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor

from nodes.ur_node import RobotNode


# Target joint positions in degrees — adjust to a safe position for your setup
TARGET_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]
VELOCITY = 0.5   # m/s
ACCEL = 0.5      # m/s²
TIMEOUT = 30.0   # seconds


def main():
    rclpy.init()
    robot = RobotNode()

    # Spin ROS in background thread
    executor = MultiThreadedExecutor()
    executor.add_node(robot)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()

    # Wait for joint state feedback
    print("Waiting for joint states...")
    timeout = 10.0
    elapsed = 0.0
    while robot.get_joint_positions() is None and elapsed < timeout:
        time.sleep(0.1)
        elapsed += 0.1

    if robot.get_joint_positions() is None:
        print("ERROR: No joint states received — is the UR driver running?")
        rclpy.shutdown()
        return

    current_deg = robot.get_joint_positions_deg()
    print(f"Current joints (deg): {[round(j, 2) for j in current_deg]}")
    print(f"Target  joints (deg): {TARGET_DEG}")

    # Convert target to radians and send movej
    target_rad = [math.radians(d) for d in TARGET_DEG]
    print("Sending movej command...")
    robot.send_movej(target_rad, accel=ACCEL, vel=VELOCITY)

    # Poll until target reached or timeout
    start = time.time()
    while time.time() - start < TIMEOUT:
        if robot.joints_at_target(target_rad, tolerance=0.02):
            print("Target reached!")
            break
        time.sleep(0.1)
    else:
        print(f"Timeout after {TIMEOUT}s — robot did not reach target")

    final_deg = robot.get_joint_positions_deg()
    print(f"Final joints (deg): {[round(j, 2) for j in final_deg]}")

    # Restore ros2_control after script execution
    print("Resending robot program to restore control...")
    robot.resend_robot_program()

    executor.shutdown()
    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
