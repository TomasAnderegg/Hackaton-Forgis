import asyncio
import logging
import os
import threading

import rclpy
import uvicorn
from rclpy.executors import MultiThreadedExecutor

from nodes.ur_node import RobotNode


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

rclpy.init()

def run_ros_executor(ros_executor: MultiThreadedExecutor) -> None:
    """Run the ROS 2 executor in a separate thread."""
    try:
        ros_executor.spin()
    except Exception as e:
        logger.error(f"ROS executor error: {e}")

robot = RobotNode()
# robot_executor = RobotExecutor(robot)
# io_robot_executor = IOExecutor(robot, executor_type="io_robot")
# logger.info("Using UR robot")

# ROS 2 executor with nodes
ros_executor = MultiThreadedExecutor()
ros_executor.add_node(robot)
# ros_executor.add_node(skills.camera.camera)

ros_executor = MultiThreadedExecutor()
ros_executor.add_node(robot)
# Run ROS 2 executor in background thread
ros_thread = threading.Thread(target=run_ros_executor, args=(ros_executor,), daemon=True)
ros_thread.start()
logger.info("ROS 2 executor started in background thread")

robot.send_movej([0, -1.57, 1.57, -1.57, -1.57, 0], accel=0.5, vel=0.5)