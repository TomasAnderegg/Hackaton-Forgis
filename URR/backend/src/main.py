import asyncio
import logging
import os
import threading
import rclpy
import uvicorn
from rclpy.executors import MultiThreadedExecutor

from api.app import create_app
from api.websocket import WebSocketManager
from executors import IOExecutor, RobotExecutor, CameraExecutor, HandExecutor
from flow.manager import FlowManager
from nodes.ur_node import RobotNode
from nodes.zivid_node import ZividNode
from nodes.covvi_hand_node import CovviHandNode

# Intentamos importar Dobot de forma segura por si faltan los mensajes
try:
    from nodes.dobot_nova5_node import DobotNova5Node
    from executors import DobotNova5Executor
    DOBOT_AVAILABLE = True
except ImportError:
    DOBOT_AVAILABLE = False
    print("⚠️ AVISO: Mensajes de Dobot no encontrados. El modo Dobot no funcionará.")

# Import skills
import skills.robot
import skills.io
import skills.camera
import skills.hand

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_ros_executor(ros_executor: MultiThreadedExecutor) -> None:
    try:
        ros_executor.spin()
    except Exception as e:
        logger.error(f"ROS executor error: {e}")

def main():
    rclpy.init()
    robot_type = os.environ.get("ROBOT_TYPE", "ur").lower()
    
    if robot_type == "dobot" and DOBOT_AVAILABLE:
        robot = DobotNova5Node()
        robot_executor = DobotNova5Executor(robot)
        io_robot_executor = None
        logger.info("Using DOBOT Nova 5 robot")
    else:
        if robot_type == "dobot" and not DOBOT_AVAILABLE:
            logger.error("Modo Dobot solicitado pero no disponible. Usando UR por defecto.")
        robot = RobotNode()
        robot_executor = RobotExecutor(robot)
        io_robot_executor = IOExecutor(robot, executor_type="io_robot")
        logger.info("Using UR robot")

    camera = ZividNode()
    hand = CovviHandNode()
    ws_manager = WebSocketManager()
    camera_executor = CameraExecutor(camera, ws_manager)
    hand_executor = HandExecutor(hand)

    executors = {"robot": robot_executor, "camera": camera_executor, "hand": hand_executor}
    if io_robot_executor: executors["io_robot"] = io_robot_executor

    flow_manager = FlowManager(executors=executors, ws_manager=ws_manager, flows_dir=os.environ.get("FLOWS_DIR", "./flows"))
    app = create_app(flow_manager, ws_manager, robot, camera_executor, io_robot_executor, hand_executor)

    ros_executor = MultiThreadedExecutor()
    ros_executor.add_node(robot)
    ros_executor.add_node(camera)
    ros_executor.add_node(hand)

    ros_thread = threading.Thread(target=run_ros_executor, args=(ros_executor,), daemon=True)
    ros_thread.start()

    async def init_executors():
        await robot_executor.initialize()
        if io_robot_executor: await io_robot_executor.initialize()
        await camera_executor.initialize()
        await hand_executor.initialize()

    asyncio.get_event_loop().run_until_complete(init_executors())
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
    finally:
        ros_executor.shutdown()
        robot.destroy_node()
        camera.destroy_node()
        hand.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()