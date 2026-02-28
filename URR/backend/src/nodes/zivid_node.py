"""ROS 2 node that receives Zivid frames from the Windows zivid_server.py via WebSocket."""

import asyncio
import logging
import os
import threading
from typing import Optional

import cv2
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Point
from ultralytics import YOLO # Added for YOLO

logger = logging.getLogger(__name__)

MAX_QUEUE_SIZE = 1  # Always use freshest frame


class ZividNode(Node):
    def __init__(self):
        super().__init__("zivid_node")
        # Added Publisher
        self.coordinates_pub = self.create_publisher(Point, '/camera/box_coordinates', 10)
        # Added Model Loading
        self.model = YOLO("roboflow_logistics.pt") 

        self._frame: Optional[np.ndarray] = None
        self._frame_jpeg: Optional[bytes] = None
        self._frame_lock = threading.Lock()
        self._recv_count = 0

        self.ws_host = os.environ.get("CAMERA_BRIDGE_HOST", "host.docker.internal")
        self.ws_port = int(os.environ.get("ZIVID_SERVER_PORT", "8766"))
        self.ws_uri = f"ws://{self.ws_host}:{self.ws_port}"

        self._connected = False
        self._reconnect_delay = 2.0
        self._running = True
        self._frame_queue: Optional[asyncio.Queue] = None

        self._ws_thread = threading.Thread(target=self._run_ws_client, daemon=True)
        self._ws_thread.start()

        self.get_logger().info(f"ZividNode initialized — connecting to {self.ws_uri}")

    def _run_ws_client(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._ws_client_loop())
        except Exception as e:
            self.get_logger().error(f"WebSocket client loop error: {e}")
        finally:
            loop.close()

    async def _ws_client_loop(self) -> None:
        try:
            import websockets
        except ImportError:
            self.get_logger().error("websockets package not installed")
            return

        while self._running:
            try:
                self.get_logger().info(f"Connecting to Zivid server at {self.ws_uri}")
                async with websockets.connect(self.ws_uri) as websocket:
                    self._connected = True
                    self.get_logger().info("Connected to Zivid server")
                    self._frame_queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
                    processor_task = asyncio.create_task(self._process_loop())
                    try:
                        async for message in websocket:
                            if not self._running: break
                            if self._frame_queue.full():
                                try: self._frame_queue.get_nowait()
                                except asyncio.QueueEmpty: pass
                            try: self._frame_queue.put_nowait(message)
                            except asyncio.QueueFull: pass
                    finally:
                        processor_task.cancel()
                        try: await processor_task
                        except asyncio.CancelledError: pass
            except Exception as e:
                self.get_logger().warn(f"Zivid server not available ({e})", throttle_duration_sec=10.0)
            finally:
                self._connected = False
                self._frame_queue = None
            if self._running: await asyncio.sleep(self._reconnect_delay)

    async def _process_loop(self) -> None:
        while self._running and self._frame_queue:
            try:
                jpeg_data = await asyncio.wait_for(self._frame_queue.get(), timeout=0.5)
                self._decode_frame(jpeg_data)
            except asyncio.TimeoutError: continue
            except asyncio.CancelledError: break
            except Exception as e: self.get_logger().error(f"Process loop error: {e}")

    def _decode_frame(self, jpeg_data: bytes) -> None:
        np_arr = np.frombuffer(jpeg_data, dtype=np.uint8)
        frame_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame_bgr is None: return

        with self._frame_lock:
            self._frame = frame_bgr
            self._frame_jpeg = jpeg_data

        # --- YOLO INFERENCE ---
        results = self.model.predict(frame_bgr, conf=0.5, verbose=False)
        msg = Point()
        
        if len(results[0].boxes) > 0:
            box = results[0].boxes[0].xyxy[0].tolist()
            msg.x = (box[0] + box[2]) / 2 # Center X
            msg.y = (box[1] + box[3]) / 2 # Center Y
            msg.z = 0.0 
        
        self.coordinates_pub.publish(msg)
        # ----------------------

        self._recv_count += 1
        if self._recv_count % 50 == 0:
            h, w = frame_bgr.shape[:2]
            self.get_logger().info(f"Received {self._recv_count} Zivid frames ({w}x{h})")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._frame is None: return None
            return self._frame.copy()

    def get_frame_jpeg(self) -> Optional[bytes]:
        with self._frame_lock: return self._frame_jpeg

    def get_frame_dimensions(self) -> Optional[tuple[int, int]]:
        with self._frame_lock:
            if self._frame is None: return None
            h, w = self._frame.shape[:2]
            return (w, h)

    def has_frame(self) -> bool:
        with self._frame_lock: return self._frame is not None

    def is_connected(self) -> bool: return self._connected

    def destroy_node(self) -> None:
        self._running = False
        super().destroy_node()