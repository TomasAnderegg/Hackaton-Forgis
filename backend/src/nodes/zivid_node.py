"""ROS 2 node for Zivid camera subscription (color + point cloud)."""

import threading
from typing import Optional

import struct

import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2


class ZividNode(Node):
    """
    ROS 2 node that subscribes to Zivid camera topics.

    Stores the latest color frame, JPEG cache, and point cloud
    in a thread-safe manner for access by the executor.

    Topics (published by zivid_ros driver):
      /zivid_camera/color/image_color   — RGB color image
      /zivid_camera/depth/points        — PointCloud2 (XYZ + RGB)
    """

    def __init__(self):
        super().__init__("zivid_node")

        self._frame: Optional[np.ndarray] = None
        self._frame_jpeg: Optional[bytes] = None
        self._point_cloud: Optional[np.ndarray] = None  # Nx6: X,Y,Z,R,G,B
        self._frame_lock = threading.Lock()
        self._pc_lock = threading.Lock()
        self._recv_count = 0

        # Zivid publishes with RELIABLE QoS
        qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(
            Image,
            "/zivid_camera/color/image_color",
            self._on_image,
            qos,
        )

        self.create_subscription(
            PointCloud2,
            "/zivid_camera/depth/points",
            self._on_point_cloud,
            qos,
        )

        self.get_logger().info(
            "ZividNode initialized — waiting for /zivid_camera/color/image_color"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_image(self, msg: Image) -> None:
        """Convert ROS Image (rgb8) to OpenCV BGR numpy array and cache JPEG."""
        if msg.encoding not in ("rgb8", "RGB8"):
            self.get_logger().warn(
                f"Unexpected encoding: {msg.encoding}", throttle_duration_sec=5.0
            )
            return

        frame_rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            (msg.height, msg.width, 3)
        )
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        success, jpeg_data = cv2.imencode(
            ".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 70]
        )

        with self._frame_lock:
            self._frame = frame_bgr
            if success:
                self._frame_jpeg = jpeg_data.tobytes()

        self._recv_count += 1
        if self._recv_count % 50 == 0:
            self.get_logger().info(
                f"Received {self._recv_count} frames ({msg.width}x{msg.height})"
            )

    def _on_point_cloud(self, msg: PointCloud2) -> None:
        """Parse PointCloud2 into a Nx6 float32 array (X, Y, Z, R, G, B)."""
        # Find byte offsets for x, y, z, rgb fields
        offsets = {f.name: f.offset for f in msg.fields}
        x_off = offsets.get("x", 0)
        y_off = offsets.get("y", 4)
        z_off = offsets.get("z", 8)
        rgb_off = offsets.get("rgb", 12)

        step = msg.point_step
        data = msg.data
        rows = []

        for i in range(msg.width * msg.height):
            base = i * step
            (x,) = struct.unpack_from("f", data, base + x_off)
            (y,) = struct.unpack_from("f", data, base + y_off)
            (z,) = struct.unpack_from("f", data, base + z_off)
            if np.isnan(x) or np.isnan(y) or np.isnan(z):
                continue
            (rgb_packed,) = struct.unpack_from("f", data, base + rgb_off)
            r, g, b = self._unpack_rgb(rgb_packed)
            rows.append((x, y, z, r, g, b))

        if not rows:
            return

        with self._pc_lock:
            self._point_cloud = np.array(rows, dtype=np.float32)

    @staticmethod
    def _unpack_rgb(packed: float) -> tuple[int, int, int]:
        """Unpack float-encoded RGB from PointCloud2."""
        rgb_int = int(packed)
        r = (rgb_int >> 16) & 0xFF
        g = (rgb_int >> 8) & 0xFF
        b = rgb_int & 0xFF
        return r, g, b

    # ------------------------------------------------------------------
    # Public accessors (thread-safe)
    # ------------------------------------------------------------------

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the latest color frame as a BGR numpy array."""
        with self._frame_lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def get_frame_jpeg(self, quality: int = 80) -> Optional[bytes]:
        """Return the latest frame as cached JPEG bytes."""
        with self._frame_lock:
            return self._frame_jpeg

    def get_point_cloud(self) -> Optional[np.ndarray]:
        """
        Return the latest point cloud as a Nx6 float32 array.
        Columns: X (m), Y (m), Z (m), R, G, B
        """
        with self._pc_lock:
            if self._point_cloud is None:
                return None
            return self._point_cloud.copy()

    def get_xyz(self) -> Optional[np.ndarray]:
        """Return only XYZ coordinates as a Nx3 float32 array (metres)."""
        with self._pc_lock:
            if self._point_cloud is None:
                return None
            return self._point_cloud[:, :3].copy()

    def get_frame_dimensions(self) -> Optional[tuple[int, int]]:
        """Return (width, height) of the color frame."""
        with self._frame_lock:
            if self._frame is None:
                return None
            h, w = self._frame.shape[:2]
            return (w, h)

    def has_frame(self) -> bool:
        """True if at least one color frame has been received."""
        with self._frame_lock:
            return self._frame is not None

    def has_point_cloud(self) -> bool:
        """True if at least one point cloud has been received."""
        with self._pc_lock:
            return self._point_cloud is not None
