"""
Windows Zivid Camera WebSocket Server.

Captures color frames from a Zivid camera and streams them as JPEG over WebSocket.
Run this script on Windows with the Zivid camera connected.

Usage:
    python zivid_server.py --port 8766 --fps 5 --width 1920 --height 1200
"""

import argparse
import asyncio
import datetime
import logging
import signal
import sys
from typing import Optional

import cv2
import numpy as np
import zivid
import websockets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ZividCapture:
    """Captures color frames from a Zivid camera."""

    def __init__(self, fps: int = 5, ip: Optional[str] = None):
        self.fps = fps
        self.ip = ip  # Optional: connect by IP (e.g. "192.168.15.107")
        self._app: Optional[zivid.Application] = None
        self._camera: Optional[zivid.Camera] = None
        self._settings: Optional[zivid.Settings2D] = None

    def start(self) -> bool:
        """Connect to the Zivid camera and configure capture settings."""
        try:
            self._app = zivid.Application()

            if self.ip:
                logger.info(f"Connecting to Zivid camera at IP {self.ip}...")
                # Try different SDK versions' IP-connect APIs
                try:
                    self._camera = self._app.connect_camera(
                        zivid.CameraInfo.Network.IPAddress(self.ip)
                    )
                except AttributeError:
                    self._camera = self._app.connect_camera(
                        zivid.NetworkCamera(self.ip)
                    )
            else:
                cameras = self._app.cameras()
                if not cameras:
                    logger.error("No Zivid cameras found")
                    return False
                for cam in cameras:
                    logger.info(
                        f"  Found: {cam.info.serial_number} | "
                        f"{cam.info.model_name} | "
                        f"status={cam.state.status}"
                    )
                self._camera = self._app.connect_camera()

            logger.info(
                f"Zivid camera connected: {self._camera.info.model_name} "
                f"(serial {self._camera.info.serial_number})"
            )

            # 2D capture settings for indoor/warehouse lighting.
            # Zivid 2+ MR130: exposure_time in [900, 20000] μs, brightness>=1.0 required for color RGB.
            # aperture=2.83 (widest), exposure=20ms (max), gain=8, projector on.
            self._settings = zivid.Settings2D(
                acquisitions=[zivid.Settings2D.Acquisition(
                    aperture=2.83,
                    exposure_time=datetime.timedelta(microseconds=10000),
                    brightness=1.0,
                    gain=3.0,
                )]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Zivid camera: {e}")
            logger.error("  → Make sure Zivid Studio is CLOSED before running this script.")
            if self.ip is None:
                logger.error("  → Or try: python zivid_server.py --ip 192.168.15.107")
            return False

    def stop(self) -> None:
        """Disconnect from the camera."""
        try:
            if self._camera:
                self._camera = None
            if self._app:
                self._app = None
            logger.info("Zivid camera disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting camera: {e}")

    def get_frame(self) -> Optional[np.ndarray]:
        """Capture a 2D color frame. Returns BGR numpy array."""
        if not self._camera or not self._settings:
            return None
        try:
            frame_2d = self._camera.capture_2d(self._settings)
            image = frame_2d.image_rgba()
            rgba = np.array(image.copy_data())          # H x W x 4 (RGBA)
            bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
            return bgr
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None


class ZividServer:
    """WebSocket server that streams Zivid color frames to connected clients."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8766,
        fps: int = 5,
        jpeg_quality: int = 80,
        ip: Optional[str] = None,
    ):
        self.host = host
        self.port = port
        self.fps = fps
        self.jpeg_quality = jpeg_quality
        self.camera = ZividCapture(fps=fps, ip=ip)
        self.clients: set = set()
        self.running = False
        self._frame_interval = 1.0 / fps

    async def register_client(self, websocket) -> None:
        """Register a new client and keep connection alive."""
        client_addr = websocket.remote_address
        logger.info(f"Client connected: {client_addr}")
        self.clients.add(websocket)
        try:
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            logger.info(f"Client disconnected: {client_addr}")

    async def broadcast_frames(self) -> None:
        """Capture and broadcast frames to all connected clients."""
        while self.running:
            if not self.clients:
                await asyncio.sleep(0.1)
                continue

            frame_bgr = await asyncio.get_event_loop().run_in_executor(
                None, self.camera.get_frame
            )
            if frame_bgr is None:
                await asyncio.sleep(0.1)
                continue

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            success, jpeg_data = cv2.imencode(".jpg", frame_bgr, encode_params)
            if not success:
                continue

            websockets.broadcast(self.clients, jpeg_data.tobytes())
            await asyncio.sleep(self._frame_interval)

    async def start(self) -> None:
        """Start the camera and WebSocket server."""
        if not self.camera.start():
            logger.error("Failed to start Zivid camera, exiting")
            return

        self.running = True
        broadcast_task = asyncio.create_task(self.broadcast_frames())

        logger.info(f"Zivid WebSocket server starting on ws://{self.host}:{self.port}")
        try:
            async with websockets.serve(
                self.register_client,
                self.host,
                self.port,
                ping_interval=None,
            ):
                await asyncio.Future()  # Run forever
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            broadcast_task.cancel()
            try:
                await broadcast_task
            except asyncio.CancelledError:
                pass
            self.camera.stop()

    def stop(self) -> None:
        self.running = False


def main():
    parser = argparse.ArgumentParser(
        description="Zivid Camera WebSocket Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="0.0.0.0", help="Server bind address")
    parser.add_argument("--port", type=int, default=8766, help="WebSocket port")
    parser.add_argument("--fps", type=int, default=5, help="Capture FPS (Zivid is slow, 5 is safe)")
    parser.add_argument("--quality", type=int, default=80, help="JPEG quality (0-100)")
    parser.add_argument("--ip", default=None, help="Zivid camera IP address (e.g. 192.168.15.107)")
    args = parser.parse_args()

    server = ZividServer(
        host=args.host,
        port=args.port,
        fps=args.fps,
        jpeg_quality=args.quality,
        ip=args.ip,
    )

    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        server.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == "__main__":
    main()
