"""
Simple pixel picker — connects to the Zivid WebSocket server and shows
live pixel coordinates on hover. Click to record a point.

Usage:
    python pixel_picker.py

Controls:
    - Move mouse: shows pixel (x, y) in real time
    - Left click: prints the pixel coordinate
    - Q: quit
"""

import asyncio
import threading
import cv2
import numpy as np
import websockets

WS_URL = "ws://localhost:8766"

latest_frame = None
frame_lock = threading.Lock()
recorded_points = []


async def receive_frames():
    global latest_frame
    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                print(f"Connected to {WS_URL}")
                while True:
                    data = await ws.recv()
                    arr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with frame_lock:
                            latest_frame = frame
        except Exception as e:
            print(f"Reconnecting... ({e})")
            await asyncio.sleep(1)


def ws_thread():
    asyncio.run(receive_frames())


def on_mouse(event, x, y, flags, param):
    pass  # handled in main loop


def main():
    t = threading.Thread(target=ws_thread, daemon=True)
    t.start()

    print("Waiting for camera frames...")
    cv2.namedWindow("Pixel Picker", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pixel Picker", 900, 750)

    mouse_x, mouse_y = 0, 0
    display_w, display_h = 900, 750

    def mouse_cb(event, x, y, flags, param):
        nonlocal mouse_x, mouse_y
        mouse_x, mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert display coords back to original image coords
            with frame_lock:
                if latest_frame is not None:
                    h, w = latest_frame.shape[:2]
                    real_x = int(x * w / display_w)
                    real_y = int(y * h / display_h)
                    recorded_points.append((real_x, real_y))
                    print(f"Point #{len(recorded_points)}: pixel=({real_x}, {real_y})")

    cv2.setMouseCallback("Pixel Picker", mouse_cb)

    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            blank = np.zeros((400, 600, 3), np.uint8)
            cv2.putText(blank, "Waiting for camera...", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Pixel Picker", blank)
        else:
            h, w = frame.shape[:2]
            display = cv2.resize(frame, (display_w, display_h))

            # Convert display mouse position to original image coordinates
            real_x = int(mouse_x * w / display_w)
            real_y = int(mouse_y * h / display_h)

            # Draw crosshair
            cv2.line(display, (mouse_x, 0), (mouse_x, display_h), (0, 0, 255), 1)
            cv2.line(display, (0, mouse_y), (display_w, mouse_y), (0, 0, 255), 1)

            # Show coordinates
            label = f"({real_x}, {real_y})"
            cv2.putText(display, label, (mouse_x + 10, mouse_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show recorded points
            for i, (px, py) in enumerate(recorded_points):
                dx = int(px * display_w / w)
                dy = int(py * display_h / h)
                cv2.circle(display, (dx, dy), 6, (0, 255, 0), -1)
                cv2.putText(display, f"#{i+1}", (dx + 8, dy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.imshow("Pixel Picker", display)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
    if recorded_points:
        print("\n--- Recorded pixel points ---")
        for i, (x, y) in enumerate(recorded_points):
            print(f"  Point #{i+1}: pixel_x={x}, pixel_y={y}")


if __name__ == "__main__":
    main()
