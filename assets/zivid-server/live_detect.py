"""
Live detection viewer — shows camera feed with YOLO detections
and robot coordinates (mm) overlaid in real time.

Usage:
    python live_detect.py

Requires:
    - zivid_server.py running (ws://localhost:8766)
    - Backend Docker container running (http://localhost:8000)
    - Camera calibrated via /api/camera/calibrate
"""

import asyncio
import threading
import time
import urllib.request
import json
import cv2
import numpy as np
import websockets

WS_URL = "ws://localhost:8766"
DETECT_URL = "http://localhost:8000/api/camera/detect"
DETECT_BODY = b'{"object_class":"cardboard box","confidence_threshold":0.3}'

latest_frame = None
frame_lock = threading.Lock()
latest_detections = []
det_lock = threading.Lock()


# ── WebSocket frame receiver ──────────────────────────────────────────────────

async def receive_frames():
    global latest_frame
    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                print("Camera connected.")
                while True:
                    data = await ws.recv()
                    arr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        with frame_lock:
                            latest_frame = frame
        except Exception as e:
            print(f"Camera reconnecting... ({e})")
            await asyncio.sleep(1)


def ws_thread():
    asyncio.run(receive_frames())


# ── Detection poller ──────────────────────────────────────────────────────────

def detection_thread():
    global latest_detections
    req = urllib.request.Request(
        DETECT_URL,
        data=DETECT_BODY,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    while True:
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
                with det_lock:
                    latest_detections = data.get("detections", [])
        except Exception as e:
            pass
        time.sleep(0.5)  # poll every 500ms


# ── Main display loop ─────────────────────────────────────────────────────────

def main():
    threading.Thread(target=ws_thread, daemon=True).start()
    threading.Thread(target=detection_thread, daemon=True).start()

    print("Starting live detection viewer... Press Q to quit.")
    cv2.namedWindow("Live Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Live Detection", 1000, 800)

    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None

        if frame is None:
            blank = np.zeros((400, 600, 3), np.uint8)
            cv2.putText(blank, "Waiting for camera...", (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("Live Detection", blank)
        else:
            h, w = frame.shape[:2]
            display = cv2.resize(frame, (1000, 800))
            sx, sy = 1000 / w, 800 / h

            with det_lock:
                dets = list(latest_detections)

            for det in dets:
                x1 = int(det["x"] * sx)
                y1 = int(det["y"] * sy)
                x2 = int((det["x"] + det["width"]) * sx)
                y2 = int((det["y"] + det["height"]) * sy)

                # Bounding box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Robot coordinates
                rx = det.get("robot_x")
                ry = det.get("robot_y")
                conf = det.get("confidence", 0)

                if rx is not None and ry is not None:
                    label1 = f"{det['class_name']} {conf:.0%}"
                    label2 = f"robot: ({rx:.1f}, {ry:.1f}) mm"
                    cv2.putText(display, label1, (x1, max(y1 - 22, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, label2, (x1, max(y1 - 4, 38)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                else:
                    label = f"{det['class_name']} {conf:.0%} (not calibrated)"
                    cv2.putText(display, label, (x1, max(y1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

                # Center dot
                cx = int(det.get("center_x", det["x"] + det["width"] / 2) * sx)
                cy = int(det.get("center_y", det["y"] + det["height"] / 2) * sy)
                cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

            # Detection count
            cv2.putText(display, f"Detections: {len(dets)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("Live Detection", display)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
