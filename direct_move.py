"""
Direct URScript sender — no ROS2, no Docker needed.

Detects boxes via the backend API (pixel coordinates), applies the
embedded calibration homography locally to get robot coordinates,
then sends movel commands directly to the UR robot over TCP (port 30002).

Usage (on Windows, outside Docker):
    python direct_move.py

Requirements:
    - Backend Docker running (http://localhost:8000) with camera streaming
    - Robot reachable at 192.168.0.10
    - numpy + opencv-python installed (comes with Anaconda)
"""

import json
import math
import socket
import time
import urllib.request

import cv2
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────
ROBOT_IP   = "192.168.0.10"
ROBOT_PORT = 30002          # UR primary client interface (URScript)

DETECT_URL  = "http://localhost:8000/api/camera/detect"
DETECT_BODY = json.dumps({
    "object_class": "cardboard box",
    "confidence_threshold": 0.3,
}).encode()

Z_HOVER_M    = -0.050       # Height above workspace (meters) — just above box
ORIENTATION  = [math.pi, 0.0, 0.0]  # TCP pointing down
VELOCITY     = 0.10         # m/s  (slow for safety)
ACCEL        = 0.3          # m/s²
HOVER_TIME   = 2.0          # seconds to pause above each box

SAFE_JOINTS_DEG = [0.0, -90.0, 90.0, -90.0, -90.0, 0.0]

# ── Calibration points: (pixel_x, pixel_y, robot_x_mm, robot_y_mm) ────────────
# Collected by placing TCP above each point and reading teach pendant (in mm).
CALIB_POINTS = [
    (470,  565,  340.0,  151.10),
    (1455, 1168, 616.6, -108.50),
    (1966, 658,  593.0,   99.30),
    (1060, 1002, 494.6, -111.40),
]

# Compute homography once at startup
_src = np.array([(p[0], p[1]) for p in CALIB_POINTS], dtype=np.float32)
_dst = np.array([(p[2], p[3]) for p in CALIB_POINTS], dtype=np.float32)
H, _ = cv2.findHomography(_src, _dst, cv2.RANSAC, 5.0)


def pixel_to_robot_mm(px: float, py: float) -> tuple[float, float]:
    """Convert pixel center to robot X/Y in mm using the embedded calibration."""
    pt = np.array([[[px, py]]], dtype=np.float32)
    result = cv2.perspectiveTransform(pt, H)
    return float(result[0, 0, 0]), float(result[0, 0, 1])


# ── URScript helpers ───────────────────────────────────────────────────────────

def deg2rad(deg_list):
    return [math.radians(d) for d in deg_list]


def send_urscript(script: str):
    """Send a URScript program to the robot and close the connection."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5)
        s.connect((ROBOT_IP, ROBOT_PORT))
        s.sendall((script + "\n").encode("utf-8"))
    print(f"  → sent to robot")


def movej(joints_rad, accel=1.0, vel=0.5):
    j = ", ".join(f"{v:.6f}" for v in joints_rad)
    send_urscript(f"movej([{j}], a={accel}, v={vel})\n")


def movel(pose, accel=0.3, vel=0.1):
    x, y, z, rx, ry, rz = pose
    send_urscript(
        f"movel(p[{x:.6f},{y:.6f},{z:.6f},{rx:.6f},{ry:.6f},{rz:.6f}],"
        f" a={accel}, v={vel})\n"
    )


def wait(seconds):
    time.sleep(seconds)


# ── Detection ──────────────────────────────────────────────────────────────────

def get_boxes():
    """Call detect API and return boxes with robot coords from local calibration."""
    req = urllib.request.Request(
        DETECT_URL,
        data=DETECT_BODY,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    boxes = []
    for det in data.get("detections", []):
        cx = det.get("center_x", det["x"] + det["width"] / 2)
        cy = det.get("center_y", det["y"] + det["height"] / 2)
        rx_mm, ry_mm = pixel_to_robot_mm(cx, cy)
        boxes.append({
            "class_name":  det["class_name"],
            "confidence":  det["confidence"],
            "pixel_cx":    cx,
            "pixel_cy":    cy,
            "robot_x_mm":  rx_mm,
            "robot_y_mm":  ry_mm,
        })
    return boxes


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Direct URScript Box Scanner ===")
    print(f"    Homography matrix computed from {len(CALIB_POINTS)} calibration points\n")

    # 1. Go to safe position
    safe_rad = deg2rad(SAFE_JOINTS_DEG)
    print("1. Moving to safe position...")
    movej(safe_rad, accel=0.5, vel=0.3)
    wait(6)

    # 2. Detect boxes
    print("2. Detecting boxes via camera API...")
    try:
        boxes = get_boxes()
    except Exception as e:
        print(f"   ERROR: {e}")
        return

    if not boxes:
        print("   No boxes detected. Check camera stream.")
        return

    boxes.sort(key=lambda b: b["robot_x_mm"])
    print(f"   Found {len(boxes)} box(es):")
    for i, b in enumerate(boxes, 1):
        print(
            f"   Box #{i}: pixel=({b['pixel_cx']:.0f}, {b['pixel_cy']:.0f})  "
            f"robot=({b['robot_x_mm']:.1f}, {b['robot_y_mm']:.1f}) mm  "
            f"conf={b['confidence']:.0%}"
        )

    # 3. Move above each box
    for i, box in enumerate(boxes, 1):
        rx_m = box["robot_x_mm"] / 1000.0
        ry_m = box["robot_y_mm"] / 1000.0
        pose = [rx_m, ry_m, Z_HOVER_M] + ORIENTATION

        print(f"\n3.{i} Moving above box #{i} → p[{rx_m:.4f}, {ry_m:.4f}, {Z_HOVER_M}] m")
        movel(pose, accel=ACCEL, vel=VELOCITY)
        wait(8)  # time for robot to reach position

        print(f"   Hovering {HOVER_TIME}s...")
        wait(HOVER_TIME)

    # 4. Return to safe position
    print("\n4. Returning to safe position...")
    movej(safe_rad, accel=0.5, vel=0.3)
    wait(6)

    print("\nDone.")


if __name__ == "__main__":
    main()
