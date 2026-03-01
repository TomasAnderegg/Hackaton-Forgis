"""REST endpoints for camera control."""

import asyncio
import cv2
from typing import Optional

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/camera", tags=["camera"])

# CameraExecutor will be injected via app state
_camera_executor = None


def set_camera_executor(executor) -> None:
    """Set the camera executor instance (called during app initialization)."""
    global _camera_executor
    _camera_executor = executor


def get_executor():
    """Get the camera executor, raising if not initialized."""
    if _camera_executor is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera executor not initialized",
        )
    return _camera_executor


# --- Request/Response Models ---


class StreamStartRequest(BaseModel):
    """Request to start streaming."""

    fps: int = Field(default=15, ge=1, le=30, description="Target FPS")


class StreamResponse(BaseModel):
    """Response for stream control operations."""

    success: bool
    streaming: bool
    message: Optional[str] = None


class CameraStateResponse(BaseModel):
    """Response for camera state."""

    connected: bool
    streaming: bool
    frame_size: Optional[dict] = None
    last_detection: Optional[dict] = None


# --- Endpoints ---


@router.post("/stream/start", response_model=StreamResponse)
async def start_stream(request: StreamStartRequest = StreamStartRequest()):
    """Start streaming camera frames over WebSocket."""
    executor = get_executor()

    if not executor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera not connected",
        )

    started = await executor.start_streaming(fps=request.fps)

    return StreamResponse(
        success=True,
        streaming=True,
        message=f"Streaming started at {request.fps} FPS" if started else "Streaming already active",
    )


@router.post("/stream/stop", response_model=StreamResponse)
async def stop_stream():
    """Stop streaming camera frames."""
    executor = get_executor()

    stopped = await executor.stop_streaming()

    return StreamResponse(
        success=True,
        streaming=False,
        message="Streaming stopped" if stopped else "Streaming was not active",
    )


@router.get("/snapshot")
async def get_snapshot(quality: int = 90):
    """
    Get a single JPEG snapshot from the camera.

    Args:
        quality: JPEG quality (1-100).

    Returns:
        JPEG image data.
    """
    executor = get_executor()

    if not executor.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Camera not connected",
        )

    jpeg_data = executor.get_snapshot_jpeg(quality=min(100, max(1, quality)))

    if jpeg_data is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No frame available",
        )

    return Response(
        content=jpeg_data,
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=snapshot.jpg"},
    )


@router.get("/state", response_model=CameraStateResponse)
async def get_camera_state():
    """Get current camera state."""
    executor = get_executor()
    return executor.get_state_summary()


@router.get("/calibrate/picker")
async def calibrate_picker():
    """
    Interactive HTML page: hover over the image to read pixel (x,y).
    Click to record a point, enter robot coords, then send calibration.
    Open in browser: http://localhost:8000/api/camera/calibrate/picker
    """
    from fastapi.responses import HTMLResponse
    html = """<!DOCTYPE html>
<html>
<head>
<title>Calibration Picker</title>
<style>
  body{font-family:monospace;background:#111;color:#eee;margin:0;padding:10px}
  #wrap{position:relative;display:inline-block;max-width:900px;border:1px solid #444;cursor:crosshair}
  #snap{display:block;width:100%;height:auto;user-select:none}
  .hline{position:absolute;left:0;right:0;height:2px;background:rgba(255,50,50,0.85);pointer-events:none;display:none}
  .vline{position:absolute;top:0;bottom:0;width:2px;background:rgba(255,50,50,0.85);pointer-events:none;display:none}
  #lbl{position:absolute;background:rgba(0,0,0,0.75);color:yellow;font:bold 13px monospace;padding:2px 6px;pointer-events:none;display:none;white-space:nowrap}
  .marker{position:absolute;pointer-events:none;color:#0f0;font:bold 12px monospace;text-shadow:1px 1px 2px #000}
  .marker-box{position:absolute;width:16px;height:16px;border:2px solid #0f0;pointer-events:none;transform:translate(-8px,-8px)}
  #img-status{color:#fa0;font-size:13px;margin:4px 0}
  #coords{font-size:22px;margin:8px 0;color:#0f0;font-weight:bold;min-height:30px}
  .point{background:#222;margin:4px 0;padding:6px;border-left:3px solid #0f0;display:flex;align-items:center;gap:8px;flex-wrap:wrap}
  .point input{width:75px;padding:3px 5px;font-size:13px;background:#333;color:#eee;border:1px solid #555}
  .point label{color:#aaa;font-size:12px}
  button{background:#0a0;color:#fff;border:none;padding:8px 16px;cursor:pointer;font-size:14px;margin:4px;border-radius:4px}
  button.del{background:#800;padding:4px 10px;font-size:12px;margin:0}
  input{background:#222;color:#eee;border:1px solid #555;padding:6px;width:90px;font-size:14px}
  #json-out{background:#000;padding:10px;margin-top:10px;white-space:pre;font-size:12px;color:#ff0;max-height:200px;overflow:auto}
  #result{margin-top:10px;font-size:18px}
  .row{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin:6px 0}
</style>
</head>
<body>
<h2>Calibration Picker</h2>
<p><b>Comment faire :</b><br>
1. Déplace le robot TCP sur une position connue dans la zone de travail<br>
2. Clique <b>Rafraîchir</b> pour avoir l'image actuelle<br>
3. Bouge la souris sur l'image → crosshair rouge + coords en vert<br>
4. Clique pour figer les coords pixel, entre les coords robot (mm), clique <b>+ Ajouter</b><br>
5. Répète pour 4+ positions, puis clique <b>Envoyer calibration</b></p>

<div id="coords">Pixel: x=— , y=—</div>
<div id="img-status">Chargement...</div>

<div id="wrap">
  <img id="snap" src="/api/camera/snapshot" draggable="false"/>
  <div class="hline" id="hline"></div>
  <div class="vline" id="vline"></div>
  <div id="lbl"></div>
</div>
<br>
<div class="row">
  Pixel X: <input id="px-input" type="number" value="0" step="1"/>
  Pixel Y: <input id="py-input" type="number" value="0" step="1"/>
  &nbsp;|&nbsp;
  Robot X (mm): <input id="rx" type="number" value="0" step="0.1"/>
  Robot Y (mm): <input id="ry" type="number" value="0" step="0.1"/>
  <button onclick="addPoint()">+ Ajouter ce point</button>
  <button onclick="refreshImg()">Rafraîchir image</button>
  <button onclick="clearPoints()" style="background:#800">Effacer tout</button>
</div>

<div id="points"></div>
<div id="json-out"></div>
<button onclick="sendCalibration()" style="background:#00a;font-size:16px;padding:12px 28px;margin-top:10px">
  Envoyer calibration
</button>
<div id="result"></div>

<script>
const snap    = document.getElementById('snap');
const wrap    = document.getElementById('wrap');
const hline   = document.getElementById('hline');
const vline   = document.getElementById('vline');
const lbl     = document.getElementById('lbl');
const pxInput = document.getElementById('px-input');
const pyInput = document.getElementById('py-input');
const statusEl= document.getElementById('img-status');
const coordsEl= document.getElementById('coords');

let points = [];
let imgW = 0, imgH = 0;
let mouseImgX = 0, mouseImgY = 0;

snap.addEventListener('load', () => {
  imgW = snap.naturalWidth;
  imgH = snap.naturalHeight;
  statusEl.textContent = 'Image chargée — ' + imgW + 'x' + imgH + ' px';
  statusEl.style.color = '#0f0';
});
snap.addEventListener('error', () => {
  statusEl.textContent = 'Image non disponible — saisie manuelle possible';
  statusEl.style.color = '#fa0';
});

// ── Mouse events directly on the image ───────────────────────
wrap.addEventListener('mousemove', e => {
  const r = snap.getBoundingClientRect();
  const dx = e.clientX - r.left;   // position in displayed image px
  const dy = e.clientY - r.top;

  mouseImgX = imgW ? Math.round(dx * imgW / r.width)  : Math.round(dx);
  mouseImgY = imgH ? Math.round(dy * imgH / r.height) : Math.round(dy);

  coordsEl.textContent = 'Pixel: x=' + mouseImgX + '   y=' + mouseImgY;
  pxInput.value = mouseImgX;
  pyInput.value = mouseImgY;

  // Move crosshair lines
  hline.style.top     = dy + 'px';
  hline.style.display = 'block';
  vline.style.left    = dx + 'px';
  vline.style.display = 'block';

  // Move label (flip if near right/bottom edge)
  const labelText = '(' + mouseImgX + ', ' + mouseImgY + ')';
  lbl.textContent = labelText;
  lbl.style.left    = (dx + 12) + 'px';
  lbl.style.top     = (dy - 24) + 'px';
  lbl.style.display = 'block';
});

wrap.addEventListener('mouseleave', () => {
  hline.style.display = 'none';
  vline.style.display = 'none';
  lbl.style.display   = 'none';
});

wrap.addEventListener('click', e => {
  const r = snap.getBoundingClientRect();
  const dx = e.clientX - r.left;
  const dy = e.clientY - r.top;
  pxInput.value = imgW ? Math.round(dx * imgW / r.width)  : Math.round(dx);
  pyInput.value = imgH ? Math.round(dy * imgH / r.height) : Math.round(dy);
});

// ── Calibration points ───────────────────────────────────────
function addPoint() {
  const px = parseFloat(pxInput.value);
  const py = parseFloat(pyInput.value);
  const rx = parseFloat(document.getElementById('rx').value);
  const ry = parseFloat(document.getElementById('ry').value);
  if (isNaN(px) || isNaN(py)) { alert('Pixel X/Y invalide'); return; }
  points.push({pixel_x: px, pixel_y: py, robot_x: rx, robot_y: ry});
  render();
  redrawMarkers();
}

function redrawMarkers() {
  // Remove old markers
  wrap.querySelectorAll('.marker,.marker-box').forEach(el => el.remove());
  if (!imgW || !imgH) return;
  const r = snap.getBoundingClientRect();
  points.forEach((p, i) => {
    const dx = p.pixel_x * r.width  / imgW;
    const dy = p.pixel_y * r.height / imgH;
    const box = document.createElement('div');
    box.className = 'marker-box';
    box.style.left = dx + 'px';
    box.style.top  = dy + 'px';
    wrap.appendChild(box);
    const tag = document.createElement('div');
    tag.className = 'marker';
    tag.textContent = '#' + (i+1);
    tag.style.left = (dx + 10) + 'px';
    tag.style.top  = (dy - 16) + 'px';
    wrap.appendChild(tag);
  });
}

function render() {
  document.getElementById('points').innerHTML = points.map((p, i) =>
    '<div class="point"><span>#' + (i+1) + '</span>' +
    '<label>px X:</label><input type="number" value="' + p.pixel_x + '" step="1" onchange="upd(' + i + ',\'pixel_x\',this.value)"/>' +
    '<label>px Y:</label><input type="number" value="' + p.pixel_y + '" step="1" onchange="upd(' + i + ',\'pixel_y\',this.value)"/>' +
    '<label>robot X mm:</label><input type="number" value="' + p.robot_x + '" step="0.1" onchange="upd(' + i + ',\'robot_x\',this.value)"/>' +
    '<label>robot Y mm:</label><input type="number" value="' + p.robot_y + '" step="0.1" onchange="upd(' + i + ',\'robot_y\',this.value)"/>' +
    '<button class="del" onclick="del(' + i + ')">Supprimer</button></div>'
  ).join('');
  document.getElementById('json-out').textContent = JSON.stringify({points}, null, 2);
}

function upd(i, f, v) { points[i][f] = parseFloat(v); document.getElementById('json-out').textContent = JSON.stringify({points}, null, 2); redrawMarkers(); }
function del(i) { points.splice(i, 1); render(); redrawMarkers(); }
function clearPoints() { points = []; render(); redrawMarkers(); }

function refreshImg() {
  statusEl.textContent = 'Chargement...'; statusEl.style.color = '#fa0';
  snap.src = '/api/camera/snapshot?t=' + Date.now();
}

async function sendCalibration() {
  if (points.length < 4) { alert('Il faut au moins 4 points !'); return; }
  const res  = await fetch('/api/camera/calibrate', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({points})});
  const data = await res.json();
  document.getElementById('result').innerHTML = res.ok
    ? '<span style="color:#0f0">Calibration OK ! ' + data.points_used + ' points utilisés.</span>'
    : '<span style="color:#f00">Erreur: ' + JSON.stringify(data) + '</span>';
}
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@router.get("/calibrate/grid")
async def calibrate_grid(step: int = 200):
    """
    Returns the current camera frame with a pixel coordinate grid overlay.
    Open in browser to visually read pixel (x,y) at any point in the workspace.
    step: grid spacing in pixels (default 200).
    """
    import numpy as _np

    executor = get_executor()
    if not executor.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Camera not connected")

    frame = executor._camera.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No frame available")

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw grid lines + coordinates
    for x in range(0, w, step):
        cv2.line(annotated, (x, 0), (x, h), (0, 200, 255), 1)
        cv2.putText(annotated, str(x), (x + 4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    for y in range(0, h, step):
        cv2.line(annotated, (0, y), (w, y), (0, 200, 255), 1)
        cv2.putText(annotated, str(y), (4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

    # Draw center cross
    cv2.line(annotated, (w // 2 - 40, h // 2), (w // 2 + 40, h // 2), (0, 255, 0), 3)
    cv2.line(annotated, (w // 2, h // 2 - 40), (w // 2, h // 2 + 40), (0, 255, 0), 3)

    display = cv2.resize(annotated, (w // 2, h // 2))
    ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(status_code=500, detail="Encoding failed")

    return Response(content=buf.tobytes(), media_type="image/jpeg",
                    headers={"Content-Disposition": "inline; filename=calibration_grid.jpg"})


class CalibrationPoint(BaseModel):
    pixel_x: float
    pixel_y: float
    robot_x: float  # In robot units (mm recommended)
    robot_y: float


class CalibrateRequest(BaseModel):
    points: list[CalibrationPoint] = Field(
        min_length=4,
        description="At least 4 (pixel_x, pixel_y, robot_x, robot_y) correspondences",
    )


@router.post("/calibrate")
async def calibrate_camera(request: CalibrateRequest):
    """
    Set the camera-to-robot coordinate transform.

    Provide at least 4 points where you know both the pixel position (from the
    camera image) and the real robot position (in mm).

    How to calibrate:
    1. Move the robot to a known position and note its x,y in mm.
    2. Look at the camera snapshot and note the pixel (x,y) at that position.
    3. Repeat for 4+ positions spread across the workspace.
    4. POST this list here — the homography is computed and saved.
    """
    executor = get_executor()
    pts = [(p.pixel_x, p.pixel_y, p.robot_x, p.robot_y) for p in request.points]
    try:
        executor.calibration.set_points(pts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {
        "success": True,
        "points_used": len(pts),
        "matrix": executor.calibration.get_matrix(),
    }


@router.get("/calibrate")
async def get_calibration():
    """Get current calibration state."""
    executor = get_executor()
    return {
        "calibrated": executor.calibration.is_calibrated(),
        "matrix": executor.calibration.get_matrix(),
    }


class DetectRequest(BaseModel):
    object_class: str = Field(default="cardboard box", description="YOLO class to detect")
    confidence_threshold: float = Field(default=0.02, ge=0.0, le=1.0)


@router.post("/detect")
async def detect_object(request: DetectRequest = DetectRequest()):
    """Run YOLO detection and return bounding boxes."""
    executor = get_executor()
    if not executor.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Camera not connected")
    detections = await executor.detect_objects(
        class_name=request.object_class,
        confidence_threshold=request.confidence_threshold,
    )
    return {
        "found": len(detections) > 0,
        "count": len(detections),
        "detections": [d.to_dict() for d in detections],
    }


@router.get("/detect/annotated")
async def detect_annotated(
    object_class: str = "cardboard box",
    confidence_threshold: float = 0.02,
    quality: int = 85,
):
    """
    Run YOLO detection and return the camera frame with bounding boxes drawn.
    Useful for visual debugging — open in browser or save as JPEG.
    """
    import cv2 as _cv2
    import numpy as _np

    executor = get_executor()
    if not executor.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Camera not connected")

    frame = executor._camera.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No frame available")

    detections = await executor.detect_objects(
        class_name=object_class,
        confidence_threshold=confidence_threshold,
    )

    # Draw bounding boxes on a copy of the frame
    annotated = frame.copy()
    for det in detections:
        x1, y1 = int(det.x), int(det.y)
        x2, y2 = int(det.x + det.width), int(det.y + det.height)
        _cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{det.class_name} {det.confidence:.2f}"
        _cv2.putText(annotated, label, (x1, max(y1 - 10, 20)),
                     _cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    encode_params = [_cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = _cv2.imencode(".jpg", annotated, encode_params)
    if not ok:
        raise HTTPException(status_code=500, detail="Image encoding failed")

    return Response(
        content=buf.tobytes(),
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=detected.jpg"},
    )


@router.get("/detect/brightest")
async def detect_brightest_object(min_area: int = 5000, quality: int = 85):
    """
    Detect the largest bright object using OpenCV contours (no ML needed).
    Works reliably with white/light objects on a dark background.
    Returns bounding box + annotated JPEG.
    """
    import numpy as _np
    import base64 as _b64

    executor = get_executor()
    if not executor.is_ready():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Camera not connected")

    frame = executor._camera.get_latest_frame()
    if frame is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="No frame available")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    kernel = _np.ones((5, 5), _np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not candidates:
        return {"found": False, "count": 0, "detections": []}

    candidates.sort(key=cv2.contourArea, reverse=True)
    best = candidates[0]
    x, y, w, h = cv2.boundingRect(best)

    detection = {
        "x": float(x), "y": float(y),
        "width": float(w), "height": float(h),
        "confidence": 1.0,
        "class_name": "bright_object",
        "center_x": float(x + w / 2),
        "center_y": float(y + h / 2),
        "area": float(cv2.contourArea(best)),
    }

    annotated = frame.copy()
    cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 4)
    cv2.putText(annotated, f"object ({w}x{h}px)", (x, max(y - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    h_f, w_f = annotated.shape[:2]
    display = cv2.resize(annotated, (w_f // 3, h_f // 3))
    ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Image encoding failed")

    return {
        "found": True,
        "count": len(candidates),
        "detections": [detection],
        "annotated_jpeg_b64": _b64.b64encode(buf.tobytes()).decode(),
    }


@router.get("/stream/detection")
async def stream_detection_mjpeg(
    object_class: str = "cardboard box",
    confidence_threshold: float = 0.02,
):
    """
    Live MJPEG stream with YOLO bounding boxes drawn in real time.
    Open directly in a browser: http://localhost:8000/api/camera/stream/detection
    """
    executor = get_executor()

    async def generate():
        while True:
            try:
                frame = executor._camera.get_latest_frame()
                if frame is None:
                    await asyncio.sleep(0.2)
                    continue

                # Run YOLO detection (uses thread executor internally)
                detections = await executor.detect_objects(
                    class_name=object_class,
                    confidence_threshold=confidence_threshold,
                )

                # Draw bounding boxes
                annotated = frame.copy()
                for det in detections:
                    x1, y1 = int(det.x), int(det.y)
                    x2, y2 = int(det.x + det.width), int(det.y + det.height)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 4)
                    label = f"{det.class_name} {det.confidence:.0%}"
                    cv2.putText(annotated, label, (x1, max(y1 - 12, 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 255, 0), 3)

                # Resize for browser display (2448x2048 is very large)
                h, w = annotated.shape[:2]
                display = cv2.resize(annotated, (w // 3, h // 3))

                ok, buf = cv2.imencode(".jpg", display, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if not ok:
                    await asyncio.sleep(0.2)
                    continue

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
            except Exception:
                await asyncio.sleep(0.5)
                continue

            await asyncio.sleep(0.5)  # ~2 FPS (YOLO is slow on CPU)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
