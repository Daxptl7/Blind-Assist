"""
object_detection.py — BlindAssist Project (OPTIMIZED)
=======================================================
YOLOv8 streaming inference with generator.
Processes frames asynchronously. No blocking per-frame.
"""

import cv2
import signal
import sys
import logging
import numpy as np
from collections import Counter, deque
from ultralytics import YOLO

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("blindassist_detection.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ObjectDetection")

# ── CONFIG ──────────────────────────────────────────────────
MODEL_PATH = "yolov8s.pt"
CONFIDENCE = 0.30
INFER_SIZE = 640
CAMERA_INDEX = 0

# ── MODEL (load once, reuse) ────────────────────────────────
logger.info(f"Loading {MODEL_PATH}...")
model = YOLO(MODEL_PATH)
logger.info("Model ready.")

# ── PREPROCESSING ───────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

def preprocess(frame: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    ycrcb[:, :, 0] = _clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

# ── STREAMING INFERENCE ─────────────────────────────────────
def stream_detect(source=0, callback=None):
    """
    Streaming object detection using YOLOv8 generator.
    Yields results as they're processed, no frame dropping.
    """
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Use generator for async processing
    results_generator = model.predict(
        source=cap,
        stream=True,
        conf=CONFIDENCE,
        imgsz=INFER_SIZE,
        verbose=False,
        device='cpu'
    )
    
    logger.info("Streaming detection active")
    
    try:
        for result in results_generator:
            frame = result.orig_img
            detections = []
            
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                
                detections.append({
                    'name': name,
                    'conf': conf,
                    'box': (x1, y1, x2, y2),
                    'center_x': (x1 + x2) / 2
                })
                
                # Draw
                color = (0, 255, 0) if conf > 0.5 else (0, 180, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (int(x1), int(y1)-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Position descriptions
            h, w = frame.shape[:2]
            descriptions = []
            for d in detections:
                pos = "left" if d['center_x'] < w/3 else ("right" if d['center_x'] > 2*w/3 else "center")
                descriptions.append(f"a {d['name']} on your {pos}")
            
            if descriptions:
                text = "I see: " + ", ".join(descriptions)
            else:
                text = "I don't see anything clearly."
            
            cv2.putText(frame, text[:80], (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            cv2.imshow("Object Detection", frame)
            
            if callback:
                callback(text, detections)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ── SINGLE SCAN API ─────────────────────────────────────────
def scan_frame(frame: np.ndarray) -> str:
    """Fast single-frame scan for TTS feedback."""
    processed = preprocess(frame)
    results = model(processed, conf=CONFIDENCE, imgsz=INFER_SIZE, verbose=False)
    
    h, w = frame.shape[:2]
    descriptions = []
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        name = model.names[int(box.cls[0])]
        center_x = (x1 + x2) / 2
        pos = "left" if center_x < w/3 else ("right" if center_x > 2*w/3 else "center")
        descriptions.append(f"a {name} on your {pos}")
    
    if not descriptions:
        return "I don't see anything clearly."
    return "I see: " + ", ".join(descriptions)

def run_detection():
    """Legacy API wrapper."""
    print("Object Detection — press Q to quit")
    stream_detect()

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    run_detection()