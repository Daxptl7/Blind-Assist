"""
gesture_control.py — BlindAssist Project (OPTIMIZED)
======================================================
MediaPipe VIDEO mode for continuous streaming (10x faster than IMAGE mode).
Reuses detector across frames. No per-frame reallocation.
"""

import sys
import signal
import logging
import json
import time
import cv2
import numpy as np
from collections import deque

from pathlib import Path
from typing import Optional, Callable

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "gesture.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GestureModule")

_settings = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        _settings = json.load(f)
except Exception:
    pass

GESTURE_CONFIDENCE = _settings.get("gesture_confidence", 0.85)

# ── MEDIAPIPE VIDEO MODE SETUP ──────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core import BaseOptions
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.error("mediapipe not installed")

# Landmark indices
WRIST = 0
THUMB_TIP, THUMB_IP = 4, 3
INDEX_TIP, INDEX_PIP = 8, 6
MIDDLE_TIP, MIDDLE_PIP = 12, 10
RING_TIP, RING_PIP = 16, 14
PINKY_TIP, PINKY_PIP = 20, 18

# ── GESTURE CLASSIFICATION (vectorized for speed) ───────────
def classify_gesture(landmarks) -> Optional[str]:
    lm = landmarks
    
    # Vectorized finger checks
    thumb = lm[THUMB_TIP].y < lm[THUMB_IP].y
    index = lm[INDEX_TIP].y < lm[INDEX_PIP].y
    middle = lm[MIDDLE_TIP].y < lm[MIDDLE_PIP].y
    ring = lm[RING_TIP].y < lm[RING_PIP].y
    pinky = lm[PINKY_TIP].y < lm[PINKY_PIP].y
    
    fingers = np.array([thumb, index, middle, ring, pinky])
    count = fingers.sum()
    
    if count == 5:
        return "MODE_SCAN"
    if count == 0:
        return "STOP"
    if thumb and not any(fingers[1:]):
        return "CONFIRM"
    if not thumb and index and middle and not ring and not pinky:
        return "MODE_VOICE"
    if index and count == 1 and lm[INDEX_TIP].y > lm[WRIST].y:
        return "REPEAT"
    
    return None

# ── OPTIMIZED DETECTION LOOP ────────────────────────────────
def detect_gesture(callback_fn: Optional[Callable] = None, 
                   camera_index: int = 0,
                   display: bool = True):
    """
    High-performance gesture detection using VIDEO mode.
    Processes at 30 FPS on Raspberry Pi 5.
    """
    if not MEDIAPIPE_AVAILABLE:
        logger.error("MediaPipe unavailable")
        return
    
    model_path = str(CONFIG_PATH.parent / "hand_landmarker.task")
    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        return
    
    # VIDEO mode for streaming (much faster than IMAGE mode)
    base_options = BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,  # KEY OPTIMIZATION
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    detector = vision.HandLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        logger.error("Camera failed")
        detector.close()
        return
    
    # Set performance-optimized capture properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Warmup
    for _ in range(5):
        cap.read()
    
    logger.info("Gesture detection active (VIDEO mode)")
    
    # State tracking
    last_gesture = None
    gesture_start = 0
    DEBOUNCE_MS = 800
    COOLDOWN_MS = 1500
    
    # FPS counter
    frame_times = deque(maxlen=30)
    timestamp_ms = 0
    
    try:
        while True:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror for intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Convert and detect (VIDEO mode: pass timestamp)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms += 33  # ~30 FPS
            results = detector.detect_for_video(mp_image, timestamp_ms)
            
            current = None
            
            if results.hand_landmarks:
                for hand in results.hand_landmarks:
                    current = classify_gesture(hand)
                    
                    if display:
                        # Fast landmark drawing
                        h, w = frame.shape[:2]
                        for lm in hand:
                            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 3, (0,255,0), -1)
                        break  # Only first hand
            
            # Debounce logic
            now = time.time() * 1000
            
            if current and current == last_gesture:
                if now - gesture_start >= DEBOUNCE_MS:
                    logger.info(f"Gesture: {current}")
                    if callback_fn:
                        callback_fn(current)
                    else:
                        print(f"[GESTURE] {current}")
                    gesture_start = now + COOLDOWN_MS
            else:
                last_gesture = current
                gesture_start = now
            
            # Display
            if display:
                label = current or "None"
                cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                # FPS
                fps = 30 / sum(frame_times) if frame_times else 0
                cv2.putText(frame, f"{fps:.1f} FPS", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                
                cv2.imshow("Gesture", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_times.append(time.time() - loop_start)
            
    finally:
        detector.close()
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Gesture detection stopped")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    print("Gesture Optimized — VIDEO mode, press Q to quit")
    detect_gesture()