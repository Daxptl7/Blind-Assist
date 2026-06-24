"""
gesture_control.py — BlindAssist Project
==========================================
Project  : Accessible Educational Terminal for Visually Impaired
Team     : Dhruv Vaghela & Dax Patel  |  CSR / Infineon 2025
Module   : Hand Gesture Recognition

Detects 21 hand landmarks via MediaPipe and classifies gestures
to control device modes without touching any button.

Gesture Map:
  Open Palm (5 fingers)  → MODE_SCAN   (Mode A: OCR)
  Thumbs Up              → CONFIRM     (Send/Accept)
  Two Fingers (V sign)   → MODE_VOICE  (Mode C: Voice Input)
  Fist (all curled)      → STOP        (Cancel)
  Point Down             → REPEAT      (Repeat last audio)
"""

import sys
import signal
import logging
import json
import time
import cv2
import numpy as np

from pathlib import Path
from typing import Optional, Callable

# ──────────────────────────────────────────────────────────────
# HARDWARE FLAGS (Pi Flag Pattern)
# ──────────────────────────────────────────────────────────────
HEADLESS = False
USE_PICAMERA = False
USE_GPIO = False

# ──────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "gesture.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("GestureModule")

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
WARMUP_FRAMES = 5
DEBOUNCE_SECONDS = 0.8  # Same gesture must hold for this long

def _load_settings() -> dict:
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Using defaults: {e}")
        return {"gesture_confidence": 0.85}

_settings = _load_settings()
GESTURE_CONFIDENCE = _settings.get("gesture_confidence", 0.85)

# ──────────────────────────────────────────────────────────────
# MEDIAPIPE INIT (Tasks API)
# ──────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_AVAILABLE = True
    logger.info("MediaPipe Tasks API loaded.")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.error("mediapipe not installed. Run: pip3 install mediapipe")


# ──────────────────────────────────────────────────────────────
# GESTURE CLASSIFICATION
# ──────────────────────────────────────────────────────────────
# MediaPipe hand landmark indices
WRIST = 0
THUMB_TIP = 4; THUMB_IP = 3; THUMB_MCP = 2
INDEX_TIP = 8; INDEX_PIP = 6
MIDDLE_TIP = 12; MIDDLE_PIP = 10
RING_TIP = 16; RING_PIP = 14
PINKY_TIP = 20; PINKY_PIP = 18


def _is_finger_extended(landmarks, tip_id, pip_id) -> bool:
    """Check if a finger is extended (tip above PIP joint in y-axis)."""
    return landmarks[tip_id].y < landmarks[pip_id].y


def _is_thumb_extended(landmarks) -> bool:
    """Check if thumb is extended (tip further from palm center than IP joint)."""
    thumb_tip = landmarks[THUMB_TIP]
    thumb_ip = landmarks[THUMB_IP]
    wrist = landmarks[WRIST]

    dist_tip = ((thumb_tip.x - wrist.x)**2 + (thumb_tip.y - wrist.y)**2)**0.5
    dist_ip = ((thumb_ip.x - wrist.x)**2 + (thumb_ip.y - wrist.y)**2)**0.5

    return dist_tip > dist_ip


def classify_gesture(landmarks) -> Optional[str]:
    """
    Classify hand gesture from 21 MediaPipe landmarks.

    Returns:
        Gesture name string or None if unrecognized.
    """
    lm = landmarks

    thumb = _is_thumb_extended(lm)
    index = _is_finger_extended(lm, INDEX_TIP, INDEX_PIP)
    middle = _is_finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring = _is_finger_extended(lm, RING_TIP, RING_PIP)
    pinky = _is_finger_extended(lm, PINKY_TIP, PINKY_PIP)

    fingers = [thumb, index, middle, ring, pinky]
    count = sum(fingers)

    # OPEN PALM — all 5 fingers extended
    if count == 5:
        return "MODE_SCAN"

    # FIST — no fingers extended
    if count == 0:
        return "STOP"

    # THUMBS UP — only thumb extended
    if thumb and not index and not middle and not ring and not pinky:
        return "CONFIRM"

    # TWO FINGERS (V/Peace sign) — index + middle only
    if not thumb and index and middle and not ring and not pinky:
        return "MODE_VOICE"

    # POINT DOWN — only index extended AND pointing below wrist
    if index and count == 1:
        if lm[INDEX_TIP].y > lm[WRIST].y:
            return "REPEAT"

    return None


# ──────────────────────────────────────────────────────────────
# PUBLIC API — detect_gesture() (blocking loop)
# ──────────────────────────────────────────────────────────────
def detect_gesture(callback_fn: Optional[Callable[[str], None]] = None):
    """
    Open webcam and continuously detect hand gestures.
    Calls callback_fn(gesture_name) when a gesture is confirmed.

    Press Q to quit the gesture detection loop.

    Args:
        callback_fn: Function to call with detected gesture name.
    """
    if not MEDIAPIPE_AVAILABLE:
        logger.error("MediaPipe not available — gesture control disabled.")
        return

    model_path = str(CONFIG_PATH.parent / "hand_landmarker.task")
    if not Path(model_path).exists():
        logger.error(f"Hand landmarker model file not found at {model_path}")
        return

    cap = None
    detector = None
    try:
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=GESTURE_CONFIDENCE
        )
        detector = vision.HandLandmarker.create_from_options(options)

        cap = cv2.VideoCapture(CAMERA_INDEX)
        if not cap.isOpened():
            logger.error("Could not open webcam for gesture detection.")
            return

        # Warmup frames
        for _ in range(WARMUP_FRAMES):
            cap.read()

        logger.info("Gesture detection started.")

        last_gesture = None
        gesture_start_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera feed lost.")
                break

            # Flip for mirror effect (more intuitive)
            frame = cv2.flip(frame, 1)

            # Convert BGR → RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect(mp_image)

            current_gesture = None

            if results.hand_landmarks:
                for hand_landmarks in results.hand_landmarks:
                    # Draw landmarks on frame
                    if not HEADLESS:
                        h, w, c = frame.shape
                        HAND_CONNECTIONS = [
                            (0, 1), (1, 2), (2, 3), (3, 4),
                            (0, 5), (5, 6), (6, 7), (7, 8),
                            (5, 9), (9, 10), (10, 11), (11, 12),
                            (9, 13), (13, 14), (14, 15), (15, 16),
                            (13, 17), (17, 18), (18, 19), (19, 20),
                            (0, 17)
                        ]
                        for lm in hand_landmarks:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
                        for start, end in HAND_CONNECTIONS:
                            lm_start = hand_landmarks[start]
                            lm_end = hand_landmarks[end]
                            cx_start, cy_start = int(lm_start.x * w), int(lm_start.y * h)
                            cx_end, cy_end = int(lm_end.x * w), int(lm_end.y * h)
                            cv2.line(frame, (cx_start, cy_start), (cx_end, cy_end), (0, 255, 255), 2)

                    # Classify
                    current_gesture = classify_gesture(hand_landmarks)

            # Debounce: gesture must hold steady for DEBOUNCE_SECONDS
            now = time.time()

            if current_gesture and current_gesture == last_gesture:
                if now - gesture_start_time >= DEBOUNCE_SECONDS:
                    logger.info(f"Gesture confirmed: {current_gesture}")

                    if callback_fn:
                        callback_fn(current_gesture)
                    else:
                        print(f"[GESTURE] {current_gesture}")

                    # Reset to prevent re-firing
                    gesture_start_time = now + 2.0  # cooldown
            else:
                last_gesture = current_gesture
                gesture_start_time = now

            # Display
            if not HEADLESS:
                label = current_gesture or "No gesture"
                cv2.putText(frame, f"Gesture: {label}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow("BlindAssist — Gesture Control", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except Exception as e:
        logger.error(f"Gesture detection error: {e}")

    finally:
        if detector is not None:
            detector.close()
        if cap is not None and cap.isOpened():
            cap.release()
        cv2.destroyAllWindows()
        logger.info("Gesture detection stopped.")


# ──────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ──────────────────────────────────────────────────────────────
def signal_handler(sig, frame):
    logger.info("Shutting down Gesture module.")
    cv2.destroyAllWindows()
    sys.exit(0)


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n" + "=" * 50)
    print("   BlindAssist Gesture Control — Live Test")
    print("   Project: CSR-DES-INFINEON-2025")
    print("=" * 50)
    print("Gesture Map:")
    print("  Open Palm (5 fingers) → MODE_SCAN")
    print("  Thumbs Up             → CONFIRM")
    print("  Two Fingers (V)       → MODE_VOICE")
    print("  Fist                  → STOP")
    print("  Point Down            → REPEAT")
    print("Press Q in the camera window to quit.")
    print("=" * 50 + "\n")

    detect_gesture()
    print("Gesture Module Closed.")
