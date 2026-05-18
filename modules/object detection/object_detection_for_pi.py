"""
object_detection.py — BlindAssist Project
==========================================
Project  : Accessible Educational Terminal for Visually Impaired
Team     : Dhruv Vaghela & Dax Patel  |  CSR / Infineon 2025
Module   : Object Recognition — Raspberry Pi 5 Production Ready

OFFLINE STATUS
--------------
This module runs 100% offline after first-time model download.
No internet needed during normal device operation.

PI-SPECIFIC OPTIMISATIONS IN THIS FILE
---------------------------------------
[PI-1]  Model pre-downloaded to USB flash drive (/media/blindassist/models/)
        so MicroSD is not worn out by repeated model reads.

[PI-2]  Frame resolution capped at 640x480 for Pi Camera.
        Full 12MP resolution is unnecessary for detection and kills FPS.

[PI-3]  Frame skip — runs inference every Nth frame to save CPU.
        Pi 5 can handle every frame at 640x480 but Coral TPU users
        can push this to 1 (every frame) for maximum speed.

[PI-4]  Headless mode — no cv2.imshow() on Pi (no monitor attached).
        Detection results go straight to TTS. Live display is
        laptop/dev-mode only, gated by HEADLESS flag.

[PI-5]  pyttsx3 offline TTS integration — reads detections aloud.
        No internet needed. Uses eSpeak engine on Pi OS.

[PI-6]  Pi Camera Module 3 support via Picamera2 library.
        Falls back to cv2.VideoCapture for USB webcam if Picamera2
        is not available (allows same code to run on laptop too).

[PI-7]  Coral USB Accelerator support — if plugged in, inference
        runs on Edge TPU instead of CPU (5-10x faster).
        Falls back to CPU automatically if Coral not present.

[PI-8]  GPIO button trigger — pressing the physical Morse button 1
        triggers a scan instead of keyboard 'S'. Works headless.

[PI-9]  Model loaded once at startup, stays in RAM. Never reloaded
        mid-session. Prevents 3-second stall on every scan.

[PI-10] Temperature watchdog — reads Pi CPU temp every 60s.
        If temp > 80°C, pauses inference for 30s to protect hardware.
"""

import cv2
import signal
import sys
import logging
import time
import os
import threading
import numpy as np
from collections import Counter
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# SECTION A: CONFIGURATION — edit these for your Pi setup
# ──────────────────────────────────────────────────────────────

# ── Model path ────────────────────────────────────────────────
# [PI-1] Store model on USB drive, not MicroSD
# On Pi:     MODEL_PATH = "/media/blindassist/models/yolov8s.pt"
# On laptop: MODEL_PATH = "yolov8s.pt"
MODEL_PATH     = "yolov8s.pt"
FALLBACK_MODEL = "yolov8n.pt"

# ── Detection settings ────────────────────────────────────────
CONFIDENCE     = 0.35
WEAK_THRESHOLD = 0.50

# ── Pi hardware flags — set these when running on Pi ─────────
HEADLESS       = False    # [PI-4] True = no display window (Pi without monitor)
USE_PICAMERA   = False    # [PI-6] True = use Pi Camera Module 3 via Picamera2
USE_CORAL      = False    # [PI-7] True = use Google Coral USB Accelerator
USE_GPIO       = False    # [PI-8] True = physical button triggers scan
GPIO_SCAN_PIN  = 17       # BCM pin number of your scan button

# ── Camera settings ───────────────────────────────────────────
CAMERA_INDEX   = 0        # USB webcam index (ignored if USE_PICAMERA=True)
FRAME_WIDTH    = 640      # [PI-2] Cap resolution — enough for detection
FRAME_HEIGHT   = 480
FRAME_SKIP     = 2        # [PI-3] Run inference every Nth frame (1=every frame)
WARMUP_FRAMES  = 10

# ── TTS settings ──────────────────────────────────────────────
TTS_RATE       = 150      # words per minute (150 = clear for blind users)
TTS_VOLUME     = 1.0      # 0.0 to 1.0

# ── Proximity thresholds ──────────────────────────────────────
NEAR_THRESHOLD = 0.45
FAR_THRESHOLD  = 0.15

# ── Thermal protection ────────────────────────────────────────
TEMP_CHECK_SEC = 60       # [PI-10] Check CPU temp every N seconds
TEMP_MAX_C     = 80       # Pause inference above this temperature
TEMP_PAUSE_SEC = 30       # How long to pause when overheating

# ── Paths ─────────────────────────────────────────────────────
LOG_FILE       = "/home/pi/blindassist_detection.log" if os.path.exists("/home/pi") else "blindassist_detection.log"

# ── Visual colours (BGR) ──────────────────────────────────────
COLOR_PERSON   = (0,   140, 255)
COLOR_STRONG   = (0,   255,  80)
COLOR_WEAK     = (0,   180, 255)
COLOR_WHITE    = (255, 255, 255)
FONT           = cv2.FONT_HERSHEY_SIMPLEX

# ──────────────────────────────────────────────────────────────
# SECTION B: LOGGER
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("BlindAssist.ObjectDetection")

# ──────────────────────────────────────────────────────────────
# SECTION C: TTS ENGINE  [PI-5]
# ──────────────────────────────────────────────────────────────
tts_engine = None

def init_tts():
    """
    Initialises pyttsx3 offline TTS.
    Works on Pi OS with eSpeak engine — no internet needed.
    """
    global tts_engine
    try:
        import pyttsx3
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate',   TTS_RATE)
        tts_engine.setProperty('volume', TTS_VOLUME)
        logger.info("TTS engine ready (offline pyttsx3).")
    except ImportError:
        logger.warning("pyttsx3 not installed — TTS disabled. Run: pip install pyttsx3")
    except Exception as e:
        logger.warning("TTS init failed: %s — running without audio.", e)

def speak(text: str):
    """Speaks text aloud. Non-blocking via thread so camera loop continues."""
    logger.info("TTS → %s", text)
    if tts_engine is None:
        print(f"[SPEAK] {text}")
        return
    def _run():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            logger.warning("TTS speak error: %s", e)
    threading.Thread(target=_run, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# SECTION D: MODEL LOADING  [PI-1, PI-7, PI-9]
# ──────────────────────────────────────────────────────────────
def _load_model():
    """
    Loads YOLO model. Tries Coral TPU first if USE_CORAL=True,
    falls back to CPU. Model is loaded ONCE and kept in RAM.
    """
    # [PI-7] Coral USB Accelerator support
    if USE_CORAL:
        coral_path = Path(MODEL_PATH).with_suffix('_edgetpu.tflite')
        if coral_path.exists():
            try:
                from ultralytics import YOLO
                m = YOLO(str(coral_path))
                logger.info("Coral Edge TPU model loaded: %s", coral_path)
                return m
            except Exception as e:
                logger.warning("Coral load failed: %s — falling back to CPU.", e)
        else:
            logger.warning(
                "Coral enabled but %s not found. "
                "Export with: yolo export model=yolov8s.pt format=edgetpu",
                coral_path
            )

    # Standard CPU model
    from ultralytics import YOLO
    for path in [MODEL_PATH, FALLBACK_MODEL]:
        try:
            # [PI-1] Check USB drive path first
            usb_path = f"/media/blindassist/models/{Path(path).name}"
            actual   = usb_path if os.path.exists(usb_path) else path
            m = YOLO(actual)
            logger.info("Model loaded from: %s | Classes: %d", actual, len(m.names))
            return m
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)

    logger.critical("All models failed. Cannot continue.")
    sys.exit(1)

logger.info("Loading model (this takes ~5s on Pi 5)...")
model = _load_model()

# ──────────────────────────────────────────────────────────────
# SECTION E: CLAHE PREPROCESSOR
# ──────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

def preprocess(frame: np.ndarray) -> np.ndarray:
    """Boosts contrast with CLAHE — helps in dim indoor lighting."""
    try:
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = _clahe.apply(ycrcb[:, :, 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    except Exception:
        return frame

# ──────────────────────────────────────────────────────────────
# SECTION F: THERMAL WATCHDOG  [PI-10]
# ──────────────────────────────────────────────────────────────
_paused_for_heat = False

def _read_cpu_temp() -> float:
    """Reads Pi CPU temperature in Celsius."""
    try:
        temp_str = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        return float(temp_str) / 1000.0
    except Exception:
        return 0.0   # not on Pi — skip thermal checks

def _thermal_watchdog():
    """Background thread that monitors CPU temp and pauses inference if too hot."""
    global _paused_for_heat
    while True:
        time.sleep(TEMP_CHECK_SEC)
        temp = _read_cpu_temp()
        if temp == 0.0:
            continue   # not on Pi
        logger.info("CPU temp: %.1f°C", temp)
        if temp > TEMP_MAX_C:
            _paused_for_heat = True
            msg = f"Warning: CPU temperature {temp:.0f} degrees. Cooling down for {TEMP_PAUSE_SEC} seconds."
            logger.warning(msg)
            speak(msg)
            time.sleep(TEMP_PAUSE_SEC)
            _paused_for_heat = False
            speak("Resuming object detection.")

threading.Thread(target=_thermal_watchdog, daemon=True).start()

# ──────────────────────────────────────────────────────────────
# SECTION G: GPIO BUTTON  [PI-8]
# ──────────────────────────────────────────────────────────────
_scan_requested = False   # set True by GPIO callback or keyboard press

def _init_gpio():
    """Sets up physical button on GPIO pin to trigger a scan."""
    if not USE_GPIO:
        return
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(GPIO_SCAN_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

        def _on_button(channel):
            global _scan_requested
            _scan_requested = True
            logger.info("GPIO scan button pressed on pin %d", GPIO_SCAN_PIN)

        GPIO.add_event_detect(GPIO_SCAN_PIN, GPIO.RISING,
                              callback=_on_button, bouncetime=500)
        logger.info("GPIO button ready on pin BCM %d.", GPIO_SCAN_PIN)
    except ImportError:
        logger.warning("RPi.GPIO not available — GPIO button disabled.")
    except Exception as e:
        logger.warning("GPIO init failed: %s", e)

# ──────────────────────────────────────────────────────────────
# SECTION H: CAMERA  [PI-2, PI-6]
# ──────────────────────────────────────────────────────────────
def open_camera():
    """
    Opens Pi Camera Module 3 via Picamera2 (if USE_PICAMERA=True),
    otherwise opens USB webcam via OpenCV.
    Resolution is capped at 640x480 to keep FPS high on Pi.
    """
    global _active_cap

    if USE_PICAMERA:
        try:
            from picamera2 import Picamera2
            cam = Picamera2()
            config = cam.create_preview_configuration(
                main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "BGR888"}
            )
            cam.configure(config)
            cam.start()
            time.sleep(2)   # allow sensor to warm up
            logger.info("Pi Camera Module 3 ready via Picamera2.")
            return cam, "picamera"
        except Exception as e:
            logger.warning("Picamera2 failed: %s — falling back to USB webcam.", e)

    # USB webcam fallback (also used on laptop)
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Camera index {CAMERA_INDEX} could not be opened. "
            "Check USB connection or change CAMERA_INDEX."
        )
    # [PI-2] Cap resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    logger.info("Warming up camera...")
    for i in range(WARMUP_FRAMES):
        ret, _ = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Camera warmup failed at frame {i}.")

    _active_cap = cap
    logger.info("USB camera ready — %dx%d",
                int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    return cap, "usb"

def read_frame(cap, cam_type: str):
    """Reads one frame regardless of camera type."""
    if cam_type == "picamera":
        frame = cap.capture_array()
        return True, frame
    else:
        return cap.read()

def release_camera(cap, cam_type: str):
    if cam_type == "picamera":
        try:
            cap.stop()
        except Exception:
            pass
    else:
        if cap.isOpened():
            cap.release()

# ──────────────────────────────────────────────────────────────
# SECTION I: GRACEFUL SHUTDOWN
# ──────────────────────────────────────────────────────────────
_active_cap = None
_cam_type   = "usb"

def _shutdown_handler(signum, frame):
    logger.info("Shutdown signal received.")
    if _active_cap is not None:
        release_camera(_active_cap, _cam_type)
    if USE_GPIO:
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
        except Exception:
            pass
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT,  _shutdown_handler)
signal.signal(signal.SIGTERM, _shutdown_handler)

# ──────────────────────────────────────────────────────────────
# SECTION J: HELPERS
# ──────────────────────────────────────────────────────────────
def _get_position(cx: float, fw: int) -> str:
    if fw <= 0:
        return "in front of you"
    if cx < fw / 3:
        return "on your left"
    if cx > 2 * fw / 3:
        return "on your right"
    return "in front of you"

def _get_proximity(bh: float, fh: int) -> str:
    if fh <= 0:
        return ""
    r = bh / fh
    if r >= NEAR_THRESHOLD:
        return ", nearby"
    if r <= FAR_THRESHOLD:
        return ", far away"
    return ""

# ──────────────────────────────────────────────────────────────
# SECTION K: DRAW BOXES (laptop/display mode only)
# ──────────────────────────────────────────────────────────────
def draw_detections(frame: np.ndarray, results) -> np.ndarray:
    fh, fw = frame.shape[:2]
    for box in results[0].boxes:
        try:
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf       = float(box.conf[0])
            class_name = model.names[int(box.cls[0])]

            if class_name == "person":
                color = COLOR_PERSON
            elif conf >= WEAK_THRESHOLD:
                color = COLOR_STRONG
            else:
                color = COLOR_WEAK

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}  {conf*100:.0f}%"
            (tw, th), bl = cv2.getTextSize(label, FONT, 0.55, 1)
            ly = max(y1 - 4, th + 6)
            cv2.rectangle(frame, (x1, ly-th-bl-2), (x1+tw+6, ly+bl), color, -1)
            cv2.putText(frame, label, (x1+3, ly-bl), FONT, 0.55, (0,0,0), 2)
            cv2.putText(frame, label, (x1+3, ly-bl), FONT, 0.55, COLOR_WHITE, 1)
        except Exception as e:
            logger.warning("Draw error: %s", e)

    # Object count
    n = len(results[0].boxes)
    cv2.putText(frame, f"Objects: {n}", (10, 30), FONT, 0.65, (0,0,0), 3)
    cv2.putText(frame, f"Objects: {n}", (10, 30), FONT, 0.65, COLOR_WHITE, 2)

    # Mode badge
    mode = "HEADLESS OFF (dev)" if not HEADLESS else "HEADLESS ON (Pi)"
    cv2.putText(frame, mode, (10, fh-10), FONT, 0.45, (0,0,0), 3)
    cv2.putText(frame, mode, (10, fh-10), FONT, 0.45, (180,180,180), 1)
    return frame

# ──────────────────────────────────────────────────────────────
# SECTION L: CORE SCAN → TTS
# ──────────────────────────────────────────────────────────────
def build_tts_string(results, fw: int, fh: int) -> str:
    """Converts YOLO results into a spoken sentence."""
    raw = []
    for box in results[0].boxes:
        try:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            name    = model.names[int(box.cls[0])]
            pos     = _get_position((x1+x2)/2, fw)
            prox    = _get_proximity(y2-y1, fh)
            article = "an" if name[0].lower() in "aeiou" else "a"
            raw.append(f"{article} {name} {pos}{prox}")
        except Exception as e:
            logger.warning("Box parse error: %s", e)

    if not raw:
        return "I don't see anything clearly."

    merged = []
    for desc, count in Counter(raw).items():
        if count == 1:
            merged.append(desc)
        else:
            parts    = desc.split()
            parts[0] = str(count)
            parts[1] = parts[1] + "s"
            merged.append(" ".join(parts))

    return "I can see: " + ", ".join(merged) + "."

# ──────────────────────────────────────────────────────────────
# SECTION M: MAIN LOOP
# ──────────────────────────────────────────────────────────────
_active_cap = None
_cam_type   = "usb"

def run():
    global _active_cap, _cam_type, _scan_requested

    init_tts()
    _init_gpio()

    speak("Blind Assist object detection starting.")

    cap, _cam_type = open_camera()
    _active_cap = cap

    frame_counter = 0
    last_results  = None

    logger.info("Detection loop started. HEADLESS=%s", HEADLESS)

    while True:
        # ── Thermal pause check ────────────────────────
        if _paused_for_heat:
            time.sleep(1)
            continue

        ret, frame = read_frame(cap, _cam_type)
        if not ret or frame is None:
            logger.error("Frame read failed.")
            break

        fh, fw  = frame.shape[:2]
        frame_counter += 1

        # ── [PI-3] Frame skip — run inference every Nth frame ──
        if frame_counter % FRAME_SKIP == 0:
            processed = preprocess(frame.copy())
            try:
                last_results = model.predict(
                    processed,
                    conf=CONFIDENCE,
                    imgsz=640,
                    verbose=False
                )
            except Exception as e:
                logger.warning("Inference failed: %s", e)

        # ── Scan trigger: GPIO button OR keyboard 'S' ──
        scan_now = False

        if USE_GPIO and _scan_requested:
            _scan_requested = False
            scan_now = True

        # ── [PI-4] Display — only if not headless ─────
        if not HEADLESS:
            if last_results is not None:
                annotated = draw_detections(frame.copy(), last_results)
            else:
                annotated = frame.copy()

            cv2.imshow("BlindAssist — Object Detection", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                scan_now = True
            elif key == ord('q'):
                logger.info("Quit key pressed.")
                break
        else:
            # Headless — keep loop alive, GPIO triggers scans
            time.sleep(0.03)

        # ── Speak result if scan triggered ────────────
        if scan_now and last_results is not None:
            tts_text = build_tts_string(last_results, fw, fh)
            speak(tts_text)
            logger.info("Scan complete: %s", tts_text)

    release_camera(cap, _cam_type)
    cv2.destroyAllWindows()
    if USE_GPIO:
        try:
            import RPi.GPIO as GPIO
            GPIO.cleanup()
        except Exception:
            pass
    logger.info("Detection session ended.")


if __name__ == "__main__":
    print("=" * 60)
    print("  BlindAssist — Pi-Ready Object Detection")
    print("  Project  : CSR-DES-INFINEON-2025")
    print("  Team     : Dhruv Vaghela & Dax Patel")
    print("=" * 60)
    print(f"  Model    : {MODEL_PATH}")
    print(f"  Headless : {HEADLESS}  (set True when running on Pi without monitor)")
    print(f"  PiCamera : {USE_PICAMERA}  (set True when using Pi Camera Module 3)")
    print(f"  Coral    : {USE_CORAL}  (set True when Coral USB is plugged in)")
    print(f"  GPIO     : {USE_GPIO}  (set True when physical buttons connected)")
    print("=" * 60)
    if not HEADLESS:
        print("  CONTROLS")
        print("  [S] Scan → speak objects aloud")
        print("  [Q] Quit")
        print("=" * 60)
    run()
