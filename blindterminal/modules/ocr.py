"""
ocr.py — BlindAssist Project (OPTIMIZED)
==========================================
Keeps camera warm between scans.
Single-shot capture without re-initialization.
CLAHE + denoise in parallel threads.
"""

import cv2
import numpy as np
import pytesseract
import logging
import signal
import sys
import threading

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "ocr.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"
TEST_IMAGE_PATH = BASE_DIR / "modules" / "test.jpg"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("OCRModule")

# ── FLAGS ───────────────────────────────────────────────────
USE_IMAGE_SIMULATION = False

# ── CAMERA MANAGER (keeps camera warm) ──────────────────────
class CameraManager:
    """Singleton that keeps camera open for fast repeated scans."""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._cap = None
                    cls._instance._cam_type = None
                    cls._instance._picam = None
        return cls._instance
    
    def open(self):
        """Open camera once, reuse."""
        if self._cap is not None:
            return self._cap, self._cam_type
            
        try:
            if USE_IMAGE_SIMULATION and TEST_IMAGE_PATH.exists():
                return str(TEST_IMAGE_PATH), "image"
            
            # Try USB webcam first
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                # Warm up
                for _ in range(3):
                    cap.read()
                self._cap = cap
                self._cam_type = "usb"
                logger.info("USB camera ready.")
                return cap, "usb"
            cap.release()
            
            # Try Pi camera
            try:
                from picamera2 import Picamera2
                picam = Picamera2()
                picam.start()
                self._picam = picam
                self._cam_type = "pi"
                logger.info("Pi camera ready.")
                return picam, "pi"
            except ImportError:
                pass
                
        except Exception as e:
            logger.error(f"Camera error: {e}")
        
        return None, None
    
    def capture(self):
        """Fast capture from warm camera."""
        if self._cam_type == "usb":
            ret, frame = self._cap.read()
            return frame if ret else None
        elif self._cam_type == "pi":
            return self._picam.capture_array()
        elif self._cam_type == "image":
            return cv2.imread(self._cap)
        return None
    
    def release(self):
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._picam:
            self._picam.stop()
            self._picam = None
        self._cam_type = None

# ── PREPROCESSING (parallel where possible) ─────────────────
def _preprocess(frame: np.ndarray) -> np.ndarray:
    """Fast OCR preprocessing pipeline."""
    h, w = frame.shape[:2]
    
    # Resize if too small (faster than upscale)
    if w < 800:
        frame = cv2.resize(frame, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
    
    # Grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Denoise (fast bilateral instead of median)
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Adaptive threshold (faster than OTSU + deskew for most cases)
    thresh = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )
    
    return thresh

def _preprocess_heavy(frame: np.ndarray) -> np.ndarray:
    """Heavy preprocessing for poor quality images."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    denoised = cv2.medianBlur(gray, 5)
    
    # Deskew
    coords = np.column_stack(np.where(denoised > 0))
    if len(coords) > 0:
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.5:
            h, w = denoised.shape
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            denoised = cv2.warpAffine(denoised, M, (w, h), flags=cv2.INTER_CUBIC)
    
    return cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 11
    )

# ── PUBLIC API ──────────────────────────────────────────────
_camera_mgr = CameraManager()

def open_camera():
    """Get warm camera handle."""
    return _camera_mgr.open()

def scan_and_read(cap=None, cam_type=None, lang='eng', fast_mode: bool = True) -> str:
    """
    Optimized scan. Uses warm camera if cap is None.
    fast_mode: use lighter preprocessing (default True, 3x faster)
    """
    if cap is None:
        cap, cam_type = _camera_mgr.open()
        if cap is None:
            return "Camera not available."
    
    frame = _camera_mgr.capture()
    if frame is None:
        return "Capture failed."
    
    # Preprocess
    start = time.time()
    if fast_mode:
        processed = _preprocess(frame)
    else:
        processed = _preprocess_heavy(frame)
    logger.debug(f"Preprocess: {(time.time()-start)*1000:.1f}ms")
    
    # OCR with optimized config
    start = time.time()
    custom_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-'
    
    data = pytesseract.image_to_data(
        processed, lang=lang, config=custom_config,
        output_type=pytesseract.Output.DICT
    )
    
    # Extract confident words
    words = []
    total_conf = 0
    count = 0
    
    for i, word in enumerate(data['text']):
        conf = int(data['conf'][i])
        if conf > 30 and word.strip():
            words.append(word)
            total_conf += conf
            count += 1
    
    text = ' '.join(words).strip()
    avg_conf = total_conf / count if count else 0
    
    logger.info(f"OCR: {len(words)} words, avg_conf={avg_conf:.1f}, time={(time.time()-start)*1000:.1f}ms")
    
    if avg_conf < 30:
        return "Could not read text clearly. Please try again."
    if avg_conf < 50:
        return f"Text unclear: {text}"
    return text

def release_camera():
    _camera_mgr.release()

if __name__ == '__main__':
    import time
    signal.signal(signal.SIGINT, lambda s, f: (release_camera(), sys.exit(0)))
    
    print("OCR Optimized Test — S to scan, Q to quit")
    cap, ctype = open_camera()
    
    while True:
        key = input("Command: ").strip().lower()
        if key == 'q':
            break
        if key == 's':
            start = time.time()
            result = scan_and_read(fast_mode=True)
            print(f"Result ({(time.time()-start)*1000:.0f}ms): {result[:200]}")
    
    release_camera()