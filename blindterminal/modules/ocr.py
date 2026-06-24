import cv2
import numpy as np
import pytesseract
import logging
import signal
import sys
import threading
import pyttsx3

from pathlib import Path

# ─────────────────────────────────────────────────────────────
# HARDWARE FLAGS
# ─────────────────────────────────────────────────────────────
HEADLESS = False
USE_PICAMERA = False
USE_GPIO = False

# IMPORTANT:
# GitHub Codespaces cannot access webcam.
# So we use image simulation mode.
USE_IMAGE_SIMULATION = False  # Set True only in Codespaces (no webcam)

# ─────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

LOG_PATH = BASE_DIR / "logs" / "ocr.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

TEST_IMAGE_PATH = BASE_DIR / "modules" / "test.jpg"

# Create logs folder automatically
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("OCRModule")

# ─────────────────────────────────────────────────────────────
# CAMERA / IMAGE SOURCE
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# CAMERA / IMAGE SOURCE
# ─────────────────────────────────────────────────────────────
def open_camera():

    try:

        # IMAGE MODE FOR CODESPACES
        if USE_IMAGE_SIMULATION:

            logger.info("Using image simulation mode.")

            if not TEST_IMAGE_PATH.exists():

                raise FileNotFoundError(
                    f"Image not found: {TEST_IMAGE_PATH}"
                )

            return str(TEST_IMAGE_PATH), "image"

        # PI CAMERA
        if USE_PICAMERA:

            logger.info("Opening Pi Camera")

            from picamera2 import Picamera2

            picam2 = Picamera2()
            picam2.start()

            return picam2, "pi"

        # USB WEBCAM
        logger.info("Opening Webcam")

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise IOError("Could not open webcam")

        return cap, "usb"

    except Exception as error:

        logger.error(f"Camera Error: {error}")

        return None, None

# ─────────────────────────────────────────────────────────────
# IMAGE PROCESSING
# ─────────────────────────────────────────────────────────────
def upscale(image):

    h, w = image.shape[:2]

    if w < 800:

        image = cv2.resize(
            image,
            (w * 2, h * 2),
            interpolation=cv2.INTER_CUBIC
        )

    return image


def to_gray(image):

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


def denoise(image):

    return cv2.medianBlur(image, 5)


def deskew(image):

    _, thresh = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = np.column_stack(
        np.where(thresh > 0)
    )

    if coords.size == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)

    else:
        angle = -angle

    (h, w) = image.shape[:2]

    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        1.0
    )

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return rotated


def threshold(image):

    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )


def morphology_cleanup(image):

    kernel = np.ones((2, 2), np.uint8)

    return cv2.morphologyEx(
        image,
        cv2.MORPH_OPEN,
        kernel
    )

# ─────────────────────────────────────────────────────────────
# OCR PIPELINE
# ─────────────────────────────────────────────────────────────
def scan_and_read(cap, cam_type, lang='eng'):

    if cap is None:

        logger.error("No camera source")

        return ""

    try:

        # USB CAMERA
        if cam_type == "usb":

            for _ in range(3):
                cap.read()

            ret, frame = cap.read()

        # PI CAMERA
        elif cam_type == "pi":

            frame = cap.capture_array()
            ret = frame is not None

        # IMAGE MODE
        elif cam_type == "image":

            frame = cv2.imread(cap)
            ret = frame is not None

        else:

            logger.error("Invalid camera type")

            return ""

        if not ret:

            logger.error("Image capture failed")

            return ""

        # PREPROCESSING
        processed = upscale(frame)

        processed = to_gray(processed)

        processed = denoise(processed)

        processed = deskew(processed)

        processed = threshold(processed)

        processed = morphology_cleanup(processed)

        # OCR
        custom_config = '--oem 3 --psm 3'

        data = pytesseract.image_to_data(
            processed,
            lang=lang,
            config=custom_config,
            output_type=pytesseract.Output.DICT
        )

        confidences = [
            int(conf)
            for conf in data['conf']
            if conf != '-1'
        ]

        avg_conf = (
            np.mean(confidences)
            if confidences else 0
        )

        text = " ".join([
            word
            for i, word in enumerate(data['text'])
            if int(data['conf'][i]) > 0
        ])

        text = text.strip()

        # CONFIDENCE CHECK
        if avg_conf < 30:

            message = (
                "Could not read text properly."
            )

            return message

        elif avg_conf < 50:

            message = (
                f"Text unclear. {text}"
            )

            return message

        else:

            return text

    except Exception as error:

        logger.error(f"OCR Error: {error}")

        return f"OCR Error: {error}"

# ─────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ─────────────────────────────────────────────────────────────
def signal_handler(sig, frame):

    logger.info("Shutting down OCR module")

    sys.exit(0)

# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':

    signal.signal(
        signal.SIGINT,
        signal_handler
    )

    cap, cam_type = open_camera()

    if cap is None:
        sys.exit(1)

    print("\n--- BlindAssist OCR ---")
    print("Press S -> Scan")
    print("Press Q -> Quit")
    print("-----------------------\n")

    try:

        while True:

            # IMAGE MODE
            if cam_type == "image":

                key = input(
                    "Press S to scan or Q to quit: "
                ).lower()

                if key == 'q':
                    break

                if key == 's':

                    print("Scanning image...")

                    result = scan_and_read(
                        cap,
                        cam_type,
                        lang='eng'
                    )

                    print(
                        f"\nExtracted Text:\n{result}\n"
                    )

            # WEBCAM MODE
            else:

                if not HEADLESS:

                    ret, frame = (
                        cap.read()
                        if cam_type == "usb"
                        else (
                            True,
                            cap.capture_array()
                        )
                    )

                    if ret:

                        cv2.imshow(
                            "OCR Feed",
                            frame
                        )

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('s'):

                    print("Scanning...")

                    result = scan_and_read(
                        cap,
                        cam_type,
                        lang='eng'
                    )

                    print(
                        f"Extracted Text: {result}"
                    )

    finally:

        if cam_type == "usb":
            cap.release()

        elif cam_type == "pi":
            cap.stop()

        cv2.destroyAllWindows()

        print("OCR Closed")