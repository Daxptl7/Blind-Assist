"""
main.py — BlindAssist Project
Production-ready orchestrator with graceful degradation.
"""

import sys
import signal
import logging
import json
import threading
import time

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "logs" / "main.log"
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
logger = logging.getLogger("MainController")

# ── SAFE IMPORTS ────────────────────────────────────────────
# Each module is optional — app starts even if some are missing

_modules = {}

def _safe_import(module_name, alias):
    """Import a module, log if missing, but don't crash."""
    try:
        mod = __import__(f"modules.{module_name}", fromlist=[alias])
        _modules[alias] = getattr(mod, alias) if hasattr(mod, alias) else mod
        logger.info(f"Module loaded: {module_name}")
        return True
    except Exception as e:
        logger.warning(f"Module unavailable: {module_name} — {e}")
        _modules[alias] = None
        return False

# Core modules (app works without these but degraded)
_safe_import("tts", "tts")
_safe_import("voice", "voice")
_safe_import("ai_query", "ai_query")
_safe_import("morse", "morse")
_safe_import("ocr", "ocr")
_safe_import("translator", "translator")

# Optional modules (app works fine without)
_safe_import("gesture_control", "gesture")
_safe_import("emotion_engine", "emotion")
_safe_import("gps_navigator", "gps")
_safe_import("confidential_mode", "privacy")
_safe_import("object_detection", "objdetect")

# Helper to check if module is available
def _has(mod_name):
    return _modules.get(mod_name) is not None

def _speak(text):
    """Safe TTS — works even if tts module failed to load."""
    if _has("tts"):
        _modules["tts"].speak(text)
    else:
        print(f"[TTS OFFLINE] {text}")

# ── MODE HANDLERS ───────────────────────────────────────────

def mode_ocr_scan():
    """Mode 1: OCR Scan → AI → TTS"""
    logger.info("Mode 1: OCR Scan")
    _speak("Starting OCR scan. Hold your document steady.")

    if not _has("ocr"):
        _speak("OCR module is not available.")
        return

    cap, cam_type = _modules["ocr"].open_camera()
    if cap is None:
        _speak("Camera not available.")
        return

    try:
        _speak("Scanning now.")
        text = _modules["ocr"].scan_and_read(cap, cam_type, lang='eng')

        if not text or "Could not" in text or "Error" in text:
            _speak("I could not read the text clearly. Please try again.")
            return

        _speak(f"I found: {text[:150]}")
        logger.info(f"OCR: {text[:200]}")

        # Index into RAG vector store for semantic retrieval
        if _has("ai_query") and hasattr(_modules["ai_query"], "index_text_in_rag"):
            _modules["ai_query"].index_text_in_rag(text)
            logger.info("OCR text indexed into RAG vector store.")

        # Ask if user wants AI explanation
        _speak("Would you like me to explain this? Say yes or press 1.")
        try:
            response = input("[Y/N/1/Enter=Yes]: ").strip().lower()
        except EOFError:
            response = 'y'

        if response in ('y', 'yes', '1', ''):
            _speak("Analyzing...")
            if _has("ai_query"):
                answer = _modules["ai_query"].ask_ai(
                    "Explain this in simple terms for a visually impaired student:",
                    context=text
                )
                _speak(answer)
            else:
                _speak("AI module is not available.")


    finally:
        if cam_type == "usb" and hasattr(cap, 'release'):
            cap.release()
        elif cam_type == "pi" and hasattr(cap, 'stop'):
            cap.stop()

def mode_morse_type():
    """Mode 2: Type question → AI → TTS"""
    logger.info("Mode 2: Morse Type")
    _speak("Type your question, then press Enter.")

    try:
        question = input("Question: ").strip()
        if not question:
            _speak("No question received.")
            return

        # Simple Morse decode if input looks like Morse
        if all(c in '.- /' for c in question):
            words = question.split(' / ') if ' / ' in question else question.split('  ')
            decoded = []
            for word in words:
                for letter in word.strip().split(' '):
                    if letter:
                        decoded.append(morse.MORSE_CODE_DICT.get(letter, '?'))
                decoded.append(' ')
            question = ''.join(decoded).strip()
            _speak(f"You typed: {question}")

        _speak("Thinking...")
        if _has("ai_query"):
            answer = _modules["ai_query"].ask_ai(question)
            _speak(answer)
        else:
            _speak("AI module is not available.")

    except EOFError:
        _speak("Morse typing cancelled.")

def mode_voice_ask():
    """Mode 3: Voice → AI → TTS"""
    logger.info("Mode 3: Voice Ask")
    
    if not _has("voice"):
        _speak("Voice recognition is not available.")
        return

    _speak("Voice mode. Ask your question now.")
    question = _modules["voice"].listen(lang='en-IN')

    if not question:
        _speak("I didn't catch that. Please try again.")
        return

    _speak(f"You asked: {question}")
    logger.info(f"Voice: {question}")

    # Emotion check (passive)
    if _has("emotion"):
        # Note: emotion engine needs audio file path, not available here
        # In full implementation, voice.py should save the recording
        pass

    _speak("Thinking...")
    if _has("ai_query"):
        answer = _modules["ai_query"].ask_ai(question)
        _speak(answer)
    else:
        _speak("AI module is not available.")

def mode_translate():
    """Mode 4: Translate text"""
    logger.info("Mode 4: Translate")
    
    if not _has("translator"):
        _speak("Translation module is not available.")
        return

    _speak("Translation mode. Enter text to translate.")
    text = input("Text: ").strip()
    if not text:
        _speak("No text entered.")
        return

    print("\n1: EN→HI  2: EN→GU  3: HI→EN  4: GU→EN")
    choice = input("Select: ").strip()
    pairs = {'1': ('en','hi'), '2': ('en','gu'), '3': ('hi','en'), '4': ('gu','en')}
    
    if choice not in pairs:
        _speak("Invalid selection.")
        return

    src, dest = pairs[choice]
    result = _modules["translator"].translate(text, from_lang=src, to_lang=dest)
    _speak(f"Translation: {result}")

def mode_gesture():
    """Mode 5: Gesture control"""
    logger.info("Mode 5: Gesture")
    
    if not _has("gesture"):
        _speak("Gesture control is not available.")
        return

    _speak("Gesture mode activated. Show your hand to the camera.")

    def on_gesture(name):
        _speak(f"Gesture: {name}")
        if name == "MODE_SCAN":
            threading.Thread(target=mode_ocr_scan, daemon=True).start()
        elif name == "MODE_VOICE":
            threading.Thread(target=mode_voice_ask, daemon=True).start()
        elif name == "STOP":
            _speak("Gesture control stopped.")

    _modules["gesture"].detect_gesture(callback_fn=on_gesture)

def mode_object_detection():
    """Mode 6: Object detection"""
    logger.info("Mode 6: Object Detection")
    
    if not _has("objdetect"):
        _speak("Object detection is not available.")
        return

    _speak("Starting object detection. Point camera at objects. Press Q to stop.")
    try:
        _modules["objdetect"].run_detection()
    except Exception as e:
        logger.error(f"Object detection error: {e}")
        _speak("Object detection error.")

def mode_gps():
    """Mode 7: GPS Navigation"""
    logger.info("Mode 7: GPS")
    
    if not _has("gps"):
        _speak("GPS navigation is not available.")
        return

    _speak("GPS mode.")
    print("\n1: Where am I?  2: Navigate to...")
    choice = input("Select: ").strip()

    if choice == '1':
        _speak("Finding your location...")
        loc = _modules["gps"].get_location()
        _speak(f"You are at {loc['address']}")

    elif choice == '2':
        dest = input("Destination: ").strip()
        if not dest:
            _speak("No destination entered.")
            return
        _speak(f"Calculating route to {dest}...")
        directions = _modules["gps"].get_directions(dest)
        _speak(directions)

# ── MAIN LOOP ───────────────────────────────────────────────

BANNER = """
╔═══════════════════════════════════════════════════════════╗
║          🦯 BlindAssist — Accessible Terminal 🦯          ║
║     CSR / Infineon 2025 — Dhruv Vaghela & Dax Patel      ║
╠═══════════════════════════════════════════════════════════╣
║  1 → OCR Scan    2 → Morse Type    3 → Voice Ask        ║
║  4 → Translate   5 → Gesture       6 → Object Detect    ║
║  7 → GPS         0 → Shutdown                              ║
╚═══════════════════════════════════════════════════════════╝
"""

MODE_MAP = {
    '1': mode_ocr_scan,
    '2': mode_morse_type,
    '3': mode_voice_ask,
    '4': mode_translate,
    '5': mode_gesture,
    '6': mode_object_detection,
    '7': mode_gps,
}

def main():
    logger.info("=" * 50)
    logger.info("BlindAssist Starting...")
    logger.info("=" * 50)

    _speak("Welcome to Blind Assist. Press a number to select a mode.")

    print(BANNER)

    running = True
    while running:
        try:
            print("\n" + "─" * 50)
            choice = input("Mode (1-7, 0=quit): ").strip()

            if choice == '0':
                running = False
                continue

            if choice in MODE_MAP:
                try:
                    MODE_MAP[choice]()
                except Exception as e:
                    logger.error(f"Mode {choice} error: {e}")
                    _speak("An error occurred. Returning to menu.")
            else:
                _speak("Invalid selection. Choose 1 to 7.")

        except (EOFError, KeyboardInterrupt):
            running = False

    shutdown()

def shutdown():
    logger.info("Shutting down...")
    _speak("Goodbye. Shutting down Blind Assist.")
    time.sleep(1)
    if _has("tts"):
        _modules["tts"].tts_manager.shutdown()
    logger.info("Shutdown complete.")
    print("\n✅ BlindAssist closed.\n")

if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: shutdown())
    signal.signal(signal.SIGTERM, lambda s, f: shutdown())
    main()