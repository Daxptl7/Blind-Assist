"""
main.py — BlindAssist Project
================================
Project  : Accessible Educational Terminal for Visually Impaired
Team     : Dhruv Vaghela & Dax Patel  |  CSR / Infineon 2025
Module   : Main Controller — State Machine Orchestrator

This is the entry point for the entire BlindAssist system.
It runs a state-machine loop that:
  1. Greets the user
  2. Presents a mode menu (keyboard shortcut or Morse)
  3. Dispatches to the selected module
  4. Routes TTS output through confidential mode check
  5. Passively runs emotion analysis on voice inputs

Mode Map:
  1 / .-     → Mode A: OCR Scan → AI → TTS
  2 / -...   → Mode B: Morse Type → AI → TTS
  3 / -.-.   → Mode C: Voice Ask → AI → TTS
  4 / -..    → Mode D: Translate → TTS
  5 / .      → Mode E: Gesture Control
  6 / ..-.   → Mode F: Object Detection
  7 / --.    → Mode G: GPS Navigate → TTS
  0 / SOS    → Graceful Shutdown
"""

import sys
import signal
import logging
import json
import threading

from pathlib import Path

# ──────────────────────────────────────────────────────────────
# HARDWARE FLAGS (Pi Flag Pattern)
# ──────────────────────────────────────────────────────────────
HEADLESS = False
USE_PICAMERA = False
USE_GPIO = False

# ──────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "logs" / "main.log"
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
logger = logging.getLogger("MainController")

# ──────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────
def _load_settings() -> dict:
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Using defaults: {e}")
        return {}

settings = _load_settings()

# ──────────────────────────────────────────────────────────────
# IMPORT ALL MODULES
# ──────────────────────────────────────────────────────────────
from modules import tts
from modules import voice
from modules import morse
from modules import ocr
from modules import ai_query
from modules import translator
from modules import gesture_control
from modules import emotion_engine
from modules import gps_navigator
from modules import confidential_mode

# Object detection uses its own internal API — we import what we need
try:
    from modules.object_detection import run_detection
    OBJECT_DETECTION_AVAILABLE = True
except ImportError:
    OBJECT_DETECTION_AVAILABLE = False
    logger.warning("Object detection module not available.")

# ──────────────────────────────────────────────────────────────
# SAFE SPEAK — routes through confidential mode
# ──────────────────────────────────────────────────────────────
def safe_speak(text: str, skip_privacy: bool = False):
    """
    Speak text through TTS, but check confidential mode first.
    If PRIVATE, the speaker output is tagged accordingly.
    """
    if not text:
        return

    if not skip_privacy:
        mode = confidential_mode.ask_privacy()
        logger.info(f"Privacy mode: {mode}")

    tts.speak(text)


# ──────────────────────────────────────────────────────────────
# EMOTION-AWARE RESPONSE
# ──────────────────────────────────────────────────────────────
def check_emotion_and_adjust():
    """
    If emotion engine detects sustained stress/confusion,
    slow down TTS and log the adaptation.
    """
    if emotion_engine.should_slow_down():
        current_rate = tts.get_rate()
        if current_rate > 110:
            tts.set_rate(110)
            logger.info("Emotion: slowing TTS for stressed/confused user.")
            tts.speak("I notice you might be having difficulty. I'll speak slower.")
    else:
        # Restore normal rate if user calms down
        if tts.get_rate() < 150:
            tts.set_rate(150)


# ──────────────────────────────────────────────────────────────
# MODE A: OCR SCAN → AI → TTS
# ──────────────────────────────────────────────────────────────
def mode_ocr_scan():
    """Scan text from camera/image, send to AI, speak result."""
    logger.info("Mode A: OCR Scan")
    tts.speak("Starting OCR scan. Hold your document steady.")

    # Open camera
    cap, cam_type = ocr.open_camera()
    if cap is None:
        tts.speak("Camera not available for scanning.")
        return

    try:
        tts.speak("Scanning now.")
        scanned_text = ocr.scan_and_read(cap, cam_type, lang='eng')

        if not scanned_text or scanned_text.startswith("Could not"):
            tts.speak("I could not read the text clearly. Please try again.")
            return

        logger.info(f"OCR result: {scanned_text[:100]}...")
        tts.speak(f"I found the following text: {scanned_text[:200]}")

        # Ask if user wants AI analysis
        tts.speak("Would you like me to explain this text? Say yes or press 1.")

        # Wait for voice or keyboard response
        try:
            response = input("[Y/N or press Enter for Yes]: ").strip().lower()
        except EOFError:
            response = 'y'

        if response in ('y', 'yes', '1', ''):
            tts.speak("Analyzing the text with AI...")
            ai_answer = ai_query.ask_ai(
                "Explain this text in simple terms for a visually impaired student:",
                context=scanned_text
            )
            safe_speak(ai_answer)

    finally:
        # Release camera
        if cam_type == "usb" and hasattr(cap, 'release'):
            cap.release()
        elif cam_type == "pi" and hasattr(cap, 'stop'):
            cap.stop()


# ──────────────────────────────────────────────────────────────
# MODE B: MORSE TYPE → AI → TTS
# ──────────────────────────────────────────────────────────────
def mode_morse_type():
    """Type a question using Morse code, send to AI, speak result."""
    logger.info("Mode B: Morse Type")
    tts.speak("Morse typing mode. Enter your question using dots and dashes. Press Enter to confirm.")

    print("\n" + "─" * 45)
    print("  MORSE TYPING MODE")
    print("  '.' → DOT  |  '-' → DASH  |  Enter → Confirm")
    print("  Type your question and press Enter.")
    print("─" * 45 + "\n")

    try:
        # For laptop simulation: just use regular text input
        # On Pi, this would use the Morse decoder with physical button
        question = input("  Type your question (or Morse): ").strip()

        if not question:
            tts.speak("No question received.")
            return

        # Check if it's actual Morse code (contains only dots, dashes, spaces)
        if all(c in '.- /\t' for c in question):
            # Decode Morse
            words = question.split(' / ') if ' / ' in question else question.split('  ')
            decoded = []
            for word in words:
                letters = word.strip().split(' ')
                for letter_code in letters:
                    letter_code = letter_code.strip()
                    if letter_code:
                        decoded_letter = morse.MORSE_CODE_DICT.get(letter_code, '?')
                        decoded.append(decoded_letter)
                decoded.append(' ')
            question = ''.join(decoded).strip()
            tts.speak(f"You typed: {question}")

        logger.info(f"Morse question: {question}")
        tts.speak("Thinking...")
        ai_answer = ai_query.ask_ai(question)
        safe_speak(ai_answer)

    except EOFError:
        tts.speak("Morse typing cancelled.")


# ──────────────────────────────────────────────────────────────
# MODE C: VOICE ASK → AI → TTS
# ──────────────────────────────────────────────────────────────
def mode_voice_ask():
    """Listen to voice question, send to AI, speak result."""
    logger.info("Mode C: Voice Ask")
    tts.speak("Voice mode. I'm listening. Ask your question now.")

    # Listen with microphone
    question = voice.listen(lang='en-IN', speak_fn=None)

    if not question:
        tts.speak("I didn't catch that. Please try again.")
        return

    logger.info(f"Voice question: {question}")
    tts.speak(f"You asked: {question}")

    # Emotion analysis on the voice input (passive)
    # Note: We'd need the audio file path for real analysis
    # For now, emotion runs as a background concern
    check_emotion_and_adjust()

    tts.speak("Let me think about that...")
    ai_answer = ai_query.ask_ai(question)
    safe_speak(ai_answer)


# ──────────────────────────────────────────────────────────────
# MODE D: TRANSLATE → TTS
# ──────────────────────────────────────────────────────────────
def mode_translate():
    """Translate text between English, Hindi, Gujarati."""
    logger.info("Mode D: Translation")
    tts.speak("Translation mode. What would you like to translate?")

    print("\n" + "─" * 45)
    print("  TRANSLATION MODE")
    print("  1 = English → Hindi")
    print("  2 = English → Gujarati")
    print("  3 = Hindi → English")
    print("  4 = Gujarati → English")
    print("─" * 45 + "\n")

    lang_pairs = {
        '1': ('en', 'hi', 'English to Hindi'),
        '2': ('en', 'gu', 'English to Gujarati'),
        '3': ('hi', 'en', 'Hindi to English'),
        '4': ('gu', 'en', 'Gujarati to English'),
    }

    try:
        choice = input("  Select (1-4): ").strip()
        if choice not in lang_pairs:
            tts.speak("Invalid selection.")
            return

        src, dest, pair_name = lang_pairs[choice]
        tts.speak(f"Selected {pair_name}. Enter text or speak now.")

        # Get text via keyboard or voice
        text = input("  Text to translate (or press Enter for voice): ").strip()
        if not text:
            tts.speak("Listening for text to translate.")
            lang_code = 'hi-IN' if src == 'hi' else ('gu-IN' if src == 'gu' else 'en-IN')
            text = voice.listen(lang=lang_code)
            if not text:
                tts.speak("I didn't hear anything to translate.")
                return

        result = translator.translate(text, src, dest)
        tts.speak(f"Translation: {result}")
        safe_speak(result, skip_privacy=True)

    except EOFError:
        tts.speak("Translation cancelled.")


# ──────────────────────────────────────────────────────────────
# MODE E: GESTURE CONTROL
# ──────────────────────────────────────────────────────────────
def mode_gesture():
    """Start gesture detection and route gestures to modes."""
    logger.info("Mode E: Gesture Control")
    tts.speak("Gesture control activated. Show your hand to the camera.")

    def on_gesture(gesture_name):
        """Callback when a gesture is confirmed."""
        logger.info(f"Gesture detected: {gesture_name}")
        tts.speak(f"Gesture: {gesture_name}")

        if gesture_name == "MODE_SCAN":
            tts.speak("Starting OCR scan from gesture.")
            # Run in a thread to not block gesture loop
            threading.Thread(target=mode_ocr_scan, daemon=True).start()
        elif gesture_name == "MODE_VOICE":
            tts.speak("Starting voice mode from gesture.")
            threading.Thread(target=mode_voice_ask, daemon=True).start()
        elif gesture_name == "CONFIRM":
            tts.speak("Confirmed.")
        elif gesture_name == "STOP":
            tts.speak("Stopping gesture control.")
        elif gesture_name == "REPEAT":
            tts.speak("Repeating last output.")

    gesture_control.detect_gesture(callback_fn=on_gesture)
    tts.speak("Gesture control ended.")


# ──────────────────────────────────────────────────────────────
# MODE F: OBJECT DETECTION
# ──────────────────────────────────────────────────────────────
def mode_object_detection():
    """Start live object detection with webcam."""
    logger.info("Mode F: Object Detection")

    if not OBJECT_DETECTION_AVAILABLE:
        tts.speak("Object detection module is not available.")
        return

    tts.speak("Starting object detection. Point the camera at objects around you. Press Q to stop.")

    try:
        run_detection()
    except Exception as e:
        logger.error(f"Object detection error: {e}")
        tts.speak("Object detection encountered an error.")

    tts.speak("Object detection ended.")


# ──────────────────────────────────────────────────────────────
# MODE G: GPS NAVIGATION
# ──────────────────────────────────────────────────────────────
def mode_gps():
    """GPS navigation — current location + walking directions."""
    logger.info("Mode G: GPS Navigation")
    tts.speak("GPS navigation mode.")

    print("\n" + "─" * 45)
    print("  GPS NAVIGATION")
    print("  1 = Where am I?")
    print("  2 = Navigate to a destination")
    print("─" * 45 + "\n")

    try:
        choice = input("  Select (1/2): ").strip()

        if choice == '1':
            tts.speak("Finding your location...")
            location = gps_navigator.get_location()
            safe_speak(f"You are at {location['address']}")

        elif choice == '2':
            tts.speak("Where would you like to go? Type or speak the destination.")
            dest = input("  Destination: ").strip()

            if not dest:
                tts.speak("Listening for your destination.")
                dest = voice.listen(lang='en-IN')
                if not dest:
                    tts.speak("I didn't hear a destination.")
                    return

            tts.speak(f"Calculating route to {dest}.")
            directions = gps_navigator.get_directions(dest)
            safe_speak(directions)

        else:
            tts.speak("Invalid selection.")

    except EOFError:
        tts.speak("Navigation cancelled.")


# ══════════════════════════════════════════════════════════════
# MAIN MENU — State Machine Loop
# ══════════════════════════════════════════════════════════════
BANNER = """
╔═══════════════════════════════════════════════════════════╗
║          🦯 BlindAssist — Accessible Terminal 🦯          ║
║     CSR / Infineon 2025 — Dhruv Vaghela & Dax Patel      ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║   1  →  📄 OCR Scan        (read documents)               ║
║   2  →  ⚡ Morse Type       (type via Morse → AI)          ║
║   3  →  🎤 Voice Ask        (speak question → AI)          ║
║   4  →  🌐 Translate        (EN ↔ HI ↔ GU)                ║
║   5  →  ✋ Gesture Control   (hand gestures)                ║
║   6  →  👁️ Object Detection (identify objects)             ║
║   7  →  📍 GPS Navigate     (walking directions)           ║
║   0  →  🛑 Shutdown                                        ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
"""

MODE_HANDLERS = {
    '1': ('OCR Scan', mode_ocr_scan),
    '2': ('Morse Type', mode_morse_type),
    '3': ('Voice Ask', mode_voice_ask),
    '4': ('Translate', mode_translate),
    '5': ('Gesture Control', mode_gesture),
    '6': ('Object Detection', mode_object_detection),
    '7': ('GPS Navigate', mode_gps),
}


def main():
    """Main entry point — runs the BlindAssist state machine."""
    logger.info("=" * 50)
    logger.info("BlindAssist System Starting...")
    logger.info("=" * 50)

    # Welcome message
    tts.speak(
        "Welcome to Blind Assist. "
        "I am your accessible educational terminal. "
        "Press a number key to select a mode, or say a command."
    )

    print(BANNER)

    # ──────────────────────────────────────────────────────────
    # MAIN LOOP
    # ──────────────────────────────────────────────────────────
    running = True
    while running:
        try:
            print("\n" + "─" * 50)
            choice = input("Select mode (1-7, 0=quit): ").strip()
            print()

            if choice == '0':
                running = False
                continue

            if choice in MODE_HANDLERS:
                mode_name, handler = MODE_HANDLERS[choice]
                logger.info(f"Entering {mode_name} (Mode {choice})")

                try:
                    handler()
                except Exception as e:
                    logger.error(f"Error in {mode_name}: {e}")
                    tts.speak(f"An error occurred in {mode_name}. Returning to main menu.")

                logger.info(f"Returned from {mode_name}")

                # Reset confidential mode after each interaction
                confidential_mode.reset_to_normal()

            else:
                print("Invalid selection. Choose 1-7 or 0 to quit.")
                tts.speak("Invalid selection. Please choose a number from 1 to 7.")

        except EOFError:
            running = False
        except KeyboardInterrupt:
            print("\n")
            running = False

    # ──────────────────────────────────────────────────────────
    # SHUTDOWN
    # ──────────────────────────────────────────────────────────
    shutdown()


def shutdown():
    """Gracefully shut down all modules."""
    logger.info("Shutting down BlindAssist...")
    tts.speak("Goodbye. Shutting down Blind Assist.")

    # Wait for TTS to finish speaking
    import time
    time.sleep(2)

    tts.tts_manager.shutdown()
    confidential_mode.reset_to_normal()
    emotion_engine.reset()

    logger.info("BlindAssist shutdown complete.")
    print("\n✅ BlindAssist shutdown complete.\n")


# ──────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ──────────────────────────────────────────────────────────────
def signal_handler(sig, frame):
    logger.info("SIGINT/SIGTERM received — shutting down.")
    shutdown()
    sys.exit(0)


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
