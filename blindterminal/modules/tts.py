"""
tts.py — BlindAssist Project
==============================
Project  : Accessible Educational Terminal for Visually Impaired
Team     : Dhruv Vaghela & Dax Patel  |  CSR / Infineon 2025
Module   : Centralized Text-To-Speech

All spoken output in the entire system goes through this module.
Uses a non-blocking background thread with a queue so callers
never freeze while audio plays.

Audio path:  text → pyttsx3 → .wav file → pygame.mixer playback
"""

import pyttsx3
import logging
import signal
import sys
import threading
import json
import time

from pathlib import Path
from queue import Queue

# ─────────────────────────────────────────────────────────────
# HARDWARE FLAGS
# ─────────────────────────────────────────────────────────────
HEADLESS = False
USE_PICAMERA = False
USE_GPIO = False

# ─────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

LOG_PATH = BASE_DIR / "logs" / "tts.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"
AUDIO_DIR = BASE_DIR / "audio"

# Create folders automatically
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

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

logger = logging.getLogger("TTSModule")

# ─────────────────────────────────────────────────────────────
# PYGAME MIXER INIT (for audio playback)
# ─────────────────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init(frequency=22050, size=-16, channels=1, buffer=1024)
    PYGAME_AVAILABLE = True
    logger.info("pygame.mixer initialized for audio playback.")
except Exception as e:
    PYGAME_AVAILABLE = False
    logger.warning(f"pygame.mixer unavailable: {e} — audio will save but not play.")


# ─────────────────────────────────────────────────────────────
# TTS MANAGER
# ─────────────────────────────────────────────────────────────
class TTSManager:

    def __init__(self):
        self.settings = self._load_settings()
        self.queue = Queue()
        self.running = True
        self.counter = 0
        self._current_rate = self.settings.get("tts_rate", 150)

        # Start background worker thread
        self.worker_thread = threading.Thread(
            target=self._process_queue,
            daemon=True
        )
        self.worker_thread.start()
        logger.info("TTS Manager initialized successfully.")

    # ─────────────────────────────────────────────────────────
    # LOAD SETTINGS
    # ─────────────────────────────────────────────────────────
    def _load_settings(self):
        try:
            with open(CONFIG_PATH, 'r') as file:
                return json.load(file)
        except Exception as error:
            logger.warning(f"Using default settings: {error}")
            return {
                "tts_rate": 150,
                "tts_volume": 1.0,
                "language": "eng"
            }

    # ─────────────────────────────────────────────────────────
    # PLAY AUDIO FILE
    # ─────────────────────────────────────────────────────────
    def _play_audio(self, audio_path: str):
        """Play a .wav file through laptop speakers using pygame."""
        if not PYGAME_AVAILABLE:
            logger.warning("Cannot play audio — pygame not available.")
            return

        try:
            sound = pygame.mixer.Sound(audio_path)
            sound.play()

            # Wait for playback to finish (non-blocking to queue, blocking within worker)
            while pygame.mixer.get_busy():
                time.sleep(0.05)

            logger.info(f"Played audio: {audio_path}")

        except Exception as e:
            logger.error(f"Audio playback error: {e}")

    # ─────────────────────────────────────────────────────────
    # PROCESS QUEUE (background thread)
    # ─────────────────────────────────────────────────────────
    def _process_queue(self):
        try:
            # Initialize pyttsx3 engine INSIDE the worker thread
            engine = pyttsx3.init()

            engine.setProperty('rate', self._current_rate)
            engine.setProperty('volume', self.settings.get("tts_volume", 1.0))

            # Pick first available voice
            voices = engine.getProperty('voices')
            if voices:
                engine.setProperty('voice', voices[0].id)
                logger.info(f"Using Voice: {voices[0].name}")

            # ─────────────────────────────────────────────
            # MAIN LOOP
            # ─────────────────────────────────────────────
            while self.running:
                text, lang = self.queue.get()

                if text is None:
                    break

                try:
                    # Update rate in case emotion engine changed it
                    engine.setProperty('rate', self._current_rate)

                    logger.info(f"Speaking: \"{text}\"")

                    # Create unique filename
                    self.counter += 1
                    audio_file = AUDIO_DIR / f"speech_{self.counter}.wav"

                    # Save to .wav file
                    engine.save_to_file(text, str(audio_file))
                    engine.runAndWait()

                    # Play the audio through speakers
                    self._play_audio(str(audio_file))

                except Exception as error:
                    logger.error(f"TTS Error: {error}")

                finally:
                    self.queue.task_done()

        except Exception as error:
            logger.critical(f"TTS Engine Failed: {error}")

    # ─────────────────────────────────────────────────────────
    # PUBLIC: SPEAK (non-blocking — adds to queue)
    # ─────────────────────────────────────────────────────────
    def speak(self, text, lang='eng'):
        if not text:
            return
        logger.info(f"Queueing: \"{text}\"")
        self.queue.put((text, lang))

    # ─────────────────────────────────────────────────────────
    # PUBLIC: SET RATE (for emotion engine integration)
    # ─────────────────────────────────────────────────────────
    def set_rate(self, rate: int):
        """Change TTS speech rate. Normal=150, Slow=100, Fast=200."""
        self._current_rate = rate
        logger.info(f"TTS rate changed to {rate}")

    # ─────────────────────────────────────────────────────────
    # PUBLIC: GET RATE
    # ─────────────────────────────────────────────────────────
    def get_rate(self):
        return self._current_rate

    # ─────────────────────────────────────────────────────────
    # PUBLIC: STOP playback
    # ─────────────────────────────────────────────────────────
    def stop(self):
        """Stop current audio playback immediately."""
        if PYGAME_AVAILABLE:
            pygame.mixer.stop()
            logger.info("Audio playback stopped.")

    # ─────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────
    def shutdown(self):
        logger.info("Shutting down TTS Manager...")
        self.running = False
        self.queue.put((None, None))
        self.worker_thread.join(timeout=3)
        if PYGAME_AVAILABLE:
            pygame.mixer.quit()


# ─────────────────────────────────────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────────────────────────────────────
tts_manager = TTSManager()


# ─────────────────────────────────────────────────────────────
# PUBLIC API FUNCTIONS (called by other modules and main.py)
# ─────────────────────────────────────────────────────────────
def speak(text, lang='eng'):
    """Queue text for TTS playback. Non-blocking."""
    tts_manager.speak(text, lang)


def set_rate(rate: int):
    """Adjust speech rate dynamically (used by emotion engine)."""
    tts_manager.set_rate(rate)


def get_rate() -> int:
    """Get current speech rate."""
    return tts_manager.get_rate()


def stop():
    """Stop current audio playback."""
    tts_manager.stop()


# ─────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ─────────────────────────────────────────────────────────────
def signal_handler(sig, frame):
    logger.info("Stopping TTS Module")
    tts_manager.shutdown()
    sys.exit(0)


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)

    print("\n" + "=" * 50)
    print("   BlindAssist TTS Module — Audio Playback Test")
    print("   Project: CSR-DES-INFINEON-2025")
    print("=" * 50)
    print("Type text and press Enter to hear it spoken.")
    print("Type QUIT to exit.")
    print("=" * 50 + "\n")

    while True:
        try:
            user_input = input("Text to speak: ").strip()

            if user_input.upper() == "QUIT":
                break

            if user_input:
                speak(user_input)
                # Wait for queue to drain so we hear it
                tts_manager.queue.join()

        except EOFError:
            break
        except KeyboardInterrupt:
            break

    tts_manager.shutdown()
    print("TTS Module Closed.")