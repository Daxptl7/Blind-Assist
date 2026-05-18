import pyttsx3
import logging
import signal
import sys
import threading
import json

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

# AUDIO DIRECTORY
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
# TTS MANAGER
# ─────────────────────────────────────────────────────────────
class TTSManager:

    def __init__(self):

        self.settings = self.load_settings()

        self.queue = Queue()

        self.running = True

        self.counter = 0

        # Start background thread
        self.worker_thread = threading.Thread(
            target=self.process_queue,
            daemon=True
        )

        self.worker_thread.start()

        logger.info(
            "TTS Manager initialized successfully."
        )

    # ─────────────────────────────────────────────────────────
    # LOAD SETTINGS
    # ─────────────────────────────────────────────────────────
    def load_settings(self):

        try:

            with open(CONFIG_PATH, 'r') as file:

                return json.load(file)

        except Exception as error:

            logger.warning(
                f"Using default settings: {error}"
            )

            return {
                "tts_rate": 110,
                "tts_volume": 1.0,
                "language": "eng"
            }

    # ─────────────────────────────────────────────────────────
    # PROCESS QUEUE
    # ─────────────────────────────────────────────────────────
    def process_queue(self):

        try:

            # Initialize engine
            engine = pyttsx3.init()

            # ─────────────────────────────────────────────
            # BETTER VOICE SETTINGS
            # ─────────────────────────────────────────────
            engine.setProperty('rate', 110)

            engine.setProperty('volume', 1.0)

            # Available voices
            voices = engine.getProperty('voices')

            # Change voice index if needed
            engine.setProperty(
                'voice',
                voices[0].id
            )

            logger.info(
                f"Using Voice: {voices[0].name}"
            )

            # ─────────────────────────────────────────────
            # MAIN LOOP
            # ─────────────────────────────────────────────
            while self.running:

                text, lang = self.queue.get()

                if text is None:
                    break

                try:

                    logger.info(
                        f"Generating Audio: {text}"
                    )

                    # Create unique filename
                    self.counter += 1

                    audio_file = (
                        AUDIO_DIR /
                        f"speech_{self.counter}.wav"
                    )

                    # Save audio file
                    engine.save_to_file(
                        text,
                        str(audio_file)
                    )

                    engine.runAndWait()

                    logger.info(
                        f"Audio Saved: {audio_file}"
                    )

                    print(
                        f"\nSaved Audio File:\n"
                        f"{audio_file}\n"
                    )

                except Exception as error:

                    logger.error(
                        f"TTS Playback Error: {error}"
                    )

                finally:

                    self.queue.task_done()

        except Exception as error:

            logger.critical(
                f"TTS Engine Failed: {error}"
            )

    # ─────────────────────────────────────────────────────────
    # PUBLIC SPEAK FUNCTION
    # ─────────────────────────────────────────────────────────
    def speak(self, text, lang='eng'):

        if not text:
            return

        logger.info(
            f"Queueing Text: {text}"
        )

        self.queue.put((text, lang))

    # ─────────────────────────────────────────────────────────
    # SHUTDOWN
    # ─────────────────────────────────────────────────────────
    def shutdown(self):

        logger.info(
            "Shutting down TTS Manager..."
        )

        self.running = False

        self.queue.put((None, None))

        self.worker_thread.join(timeout=2)

# ─────────────────────────────────────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────────────────────────────────────
tts_manager = TTSManager()

# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────
def speak(text, lang='eng'):

    tts_manager.speak(text, lang)

# ─────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ─────────────────────────────────────────────────────────────
def signal_handler(sig, frame):

    logger.info("Stopping TTS Module")

    tts_manager.shutdown()

    sys.exit(0)

# ─────────────────────────────────────────────────────────────
# MAIN TEST BLOCK
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':

    signal.signal(
        signal.SIGINT,
        signal_handler
    )

    print("\n--- BlindAssist TTS Module ---")
    print("Type text and press Enter")
    print("Type QUIT to exit")
    print("--------------------------------\n")

    while True:

        try:

            user_input = input(
                "Text to speak: "
            )

            if user_input.upper() == "QUIT":
                break

            speak(user_input)

        except EOFError:
            break

        except KeyboardInterrupt:

            signal_handler(None, None)

            break

    tts_manager.shutdown()

    print("TTS Module Closed")