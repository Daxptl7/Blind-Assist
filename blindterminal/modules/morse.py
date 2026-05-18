import json
import logging
import signal
import sys
import time

from pathlib import Path
from threading import Lock

# ─────────────────────────────────────────────────────────────
# HARDWARE FLAGS
# ─────────────────────────────────────────────────────────────
HEADLESS = False
USE_PICAMERA = False
USE_GPIO = False

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# BASE DIRECTORY
# ─────────────────────────────────────────────────────────────
# Current file:
# blindterminal/modules/morse.py
#
# parent        -> modules/
# parent.parent -> blindterminal/
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
CONFIG_PATH = BASE_DIR / "config" / "settings.json"
LOG_PATH = BASE_DIR / "logs" / "morse.log"

# Create logs directory automatically
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
# ─────────────────────────────────────────────────────────────
# DEFAULT SETTINGS
# ─────────────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "morse_dot_max_ms": 400,
    "morse_dash_min_ms": 400,
    "morse_letter_gap_ms": 1500,
    "morse_word_gap_ms": 3000,
}

# ─────────────────────────────────────────────────────────────
# MORSE CODE MAP
# ─────────────────────────────────────────────────────────────
MORSE_CODE_DICT = {
    '.-': 'A',
    '-...': 'B',
    '-.-.': 'C',
    '-..': 'D',
    '.': 'E',
    '..-.': 'F',
    '--.': 'G',
    '....': 'H',
    '..': 'I',
    '.---': 'J',
    '-.-': 'K',
    '.-..': 'L',
    '--': 'M',
    '-.': 'N',
    '---': 'O',
    '.--.': 'P',
    '--.-': 'Q',
    '.-.': 'R',
    '...': 'S',
    '-': 'T',
    '..-': 'U',
    '...-': 'V',
    '.--': 'W',
    '-..-': 'X',
    '-.--': 'Y',
    '--..': 'Z',

    '-----': '0',
    '.----': '1',
    '..---': '2',
    '...--': '3',
    '....-': '4',
    '.....': '5',
    '-....': '6',
    '--...': '7',
    '---..': '8',
    '----.': '9',

    '.-.-.-': '.',
    '--..--': ',',
    '..--..': '?',
    '-....-': '-',
    '-..-.': '/',
}

# ─────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("MorseModule")


# ─────────────────────────────────────────────────────────────
# MORSE DECODER
# ─────────────────────────────────────────────────────────────
class MorseDecoder:

    def __init__(self):
        self.settings = self.load_settings()

        self.current_sequence = ""
        self.current_word = ""

        self.last_input_time = time.time()

        self.lock = Lock()
        self.running = True

    def load_settings(self):
        try:
            with open(CONFIG_PATH, "r") as file:
                return json.load(file)

        except Exception as error:
            logger.warning(f"Using default settings: {error}")
            return DEFAULT_SETTINGS.copy()

    def add_symbol(self, symbol: str):
        with self.lock:
            self.current_sequence += symbol
            self.last_input_time = time.time()

            logger.info(
                f"Added Symbol: {symbol} | "
                f"Current Sequence: {self.current_sequence}"
            )

    def process_gap(self):
        with self.lock:

            current_time = time.time()

            gap_ms = (
                current_time - self.last_input_time
            ) * 1000

            word_gap = self.settings["morse_word_gap_ms"]
            letter_gap = self.settings["morse_letter_gap_ms"]

            if gap_ms >= word_gap:

                self.decode_letter()
                return "WORD_GAP"

            if gap_ms >= letter_gap:

                self.decode_letter()
                return "LETTER_GAP"

            return None

    def decode_letter(self):

        if not self.current_sequence:
            return

        letter = MORSE_CODE_DICT.get(
            self.current_sequence,
            "?"
        )

        self.current_word += letter

        logger.info(
            f"Decoded: {self.current_sequence} -> {letter}"
        )

        self.current_sequence = ""

    def get_letter(self):

        with self.lock:

            if not self.current_word:
                return None

            return self.current_word[-1]

    def get_word(self):

        with self.lock:

            word = self.current_word
            self.current_word = ""

            return word

    def delete_char(self):

        with self.lock:

            if self.current_word:

                self.current_word = (
                    self.current_word[:-1]
                )

                logger.info(
                    f"Deleted Character | "
                    f"Current Word: {self.current_word}"
                )

    def add_space(self):

        with self.lock:

            self.current_word += " "

            logger.info(
                f"Space Added | "
                f"Current Word: {self.current_word}"
            )


# ─────────────────────────────────────────────────────────────
# GLOBAL DECODER
# ─────────────────────────────────────────────────────────────
decoder = MorseDecoder()


# ─────────────────────────────────────────────────────────────
# API FUNCTIONS
# ─────────────────────────────────────────────────────────────
def get_letter():
    return decoder.get_letter()


def get_word():
    return decoder.get_word()


# ─────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ─────────────────────────────────────────────────────────────
def signal_handler(sig, frame):

    logger.info("Shutting down Morse module...")

    decoder.running = False
    sys.exit(0)


# ─────────────────────────────────────────────────────────────
# TERMINAL SIMULATION
# ─────────────────────────────────────────────────────────────
def run_simulation():

    signal.signal(signal.SIGINT, signal_handler)

    print("\n--- BlindAssist Morse Simulation ---")
    print("'.'  -> DOT")
    print("'-'  -> DASH")
    print("'Enter' -> Confirm Word")
    print("'Backspace' -> Delete Character")
    print("'Tab' -> Add Space")
    print("'Ctrl+C' -> Exit")
    print("------------------------------------\n")

    try:
        import readchar

    except ImportError:

        import subprocess

        print("Installing readchar...")

        subprocess.check_call([
            sys.executable,
            "-m",
            "pip",
            "install",
            "readchar"
        ])

        import readchar

    while decoder.running:

        try:

            decoder.process_gap()

            char = readchar.readchar()

            if char == '.':
                decoder.add_symbol('.')

            elif char == '-':
                decoder.add_symbol('-')

            elif char in ['\r', '\n']:

                word = decoder.get_word()

                print(f"\n[CONFIRMED]: {word}\n")

            elif char == '\x7f':
                decoder.delete_char()

            elif char == '\t':
                decoder.add_space()

        except Exception as error:

            logger.error(f"Simulation Error: {error}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_simulation()