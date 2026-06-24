"""
translator.py — BlindAssist Project
=====================================
Project  : Accessible Educational Terminal for Visually Impaired
Team     : Dhruv Vaghela & Dax Patel  |  CSR / Infineon 2025
Module   : Multilingual Translation

Translates text between English, Hindi, and Gujarati using
Google Translate (via googletrans library). Falls back gracefully
if internet is unavailable.
"""

import sys
import signal
import logging
import json

from pathlib import Path
from typing import Optional

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
LOG_PATH = BASE_DIR / "logs" / "translator.log"
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
logger = logging.getLogger("TranslatorModule")

# ──────────────────────────────────────────────────────────────
# SUPPORTED LANGUAGES
# ──────────────────────────────────────────────────────────────
SUPPORTED_LANGS = {
    'en': 'English',
    'hi': 'Hindi',
    'gu': 'Gujarati',
}

# Map settings.json language codes to googletrans codes
SETTINGS_TO_GOOGLE = {
    'eng': 'en',
    'hin': 'hi',
    'guj': 'gu',
    'en': 'en',
    'hi': 'hi',
    'gu': 'gu',
}

# ──────────────────────────────────────────────────────────────
# TRANSLATOR INIT
# ──────────────────────────────────────────────────────────────
_translator = None

def _init_translator():
    """Lazy-initialize the Google Translator."""
    global _translator
    if _translator is not None:
        return True

    try:
        from googletrans import Translator
        _translator = Translator()
        logger.info("Google Translator initialized.")
        return True
    except ImportError:
        logger.error("googletrans not installed. Run: pip3 install googletrans==4.0.0-rc1")
        return False
    except Exception as e:
        logger.error(f"Translator init failed: {e}")
        return False


# ──────────────────────────────────────────────────────────────
# PUBLIC API — translate()
# ──────────────────────────────────────────────────────────────
def translate(text: str, from_lang: str = 'en', to_lang: str = 'hi') -> str:
    """
    Translate text between supported languages.

    Args:
        text: The text to translate.
        from_lang: Source language code ('en', 'hi', 'gu', 'eng', 'hin', 'guj').
        to_lang: Target language code ('en', 'hi', 'gu', 'eng', 'hin', 'guj').

    Returns:
        Translated text, or original text with error message if translation fails.
    """
    if not text or not text.strip():
        return "No text to translate."

    # Normalize language codes
    src = SETTINGS_TO_GOOGLE.get(from_lang, from_lang)
    dest = SETTINGS_TO_GOOGLE.get(to_lang, to_lang)

    if src not in SUPPORTED_LANGS:
        logger.warning(f"Unsupported source language: {from_lang}")
        return f"Unsupported source language: {from_lang}"

    if dest not in SUPPORTED_LANGS:
        logger.warning(f"Unsupported target language: {to_lang}")
        return f"Unsupported target language: {to_lang}"

    if src == dest:
        logger.info("Source and target language are the same.")
        return text

    if not _init_translator():
        return text

    try:
        logger.info(
            f"Translating from {SUPPORTED_LANGS[src]} to "
            f"{SUPPORTED_LANGS[dest]}: \"{text[:80]}...\""
        )

        result = _translator.translate(text, src=src, dest=dest)

        if result and result.text:
            translated = result.text.strip()
            logger.info(f"Translation result: \"{translated[:80]}...\"")
            return translated
        else:
            logger.warning("Translation returned empty result.")
            return text

    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return f"Translation failed. Original text: {text}"


# ──────────────────────────────────────────────────────────────
# PUBLIC API — detect_language()
# ──────────────────────────────────────────────────────────────
def detect_language(text: str) -> Optional[str]:
    """
    Detect the language of given text.

    Returns:
        Language code ('en', 'hi', 'gu') or None.
    """
    if not _init_translator():
        return None

    try:
        detected = _translator.detect(text)
        if detected and detected.lang:
            lang = detected.lang
            logger.info(f"Detected language: {lang} (confidence: {detected.confidence})")
            return lang
        return None
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ──────────────────────────────────────────────────────────────
def signal_handler(sig, frame):
    logger.info("Shutting down Translator module.")
    sys.exit(0)


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n" + "=" * 50)
    print("   BlindAssist Translation Module — Test")
    print("   Project: CSR-DES-INFINEON-2025")
    print("=" * 50)
    print("Supported Languages:")
    print(" 1. English (en)")
    print(" 2. Hindi (hi)")
    print(" 3. Gujarati (gu)")
    print("Type QUIT to exit.")
    print("=" * 50 + "\n")

    lang_map = {'1': 'en', '2': 'hi', '3': 'gu'}

    while True:
        try:
            text = input("Text to translate: ").strip()
            if text.upper() == "QUIT":
                break
            if not text:
                continue

            src = input("From language (1=EN, 2=HI, 3=GU): ").strip()
            dest = input("To language   (1=EN, 2=HI, 3=GU): ").strip()

            src_code = lang_map.get(src, 'en')
            dest_code = lang_map.get(dest, 'hi')

            result = translate(text, src_code, dest_code)
            print(f"\n>>> Translation: {result}\n")

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting Translator Test...")
            break

    print("Translator Module Closed.")
