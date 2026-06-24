"""
voice.py — BlindAssist Project
===============================
Project  : Accessible Educational Terminal for Visually Impaired
Team     : Dhruv Vaghela & Dax Patel  |  CSR / Infineon 2025
Module   : Voice Input (Speech-to-Text)

Captures voice commands or questions from the default microphone and
transcribes them to text using Google Speech Recognition. Supports English,
Hindi, and Gujarati localization.
"""

import sys
import signal
import logging
import time
from pathlib import Path
from typing import Optional, Callable
import speech_recognition as sr

# ──────────────────────────────────────────────────────────────
# HARDWARE FLAGS (Pi Flag Pattern)
# ──────────────────────────────────────────────────────────────
# Currently False on laptop/Mac environment.
HEADLESS = False
USE_PICAMERA = False
USE_GPIO = False

# ──────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "voice.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

# Create folders automatically
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
logger = logging.getLogger("VoiceModule")

# ──────────────────────────────────────────────────────────────
# VOICE TO TEXT FUNCTION
# ──────────────────────────────────────────────────────────────
def listen(lang: str = 'en-IN', speak_fn: Optional[Callable[[str], None]] = None) -> Optional[str]:
    """
    Listens to the default system microphone and transcribes the speech.
    
    Args:
        lang (str): The language locale code.
                    - English: 'en-IN' or 'en-US'
                    - Hindi: 'hi-IN'
                    - Gujarati: 'gu-IN'
        speak_fn (Callable): Optional external callback function to speak prompts (non-blocking).
        
    Returns:
        Optional[str]: Transcribed text or None if recognition failed.
    """
    recognizer = sr.Recognizer()
    
    # Optional audio prompt
    prompt = "Speak now"
    logger.info(f"User Prompt: {prompt}")
    
    if speak_fn:
        # Use centralized TTS if provided
        speak_fn(prompt)
    else:
        # Fallback to local offline TTS prompt
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Set slightly slower rate for clean prompt
            engine.setProperty('rate', 140)
            engine.say(prompt)
            engine.runAndWait()
        except Exception as e:
            logger.warning(f"Local TTS prompt failed: {e}. Printing prompt only.")
            print(f"\n>>> {prompt} <<<\n")

    try:
        # Initializing microphone
        with sr.Microphone() as source:
            logger.info("Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            logger.info("Listening for voice input...")
            # listen with timeout of 10s and max phrase limit of 8s
            audio = recognizer.listen(source, timeout=8, phrase_time_limit=8)
            
        logger.info("Speech captured. Processing recognition...")
        
        # Google Speech Recognition (free/public tier)
        text = recognizer.recognize_google(audio, language=lang)
        text = text.strip()
        logger.info(f"Transcribed Result: '{text}' (lang={lang})")
        return text

    except sr.WaitTimeoutError:
        logger.warning("Listening timed out. No speech detected.")
        return None
    except sr.UnknownValueError:
        logger.warning("Speech recognition could not understand the audio.")
        return None
    except sr.RequestError as e:
        logger.error(f"Could not request results from Google Speech Recognition service; {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in Voice capture: {e}")
        return None

# ──────────────────────────────────────────────────────────────
# SHUTDOWN HANDLER
# ──────────────────────────────────────────────────────────────
def signal_handler(sig, frame):
    logger.info("Stopping Voice Module gracefully.")
    sys.exit(0)

# ──────────────────────────────────────────────────────────────
# STANDALONE TEST MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n" + "="*45)
    print("   BlindAssist Voice Recognition (Speech-to-Text)")
    print("   Project: CSR-DES-INFINEON-2025")
    print("="*45)
    print("Testing locally on Mac Microphone...")
    print("Supported Languages:")
    print(" 1. English (en-IN)")
    print(" 2. Hindi (hi-IN)")
    print(" 3. Gujarati (gu-IN)")
    print(" Press Ctrl+C or type 'q' to Quit.")
    print("="*45 + "\n")

    # Mappings
    langs = {
        '1': ('en-IN', 'English'),
        '2': ('hi-IN', 'Hindi'),
        '3': ('gu-IN', 'Gujarati')
    }

    while True:
        try:
            choice = input("Select Language (1-3) or Q to Quit: ").strip().lower()
            if choice == 'q':
                break
            if choice not in langs:
                print("Invalid choice. Please select 1, 2, or 3.")
                continue

            lang_code, lang_name = langs[choice]
            print(f"\n[Language Selected: {lang_name}]")
            print("Preparing microphone... stand by.")
            
            result = listen(lang=lang_code)
            
            if result:
                print(f"\n>>> SUCCESS: {result}\n")
            else:
                print("\n>>> FAILED: Could not transcribe any speech.\n")
                
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting Voice Module Test...")
            break
            
    print("Voice Module Closed.")
