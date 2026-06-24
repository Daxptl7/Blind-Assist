"""
ai_query.py — BlindAssist Project
====================================
Project  : Accessible Educational Terminal for Visually Impaired
Team     : Dhruv Vaghela & Dax Patel  |  CSR / Infineon 2025
Module   : AI Query — Online (Gemini) + Offline (TinyLlama stub)

Routes user questions to the best available AI:
  1. Google Gemini API (primary — free tier)
  2. Offline LLM via llama-cpp-python (when no internet)
  3. Graceful fallback message if both unavailable
"""

import sys
import signal
import logging
import json
import os

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
LOG_PATH = BASE_DIR / "logs" / "ai_query.log"
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
logger = logging.getLogger("AIQueryModule")

# ──────────────────────────────────────────────────────────────
# LOAD SETTINGS
# ──────────────────────────────────────────────────────────────
def _load_settings() -> dict:
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Could not load settings: {e}")
        return {}

_settings = _load_settings()

# ──────────────────────────────────────────────────────────────
# GEMINI CLIENT (Online — Primary)
# ──────────────────────────────────────────────────────────────
_gemini_model = None

def _init_gemini():
    global _gemini_model
    if _gemini_model is not None:
        return True

    api_key = _settings.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")

    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        logger.warning("Gemini API key not configured.")
        return False

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        _gemini_model = genai.GenerativeModel(
            model_name='gemini-2.0-flash',
            system_instruction=(
                "You are BlindAssist, a supportive and concise AI assistant "
                "for visually impaired users. Keep responses short, clear, "
                "and easy to understand when read aloud via text-to-speech. "
                "Avoid markdown formatting, bullet points, or special characters. "
                "Speak naturally as if talking to the user."
            )
        )
        logger.info("Gemini AI initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Gemini init failed: {e}")
        return False


def _ask_gemini(prompt: str, context: str = '') -> Optional[str]:
    """Send query to Google Gemini API."""
    if not _init_gemini():
        return None

    try:
        full_prompt = prompt
        if context:
            full_prompt = f"Context:\n{context}\n\nUser Question: {prompt}"

        response = _gemini_model.generate_content(full_prompt)

        if response and response.text:
            result = response.text.strip()
            logger.info(f"Gemini response: {result[:100]}...")
            return result

        return None

    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# OFFLINE LLM CLIENT (TinyLlama via llama-cpp-python)
# ──────────────────────────────────────────────────────────────
_offline_model = None

def _init_offline():
    """Try to load local GGUF model for offline AI."""
    global _offline_model
    if _offline_model is not None:
        return True

    model_path = _settings.get("offline_model_path", "")

    if not model_path or not Path(model_path).exists():
        logger.info("Offline model file not found — offline AI unavailable.")
        return False

    try:
        from llama_cpp import Llama
        _offline_model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False
        )
        logger.info(f"Offline LLM loaded from {model_path}")
        return True
    except ImportError:
        logger.info("llama-cpp-python not installed — offline AI unavailable.")
        return False
    except Exception as e:
        logger.error(f"Offline LLM load failed: {e}")
        return False


def _ask_offline(prompt: str) -> Optional[str]:
    """Query local LLM model."""
    if not _init_offline():
        return None

    try:
        response = _offline_model(
            f"User: {prompt}\nAssistant:",
            max_tokens=256,
            stop=["User:", "\n\n"],
            echo=False
        )
        text = response['choices'][0]['text'].strip()
        logger.info(f"Offline response: {text[:100]}...")
        return text

    except Exception as e:
        logger.error(f"Offline LLM error: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# PUBLIC API — ask_ai()
# ──────────────────────────────────────────────────────────────
def ask_ai(prompt: str, context: str = '') -> str:
    """
    Send a question to the best available AI and return the answer.

    Tries online (Gemini) first, then offline (TinyLlama), then
    returns a graceful fallback message.

    Args:
        prompt: The user's question text.
        context: Optional OCR/scanned text context.

    Returns:
        AI response string.
    """
    if not prompt or not prompt.strip():
        return "I didn't receive a question. Please try again."

    logger.info(f"Query: \"{prompt}\"")

    # Try 1: Online — Gemini
    result = _ask_gemini(prompt, context)
    if result:
        return result

    # Try 2: Offline — Local LLM
    result = _ask_offline(prompt)
    if result:
        return result

    # Try 3: Fallback
    logger.warning("All AI backends unavailable.")
    return (
        "I'm sorry, I cannot answer right now. "
        "Please check your internet connection or "
        "ensure the offline AI model is installed."
    )


# ──────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ──────────────────────────────────────────────────────────────
def signal_handler(sig, frame):
    logger.info("Shutting down AI Query module.")
    sys.exit(0)


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n" + "=" * 50)
    print("   BlindAssist AI Query Module — Test")
    print("   Project: CSR-DES-INFINEON-2025")
    print("=" * 50)
    print("Type a question and press Enter.")
    print("Type QUIT to exit.")
    print("=" * 50 + "\n")

    while True:
        try:
            question = input("Your question: ").strip()

            if question.upper() == "QUIT":
                break

            if question:
                print("Thinking...")
                answer = ask_ai(question)
                print(f"\nAI: {answer}\n")

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting AI Query Test...")
            break

    print("AI Query Module Closed.")
