"""
confidential_mode.py — BlindAssist Project
============================================
Project  : Accessible Educational Terminal for Visually Impaired
Team     : Dhruv Vaghela & Dax Patel  |  CSR / Infineon 2025
Module   : Privacy / Confidential Mode

Before every TTS output, this module asks the user:
  "Is this confidential? Press 1 for PRIVATE, 2 for NORMAL"

PRIVATE  → Audio routes ONLY through bone conduction (silent to bystanders)
NORMAL   → Audio plays through regular speaker

LAPTOP SIMULATION — prints [PRIVATE MODE] or [NORMAL MODE] tags.
On Pi with hardware: controls ALSA audio routing and PAM8403 amp GPIO pin.

Privacy question itself is always delivered via bone conduction
(simulated as whisper-style print on laptop).
"""

import sys
import signal
import logging
import json
import time

from pathlib import Path

# ──────────────────────────────────────────────────────────────
# HARDWARE FLAGS (Pi Flag Pattern)
# ──────────────────────────────────────────────────────────────
HEADLESS = False
USE_PICAMERA = False
USE_GPIO = False    # True = use ALSA + GPIO for real audio routing

# ──────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "confidential.log"
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
logger = logging.getLogger("ConfidentialModule")

# ──────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────
def _load_settings() -> dict:
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

_settings = _load_settings()
BONE_DEVICE = _settings.get("bone_device", "hw:1,0")
SPEAKER_DEVICE = _settings.get("speaker_device", "hw:2,0")

# Timeout in seconds — defaults to NORMAL if no response
PRIVACY_TIMEOUT = 8

# Current routing state
_current_mode = "NORMAL"


# ──────────────────────────────────────────────────────────────
# ALSA AUDIO ROUTING (Pi hardware only)
# ──────────────────────────────────────────────────────────────
def _mute_speaker():
    """Mute the main speaker via ALSA (Pi only)."""
    if not USE_GPIO:
        return
    try:
        import subprocess
        subprocess.run(
            ['amixer', '-D', SPEAKER_DEVICE, 'set', 'Master', 'mute'],
            capture_output=True, timeout=3
        )
        logger.info("Speaker muted via ALSA.")
    except Exception as e:
        logger.error(f"Failed to mute speaker: {e}")


def _unmute_speaker():
    """Unmute the main speaker via ALSA (Pi only)."""
    if not USE_GPIO:
        return
    try:
        import subprocess
        subprocess.run(
            ['amixer', '-D', SPEAKER_DEVICE, 'set', 'Master', 'unmute'],
            capture_output=True, timeout=3
        )
        logger.info("Speaker unmuted via ALSA.")
    except Exception as e:
        logger.error(f"Failed to unmute speaker: {e}")


def _enable_bone_conduction():
    """Enable PAM8403 amplifier via GPIO pin (Pi only)."""
    if not USE_GPIO:
        return
    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(18, GPIO.OUT)  # GPIO 18 = PAM8403 enable pin
        GPIO.output(18, GPIO.HIGH)
        logger.info("Bone conduction amplifier enabled (GPIO 18 HIGH).")
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"GPIO error: {e}")


def _disable_bone_conduction():
    """Disable PAM8403 amplifier via GPIO pin (Pi only)."""
    if not USE_GPIO:
        return
    try:
        import RPi.GPIO as GPIO
        GPIO.output(18, GPIO.LOW)
        logger.info("Bone conduction amplifier disabled (GPIO 18 LOW).")
    except ImportError:
        pass
    except Exception as e:
        logger.error(f"GPIO error: {e}")


# ──────────────────────────────────────────────────────────────
# PUBLIC API — ask_privacy()
# ──────────────────────────────────────────────────────────────
def ask_privacy() -> str:
    """
    Ask the user whether the upcoming content is confidential.

    Returns:
        'PRIVATE' or 'NORMAL'
    """
    global _current_mode

    # The privacy question itself is delivered via bone conduction
    # (simulated as a whisper-style prompt on laptop)
    if USE_GPIO:
        # On Pi: play prompt through bone conduction only
        _mute_speaker()
        _enable_bone_conduction()
        # TTS would play here on Pi
        _disable_bone_conduction()
        _unmute_speaker()

    # LAPTOP SIMULATION — keyboard input
    print("\n" + "─" * 45)
    print("  🔒 [WHISPER] Is this confidential?")
    print("     Press 1 → PRIVATE (bone conduction only)")
    print("     Press 2 → NORMAL  (speaker)")
    print(f"     ({PRIVACY_TIMEOUT}s timeout → defaults to NORMAL)")
    print("─" * 45)

    try:
        if USE_GPIO:
            # On Pi: read GPIO button presses with timeout
            # Button 1 (GPIO 17) = PRIVATE
            # Button 2 (GPIO 27) = NORMAL
            # Stub — would use GPIO.wait_for_edge() with timeout
            _current_mode = "NORMAL"
        else:
            # LAPTOP SIMULATION — keyboard input with timeout
            import select

            # Simple input with manual timeout
            choice = None
            start = time.time()

            # On macOS, select.select on stdin works
            try:
                import threading

                result = [None]

                def _get_input():
                    try:
                        result[0] = input("  Your choice (1/2): ").strip()
                    except EOFError:
                        result[0] = '2'

                input_thread = threading.Thread(target=_get_input, daemon=True)
                input_thread.start()
                input_thread.join(timeout=PRIVACY_TIMEOUT)

                choice = result[0]

            except Exception:
                choice = '2'

            if choice == '1':
                _current_mode = "PRIVATE"
            else:
                _current_mode = "NORMAL"

    except Exception as e:
        logger.error(f"Privacy prompt error: {e}")
        _current_mode = "NORMAL"

    # Apply audio routing
    if _current_mode == "PRIVATE":
        logger.info("Privacy mode: PRIVATE — bone conduction only.")
        print("  🔒 [PRIVATE MODE] Audio routed to bone conduction only.")
        if USE_GPIO:
            _mute_speaker()
            _enable_bone_conduction()
    else:
        logger.info("Privacy mode: NORMAL — speaker active.")
        print("  🔊 [NORMAL MODE] Audio routed to speaker.")
        if USE_GPIO:
            _unmute_speaker()
            _disable_bone_conduction()

    print()
    return _current_mode


# ──────────────────────────────────────────────────────────────
# PUBLIC API — reset_to_normal()
# ──────────────────────────────────────────────────────────────
def reset_to_normal():
    """Reset audio routing back to NORMAL (speaker) after playback."""
    global _current_mode
    _current_mode = "NORMAL"

    if USE_GPIO:
        _unmute_speaker()
        _disable_bone_conduction()

    logger.info("Audio routing reset to NORMAL.")


# ──────────────────────────────────────────────────────────────
# PUBLIC API — get_mode()
# ──────────────────────────────────────────────────────────────
def get_mode() -> str:
    """Return current privacy mode."""
    return _current_mode


# ──────────────────────────────────────────────────────────────
# PUBLIC API — is_private()
# ──────────────────────────────────────────────────────────────
def is_private() -> bool:
    """Return True if currently in private mode."""
    return _current_mode == "PRIVATE"


# ──────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ──────────────────────────────────────────────────────────────
def signal_handler(sig, frame):
    logger.info("Shutting down Confidential Mode module.")
    reset_to_normal()
    sys.exit(0)


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n" + "=" * 50)
    print("   BlindAssist Confidential Mode — Test")
    print("   Project: CSR-DES-INFINEON-2025")
    print("=" * 50)
    print("[SIMULATION] Using keyboard for privacy selection")
    print("Press ENTER to trigger a privacy prompt.")
    print("Type QUIT to exit.")
    print("=" * 50 + "\n")

    while True:
        try:
            cmd = input("Press Enter for privacy prompt (or QUIT): ").strip()
            if cmd.upper() == "QUIT":
                break

            mode = ask_privacy()
            print(f">>> Mode selected: {mode}")
            print(f">>> Is private: {is_private()}")

            # Simulate playback complete
            reset_to_normal()
            print(">>> Reset to NORMAL after playback.\n")

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting Confidential Mode Test...")
            break

    reset_to_normal()
    print("Confidential Mode Closed.")
