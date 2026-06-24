"""
emotion_engine.py — BlindAssist Project
=========================================
Project  : Accessible Educational Terminal for Visually Impaired
Team     : Dhruv Vaghela & Dax Patel  |  CSR / Infineon 2025
Module   : Emotion-Adaptive AI

Passively analyses user voice for stress/confusion signals.
If the user sounds confused or stressed in 2 consecutive utterances,
the system automatically slows TTS speed and tells the AI to respond
more simply.

Pipeline: audio_file → librosa MFCCs + spectral features → SVM → emotion label
"""

import sys
import signal
import logging
import json
import numpy as np

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
LOG_PATH = BASE_DIR / "logs" / "emotion.log"
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
logger = logging.getLogger("EmotionModule")

# ──────────────────────────────────────────────────────────────
# SETTINGS
# ──────────────────────────────────────────────────────────────
def _load_settings() -> dict:
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return {"emotion_consecutive_threshold": 2}

_settings = _load_settings()
CONSECUTIVE_THRESHOLD = _settings.get("emotion_consecutive_threshold", 2)

# ──────────────────────────────────────────────────────────────
# LIBROSA + SVM INIT
# ──────────────────────────────────────────────────────────────
LIBROSA_AVAILABLE = False
_classifier = None

try:
    import librosa
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    LIBROSA_AVAILABLE = True
    logger.info("librosa + scikit-learn loaded.")
except ImportError as e:
    logger.warning(f"Audio analysis unavailable: {e}")


def _build_default_classifier():
    """
    Build a simple SVM classifier with synthetic threshold-based
    feature boundaries. This gives us a working emotion detector
    without needing an external training dataset.

    Feature logic:
      - CALM: moderate energy, stable pitch, smooth MFCCs
      - STRESSED: high energy, high pitch variance
      - CONFUSED: low energy, hesitant pauses (low spectral centroid)
    """
    global _classifier

    if not LIBROSA_AVAILABLE:
        return False

    try:
        from sklearn.pipeline import make_pipeline

        # Synthetic training data (15 features: 13 MFCCs mean + spectral_centroid mean + rms mean)
        # These represent approximate feature distributions for each class
        np.random.seed(42)

        n_samples = 50

        # CALM: moderate values
        calm_data = np.random.normal(loc=0.0, scale=0.3, size=(n_samples, 15))
        calm_data[:, 13] = np.random.normal(2000, 300, n_samples)  # spectral centroid
        calm_data[:, 14] = np.random.normal(0.05, 0.01, n_samples)  # rms energy

        # STRESSED: high energy, high spectral centroid
        stressed_data = np.random.normal(loc=0.5, scale=0.4, size=(n_samples, 15))
        stressed_data[:, 13] = np.random.normal(3500, 500, n_samples)
        stressed_data[:, 14] = np.random.normal(0.12, 0.03, n_samples)

        # CONFUSED: low energy, low spectral centroid, high MFCC variance
        confused_data = np.random.normal(loc=-0.3, scale=0.5, size=(n_samples, 15))
        confused_data[:, 13] = np.random.normal(1200, 400, n_samples)
        confused_data[:, 14] = np.random.normal(0.02, 0.008, n_samples)

        X = np.vstack([calm_data, stressed_data, confused_data])
        y = (['CALM'] * n_samples +
             ['STRESSED'] * n_samples +
             ['CONFUSED'] * n_samples)

        _classifier = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
        _classifier.fit(X, y)

        logger.info("Emotion SVM classifier trained (synthetic data).")
        return True

    except Exception as e:
        logger.error(f"Failed to build classifier: {e}")
        return False


# Build classifier on module load
if LIBROSA_AVAILABLE:
    _build_default_classifier()


# ──────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ──────────────────────────────────────────────────────────────
def _extract_features(audio_path: str) -> Optional[np.ndarray]:
    """
    Extract audio features from a .wav file for emotion classification.

    Features (15 total):
      - 13 MFCC coefficients (mean across time)
      - 1 spectral centroid (mean)
      - 1 RMS energy (mean)
    """
    if not LIBROSA_AVAILABLE:
        return None

    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=22050, duration=5.0)

        if len(y) < sr * 0.3:  # Less than 0.3 seconds
            logger.warning("Audio clip too short for analysis.")
            return None

        # 13 MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)

        # Spectral centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_cent_mean = np.mean(spec_cent)

        # RMS energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)

        features = np.hstack([mfcc_means, spec_cent_mean, rms_mean])
        return features.reshape(1, -1)

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        return None


# ──────────────────────────────────────────────────────────────
# EMOTION STATE TRACKER
# ──────────────────────────────────────────────────────────────
_consecutive_non_calm = 0
_last_emotion = "CALM"


def _reset_tracker():
    global _consecutive_non_calm, _last_emotion
    _consecutive_non_calm = 0
    _last_emotion = "CALM"


# ──────────────────────────────────────────────────────────────
# PUBLIC API — analyze_emotion()
# ──────────────────────────────────────────────────────────────
def analyze_emotion(audio_path: str) -> str:
    """
    Analyze voice emotion from an audio file.

    Args:
        audio_path: Path to a .wav audio file.

    Returns:
        'CALM', 'STRESSED', or 'CONFUSED'
    """
    global _consecutive_non_calm, _last_emotion

    if not LIBROSA_AVAILABLE or _classifier is None:
        logger.info("Emotion analysis unavailable — returning CALM.")
        return "CALM"

    features = _extract_features(audio_path)
    if features is None:
        return "CALM"

    try:
        prediction = _classifier.predict(features)[0]
        probabilities = _classifier.predict_proba(features)[0]

        logger.info(
            f"Emotion detected: {prediction} "
            f"(probabilities: {dict(zip(_classifier.classes_, probabilities.round(2)))})"
        )

        _last_emotion = prediction

        if prediction in ('STRESSED', 'CONFUSED'):
            _consecutive_non_calm += 1
        else:
            _consecutive_non_calm = 0

        return prediction

    except Exception as e:
        logger.error(f"Emotion classification failed: {e}")
        return "CALM"


# ──────────────────────────────────────────────────────────────
# PUBLIC API — should_slow_down()
# ──────────────────────────────────────────────────────────────
def should_slow_down() -> bool:
    """
    Returns True if the user has been stressed/confused for
    CONSECUTIVE_THRESHOLD utterances in a row.
    """
    result = _consecutive_non_calm >= CONSECUTIVE_THRESHOLD
    if result:
        logger.info(
            f"Slow-down triggered: {_consecutive_non_calm} consecutive "
            f"non-CALM detections."
        )
    return result


# ──────────────────────────────────────────────────────────────
# PUBLIC API — get_emotion()
# ──────────────────────────────────────────────────────────────
def get_emotion() -> str:
    """Return the last detected emotion."""
    return _last_emotion


# ──────────────────────────────────────────────────────────────
# PUBLIC API — reset()
# ──────────────────────────────────────────────────────────────
def reset():
    """Reset the consecutive counter (when user returns to calm)."""
    _reset_tracker()
    logger.info("Emotion tracker reset.")


# ──────────────────────────────────────────────────────────────
# SIGNAL HANDLER
# ──────────────────────────────────────────────────────────────
def signal_handler(sig, frame):
    logger.info("Shutting down Emotion Engine.")
    sys.exit(0)


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("\n" + "=" * 50)
    print("   BlindAssist Emotion Engine — Test")
    print("   Project: CSR-DES-INFINEON-2025")
    print("=" * 50)

    if not LIBROSA_AVAILABLE:
        print("ERROR: librosa not installed.")
        print("Run: pip3 install librosa")
        sys.exit(1)

    print("This module analyses .wav audio files for emotion.")
    print("Enter a path to a .wav file, or type QUIT to exit.")
    print("=" * 50 + "\n")

    # Quick self-test with a synthetic audio file
    print("[Self-Test] Generating synthetic test audio...")
    try:
        import soundfile as sf

        sr = 22050
        duration = 2.0

        # Calm: smooth sine wave
        t = np.linspace(0, duration, int(sr * duration))
        calm_audio = 0.3 * np.sin(2 * np.pi * 200 * t)

        test_path = str(BASE_DIR / "audio" / "emotion_test.wav")
        sf.write(test_path, calm_audio, sr)

        result = analyze_emotion(test_path)
        print(f"[Self-Test] Synthetic audio emotion: {result}")
        print(f"[Self-Test] Should slow down: {should_slow_down()}\n")

    except ImportError:
        print("[Self-Test] soundfile not installed — skipping synthetic test.")
        print("You can still test with an existing .wav file.\n")
    except Exception as e:
        print(f"[Self-Test] Error: {e}\n")

    while True:
        try:
            path = input("WAV file path (or QUIT): ").strip()
            if path.upper() == "QUIT":
                break
            if not path:
                continue

            if not Path(path).exists():
                print(f"File not found: {path}")
                continue

            emotion = analyze_emotion(path)
            print(f"\n>>> Emotion: {emotion}")
            print(f">>> Should slow down: {should_slow_down()}\n")

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\nExiting Emotion Engine Test...")
            break

    print("Emotion Engine Closed.")
