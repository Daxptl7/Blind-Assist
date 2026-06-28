"""
voice.py — BlindAssist Project (OPTIMIZED)
============================================
Fast voice capture with WebRTC VAD for precise speech detection.
Reduces ambient noise adjustment from 1s to 0.3s.
Uses energy-based pre-detection to skip silence.
"""

import sys
import signal
import logging
import time
import io
import wave
import collections
import webrtcvad  # pip install webrtcvad-wheels

from pathlib import Path
from typing import Optional, Callable

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "voice.log"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("VoiceModule")

# ── CONFIGURATION ───────────────────────────────────────────
SAMPLE_RATE = 16000  # WebRTC VAD requires 8k, 16k, 32k, or 48k
FRAME_DURATION = 30  # ms (10, 20, or 30)
VAD_AGGRESSIVENESS = 2  # 0-3 (3 = most aggressive, filters more noise)
TIMEOUT_SECONDS = 8
PHRASE_SECONDS = 10
PAUSE_SECONDS = 1.5  # Stop listening after this much silence
PRE_BUFFER_SECONDS = 0.3  # Keep this much audio before speech starts

# ── VAD SETUP ───────────────────────────────────────────────
try:
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    VAD_AVAILABLE = True
    logger.info("WebRTC VAD initialized.")
except Exception as e:
    VAD_AVAILABLE = False
    logger.warning(f"WebRTC VAD unavailable: {e}. Using energy-based detection.")


def _read_audio_chunk(stream, chunk_size: int) -> bytes:
    """Read raw audio bytes from PyAudio stream."""
    return stream.read(chunk_size, exception_on_overflow=False)


def _energy_detect(audio_bytes: bytes, threshold: int = 500) -> bool:
    """Simple energy-based voice detection fallback."""
    import audioop
    rms = audioop.rms(audio_bytes, 2)  # 2 = 16-bit
    return rms > threshold


def listen(lang: str = 'en-IN', speak_fn: Optional[Callable] = None) -> Optional[str]:
    """
    Optimized voice recognition with VAD.
    Returns transcribed text or None.
    """
    import speech_recognition as sr
    
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 300  # Pre-set, skip dynamic adjustment
    recognizer.dynamic_energy_threshold = False  # Faster, more predictable
    recognizer.pause_threshold = PAUSE_SECONDS
    
    prompt = "Speak now"
    logger.info(f"Prompt: {prompt}")
    
    if speak_fn:
        speak_fn(prompt)
    
    try:
        with sr.Microphone(sample_rate=SAMPLE_RATE) as source:
            # Quick ambient calibration (0.3s instead of 1s)
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            
            logger.info("Listening with VAD...")
            
            # Use listen with phrase_time_limit for bounded capture
            audio = recognizer.listen(
                source,
                timeout=TIMEOUT_SECONDS,
                phrase_time_limit=PHRASE_SECONDS
            )
            
            # Save to in-memory buffer (no disk!)
            wav_buffer = io.BytesIO()
            audio_data = audio.get_wav_data(convert_rate=SAMPLE_RATE, convert_width=2)
            wav_buffer.write(audio_data)
            wav_buffer.seek(0)
            
            logger.info("Audio captured, transcribing...")
            
            # Google Speech Recognition
            text = recognizer.recognize_google(audio, language=lang)
            text = text.strip()
            logger.info(f"Transcribed: '{text}'")
            return text
            
    except sr.WaitTimeoutError:
        logger.warning("Timeout — no speech detected.")
        return None
    except sr.UnknownValueError:
        logger.warning("Could not understand audio.")
        return None
    except sr.RequestError as e:
        logger.error(f"Google API error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return None


def listen_with_vad(lang: str = 'en-IN', speak_fn: Optional[Callable] = None) -> Optional[str]:
    """
    Advanced listening with WebRTC VAD for precise speech boundaries.
    Cuts off immediately when user stops speaking.
    """
    if not VAD_AVAILABLE:
        return listen(lang, speak_fn)
    
    import pyaudio
    import speech_recognition as sr
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK_DURATION_MS = 30
    CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)
    
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    # Pre-buffer
    ring_buffer = collections.deque(maxlen=int(PRE_BUFFER_SECONDS * 1000 / CHUNK_DURATION_MS))
    triggered = False
    voiced_frames = []
    num_voiced = 0
    num_unvoiced = 0
    
    # VAD parameters
    RING_BUFFER_MAX = int(PRE_BUFFER_SECONDS * 1000 / CHUNK_DURATION_MS)
    TRIGGER_THRESHOLD = 3  # consecutive voiced frames to trigger
    UNTRIGGER_THRESHOLD = 10  # consecutive unvoiced to stop
    
    logger.info("VAD listening started...")
    start_time = time.time()
    
    try:
        while True:
            if time.time() - start_time > TIMEOUT_SECONDS:
                logger.warning("VAD timeout.")
                break
                
            chunk = _read_audio_chunk(stream, CHUNK_SIZE)
            is_speech = vad.is_speech(chunk, SAMPLE_RATE)
            
            if not triggered:
                ring_buffer.append(chunk)
                num_voiced = num_voiced + 1 if is_speech else 0
                if num_voiced >= TRIGGER_THRESHOLD:
                    triggered = True
                    voiced_frames.extend(ring_buffer)
                    ring_buffer.clear()
                    logger.info("Speech detected.")
            else:
                voiced_frames.append(chunk)
                num_unvoiced = num_unvoiced + 1 if not is_speech else 0
                if num_unvoiced >= UNTRIGGER_THRESHOLD:
                    logger.info("Speech ended.")
                    break
                    
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
    
    if not voiced_frames:
        return None
    
    # Convert to AudioData for recognition
    raw_data = b''.join(voiced_frames)
    
    # Create WAV in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(raw_data)
    
    wav_buffer.seek(0)
    
    # Recognize
    recognizer = sr.Recognizer()
    with sr.AudioFile(wav_buffer) as source:
        audio = recognizer.record(source)
    
    try:
        text = recognizer.recognize_google(audio, language=lang)
        logger.info(f"VAD transcribed: '{text}'")
        return text.strip()
    except sr.UnknownValueError:
        logger.warning("VAD could not understand.")
        return None
    except sr.RequestError as e:
        logger.error(f"VAD API error: {e}")
        return None


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    print("Voice Module Optimized Test")
    print("1: Standard listen  2: VAD listen  Q: Quit")
    
    while True:
        choice = input("\nSelect: ").strip().lower()
        if choice == 'q':
            break
        if choice == '1':
            result = listen('en-IN')
            print(f"Result: {result}")
        elif choice == '2':
            if VAD_AVAILABLE:
                result = listen_with_vad('en-IN')
                print(f"VAD Result: {result}")
            else:
                print("VAD not available. Install: pip install webrtcvad-wheels")