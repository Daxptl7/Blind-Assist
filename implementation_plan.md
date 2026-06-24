# Implement Voice Recognition Module (voice.py)

This plan outlines the implementation of the voice-to-text input module, [`voice.py`](file:///Users/daxpatel/Desktop/Blind/Blind-Assist/Blindterminal/modules/voice.py), which will capture speech from the laptop's microphone and convert it to text using Google Speech Recognition.

## User Review Required

No breaking changes or major design decisions are required. The module will fall back gracefully if the microphone fails or if there is no internet connection.

## Proposed Changes

### Dependencies

We need to install the audio recording dependencies on macOS:
1. `brew install portaudio` (Audio I/O library)
2. `pip3 install PyAudio` (Python bindings for PortAudio)
3. `pip3 install SpeechRecognition` (Wrapper library for speech-to-text engines)

---

### BlindAssist Modules

#### [NEW] [voice.py](file:///Users/daxpatel/Desktop/Blind/Blind-Assist/Blindterminal/modules/voice.py)

We will create a new Python module inside `Blindterminal/modules/` containing:
- **Pi Hardware Flags**: Kept at the top for architectural consistency with other modules.
- **Logging**: Configured to log speech events to `logs/voice.log`.
- **`listen(lang: str = 'en-IN') -> Optional[str]`**:
  - Initializes the speech recognizer.
  - Opens the default microphone.
  - Alerts the user using the centralized `tts.py` module (speaks "Speak now" or plays a beep).
  - Captures the audio with a configured timeout (e.g., 5 seconds of silence, 10 seconds total).
  - Sends the audio to the Google Speech Recognition API (with English, Hindi, or Gujarati locale codes).
  - Returns the transcribed string.
  - Returns `None` if translation failed or was silent, without throwing an unhandled exception.
- **Graceful Shutdown**: Handled via SIGINT/SIGTERM handlers.
- **Interactive Test Block**: Allows direct verification on macOS (`python3 Blindterminal/modules/voice.py`).

## Verification Plan

### Manual Verification
1. Run standalone module:
   ```bash
   python3 Blindterminal/modules/voice.py
   ```
2. Speak a message into your Mac's microphone (e.g., "Hello, testing BlindAssist").
3. Verify that the output prints the correct transcription on screen.
