# 👁️ BlindAssist — Accessible Educational Terminal

[![Project Status](https://img.shields.io/badge/status-ready-success.svg)](#)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.13-blue.svg)](#)
[![Hardware Target](https://img.shields.io/badge/hardware-macOS%20%7C%20Raspberry%20Pi%205-red.svg)](#)

An advanced, voice-controlled, and tactile educational terminal designed to assist visually impaired individuals. The system provides reading aids, AI queries, translation services, navigation systems, object detection, and gesture-based controls. Built to run locally on standard hardware (laptops) with a structured path to Raspberry Pi deployment.

---

## 🏛️ System Architecture

BlindAssist follows a **Star Topology** for software decoupling:
*   **Orchestrator (`main.py`)**: The central hub that holds the system state machine. It manages inputs from Morse codes, gestures, and audio.
*   **Decoupled Modules (`modules/`)**: Individual capabilities are completely isolated. **No module is allowed to import another module.** All cross-module communication is piped back through the Orchestrator.
*   **Pi Flag Pattern**: Every hardware-facing module includes top-level flags to run seamlessly in **Laptop Simulation Mode** (webcam, microphone, pygame speakers, and keyboard simulation) or **Pi Production Mode** (Pi Camera v3, GPIO buttons, bone conduction transducers, and hardware serial ports).

---

## ⚙️ Core Modules & Functionality

| Module | Purpose | Tech Stack | Laptop Simulation | Pi Production |
|---|---|---|---|---|
| **TTS (Text-to-Speech)** | Non-blocking spoken readout of system states, text, and alerts. | `pyttsx3`, `pygame.mixer` | Stereo Laptop Speakers | Bone conduction audio device |
| **Voice Recognition** | Records voice prompts and transcribes queries. | `speech_recognition` | Internal Laptop Microphone | USB Audio interface |
| **Morse Input** | Tactile menu navigation and typing interface. | Keyboard hooks | Keys: `.` (dot), `-` (dash), `Space`, `Enter` | Tactile GPIO Button |
| **OCR Reader** | Captures documents, processes text, and reads aloud. | `cv2`, `pytesseract` | Laptop Webcam + OCR | Pi Camera v3 + Tesseract |
| **AI Query (Gemini)** | Answers questions and runs local RAG searches. | `google-generativeai` | Online Gemini API | Offline Llama-2 (GGUF via llama-cpp) |
| **Translator** | Translates texts into local regional languages. | `googletrans` | English ⇄ Hindi ⇄ Gujarati | Same |
| **Gesture Control** | Recognizes hand gestures for trigger modes. | `mediapipe` Tasks, `cv2` | Mediapipe Tasks model | Headless camera worker thread |
| **Emotion Engine** | Slows speech rate dynamically if stress or confusion is detected. | `librosa`, `scikit-learn` | Voice tone MFCC analyzer | Real-time audio analyzer |
| **GPS Navigation** | Spoken directions, routes, and coordinates. | `requests` (OSRM/Nominatim) | Indore GPS Simulation | Hardware GPS HAT (UART Serial) |
| **Confidential Mode** | Mutes readout and requires confirmation for private alerts. | System Muting | Console prompts | Physical Toggle Button |
| **Object Detection** | Announces surrounding items and obstacles. | OpenCV, `ultralytics` YOLOv8 | YOLOv8s on Laptop Camera | Coral Edge TPU / yolov8n |

---

## 🚀 Setup & Installation

### 📋 Prerequisites

1. **Python 3.10+** (tested up to Python 3.13)
2. **Tesseract OCR Engine** (required for OCR text scanning):
   - **macOS**: `brew install tesseract`
   - **Linux**: `sudo apt install tesseract-ocr`

### 💻 Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/Daxptl7/Blind-Assist.git
   cd Blind-Assist
   ```

2. Install dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Download the model files (done automatically on first launch or run these helper commands):
   - **YOLOv8 Model**: Saved at `Blindterminal/yolov8s.pt`
   - **MediaPipe Hand Landmarker**: Download [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task) and place it at `Blindterminal/config/hand_landmarker.task`

4. Add your Google Gemini API Key to [Blindterminal/config/settings.json](file:///Users/daxpatel/Desktop/Blind/Blind-Assist/Blindterminal/config/settings.json):
   ```json
   "gemini_api_key": "YOUR_GEMINI_API_KEY_HERE"
   ```

---

## 🕹️ Laptop Simulation Controls

When running the system on a macOS/Windows/Linux laptop without physical hardware, use the following interactive console mappings:

### 📟 Mode Selection
Press the **number keys** in the console to select states:
*   `1` — **Scan OCR Mode** (Captures frame, extracts text, reads aloud)
*   `2` — **Morse Query Mode** (Allows inputting Morse text, queries Gemini)
*   `3` — **Voice Query Mode** (Listens to voice, queries Gemini with RAG context)
*   `4` — **Translation Mode** (Translates text input)
*   `5` — **Gesture Control Mode** (Opens camera window, detects gestures)
*   `6` — **Object Recognition Mode** (Detects surrounding obstacles)
*   `7` — **GPS Mode** (Indore simulated coordinate route planner)
*   `0` — **SOS / Shutdown** (Graceful system termination)

### 🖐️ Gesture Commands (Mode 5)
Point your hand at the camera:
*   **Open Palm** 🖐️ → Mapped to **OCR Scan Mode**
*   **Two Fingers (V Sign)** ✌️ → Mapped to **Voice Query Mode**
*   **Thumbs Up** 👍 → Mapped to **Confirm / Send**
*   **Point Down** 👇 → Mapped to **Repeat Last Spoken Response**
*   **Fist** ✊ → Mapped to **Cancel / Stop Current Readout**

---

## 🛠️ Verification & Testing

Every module contains an independent `__main__` test block to test its functionality in isolation:

```bash
# Test Text-to-Speech audio playback
python3 Blindterminal/modules/tts.py

# Test Tesseract OCR and Webcam Capture
python3 Blindterminal/modules/ocr.py

# Test MediaPipe Hand Landmarker and Webcam Gestures
python3 Blindterminal/modules/gesture_control.py

# Test Voice Recognition and microphone sensitivity
python3 Blindterminal/modules/voice.py

# Test YOLOv8 Live Object Detection
python3 Blindterminal/modules/object_detection.py
```

To run the complete integrated system:
```bash
python3 Blindterminal/main.py
```

---

## 👥 Authors & Contributors
*   **Dhruv Vaghela** — CSR/Infineon Technologies 2025
*   **Dax Patel** — CSR/Infineon Technologies 2025