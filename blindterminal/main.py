import sys
import logging
import json
from pathlib import Path

# Import Hardware Modules
from modules.ocr import open_camera, scan_and_read
from modules.tts import speak

# Import RAG Services
from services.embedder import Embedder
from services.vector_store import VectorStore
from services.retriever import Retriever
from services.gemini_agent import GeminiAgent
from services.rag_pipeline import RAGPipeline

# ─────────────────────────────────────────────────────────────
# CONFIGURATION & LOGGING
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("BlindAssistMain")

def load_config():
    try:
        with open(CONFIG_PATH, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Could not load config: {e}")
        return {}

def main():
    config = load_config()

    # API KEY handling - prioritizes config, then env var, then a placeholder
    api_key = config.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY") or "YOUR_GEMINI_API_KEY_HERE"

    if api_key == "YOUR_GEMINI_API_KEY_HERE":
        logger.warning("Gemini API Key not found! Please update config/settings.json")

    # 1. Initialize RAG Stack
    try:
        embedder = Embedder()
        vector_store = VectorStore(store_path=str(BASE_DIR / "data" / "vector_store"))
        retriever = Retriever(embedder, vector_store)
        gemini_agent = GeminiAgent(api_key=api_key)

        rag = RAGPipeline(embedder, vector_store, retriever, gemini_agent)
        logger.info("RAG Pipeline successfully initialized.")
    except Exception as e:
        logger.critical(f"Failed to initialize RAG system: {e}")
        sys.exit(1)

    # 2. Initialize Hardware
    cap, cam_type = open_camera()
    if cap is None:
        logger.error("Camera failed to initialize.")
        sys.exit(1)

    print("\n" + "="*40)
    print("   BLINDAssist: Contextual AI RAG")
    print("="*40)
    print("Controls:")
    print(" [S] Scan Document")
    print(" [Q] Quit")
    print(" After scanning, type your question.")
    print("="*40 + "\n")

    try:
        while True:
            # In a real Pi environment, this would be a GPIO button press
            # Here we simulate with keyboard input
            cmd = input("Action (S/Q): ").lower().strip()

            if cmd == 'q':
                break

            if cmd == 's':
                print("\nScanning...")
                text = scan_and_read(cap, cam_type)

                if text:
                    print(f"Extracted Text: {text[:100]}...")
                    # Feed text into RAG pipeline
                    rag.process_scan(text)
                    speak("Document scanned and indexed. You can now ask me questions about it.")
                else:
                    speak("I couldn't find any text to scan.")

                # Enter Question Loop
                while True:
                    query = input("\nAsk a question about the document (or press Enter to go back): ").strip()
                    if not query:
                        break

                    print("Thinking...")
                    response = rag.ask_question(query)
                    print(f"\nAI: {response}")
                    speak(response)

    except KeyboardInterrupt:
        pass
    finally:
        if cam_type == "usb":
            cap.release()
        elif cam_type == "pi":
            cap.stop()
        print("\nShutting down BlindAssist...")

if __name__ == "__main__":
    import os # Needed for environ
    main()
