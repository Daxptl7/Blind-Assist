"""
ai_query.py — BlindAssist Project (OPTIMIZED)
===============================================
Async API calls with concurrent fallback.
Preloads offline model at import time (hidden in thread).
Non-blocking for online APIs.
"""

import sys
import signal
import logging
import json
import os
import threading
import concurrent.futures
import time

from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "logs" / "ai_query.log"
CONFIG_PATH = BASE_DIR / "config" / "settings.json"

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AIQueryModule")

_settings = {}
try:
    with open(CONFIG_PATH, 'r') as f:
        _settings = json.load(f)
except Exception as e:
    logger.warning(f"Settings load failed: {e}")

_rag_pipeline_instance = None

def get_rag_pipeline():
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        try:
            from services.embedder import Embedder
            from services.vector_store import VectorStore
            from services.retriever import Retriever
            from services.gemini_agent import GeminiAgent
            from services.rag_pipeline import RAGPipeline

            emb = Embedder(
                provider=_settings.get("embedding_provider", "local"),
                api_key=_settings.get("gemini_api_key") or os.getenv("GEMINI_API_KEY")
            )
            store_path = str(BASE_DIR / "data" / "vector_store")
            store = VectorStore(store_path=store_path)
            retriever = Retriever(
                embedder=emb,
                vector_store=store,
                cohere_api_key=_settings.get("cohere_api_key") or os.getenv("COHERE_API_KEY")
            )
            gemini_agent = GeminiAgent(
                api_key=_settings.get("gemini_api_key") or os.getenv("GEMINI_API_KEY"),
                model_name=_settings.get("gemini_model_name", "gemini-1.5-flash")
            )
            _rag_pipeline_instance = RAGPipeline(emb, store, retriever, gemini_agent)
            logger.info("RAG Pipeline successfully initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline in ai_query: {e}")
    return _rag_pipeline_instance


SYSTEM_PROMPT = (
    "You are BlindAssist, a concise assistant for visually impaired users. "
    "Answer in 1-3 short sentences. No markdown, no lists, no special characters. "
    "Speak naturally."
)

# ── CLIENT INITIALIZATION (lazy but cached) ─────────────────
_groq_client = None
_openai_client = None
_gemini_model = None
_offline_model = None
_offline_loading = threading.Event()
_offline_ready = False

def _init_groq():
    global _groq_client
    if _groq_client:
        return True
    key = _settings.get("groq_api_key") or os.environ.get("GROQ_API_KEY")
    if not key or key.startswith("YOUR_"):
        return False
    try:
        from groq import Groq
        _groq_client = Groq(api_key=key)
        return True
    except Exception as e:
        logger.debug(f"Groq unavailable: {e}")
        return False

def _init_openai():
    global _openai_client
    if _openai_client:
        return True
    key = _settings.get("openai_api_key") or os.environ.get("OPENAI_API_KEY")
    if not key or key.startswith("YOUR_"):
        return False
    try:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=key)
        return True
    except Exception as e:
        logger.debug(f"OpenAI unavailable: {e}")
        return False

def _init_gemini():
    global _gemini_model
    if _gemini_model:
        return True
    key = _settings.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    if not key or key.startswith("YOUR_"):
        return False
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        _gemini_model = genai.GenerativeModel(
            'gemini-1.5-flash',
            system_instruction=SYSTEM_PROMPT
        )
        return True
    except Exception as e:
        logger.debug(f"Gemini unavailable: {e}")
        return False

def _preload_offline_model():
    """Background thread: preload offline model to avoid 10-30s cold start."""
    global _offline_model, _offline_ready
    model_path = _settings.get("offline_model_path", "")
    if not model_path:
        _offline_loading.set()
        return
    
    # Try relative path
    if not Path(model_path).exists():
        alt = BASE_DIR / "models" / Path(model_path).name
        if alt.exists():
            model_path = str(alt)
        else:
            logger.info("Offline model not found, skipping preload.")
            _offline_loading.set()
            return
    
    try:
        from llama_cpp import Llama
        logger.info("Preloading offline model (this may take 20-30s)...")
        start = time.time()
        _offline_model = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
            n_batch=512,
        )
        _offline_ready = True
        logger.info(f"Offline model ready in {time.time()-start:.1f}s")
    except Exception as e:
        logger.warning(f"Offline preload failed: {e}")
    finally:
        _offline_loading.set()

# Start preload in background immediately
_preload_thread = threading.Thread(target=_preload_offline_model, daemon=True)
_preload_thread.start()

# ── API CALLERS (with timeouts) ─────────────────────────────

def _ask_groq(prompt: str, simplify: bool = False, timeout: float = 5.0) -> Optional[str]:
    if not _init_groq():
        return None
    try:
        system = SYSTEM_PROMPT
        if simplify:
            system += " Use very simple words."
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                _groq_client.chat.completions.create,
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            response = future.result(timeout=timeout)
            return response.choices[0].message.content.strip()
    except concurrent.futures.TimeoutError:
        logger.warning("Groq timeout.")
        return None
    except Exception as e:
        logger.error(f"Groq error: {e}")
        return None

def _ask_openai(prompt: str, simplify: bool = False, timeout: float = 8.0) -> Optional[str]:
    if not _init_openai():
        return None
    try:
        system = SYSTEM_PROMPT
        if simplify:
            system += " Use very simple words."
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                _openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=200,
                temperature=0.7,
            )
            response = future.result(timeout=timeout)
            return response.choices[0].message.content.strip()
    except concurrent.futures.TimeoutError:
        logger.warning("OpenAI timeout.")
        return None
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return None

def _ask_gemini(prompt: str, simplify: bool = False, timeout: float = 10.0) -> Optional[str]:
    if not _init_gemini():
        return None
    try:
        full = "Explain simply: " + prompt if simplify else prompt
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(_gemini_model.generate_content, full)
            response = future.result(timeout=timeout)
            return response.text.strip() if response and response.text else None
    except concurrent.futures.TimeoutError:
        logger.warning("Gemini timeout.")
        return None
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return None

def _ask_offline(prompt: str, simplify: bool = False) -> Optional[str]:
    global _offline_ready
    _offline_loading.wait(timeout=60)  # Wait for preload (or fail fast)
    if not _offline_ready or _offline_model is None:
        return None
    
    try:
        model_path = _settings.get("offline_model_path", "").lower()
        if "qwen" in model_path:
            full = (
                "<|im_start|>system\n" + SYSTEM_PROMPT + "\n"
                "<|im_start|>user\n" + prompt + "\n"
                "<|im_start|>assistant\n"
            )
            stops = ["<|im_end|>", "<|im_start|>"]
        elif "llama-3" in model_path:
            full = (
                "<|begin_of_text|>system\n" + SYSTEM_PROMPT + "\n<|eot_id|>"
                "user\n" + prompt + "\n<|eot_id|>"
                "assistant\n"
            )
            stops = ["<|eot_id|>"]
        else:
            full = f"User: {prompt}\nAssistant:"
            stops = ["User:", "\n\n"]
        
        response = _offline_model(
            full,
            max_tokens=150,
            stop=stops,
            echo=False,
            temperature=0.7,
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        logger.error(f"Offline error: {e}")
        return None

# ── PUBLIC API ──────────────────────────────────────────────

def ask_ai(prompt: str, context: str = '', simplify: bool = False) -> str:
    """
    Concurrent AI query with fastest-response-wins strategy.
    Tries Groq + OpenAI simultaneously, uses whichever answers first.
    Falls back to Gemini, then offline.
    """
    if not prompt or not prompt.strip():
        return "I didn't receive a question. Please try again."
    
    full_prompt = f"Context:\n{context}\n\nQuestion: {prompt}" if context else prompt
    logger.info(f"Query: \"{prompt[:50]}...\"")
    
    # Phase 1: Race Groq vs OpenAI (fastest wins)
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_ask_groq, full_prompt, simplify): 'groq',
            executor.submit(_ask_openai, full_prompt, simplify): 'openai',
        }
        for future in concurrent.futures.as_completed(futures, timeout=10):
            source = futures[future]
            try:
                result = future.result()
                if result:
                    results[source] = result
                    logger.info(f"First response from {source}")
                    break  # First valid answer wins
            except Exception:
                pass
    
    if results:
        return list(results.values())[0]
    
    # Phase 2: Try Gemini
    result = _ask_gemini(full_prompt, simplify)
    if result:
        return result
    
    # Phase 3: Offline (already preloaded)
    result = _ask_offline(full_prompt, simplify)
    if result:
        return result
    
    return "I'm sorry, I cannot answer right now. Please check your connection."

def index_text_in_rag(text: str):
    """Indexes raw text (e.g. OCR scan) into the RAG vector database."""
    pipeline = get_rag_pipeline()
    if pipeline:
        pipeline.index_document(text)

def index_file_in_rag(file_path: str) -> bool:
    """Indexes a text file (e.g. NCERT textbook chapter) into the RAG vector database."""
    pipeline = get_rag_pipeline()
    if pipeline:
        return pipeline.index_file(file_path)
    return False


if __name__ == '__main__':
    signal.signal(signal.SIGINT, lambda s, f: sys.exit(0))
    print("AI Query Optimized — racing Groq vs OpenAI")
    while True:
        q = input("Question (QUIT): ").strip()
        if q.upper() == "QUIT":
            break
        if q:
            print("Racing APIs...")
            start = time.time()
            print(f"Answer ({time.time()-start:.2f}s): {ask_ai(q)}\n")