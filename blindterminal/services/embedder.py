import os
import logging
from typing import List, Optional
import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("EmbedderService")

class Embedder:
    """
    Dual-mode embedding service supporting:
    1. Gemini Embedding-001 (cloud API as specified in purchase sheet)
    2. sentence-transformers / all-MiniLM-L6-v2 (local offline fallback)
    """
    def __init__(self, provider: str = 'local', model_name: str = 'all-MiniLM-L6-v2', api_key: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        self.local_model = None
        self.use_gemini = False

        if provider == 'gemini':
            key = api_key or os.getenv("GEMINI_API_KEY")
            if key and not key.startswith("YOUR_"):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=key)
                    self.use_gemini = True
                    logger.info("Embedder initialized with Gemini Embedding API.")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini Embeddings, falling back to local: {e}")

        if not self.use_gemini:
            self._init_local_model()

        # Dynamic dimension property
        if self.use_gemini:
            self.dimension = 768
        elif self.local_model:
            try:
                if hasattr(self.local_model, 'get_embedding_dimension'):
                    self.dimension = self.local_model.get_embedding_dimension()
                else:
                    self.dimension = self.local_model.get_sentence_embedding_dimension()
            except Exception:
                self.dimension = 384
        else:
            self.dimension = 384

    def _init_local_model(self):
        if self.local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Initializing local embedding model: {self.model_name}...")
                self.local_model = SentenceTransformer(self.model_name)
                logger.info("Local embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load local embedding model: {e}")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Converts a single string into a dense vector embedding.
        """
        if not text or not text.strip():
            return np.array([])

        if self.use_gemini:
            try:
                import google.generativeai as genai
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_document"
                )
                return np.array(result['embedding'], dtype=np.float32)
            except Exception as e:
                logger.error(f"Gemini embedding error, falling back to local: {e}")

        self._init_local_model()
        if self.local_model:
            try:
                return np.array(self.local_model.encode(text), dtype=np.float32)
            except Exception as e:
                logger.error(f"Local embedding error: {e}")

        return np.array([])

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Converts a list of strings into a matrix of dense vectors.
        """
        if not texts:
            return np.array([])

        embeddings = [self.get_embedding(t) for t in texts if t]
        valid = [e for e in embeddings if e.size > 0]
        if valid:
            return np.vstack(valid)
        return np.array([])

