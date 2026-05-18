import logging
from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

logger = logging.getLogger("EmbedderService")

class Embedder:
    """
    Lightweight embedding service using sentence-transformers.
    Optimized for Raspberry Pi 5.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        try:
            logger.info(f"Initializing embedding model: {model_name}...")
            self.model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Converts a single string into a dense vector.
        """
        if not text or not text.strip():
            return np.array([])

        try:
            embedding = self.model.encode(text)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.array([])

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Converts a list of strings into a matrix of dense vectors.
        """
        if not texts:
            return np.array([])

        try:
            embeddings = self.model.encode(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return np.array([])
