import faiss
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

logger = logging.getLogger("VectorStoreService")

class VectorStore:
    """
    FAISS-based vector storage for local semantic retrieval.
    Handles indexing, storage, and retrieval of text chunks.
    """
    def __init__(self, store_path: str, dimension: int = 384):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.store_path / "index.faiss"
        self.metadata_file = self.store_path / "metadata.pkl"

        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)
        self.metadata = []  # Stores the original text chunks

        self.load_index()

    def add_text(self, text: str, embedding: np.ndarray):
        """
        Adds a text chunk and its corresponding embedding to the store.
        """
        if embedding.size == 0:
            return

        # FAISS expects float32
        vec = embedding.astype('float32').reshape(1, -1)

        self.index.add(vec)
        self.metadata.append(text)

        # Periodic save can be implemented here, or call save_index manually
        self.save_index()

    def add_batch(self, texts: List[str], embeddings: np.ndarray):
        """
        Adds multiple text chunks and their embeddings.
        """
        if embeddings.size == 0:
            return

        vecs = embeddings.astype('float32')
        self.index.add(vecs)
        self.metadata.extend(texts)
        self.save_index()

    def retrieve(self, query_embedding: np.ndarray, k: int = 3) -> List[Tuple[int, float, str]]:
        """
        Finds the top-k nearest neighbors for a query embedding.
        Returns: List of (index, distance, text)
        """
        if self.index.ntotal == 0:
            return []

        vec = query_embedding.astype('float32').reshape(1, -1)
        distances, indices = self.index.search(vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                results.append((int(idx), float(dist), self.metadata[idx]))

        return results

    def save_index(self):
        """
        Persists the FAISS index and metadata to disk.
        """
        try:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            logger.info(f"Vector store saved to {self.store_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def load_index(self):
        """
        Loads the FAISS index and metadata from disk.
        """
        if self.index_file.exists() and self.metadata_file.exists():
            try:
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Vector store loaded from {self.store_path}")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                self.index = faiss.IndexFlatL2(self.dimension)
                self.metadata = []
        else:
            logger.info("No existing vector store found. Starting fresh.")
