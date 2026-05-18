import logging
from typing import List
import numpy as np
from .embedder import Embedder
from .vector_store import VectorStore

logger = logging.getLogger("RetrieverService")

class Retriever:
    """
    Handles the semantic search process by coordinating the Embedder and VectorStore.
    """
    def __init__(self, embedder: Embedder, vector_store: VectorStore):
        self.embedder = embedder
        self.vector_store = vector_store

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """
        Retrieves the top-k most relevant text chunks for a given query.
        Returns the chunks joined as a single context string.
        """
        if not query or not query.strip():
            return ""

        try:
            # 1. Embed the query
            query_vec = self.embedder.get_embedding(query)

            if query_vec.size == 0:
                return ""

            # 2. Search in FAISS
            results = self.vector_store.retrieve(query_vec, k=k)

            if not results:
                return ""

            # 3. Extract just the text from the results (index, distance, text)
            context_chunks = [text for _, _, text in results]

            # Join chunks with a separator for the LLM
            return "\n---\n".join(context_chunks)

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""
