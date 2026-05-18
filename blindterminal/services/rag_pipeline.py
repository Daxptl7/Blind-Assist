import logging
from typing import List
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .gemini_agent import GeminiAgent

logger = logging.getLogger("RAGPipelineService")

class RAGPipeline:
    """
    Orchestrates the flow from OCR text to AI response.
    """
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        retriever: Retriever,
        gemini_agent: GeminiAgent
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.retriever = retriever
        self.gemini_agent = gemini_agent

    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """
        Splits long text into smaller chunks for better retrieval accuracy.
        """
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i : i + chunk_size])
        return chunks

    def process_scan(self, text: str):
        """
        Takes raw OCR text, chunks it, embeds it, and stores it in the vector DB.
        """
        if not text or not text.strip():
            logger.warning("No text provided for processing.")
            return

        try:
            # 1. Chunk the text
            chunks = self._chunk_text(text)
            logger.info(f"Text split into {len(chunks)} chunks.")

            # 2. Generate embeddings for all chunks
            embeddings = self.embedder.get_embeddings(chunks)

            # 3. Store in FAISS
            self.vector_store.add_batch(chunks, embeddings)
            logger.info("OCR text successfully indexed in vector store.")

        except Exception as e:
            logger.error(f"Error processing scan: {e}")

    def ask_question(self, query: str) -> str:
        """
        Performs semantic retrieval and generates a contextual response via Gemini.
        """
        try:
            # 1. Retrieve relevant context
            context = self.retriever.get_relevant_context(query)
            logger.info(f"Retrieved context length: {len(context)} chars.")

            # 2. Generate AI response using context
            response = self.gemini_agent.generate_response(query, context)
            return response

        except Exception as e:
            logger.error(f"Error in RAG query flow: {e}")
            return "I'm sorry, I had trouble processing your question."
