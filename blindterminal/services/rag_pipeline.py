import logging
from pathlib import Path
from typing import List, Optional
from .embedder import Embedder
from .vector_store import VectorStore
from .retriever import Retriever
from .gemini_agent import GeminiAgent

logger = logging.getLogger("RAGPipelineService")

class RAGPipeline:
    """
    Orchestrates the complete RAG flow:
    Document / OCR text -> Chunking -> Embedding -> VectorStore -> Cohere Reranking -> Gemini LLM Response.
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
        Splits long text hierarchically across natural semantic boundaries 
        (paragraphs, sentences, spaces) to prevent severing words or thoughts mid-sentence.
        """
        if not text or not text.strip():
            return []

        separators = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

        def _split_text(text_to_split: str, seps: List[str]) -> List[str]:
            if len(text_to_split) <= chunk_size or not seps:
                return [text_to_split.strip()] if text_to_split.strip() else []

            sep = seps[0]
            next_seps = seps[1:]

            if sep == "":
                splits = [text_to_split[i:i + chunk_size] for i in range(0, len(text_to_split), chunk_size - overlap)]
                return [s.strip() for s in splits if s.strip()]

            parts = text_to_split.split(sep)
            docs = []
            current_chunk = []
            current_length = 0

            for part in parts:
                part_text = part + sep if sep in ["\n\n", "\n", ". ", "? ", "! "] else part + (" " if sep == " " else "")
                part_len = len(part_text)

                if part_len > chunk_size:
                    if current_chunk:
                        joined = "".join(current_chunk).strip()
                        if joined:
                            docs.append(joined)
                        current_chunk = []
                        current_length = 0
                    docs.extend(_split_text(part, next_seps))
                elif current_length + part_len > chunk_size:
                    joined = "".join(current_chunk).strip()
                    if joined:
                        docs.append(joined)
                    current_chunk = [part_text]
                    current_length = part_len
                else:
                    current_chunk.append(part_text)
                    current_length += part_len

            if current_chunk:
                joined = "".join(current_chunk).strip()
                if joined:
                    docs.append(joined)

            return docs

        return _split_text(text, separators)

    def index_document(self, text: str):
        """
        Takes raw OCR text or document text, chunks it, embeds it, and stores it in vector store.
        """
        if not text or not text.strip():
            logger.warning("No text provided for indexing.")
            return

        try:
            chunks = self._chunk_text(text)
            logger.info(f"Indexing text into {len(chunks)} chunks.")

            embeddings = self.embedder.get_embeddings(chunks)
            if embeddings.size > 0:
                self.vector_store.add_batch(chunks, embeddings)
                logger.info("Text successfully indexed in vector store.")
            else:
                logger.warning("Embedding generation returned empty matrix.")

        except Exception as e:
            logger.error(f"Error indexing document: {e}")

    def index_file(self, file_path: str):
        """
        Reads a local text file (e.g., NCERT chapter) and indexes its content.
        """
        path = Path(file_path)
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return False

        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            self.index_document(content)
            return True
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            return False

    def process_scan(self, text: str):
        """Alias for index_document (backwards compatibility)."""
        self.index_document(text)

    def ask_question(self, query: str, top_k: int = 3) -> str:
        """
        Performs semantic retrieval, reranking, and generates a contextual response via Gemini / LLM.
        """
        try:
            context = self.retriever.get_relevant_context(query, k=top_k)
            logger.info(f"Retrieved context length: {len(context)} chars.")

            response = self.gemini_agent.generate_response(query, context)
            return response

        except Exception as e:
            logger.error(f"Error in RAG query flow: {e}")
            return "I'm sorry, I encountered an issue retrieving information from the textbook database."

