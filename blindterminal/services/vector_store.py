import os
import re
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Optional
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("VectorStoreService")

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False
    logger.warning("FAISS not installed. VectorStore will use numpy cosine similarity as local fallback.")

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    logger.warning("rank-bm25 not installed. Hybrid search will rely solely on dense vectors.")


class VectorStore:
    """
    Production-grade Vector Storage & Hybrid Search Engine.
    Supports:
    1. Pinecone Cloud Vector DB
    2. FAISS Index Flat (local dense vector search)
    3. BM25 Keyword Search (local sparse search)
    4. Reciprocal Rank Fusion (RRF) combining Dense + Sparse rankings
    """
    def __init__(self, store_path: str, dimension: int = 384):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.index_file = self.store_path / "index.faiss"
        self.metadata_file = self.store_path / "metadata.pkl"
        self.vectors_file = self.store_path / "vectors.npy"

        self.dimension = dimension
        self.metadata = []
        self.vectors_list = []
        self.bm25 = None
        
        self.pinecone_key = os.getenv("PINECONE_API_KEY") or os.getenv("PINECONE_KEY")
        self.pinecone_index_name = os.getenv("PINECONE_INDEX", "ncert-books")
        self.pinecone_index = None

        if self.pinecone_key and not self.pinecone_key.startswith("YOUR_"):
            try:
                from pinecone import Pinecone
                pc = Pinecone(api_key=self.pinecone_key)
                self.pinecone_index = pc.Index(self.pinecone_index_name)
                logger.info(f"Connected to Pinecone Cloud Vector DB: {self.pinecone_index_name}")
            except Exception as e:
                logger.warning(f"Pinecone Cloud setup fallback: {e}")

        self.index = None
        self.load_index()

    def _tokenize(self, text: str) -> List[str]:
        """Simple alphanumeric tokenizer for BM25 keyword matching."""
        return re.findall(r'\w+', text.lower())

    def _rebuild_bm25(self):
        """Rebuilds BM25 index over self.metadata."""
        if HAS_BM25 and self.metadata:
            tokenized_corpus = [self._tokenize(doc) for doc in self.metadata]
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.debug(f"BM25 index rebuilt over {len(self.metadata)} documents.")

    def add_text(self, text: str, embedding: np.ndarray):
        if embedding.size == 0:
            return
        self.add_batch([text], embedding.reshape(1, -1))

    def add_batch(self, texts: List[str], embeddings: np.ndarray):
        if embeddings.size == 0 or not texts:
            return

        vecs = embeddings.astype('float32')
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)

        # Dynamic dimension adjustment for FAISS if needed
        actual_dim = vecs.shape[1]
        if actual_dim != self.dimension:
            self.dimension = actual_dim
            if HAS_FAISS:
                self.index = faiss.IndexFlatL2(self.dimension)
                if self.vectors_list:
                    existing_mat = np.array(self.vectors_list, dtype=np.float32)
                    if existing_mat.shape[1] == self.dimension:
                        self.index.add(existing_mat)

        if self.pinecone_index is not None:
            try:
                import uuid
                vectors_to_upsert = []
                for vec, text in zip(vecs, texts):
                    doc_id = str(uuid.uuid4())
                    vectors_to_upsert.append((doc_id, vec.tolist(), {"text": text}))
                self.pinecone_index.upsert(vectors=vectors_to_upsert)
                logger.info(f"Upserted {len(vectors_to_upsert)} vectors to Pinecone Cloud.")
            except Exception as e:
                logger.error(f"Pinecone Cloud upsert error: {e}")

        if HAS_FAISS:
            if self.index is None:
                self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(vecs)
        
        for vec, text in zip(vecs, texts):
            self.vectors_list.append(vec)
            self.metadata.append(text)

        self._rebuild_bm25()
        self.save_index()

    def retrieve(self, query_embedding: np.ndarray, query_text: Optional[str] = None, k: int = 3) -> List[Tuple[int, float, str]]:
        if len(self.metadata) == 0 and self.pinecone_index is None:
            return []

        vec = query_embedding.astype('float32').reshape(-1)
        vector_results = []

        # 1. Pinecone Retrieval
        if self.pinecone_index is not None:
            try:
                res = self.pinecone_index.query(vector=vec.tolist(), top_k=max(k*3, 10), include_metadata=True)
                for idx, match in enumerate(res.matches):
                    text = match.metadata.get("text", "")
                    score = match.score
                    vector_results.append((idx, float(1.0 - score), text))
            except Exception as e:
                logger.error(f"Pinecone Cloud query error, falling back: {e}")

        # 2. Local Dense Retrieval (FAISS or Numpy) if Pinecone returned empty
        if not vector_results and self.metadata:
            vec_mat = vec.reshape(1, -1)
            if HAS_FAISS and self.index is not None and self.index.ntotal > 0:
                search_k = min(max(k * 3, 10), self.index.ntotal)
                distances, indices = self.index.search(vec_mat, search_k)
                for dist, idx in zip(distances[0], indices[0]):
                    if idx != -1 and idx < len(self.metadata):
                        vector_results.append((int(idx), float(dist), self.metadata[idx]))
            elif self.vectors_list:
                matrix = np.array(self.vectors_list)
                if matrix.shape[1] == vec.shape[0]:
                    norm_q = vec / (np.linalg.norm(vec) + 1e-10)
                    norm_m = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
                    similarities = np.dot(norm_m, norm_q.T).flatten()
                    top_idx = np.argsort(similarities)[::-1][:max(k * 3, 10)]
                    for idx in top_idx:
                        vector_results.append((int(idx), float(1.0 - similarities[idx]), self.metadata[idx]))

        # 3. Hybrid Search Fusion (Reciprocal Rank Fusion) if query_text & BM25 available
        if query_text and HAS_BM25 and self.bm25 and self.metadata:
            tokenized_query = self._tokenize(query_text)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            bm25_top_indices = np.argsort(bm25_scores)[::-1][:max(k * 3, 10)]

            # Map doc text -> RRF score
            rrf_scores = {}
            c = 60 # standard RRF constant

            # Vector ranks
            for rank, item in enumerate(vector_results):
                doc_text = item[2]
                rrf_scores[doc_text] = rrf_scores.get(doc_text, 0.0) + (1.0 / (c + rank + 1))

            # BM25 ranks
            for rank, idx in enumerate(bm25_top_indices):
                if idx < len(self.metadata) and bm25_scores[idx] > 0:
                    doc_text = self.metadata[idx]
                    rrf_scores[doc_text] = rrf_scores.get(doc_text, 0.0) + (1.0 / (c + rank + 1))

            if rrf_scores:
                sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
                return [(i, float(1.0 - score), text) for i, (text, score) in enumerate(sorted_docs)]

        # Fallback to standard vector results
        return vector_results[:k]

    def save_index(self):
        try:
            if HAS_FAISS and self.index is not None:
                faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
            if self.vectors_list:
                np.save(self.vectors_file, np.array(self.vectors_list))
            logger.info(f"Vector store saved to {self.store_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")

    def load_index(self):
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                if self.vectors_file.exists():
                    vecs = np.load(self.vectors_file)
                    self.vectors_list = list(vecs)
                    if vecs.ndim > 1:
                        self.dimension = vecs.shape[1]

                if HAS_FAISS and self.index_file.exists():
                    self.index = faiss.read_index(str(self.index_file))
                    self.dimension = self.index.d
                elif HAS_FAISS and self.vectors_list:
                    self.index = faiss.IndexFlatL2(self.dimension)
                    self.index.add(np.array(self.vectors_list, dtype=np.float32))

                self._rebuild_bm25()
                logger.info(f"Vector store loaded successfully from {self.store_path}")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")
                self.metadata = []
                self.vectors_list = []

