import hashlib
import logging
from typing import Any

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class _HashingEmbeddingModel:
    """Small deterministic embedding fallback for low-memory demos."""

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def encode(self, texts: list[str], show_progress_bar: bool = False) -> np.ndarray:
        _ = show_progress_bar
        rows = []
        for text in texts:
            vec = np.zeros(self.dim, dtype="float32")
            tokens = str(text).lower().split()
            for token in tokens or [""]:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            norm = float(np.linalg.norm(vec))
            if norm:
                vec /= norm
            rows.append(vec)
        return np.array(rows, dtype="float32")


class EmbeddingPipeline:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self.model_name = model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model: Any
        if model_name.lower() in {"hashing", "hashing-384", "lightweight"}:
            self.model = _HashingEmbeddingModel()
        else:
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer(model_name)
            except ImportError:
                logger.warning(
                    "sentence-transformers is not installed; using hashing embeddings."
                )
                self.model = _HashingEmbeddingModel()
        logger.info("Loaded embedding model: %s", model_name)

    def chunk_documents(self, documents: list[Any]) -> list[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(documents)
        logger.info("Split %s docs into %s chunks", len(documents), len(chunks))
        return chunks

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings, dtype="float32")

    def embed_chunks(self, chunks: list[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        return self.embed_texts(texts)
