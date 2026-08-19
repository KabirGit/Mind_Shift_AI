import logging
from typing import Any

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


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
        self.model = SentenceTransformer(model_name)
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
