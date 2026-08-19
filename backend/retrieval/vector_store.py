from __future__ import annotations

import hashlib
import logging
import os
import pickle
import threading
from typing import Any

import faiss

from backend.embedding.pipeline import EmbeddingPipeline

logger = logging.getLogger(__name__)


class FaissVectorStore:
    def __init__(
        self,
        persist_dir: str,
        embedding_pipeline: EmbeddingPipeline,
        ltm_max_entries: int = 0,
    ) -> None:
        self.persist_dir = persist_dir
        self.embedding_pipeline = embedding_pipeline
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []
        self._known_hashes: set[str] = set()
        self._lock = threading.Lock()
        self.ltm_max_entries = max(0, int(ltm_max_entries))
        os.makedirs(self.persist_dir, exist_ok=True)

    @property
    def faiss_path(self) -> str:
        return os.path.join(self.persist_dir, "faiss.index")

    @property
    def meta_path(self) -> str:
        return os.path.join(self.persist_dir, "metadata.pkl")

    def exists(self) -> bool:
        return os.path.exists(self.faiss_path) and os.path.exists(self.meta_path)

    def load(self) -> None:
        if not self.exists():
            raise FileNotFoundError("Vector store files not found.")
        self.index = faiss.read_index(self.faiss_path)
        with open(self.meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        self._known_hashes = {
            m.get("entry_hash", "") for m in self.metadata if m.get("entry_hash")
        }
        logger.info("Loaded vector store with %s items", len(self.metadata))

    def save(self) -> None:
        if self.index is None:
            raise ValueError("Cannot save empty index.")
        faiss.write_index(self.index, self.faiss_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        logger.info("Persisted vector store to %s", self.persist_dir)

    def _ensure_index(self, dim: int) -> None:
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

    def build_from_documents(self, documents: list[Any]) -> None:
        chunks = self.embedding_pipeline.chunk_documents(documents)
        records = [
            {
                "text": chunk.page_content,
                "timestamp": None,
                "emotion": "neutral",
                "emotion_intensity": 0.0,
                "tags": [],
            }
            for chunk in chunks
        ]
        if not records:
            dim = self.embedding_pipeline.model.get_sentence_embedding_dimension()
            self._ensure_index(int(dim or 384))
            self.save()
            return
        self.add_entries(records, save=False)
        self.save()

    @staticmethod
    def _hash_entry(text: str, timestamp: str | None) -> str:
        # Deduplicate by content only (not timestamp), so repeated submissions
        # don't cause repeated embedding work.
        _ = timestamp  # timestamp still stored in metadata; not used for dedup key
        normalized = " ".join(str(text).strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def add_entries(self, entries: list[dict[str, Any]], save: bool = True) -> int:
        if not entries:
            return 0

        with self._lock:
            unique_entries: list[dict[str, Any]] = []
            for entry in entries:
                text = str(entry.get("text", "")).strip()
                if not text:
                    continue
                entry_hash = self._hash_entry(text, entry.get("timestamp"))
                if entry_hash in self._known_hashes:
                    continue
                entry["entry_hash"] = entry_hash
                unique_entries.append(entry)

            if not unique_entries:
                logger.info("No new entries to add (all duplicates or empty).")
                return 0

            if self.ltm_max_entries > 0:
                remaining = max(0, self.ltm_max_entries - len(self.metadata))
                if remaining <= 0:
                    logger.info(
                        "LTM max entries reached (%s). Skipping new additions.",
                        self.ltm_max_entries,
                    )
                    return 0
                unique_entries = unique_entries[:remaining]

            texts = [entry["text"] for entry in unique_entries]
            embeddings = self.embedding_pipeline.embed_texts(texts)
            self._ensure_index(embeddings.shape[1])
            self.index.add(embeddings)
            self.metadata.extend(unique_entries)
            self._known_hashes.update(entry["entry_hash"] for entry in unique_entries)

            if save:
                self.save()
            logger.info("Added %s entries to vector store", len(unique_entries))
            return len(unique_entries)

    def query(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        with self._lock:
            if self.index is None or self.index.ntotal == 0:
                return []
            query_emb = self.embedding_pipeline.embed_texts([query_text])
            distances, indices = self.index.search(query_emb, top_k)

            results: list[dict[str, Any]] = []
            for idx, dist in zip(indices[0], distances[0], strict=False):
                if idx < 0 or idx >= len(self.metadata):
                    continue
                results.append(
                    {
                        "index": int(idx),
                        "distance": float(dist),
                        "metadata": self.metadata[idx],
                    }
                )
            return results
