from __future__ import annotations

from collections import deque
from typing import Any

from backend.emotion.detector import EmotionDetector
from backend.memory.schema import MemoryEntry
from backend.retrieval.vector_store import FaissVectorStore


class MemoryManager:
    def __init__(
        self,
        vector_store: FaissVectorStore,
        emotion_detector: EmotionDetector,
        stm_size: int = 10,
        stm: deque[MemoryEntry] | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.emotion_detector = emotion_detector
        self.stm = stm if stm is not None else deque(maxlen=stm_size)

    def store_entry(
        self,
        text: str,
        tags: list[str] | None = None,
        emotion_signal: dict[str, Any] | None = None,
        topics: list[str] | None = None,
        person_relationship_types: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        signal = emotion_signal or self.emotion_detector.detect(text)
        entry = MemoryEntry(
            text=text,
            emotion=signal["emotion"],
            emotion_intensity=signal["confidence"],
            tags=tags or [],
            topics=topics or [],
            person_relationship_types=person_relationship_types or {},
        )
        self.stm.append(entry)
        self.vector_store.add_entries([entry.to_metadata()], save=True)
        return entry.to_metadata()

    def get_recent_memory(self, limit: int = 5) -> list[dict[str, Any]]:
        entries = list(self.stm)[-limit:]
        return [entry.to_metadata() for entry in entries]

    def search_long_term_memory(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.vector_store.query(query_text=query, top_k=top_k)
