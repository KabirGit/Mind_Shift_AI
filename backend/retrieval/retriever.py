from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from backend.retrieval.vector_store import FaissVectorStore


class Retriever:
    def __init__(
        self,
        vector_store: FaissVectorStore,
        semantic_weight: float = 0.6,
        emotion_weight: float = 0.25,
        recency_weight: float = 0.15,
        half_life_hours: float = 72.0,
        candidate_pool: int = 20,
    ) -> None:
        self.vector_store = vector_store
        self.semantic_weight = semantic_weight
        self.emotion_weight = emotion_weight
        self.recency_weight = recency_weight
        self.half_life_hours = max(1.0, half_life_hours)
        self.candidate_pool = max(candidate_pool, 5)

    def retrieve(
        self,
        query: str,
        query_emotion: str = "neutral",
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        candidates = self.vector_store.query(
            query_text=query,
            top_k=max(top_k, self.candidate_pool),
        )
        if not candidates:
            return []

        semantic_scores = self._normalize_semantic(candidates)
        ranked = []
        for item, semantic in zip(candidates, semantic_scores):
            meta = item.get("metadata", {})
            emotion = self._emotion_similarity(
                query_emotion=query_emotion,
                memory_emotion=str(meta.get("emotion", "neutral")),
            )
            recency = self._recency_weight(str(meta.get("timestamp", "")))
            score = (
                self.semantic_weight * semantic
                + self.emotion_weight * emotion
                + self.recency_weight * recency
            )
            ranked.append(
                {
                    **item,
                    "scores": {
                        "semantic": round(semantic, 4),
                        "emotion": round(emotion, 4),
                        "recency": round(recency, 4),
                        "combined": round(score, 4),
                    },
                }
            )

        ranked.sort(key=lambda x: x["scores"]["combined"], reverse=True)
        return ranked[:top_k]

    def _normalize_semantic(self, candidates: list[dict[str, Any]]) -> list[float]:
        distances = [float(c.get("distance", 0.0)) for c in candidates]
        d_min, d_max = min(distances), max(distances)
        if d_max == d_min:
            return [1.0 for _ in distances]
        return [1.0 - ((d - d_min) / (d_max - d_min)) for d in distances]

    def _emotion_similarity(self, query_emotion: str, memory_emotion: str) -> float:
        query_emotion = (query_emotion or "neutral").lower()
        memory_emotion = (memory_emotion or "neutral").lower()
        if query_emotion == memory_emotion:
            return 1.0
        if query_emotion == "neutral" or memory_emotion == "neutral":
            return 0.5
        return 0.0

    def _recency_weight(self, timestamp: str) -> float:
        if not timestamp:
            return 0.3
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            age_hours = max(0.0, (datetime.now(UTC) - dt).total_seconds() / 3600.0)
            return 0.5 ** (age_hours / self.half_life_hours)
        except ValueError:
            return 0.3
