from __future__ import annotations

from typing import Any

from backend.analytics._stats_utils import recency_decay
from backend.retrieval.vector_store import FaissVectorStore

# Coarse valence grouping for fine-grained (GoEmotions-style) labels. Used to
# give related emotions a partial similarity score instead of a hard zero.
_POSITIVE = {
    "admiration", "amusement", "approval", "caring", "desire", "excitement",
    "gratitude", "joy", "love", "optimism", "pride", "relief", "curiosity",
    "happiness",
}
_NEGATIVE = {
    "anger", "annoyance", "disappointment", "disapproval", "disgust",
    "embarrassment", "fear", "grief", "nervousness", "remorse", "sadness",
}
_AMBIGUOUS = {"confusion", "realization", "surprise"}

_EMOTION_VALENCE: dict[str, str] = {
    **{e: "positive" for e in _POSITIVE},
    **{e: "negative" for e in _NEGATIVE},
    **{e: "ambiguous" for e in _AMBIGUOUS},
}


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
        for item, semantic in zip(candidates, semantic_scores, strict=False):
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
        # Valence-aware partial match: with a fine-grained (e.g. 28-label)
        # taxonomy, distinct-but-related emotions should still score above 0.
        q_val = _EMOTION_VALENCE.get(query_emotion)
        m_val = _EMOTION_VALENCE.get(memory_emotion)
        if q_val is not None and m_val is not None and q_val == m_val:
            return 0.6
        return 0.0

    def _recency_weight(self, timestamp: str) -> float:
        return recency_decay(timestamp, self.half_life_hours)
