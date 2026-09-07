from __future__ import annotations

import re
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
        *,
        exclude_entry_hashes: set[str] | None = None,
        diversify: bool = False,
        diversity_penalty: float = 0.15,
    ) -> list[dict[str, Any]]:
        candidates = self.vector_store.query(
            query_text=query,
            top_k=max(top_k, self.candidate_pool),
        )
        excluded = exclude_entry_hashes or set()
        if excluded:
            candidates = [
                item
                for item in candidates
                if str(item.get("metadata", {}).get("entry_hash", "")) not in excluded
            ]
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
        if diversify:
            return self._diverse_top_k(ranked, top_k, max(0.0, diversity_penalty))
        return ranked[:top_k]

    @classmethod
    def _diverse_top_k(
        cls,
        ranked: list[dict[str, Any]],
        top_k: int,
        penalty: float,
    ) -> list[dict[str, Any]]:
        """Penalize near-repeated text while preserving the existing ranking signal."""

        remaining = list(enumerate(ranked))
        selected: list[dict[str, Any]] = []
        while remaining and len(selected) < top_k:
            best_position = max(
                range(len(remaining)),
                key=lambda position: (
                    float(remaining[position][1]["scores"]["combined"])
                    - penalty
                    * max(
                        (
                            cls._token_overlap(
                                remaining[position][1], prior
                            )
                            for prior in selected
                        ),
                        default=0.0,
                    ),
                    -remaining[position][0],
                ),
            )
            _, item = remaining.pop(best_position)
            if selected and max(cls._token_overlap(item, prior) for prior in selected) >= 0.82:
                continue
            selected.append(item)
        return selected

    @staticmethod
    def _token_overlap(left: dict[str, Any], right: dict[str, Any]) -> float:
        def tokens(item: dict[str, Any]) -> set[str]:
            text = str(item.get("metadata", {}).get("text", "")).casefold()
            return set(re.findall(r"[a-z0-9]+", text))

        left_tokens = tokens(left)
        right_tokens = tokens(right)
        union = left_tokens | right_tokens
        return len(left_tokens & right_tokens) / len(union) if union else 0.0

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
