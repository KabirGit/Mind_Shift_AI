from __future__ import annotations

from backend.retrieval.retriever import Retriever


class _StubStore:
    def query(self, query_text, top_k=5):
        return []


def _r():
    return Retriever(_StubStore())


def test_exact_match_full_score():
    assert _r()._emotion_similarity("joy", "joy") == 1.0


def test_neutral_partial():
    assert _r()._emotion_similarity("joy", "neutral") == 0.5
    assert _r()._emotion_similarity("neutral", "anger") == 0.5


def test_same_valence_partial():
    # Both positive (GoEmotions) -> partial similarity, not zero.
    assert _r()._emotion_similarity("joy", "gratitude") == 0.6
    # Both negative.
    assert _r()._emotion_similarity("sadness", "fear") == 0.6


def test_opposite_valence_zero():
    assert _r()._emotion_similarity("joy", "anger") == 0.0
