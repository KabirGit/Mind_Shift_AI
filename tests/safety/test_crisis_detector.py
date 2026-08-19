from __future__ import annotations

import hashlib

from backend.api.rag_service import RAGService
from backend.safety.crisis_detector import CRISIS_MESSAGE, CrisisDetector
from backend.storage.db import JournalDB


def test_trigger_phrase_flagged():
    out = CrisisDetector().check("Sometimes I just want to kill myself.")
    assert out["flagged"] is True
    assert "kill myself" in out["matched_terms"]


def test_neutral_text_not_flagged():
    out = CrisisDetector().check("I had a calm and productive day at work.")
    assert out["flagged"] is False
    assert out["matched_terms"] == []


def test_empty_text_not_flagged():
    out = CrisisDetector().check("")
    assert out["flagged"] is False


# --- full pipeline crisis integration (with fakes, no models/network) ---


class FakeVectorStore:
    @staticmethod
    def _hash_entry(text, timestamp):
        return hashlib.sha256(text.lower().encode()).hexdigest()


class FakeEmotion:
    def detect(self, text):
        return {"emotion": "sadness", "confidence": 0.9}


class FakeRetriever:
    def retrieve(self, query, query_emotion="neutral", top_k=5):
        return []


class FakeMemory:
    def store_entry(self, text, tags=None, emotion_signal=None, topics=None):
        return {"text": text, "emotion": "sadness", "timestamp": "2026-01-01T00:00:00Z"}

    def get_recent_memory(self, limit=3):
        return []


class FakePrompt:
    def build(self, **kwargs):
        return "PROMPT"


class FakeLLM:
    def generate(self, prompt):
        return "normal empathetic reply"


class FakeTextProcessor:
    def extract(self, text):
        return {
            "entities_people": [], "entities_places": [], "entities_orgs": [],
            "keywords": [], "topics": [], "habits": [], "sentiment_compound": 0.0,
            "sentiment_valence": 0.0,
        }


def test_run_pipeline_flagged_prepends_safety_message(tmp_path):
    svc = RAGService(
        vector_store=FakeVectorStore(),
        emotion_detector=FakeEmotion(),
        retriever=FakeRetriever(),
        prompt_builder=FakePrompt(),
        llm_client=FakeLLM(),
        memory_manager=FakeMemory(),
        journal_db=JournalDB(str(tmp_path / "j.db")),
        text_processor=FakeTextProcessor(),
    )
    out = svc.run_pipeline(text="I want to die and end my life", chat_history=[])
    assert out["crisis"]["flagged"] is True
    assert CRISIS_MESSAGE in out["response"]
    # The normal pipeline still ran (reply appended after the safety message).
    assert "normal empathetic reply" in out["response"]
