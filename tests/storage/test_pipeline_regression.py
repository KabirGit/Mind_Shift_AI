from __future__ import annotations

import hashlib

from backend.api.rag_service import RAGService
from backend.storage.db import JournalDB


class FakeVectorStore:
    @staticmethod
    def _hash_entry(text, timestamp):
        normalized = " ".join(str(text).strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class FakeEmotion:
    def detect(self, text):
        return {"emotion": "joy", "confidence": 0.8}


class FakeRetriever:
    def retrieve(self, query, query_emotion="neutral", top_k=5):
        return [{"metadata": {"text": "past", "emotion": "joy", "timestamp": ""}, "scores": {"combined": 0.5}}]


class FakeMemory:
    def store_entry(self, text, tags=None, emotion_signal=None, topics=None):
        return {
            "text": text,
            "emotion": "joy",
            "timestamp": "2026-01-01T00:00:00Z",
            "topics": topics or [],
        }

    def get_recent_memory(self, limit=3):
        return [{"text": "recent", "emotion": "joy", "timestamp": "2026-01-01T00:00:00Z"}]


class FakePrompt:
    def build(self, **kwargs):
        return "PROMPT"


class FakeLLM:
    def generate(self, prompt):
        return "a response"


class FakeTextProcessor:
    def extract(self, text):
        return {
            "entities_people": [],
            "entities_places": [],
            "entities_orgs": [],
            "keywords": [],
            "topics": [],
            "habits": [],
            "sentiment_compound": 0.0,
            "sentiment_valence": 0.0,
        }


def _service(tmp_path):
    return RAGService(
        vector_store=FakeVectorStore(),
        emotion_detector=FakeEmotion(),
        retriever=FakeRetriever(),
        prompt_builder=FakePrompt(),
        llm_client=FakeLLM(),
        memory_manager=FakeMemory(),
        journal_db=JournalDB(str(tmp_path / "j.db")),
        text_processor=FakeTextProcessor(),
    )


def test_run_pipeline_shape_unchanged(tmp_path):
    svc = _service(tmp_path)
    out = svc.run_pipeline(text="I feel great today", chat_history=[], top_k=3)
    assert {
        "emotion",
        "stored_entry",
        "retrieved_memories",
        "prompt",
        "response",
    }.issubset(out.keys())
    assert out["response"] == "a response"
    assert out["emotion"] == {"emotion": "joy", "confidence": 0.8}
    assert "topics" in out["stored_entry"]


def test_run_pipeline_writes_journal_record(tmp_path):
    svc = _service(tmp_path)
    svc.run_pipeline(text="I feel great today", chat_history=[], top_k=3)
    rows = svc.journal_db.get_all()
    assert len(rows) == 1
    assert rows[0].emotion == "joy"
