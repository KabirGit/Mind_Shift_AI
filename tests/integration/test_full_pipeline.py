from __future__ import annotations

import hashlib

import pytest

from backend.api.rag_service import RAGService
from backend.nlp.text_processor import TextProcessor
from backend.storage.db import JournalDB

# Realistic sample entries spanning topics/emotions.
SAMPLES = [
    "I had a stressful job interview today and my boss criticized my project.",
    "Another rough day at work, the deadline at the office is crushing me.",
    "My manager praised my project at work, I finally feel good about my career.",
    "Spent the evening with my friend Alice, we talked about our relationship struggles.",
    "I'm worried about money this month, rent and bills are piling up.",
    "Work has been better lately, the promotion talk gives me hope.",
]


class FakeEmotion:
    _MAP = ["sadness", "fear", "joy", "neutral", "fear", "joy"]

    def __init__(self):
        self.i = 0

    def detect(self, text):
        emo = self._MAP[self.i % len(self._MAP)]
        self.i += 1
        return {"emotion": emo, "confidence": 0.85}


class FakeRetriever:
    def retrieve(self, query, query_emotion="neutral", top_k=5):
        return []


class FakeMemory:
    def __init__(self):
        self.entries = []

    def store_entry(self, text, tags=None, emotion_signal=None, topics=None):
        meta = {
            "text": text,
            "emotion": (emotion_signal or {}).get("emotion", "neutral"),
            "timestamp": "2026-06-20T00:00:00Z",
        }
        self.entries.append(meta)
        return meta

    def get_recent_memory(self, limit=3):
        return self.entries[-limit:]


class FakeVectorStore:
    @staticmethod
    def _hash_entry(text, timestamp):
        normalized = " ".join(text.strip().lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()


class FakeLLM:
    def generate(self, prompt):
        return "I hear you, and that sounds really hard. Be gentle with yourself."


@pytest.fixture
def service(tmp_path):
    from backend.llm.prompt_builder import PromptBuilder

    return RAGService(
        vector_store=FakeVectorStore(),
        emotion_detector=FakeEmotion(),
        retriever=FakeRetriever(),
        prompt_builder=PromptBuilder(),
        llm_client=FakeLLM(),
        memory_manager=FakeMemory(),
        journal_db=JournalDB(str(tmp_path / "journal.db")),
        text_processor=TextProcessor(),
    )


def test_full_pipeline_end_to_end(service):
    outputs = []
    for text in SAMPLES:
        out = service.run_pipeline(text=text, chat_history=[], top_k=3)
        outputs.append(out)

    # No exceptions, all responses non-empty.
    assert all(o["response"] for o in outputs)

    # Phase 1+2: SQLite has records with enrichment.
    records = service.journal_db.get_all()
    assert len(records) == len(SAMPLES)
    assert any(r.topics for r in records)
    assert any(r.entities_people for r in records)

    # Phase 3: PatternEngine finds at least one trigger (career appears >= 2).
    summary = service.pattern_engine.analyze(lookback_days=36500)
    assert summary.period_entry_count == len(SAMPLES)
    assert len(summary.triggers) >= 1

    # Phase 4: InsightEngine returns insights.
    insights = service.insight_engine.generate(lookback_days=36500)
    assert isinstance(insights, list) and len(insights) >= 1
    assert insights != ["Not enough journal history yet to generate insights."]

    # Original return contract preserved.
    assert {"emotion", "stored_entry", "retrieved_memories", "prompt", "response"}.issubset(
        outputs[-1].keys()
    )
