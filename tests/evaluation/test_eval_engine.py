from __future__ import annotations

import json

from backend.evaluation.eval_engine import EvalEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


class _FakeStore:
    """Returns results whose topics always overlap with the query record."""

    def query(self, text, top_k=3):
        return [
            {"metadata": {"topics": ["career"]}},
            {"metadata": {"topics": ["health"]}},  # no overlap
            {"metadata": {"topics": ["career"]}},
        ][:top_k]


def _db(tmp_path, confs):
    db = JournalDB(str(tmp_path / "j.db"))
    for i, c in enumerate(confs):
        db.insert(JournalRecord(id=str(i), text="career stuff",
                                timestamp="2026-06-20T12:00:00Z", emotion="joy",
                                emotion_confidence=c, topics=["career"],
                                sentiment_compound=0.1))
    return db


def test_precision_at_k_math(tmp_path):
    db = _db(tmp_path, [0.9])  # 1 record, all topics=career
    eng = EvalEngine(_FakeStore(), db, str(tmp_path / "lat.jsonl"))
    res = eng.retrieval_precision_at_k(k=3, n_samples=10)
    # 1 sample, k=3: 2 of 3 results overlap topic -> 2/3.
    assert res["n_samples_used"] == 1
    assert abs(res["precision_at_k"] - (2 / 3)) < 1e-3
    assert "topic overlap" in res["note"]


def test_confidence_stats(tmp_path):
    db = _db(tmp_path, [0.9, 0.3, 0.4, 0.8])
    eng = EvalEngine(_FakeStore(), db, str(tmp_path / "lat.jsonl"))
    stats = eng.emotion_confidence_stats()
    assert stats["low_confidence_ratio"] == 0.5  # 2 of 4 < 0.5
    assert stats["min"] == 0.3
    assert stats["max"] == 0.9


def test_latency_summary(tmp_path):
    log = tmp_path / "lat.jsonl"
    values = [10, 20, 30, 40, 100]
    log.write_text("\n".join(json.dumps({"elapsed_ms": v}) for v in values))
    eng = EvalEngine(_FakeStore(), _db(tmp_path, [0.9]), str(log))
    summary = eng.latency_summary()
    assert summary["sample_count"] == 5
    assert abs(summary["avg_ms"] - 40.0) < 1e-6
    assert summary["p95_ms"] == 100


def test_latency_missing_file(tmp_path):
    eng = EvalEngine(_FakeStore(), _db(tmp_path, [0.9]), str(tmp_path / "none.jsonl"))
    assert eng.latency_summary() == {"avg_ms": 0.0, "p95_ms": 0.0, "sample_count": 0}
