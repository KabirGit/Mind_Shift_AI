from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.pattern_engine import PatternEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    dt = datetime.now(UTC) - timedelta(days=days_ago)
    return dt.isoformat().replace("+00:00", "Z")


def _seed(db: JournalDB) -> None:
    records = [
        # career: 4 records, sentiment trends up, joy dominant
        JournalRecord(id="c1", text="t", timestamp=_ts(20), emotion="sadness",
                      emotion_confidence=0.9, topics=["career"],
                      entities_people=["Alice"], sentiment_compound=-0.5),
        JournalRecord(id="c2", text="t", timestamp=_ts(18), emotion="fear",
                      emotion_confidence=0.9, topics=["career"],
                      sentiment_compound=-0.3),
        JournalRecord(id="c3", text="t", timestamp=_ts(8), emotion="joy",
                      emotion_confidence=0.9, topics=["career"],
                      entities_people=["Bob"], sentiment_compound=0.4),
        JournalRecord(id="c4", text="t", timestamp=_ts(6), emotion="joy",
                      emotion_confidence=0.9, topics=["career"],
                      sentiment_compound=0.6),
        # health: 2 records, stable
        JournalRecord(id="h1", text="t", timestamp=_ts(15), emotion="joy",
                      emotion_confidence=0.9, topics=["health"],
                      sentiment_compound=0.2),
        JournalRecord(id="h2", text="t", timestamp=_ts(12), emotion="joy",
                      emotion_confidence=0.9, topics=["health"],
                      sentiment_compound=0.2),
        # relationship: 1 record, below trigger threshold
        JournalRecord(id="r1", text="t", timestamp=_ts(11), emotion="neutral",
                      emotion_confidence=0.9, topics=["relationship"],
                      entities_people=["Alice"], sentiment_compound=0.1),
    ]
    for r in records:
        db.insert(r)


def test_analyze_deterministic(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    _seed(db)
    summary = PatternEngine(db).analyze(lookback_days=30)

    assert summary.period_entry_count == 7
    assert summary.recurring_emotions == {"sadness": 1, "fear": 1, "joy": 4, "neutral": 1}
    assert summary.recurring_topics == {"career": 4, "health": 2, "relationship": 1}
    assert summary.recurring_people == {"Alice": 2, "Bob": 1}
    assert summary.topic_trends["career"] == "increasing"
    assert summary.emotion_trends["joy"] == "increasing"

    triggers = {t.topic: t for t in summary.triggers}
    # relationship excluded (freq 1 < 2)
    assert set(triggers.keys()) == {"career", "health"}

    career = triggers["career"]
    assert career.frequency == 4
    assert career.dominant_emotion == "joy"
    assert career.trend == "increasing"
    assert abs(career.avg_sentiment - 0.05) < 1e-6

    health = triggers["health"]
    assert health.frequency == 2
    assert health.trend == "stable"


def test_analyze_zero_records(tmp_path):
    db = JournalDB(str(tmp_path / "empty.db"))
    summary = PatternEngine(db).analyze(lookback_days=30)
    assert summary.period_entry_count == 0
    assert summary.recurring_emotions == {}
    assert summary.recurring_topics == {}
    assert summary.recurring_people == {}
    assert summary.triggers == []


def test_lookback_excludes_old(tmp_path):
    db = JournalDB(str(tmp_path / "old.db"))
    db.insert(JournalRecord(id="old", text="t", timestamp=_ts(400),
                            emotion="joy", emotion_confidence=0.9, topics=["career"]))
    summary = PatternEngine(db).analyze(lookback_days=30)
    assert summary.period_entry_count == 0
