from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.habit_engine import HabitEngine
from backend.analytics.models import compute_confidence
from backend.analytics.pattern_engine import PatternEngine
from backend.analytics.relationship_engine import RelationshipEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


def test_compute_confidence_math():
    assert compute_confidence(5) == 0.5
    assert compute_confidence(10) == 1.0
    assert compute_confidence(20) == 1.0
    assert compute_confidence(0) == 0.0


def test_trigger_has_confidence_and_explanation(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    for i in range(5):
        db.insert(JournalRecord(id=f"c{i}", text="t", timestamp=_ts(i + 1),
                                emotion="joy", emotion_confidence=0.9,
                                topics=["career"], sentiment_compound=0.2))
    summary = PatternEngine(db).analyze(lookback_days=30)
    trig = summary.triggers[0]
    assert trig.confidence == 0.5  # 5 / 10
    assert trig.explanation != ""


def test_habit_and_relationship_fields(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    for i in range(3):
        db.insert(JournalRecord(id=f"h{i}", text="t", timestamp=_ts(i + 1),
                                emotion="joy", emotion_confidence=0.9,
                                habits=["exercise"], entities_people=["Sam"],
                                sentiment_compound=0.5))
    for i in range(2):
        db.insert(JournalRecord(id=f"o{i}", text="t", timestamp=_ts(i + 10),
                                emotion="sadness", emotion_confidence=0.9,
                                sentiment_compound=-0.3))
    habits = HabitEngine(db).analyze(lookback_days=30)
    assert habits and habits[0].confidence == 0.3 and habits[0].explanation
    people = RelationshipEngine(db).analyze(lookback_days=30)
    assert people and people[0].confidence == 0.3 and people[0].explanation
