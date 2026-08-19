from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.habit_engine import HabitEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


def _seed(db: JournalDB) -> None:
    # exercise: mentioned in 3 records with high sentiment; others low.
    recs = [
        JournalRecord(id="e1", text="t", timestamp=_ts(10), emotion="joy",
                      emotion_confidence=0.9, habits=["exercise"], sentiment_compound=0.6),
        JournalRecord(id="e2", text="t", timestamp=_ts(9), emotion="joy",
                      emotion_confidence=0.9, habits=["exercise"], sentiment_compound=0.8),
        JournalRecord(id="e3", text="t", timestamp=_ts(8), emotion="joy",
                      emotion_confidence=0.9, habits=["exercise"], sentiment_compound=0.4),
        JournalRecord(id="o1", text="t", timestamp=_ts(7), emotion="sadness",
                      emotion_confidence=0.9, habits=[], sentiment_compound=-0.2),
        JournalRecord(id="o2", text="t", timestamp=_ts(6), emotion="sadness",
                      emotion_confidence=0.9, habits=[], sentiment_compound=-0.4),
        # sleep mentioned only once -> excluded
        JournalRecord(id="s1", text="t", timestamp=_ts(5), emotion="neutral",
                      emotion_confidence=0.9, habits=["sleep"], sentiment_compound=0.0),
    ]
    for r in recs:
        db.insert(r)


def test_habit_correlation_exact(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    _seed(db)
    out = HabitEngine(db).analyze(lookback_days=30)

    habits = {c.habit: c for c in out}
    # sleep excluded (only 1 mention)
    assert "sleep" not in habits
    assert "exercise" in habits

    ex = habits["exercise"]
    assert ex.mention_count == 3
    # avg when mentioned = (0.6+0.8+0.4)/3 = 0.6
    assert abs(ex.avg_sentiment_when_mentioned - 0.6) < 1e-6
    # avg others = (-0.2 + -0.4 + 0.0)/3 = -0.2
    assert abs(ex.avg_sentiment_other_days - (-0.2)) < 1e-6
    assert abs(ex.delta - 0.8) < 1e-6
    assert ex.correlation_label == "positive"


def test_fewer_than_two_mentions_excluded(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    db.insert(JournalRecord(id="x", text="t", timestamp=_ts(1), emotion="joy",
                            emotion_confidence=0.9, habits=["coffee"], sentiment_compound=0.5))
    assert HabitEngine(db).analyze(lookback_days=30) == []


def test_zero_data(tmp_path):
    db = JournalDB(str(tmp_path / "empty.db"))
    assert HabitEngine(db).analyze(lookback_days=30) == []
