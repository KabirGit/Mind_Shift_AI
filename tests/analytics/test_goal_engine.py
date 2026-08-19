from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.goal_engine import GoalEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


def test_improving_goal_progress(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    vals = [(30, -0.3), (20, 0.1), (10, 0.4), (3, 0.6)]
    for i, (d, s) in enumerate(vals):
        db.insert(JournalRecord(id=f"g{i}", text="had a job interview today",
                                timestamp=_ts(d), emotion="joy",
                                emotion_confidence=0.9, sentiment_compound=s))
    goals = GoalEngine(db).analyze(lookback_days=90)
    js = [g for g in goals if g.goal_keyword == "job_search"]
    assert js
    assert js[0].sentiment_trend == "improving"
    assert js[0].estimated_progress > 0.5


def test_single_mention_excluded(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    db.insert(JournalRecord(id="x", text="my exam is coming", timestamp=_ts(1),
                            emotion="fear", emotion_confidence=0.9,
                            sentiment_compound=-0.2))
    assert GoalEngine(db).analyze() == []


def test_zero_data(tmp_path):
    db = JournalDB(str(tmp_path / "empty.db"))
    assert GoalEngine(db).analyze() == []
