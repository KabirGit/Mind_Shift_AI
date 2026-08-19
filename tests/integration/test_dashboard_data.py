from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.insight_engine import InsightEngine
from backend.analytics.models import PatternSummary
from backend.analytics.pattern_engine import PatternEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


def _seed(db: JournalDB) -> None:
    for i in range(4):
        db.insert(
            JournalRecord(
                id=f"c{i}",
                text="work stuff",
                timestamp=_ts(10 - i),
                emotion="joy" if i % 2 == 0 else "sadness",
                emotion_confidence=0.9,
                topics=["career"],
                entities_people=["Alice"],
                sentiment_compound=0.1 * i,
            )
        )


def test_dashboard_data_layer_well_formed(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    _seed(db)

    # Same calls the dashboard tab makes.
    records = db.get_all()
    assert len(records) == 4

    engine = PatternEngine(db)
    summary = engine.analyze(lookback_days=30)
    assert isinstance(summary, PatternSummary)
    assert summary.period_entry_count == 4
    assert summary.recurring_topics.get("career") == 4

    insights = InsightEngine(engine).generate(lookback_days=30)
    assert isinstance(insights, list)
    assert all(isinstance(s, str) for s in insights)
    assert len(insights) >= 1


def test_dashboard_empty_db_is_safe(tmp_path):
    db = JournalDB(str(tmp_path / "empty.db"))
    assert db.get_all() == []
    summary = PatternEngine(db).analyze(lookback_days=30)
    assert summary.period_entry_count == 0
    insights = InsightEngine(PatternEngine(db)).generate(lookback_days=30)
    assert insights == ["Not enough journal history yet to generate insights."]
