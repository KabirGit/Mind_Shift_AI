from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.causal_engine import CausalEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


def test_exercise_lifts_mood(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    # 4 exercise days, all with positive same-day sentiment.
    for i in range(4):
        db.insert(JournalRecord(id=f"e{i}", text="gym", timestamp=_ts(20 - i * 2),
                                emotion="joy", emotion_confidence=0.9,
                                habits=["exercise"], sentiment_compound=0.6))
    # Several non-exercise days, mostly negative -> low base rate.
    for i in range(5):
        db.insert(JournalRecord(id=f"o{i}", text="meh", timestamp=_ts(5 - i if i < 5 else 0),
                                emotion="sadness", emotion_confidence=0.9,
                                habits=[], sentiment_compound=-0.3))
    links = CausalEngine(db).analyze(lookback_days=60)
    ex = [link for link in links if link.cause == "exercise" and link.effect == "positive_mood"]
    assert ex
    assert ex[0].probability > ex[0].base_rate
    assert ex[0].lift > 0
    assert ex[0].lag_lifts
    assert 1 <= ex[0].strongest_lag_days <= 3
    assert ex[0].sample_size == 4
    assert "%" in ex[0].explanation


def test_zero_data(tmp_path):
    db = JournalDB(str(tmp_path / "empty.db"))
    assert CausalEngine(db).analyze() == []
