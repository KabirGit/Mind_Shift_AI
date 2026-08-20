from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.temporal_engine import TemporalEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _last_weekday(weekday: int, weeks_back: int) -> str:
    """A timestamp on a specific weekday, weeks_back weeks ago."""
    today = datetime.now(UTC)
    # step back to the desired weekday
    delta_days = (today.weekday() - weekday) % 7 + weeks_back * 7
    dt = today - timedelta(days=delta_days)
    return dt.isoformat().replace("+00:00", "Z")


def test_peak_day_detected(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    # 5 career entries; Mondays strongly negative, other days neutral/positive.
    rows = [
        (_last_weekday(0, 1), -0.8),  # Monday
        (_last_weekday(0, 2), -0.7),  # Monday
        (_last_weekday(2, 1), 0.2),   # Wednesday
        (_last_weekday(3, 1), 0.3),   # Thursday
        (_last_weekday(4, 1), 0.1),   # Friday
    ]
    for i, (ts, s) in enumerate(rows):
        db.insert(JournalRecord(id=f"c{i}", text="t", timestamp=ts, emotion="fear",
                                emotion_confidence=0.9, topics=["career"],
                                sentiment_compound=s))
    out = TemporalEngine(db).analyze(lookback_days=60)
    career = [p for p in out if p.topic == "career"]
    assert career
    p = career[0]
    assert p.peak_day_of_week == "Monday"
    assert p.delta < 0  # Monday below baseline
    assert p.day_time_crossing is not None
    assert p.day_time_sample_size >= 2
    assert p.confidence > 0


def test_below_threshold_topic_skipped(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    # only 3 records -> < 5, skipped
    for i in range(3):
        db.insert(JournalRecord(id=f"h{i}", text="t", timestamp=_last_weekday(0, i),
                                emotion="joy", emotion_confidence=0.9,
                                topics=["health"], sentiment_compound=0.1))
    assert TemporalEngine(db).analyze(lookback_days=60) == []


def test_zero_data(tmp_path):
    db = JournalDB(str(tmp_path / "empty.db"))
    assert TemporalEngine(db).analyze() == []
