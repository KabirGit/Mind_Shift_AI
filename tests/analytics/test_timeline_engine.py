from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.timeline_engine import TimelineEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


def test_event_type_classification(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    db.insert(JournalRecord(id="pos", text="great day", timestamp=_ts(5),
                            emotion="joy", emotion_confidence=0.9,
                            topics=["career"], sentiment_compound=0.7))
    db.insert(JournalRecord(id="neg", text="awful day", timestamp=_ts(4),
                            emotion="sadness", emotion_confidence=0.9,
                            sentiment_compound=-0.6))
    db.insert(JournalRecord(id="mid", text="ok day", timestamp=_ts(3),
                            emotion="neutral", emotion_confidence=0.9,
                            sentiment_compound=0.05))
    events = {e.timestamp[:10]: e for e in TimelineEngine(db).build()}
    types = {e.event_type for e in events.values()}
    assert "positive_peak" in types
    assert "negative_peak" in types
    assert "normal" in types


def test_peaks_always_kept(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    # 3 peaks + many normals; max_events small should still keep peaks.
    db.insert(JournalRecord(id="p1", text="t", timestamp=_ts(30), emotion="joy",
                            emotion_confidence=0.9, sentiment_compound=0.8))
    db.insert(JournalRecord(id="p2", text="t", timestamp=_ts(29), emotion="sadness",
                            emotion_confidence=0.9, sentiment_compound=-0.9))
    for i in range(10):
        db.insert(JournalRecord(id=f"n{i}", text="t", timestamp=_ts(20 - i),
                                emotion="neutral", emotion_confidence=0.9,
                                sentiment_compound=0.05))
    events = TimelineEngine(db).build(max_events=3)
    peak_ids = {e.event_type for e in events}
    assert "positive_peak" in peak_ids and "negative_peak" in peak_ids


def test_zero_data(tmp_path):
    db = JournalDB(str(tmp_path / "empty.db"))
    assert TimelineEngine(db).build() == []
