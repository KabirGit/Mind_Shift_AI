from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.alert_engine import AlertEngine
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


class _FakeDB:
    def __init__(self, recent):
        self._recent = recent

    def get_recent(self, limit=50):
        return self._recent[:limit]

    def get_all(self):
        return self._recent


def _rec(i, emotion, sentiment, topics=None, habits=None, days_ago=0):
    return JournalRecord(
        id=str(i), text="t", timestamp=_ts(days_ago), emotion=emotion,
        emotion_confidence=0.9, topics=topics or [], habits=habits or [],
        sentiment_compound=sentiment,
    )


def test_consecutive_stress_alert():
    recent = [
        _rec(1, "sadness", -0.5, days_ago=1),
        _rec(2, "fear", -0.4, days_ago=2),
        _rec(3, "sadness", -0.3, days_ago=3),
    ]
    alerts = AlertEngine(_FakeDB(recent), None).check()
    assert any(a.severity == "watch" and "consecutive" in a.message for a in alerts)


def test_positive_streak_alert():
    recent = [_rec(i, "joy", 0.5, days_ago=i) for i in range(4)]
    alerts = AlertEngine(_FakeDB(recent), None).check()
    assert any(a.severity == "info" and "positive entries in a row" in a.message for a in alerts)


def test_neutral_no_alerts():
    recent = [
        _rec(1, "neutral", 0.05, days_ago=1),
        _rec(2, "joy", 0.15, days_ago=2),
        _rec(3, "neutral", -0.05, days_ago=3),
    ]
    assert AlertEngine(_FakeDB(recent), None).check() == []
