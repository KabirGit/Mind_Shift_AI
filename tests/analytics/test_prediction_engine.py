from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.prediction_engine import PredictionEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


def test_declining_trend(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    # Sentiment declines over time: older positive, recent negative.
    vals = [(12, 0.6), (10, 0.4), (8, 0.1), (6, -0.2), (4, -0.4), (2, -0.6)]
    for i, (d, s) in enumerate(vals):
        db.insert(JournalRecord(id=f"r{i}", text="t", timestamp=_ts(d),
                                emotion="sadness", emotion_confidence=0.9,
                                sentiment_compound=s))
    fc = PredictionEngine(db).forecast_sentiment(days_back=14, horizon=7)
    assert fc.direction == "declining"
    assert fc.confidence > 0
    assert "not tracked yet" in fc.forecast_accuracy_note


def test_burnout_medium_or_high_for_negative(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    for i in range(8):
        db.insert(JournalRecord(id=f"n{i}", text="t", timestamp=_ts(i + 1),
                                emotion="sadness", emotion_confidence=0.9,
                                topics=["career"], sentiment_compound=-0.5))
    risk = PredictionEngine(db).assess_burnout_risk()
    assert risk.risk_level in {"medium", "high"}
    assert "not a clinical assessment" in risk.explanation


def test_insufficient_data_defaults(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    db.insert(JournalRecord(id="x", text="t", timestamp=_ts(1), emotion="joy",
                            emotion_confidence=0.9, sentiment_compound=0.2))
    fc = PredictionEngine(db).forecast_sentiment()
    assert fc.confidence == 0.0 and fc.direction == "stable"
    risk = PredictionEngine(db).assess_burnout_risk()
    assert risk.confidence == 0.0 and risk.risk_level == "low"
    assert "not a clinical assessment" in risk.explanation


def test_disclaimer_always_present(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    for i in range(5):
        db.insert(JournalRecord(id=f"p{i}", text="t", timestamp=_ts(i + 1),
                                emotion="joy", emotion_confidence=0.9,
                                sentiment_compound=0.5))
    risk = PredictionEngine(db).assess_burnout_risk()
    assert "This is a statistical pattern only, not a clinical assessment." in risk.explanation
