from __future__ import annotations

from backend.analytics.growth_tracker import GrowthTracker
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def test_snapshots_and_deltas(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    # 3 months, improving avg sentiment.
    months = {
        "2026-01": (-0.4, "sadness"),
        "2026-02": (0.0, "neutral"),
        "2026-03": (0.5, "joy"),
    }
    i = 0
    for label, (sent, emo) in months.items():
        for _ in range(2):
            db.insert(JournalRecord(id=f"r{i}", text="t",
                                    timestamp=f"{label}-15T12:00:00Z", emotion=emo,
                                    emotion_confidence=0.9, topics=["career"],
                                    sentiment_compound=sent))
            i += 1
    gt = GrowthTracker(db)
    snaps = gt.compute_snapshots()
    assert len(snaps) == 3
    assert snaps[0].period_label == "2026-01"
    assert snaps[-1].dominant_emotion == "joy"

    deltas = gt.compute_growth_deltas()
    assert len(deltas) == 2
    assert deltas[-1]["sentiment_delta"] > 0.1
    assert deltas[-1]["emotion_changed"] is True

    assert "improving" in gt.narrative()
    assert "6 entries total" in gt.narrative()


def test_single_month_narrative(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    db.insert(JournalRecord(id="a", text="t", timestamp="2026-03-01T12:00:00Z",
                            emotion="joy", emotion_confidence=0.9,
                            sentiment_compound=0.3))
    assert "starts here" in GrowthTracker(db).narrative()


def test_zero_data(tmp_path):
    db = JournalDB(str(tmp_path / "empty.db"))
    assert GrowthTracker(db).compute_snapshots() == []
    assert "starts here" in GrowthTracker(db).narrative()
