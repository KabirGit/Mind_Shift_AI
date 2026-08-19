from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.habit_engine import HabitEngine
from backend.analytics.pattern_engine import PatternEngine
from backend.analytics.relationship_engine import RelationshipEngine
from backend.profile.profile_manager import ProfileManager
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


def _mgr(db: JournalDB) -> ProfileManager:
    return ProfileManager(
        db, PatternEngine(db), HabitEngine(db), RelationshipEngine(db)
    )


def test_default_when_empty(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    mgr = _mgr(db)
    assert mgr.load().entry_count == 0
    profile = mgr.update()
    assert profile.entry_count == 0


def test_profile_fields_compute(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    # Older entries negative, recent entries positive -> growth > 0.5.
    db.insert(JournalRecord(id="o1", text="x" * 50, timestamp=_ts(40), emotion="sadness",
                            emotion_confidence=0.9, topics=["career"],
                            entities_people=["Sarah"], sentiment_compound=-0.6))
    db.insert(JournalRecord(id="o2", text="y" * 50, timestamp=_ts(35), emotion="sadness",
                            emotion_confidence=0.9, topics=["career"],
                            entities_people=["Sarah"], sentiment_compound=-0.4))
    db.insert(JournalRecord(id="n1", text="z" * 50, timestamp=_ts(2), emotion="joy",
                            emotion_confidence=0.9, topics=["career"],
                            entities_people=["Sarah"], habits=["exercise"],
                            sentiment_compound=0.6))
    profile = _mgr(db).update()

    assert profile.entry_count == 3
    # current (last 7d) positive, baseline mixed -> growth above neutral 0.5.
    assert profile.current_sentiment > profile.baseline_sentiment
    assert profile.growth_score > 0.5
    assert profile.communication_style == "brief"  # 50-char entries
    assert "career" in profile.top_triggers
    assert "Sarah" in profile.top_people


def test_communication_style_thresholds(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    db.insert(JournalRecord(id="d1", text="w" * 400, timestamp=_ts(1), emotion="joy",
                            emotion_confidence=0.9, sentiment_compound=0.2))
    assert _mgr(db).update().communication_style == "detailed"


def test_persist_and_reload(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    db.insert(JournalRecord(id="a", text="hello there friend", timestamp=_ts(1),
                            emotion="joy", emotion_confidence=0.9, sentiment_compound=0.3))
    mgr = _mgr(db)
    mgr.update()
    # New manager instance reads the persisted blob.
    reloaded = _mgr(db).load()
    assert reloaded.entry_count == 1
