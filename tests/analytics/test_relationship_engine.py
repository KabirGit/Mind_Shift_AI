from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.relationship_engine import RelationshipEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


def _seed(db: JournalDB) -> None:
    recs = [
        # Alice: 3 mentions, sentiment improving, joy dominant
        JournalRecord(id="a1", text="t", timestamp=_ts(20), emotion="sadness",
                      emotion_confidence=0.9, entities_people=["Alice"],
                      sentiment_compound=-0.4),
        JournalRecord(id="a2", text="t", timestamp=_ts(10), emotion="joy",
                      emotion_confidence=0.9, entities_people=["Alice"],
                      sentiment_compound=0.3),
        JournalRecord(id="a3", text="t", timestamp=_ts(5), emotion="joy",
                      emotion_confidence=0.9, entities_people=["Alice", "Bob"],
                      sentiment_compound=0.5),
        # Bob: 2 mentions
        JournalRecord(id="b1", text="t", timestamp=_ts(8), emotion="fear",
                      emotion_confidence=0.9, entities_people=["Bob"],
                      sentiment_compound=-0.1),
        # Carol: 1 mention -> excluded
        JournalRecord(id="c1", text="t", timestamp=_ts(3), emotion="neutral",
                      emotion_confidence=0.9, entities_people=["Carol"],
                      sentiment_compound=0.0),
    ]
    for r in recs:
        db.insert(r)


def test_relationship_profiles_exact(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    _seed(db)
    profiles = {p.person: p for p in RelationshipEngine(db).analyze(lookback_days=30)}

    assert "Carol" not in profiles  # only 1 mention
    assert set(profiles.keys()) == {"Alice", "Bob"}

    alice = profiles["Alice"]
    assert alice.mention_count == 3
    assert alice.dominant_emotion == "joy"
    # avg (-0.4 + 0.3 + 0.5)/3 = 0.1333
    assert abs(alice.avg_sentiment - 0.1333) < 1e-3
    assert alice.trend == "improving"
    all_alice_ts = [
        r.timestamp for r in db.get_all() if "Alice" in (r.entities_people or [])
    ]
    assert alice.last_mentioned == max(all_alice_ts)

    bob = profiles["Bob"]
    assert bob.mention_count == 2


def test_sorted_by_mention_count(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    _seed(db)
    profiles = RelationshipEngine(db).analyze(lookback_days=30)
    counts = [p.mention_count for p in profiles]
    assert counts == sorted(counts, reverse=True)


def test_zero_data(tmp_path):
    db = JournalDB(str(tmp_path / "empty.db"))
    assert RelationshipEngine(db).analyze(lookback_days=30) == []
