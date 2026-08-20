from __future__ import annotations

from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _rec(rid: str, ts: str, **kw) -> JournalRecord:
    base = dict(
        id=rid,
        text=f"entry {rid}",
        timestamp=ts,
        emotion="joy",
        emotion_confidence=0.9,
    )
    base.update(kw)
    return JournalRecord(**base)


def test_insert_retrieve_roundtrip(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    rec = _rec(
        "h1",
        "2026-01-01T10:00:00Z",
        entities_people=["Alice"],
        entities_places=["Paris"],
        entities_orgs=["Acme"],
        keywords=["work", "trip"],
        topics=["career"],
        person_relationship_types={"Alice": "friend"},
        sentiment_compound=0.5,
        sentiment_valence=0.5,
    )
    db.insert(rec)
    out = db.get_all()
    assert len(out) == 1
    got = out[0]
    assert got.id == "h1"
    assert got.entities_people == ["Alice"]
    assert got.entities_places == ["Paris"]
    assert got.entities_orgs == ["Acme"]
    assert got.keywords == ["work", "trip"]
    assert got.topics == ["career"]
    assert got.person_relationship_types == {"Alice": "friend"}
    assert got.sentiment_compound == 0.5


def test_upsert_no_duplicate(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    db.insert(_rec("dup", "2026-01-01T10:00:00Z", emotion="joy"))
    db.insert(_rec("dup", "2026-01-02T10:00:00Z", emotion="sadness"))
    out = db.get_all()
    assert len(out) == 1
    assert out[0].emotion == "sadness"


def test_get_recent_limit_and_order(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    for i in range(5):
        db.insert(_rec(f"h{i}", f"2026-01-0{i + 1}T10:00:00Z"))
    recent = db.get_recent(limit=2)
    assert len(recent) == 2
    # Newest first.
    assert recent[0].timestamp == "2026-01-05T10:00:00Z"
    assert recent[1].timestamp == "2026-01-04T10:00:00Z"


def test_db_failure_is_swallowed(tmp_path):
    # Parent is a regular file, so the db path is unusable -> errors must be
    # caught internally and never raise out of JournalDB methods.
    blocker = tmp_path / "blocker"
    blocker.write_text("x")
    bad = JournalDB(str(blocker / "sub" / "j.db"))
    bad.insert(_rec("x", "2026-01-01T10:00:00Z"))
    assert bad.get_all() == []
    assert bad.get_recent() == []
