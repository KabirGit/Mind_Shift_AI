from __future__ import annotations

import sqlite3

from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _make_legacy_db(path: str) -> None:
    """Create a pre-migration table without the habits column."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE journal_records (
            id TEXT PRIMARY KEY,
            text TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            emotion TEXT NOT NULL,
            emotion_confidence REAL NOT NULL,
            entities_people TEXT NOT NULL DEFAULT '[]',
            entities_places TEXT NOT NULL DEFAULT '[]',
            entities_orgs TEXT NOT NULL DEFAULT '[]',
            keywords TEXT NOT NULL DEFAULT '[]',
            topics TEXT NOT NULL DEFAULT '[]',
            sentiment_compound REAL NOT NULL DEFAULT 0.0,
            sentiment_valence REAL NOT NULL DEFAULT 0.0
        )
        """
    )
    conn.execute(
        "INSERT INTO journal_records (id, text, timestamp, emotion, emotion_confidence) "
        "VALUES ('old', 'legacy', '2026-06-20T00:00:00Z', 'joy', 0.9)"
    )
    conn.commit()
    conn.close()


def test_migration_adds_habits_column(tmp_path):
    path = str(tmp_path / "legacy.db")
    _make_legacy_db(path)

    db = JournalDB(path)  # triggers migration
    cols = {row[1] for row in sqlite3.connect(path).execute("PRAGMA table_info(journal_records)")}
    assert "habits" in cols
    assert "person_relationship_types" in cols

    # Legacy row still readable, additive fields default safely.
    rows = db.get_all()
    assert len(rows) == 1
    assert rows[0].habits == []
    assert rows[0].person_relationship_types == {}

    # New insert with habits round-trips.
    db.insert(JournalRecord(id="new", text="t", timestamp="2026-06-21T00:00:00Z",
                            emotion="joy", emotion_confidence=0.9, habits=["exercise"],
                            person_relationship_types={"Alice": "friend"}))
    got = {r.id: r for r in db.get_all()}
    assert got["new"].habits == ["exercise"]
    assert got["new"].person_relationship_types == {"Alice": "friend"}


def test_migration_idempotent_on_fresh_and_repeated(tmp_path):
    path = str(tmp_path / "fresh.db")
    JournalDB(path)
    JournalDB(path)  # second init must not raise
    db = JournalDB(path)
    db.insert(JournalRecord(id="a", text="t", timestamp="2026-06-21T00:00:00Z",
                            emotion="joy", emotion_confidence=0.9, habits=["sleep"]))
    assert db.get_all()[0].habits == ["sleep"]
