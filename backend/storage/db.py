from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any

from backend.storage.models import JournalRecord

logger = logging.getLogger(__name__)

_LIST_FIELDS = (
    "entities_people",
    "entities_places",
    "entities_orgs",
    "keywords",
    "topics",
    "habits",
)

# New list-type columns added after the initial schema; migrated idempotently.
_MIGRATION_LIST_COLUMNS = ("habits",)


class JournalDB:
    """SQLite-backed structured store for journal records.

    All DB calls are wrapped in try/except: a storage failure must never crash
    the main chat pipeline.
    """

    def __init__(self, db_path: str = "data/journal.db") -> None:
        self.db_path = db_path
        try:
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            self._create_table()
            self._migrate()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("JournalDB init failed: %s", exc)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS journal_records (
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
            conn.commit()

    def _migrate(self) -> None:
        """Idempotently add new list-type columns to an existing table.

        Safe on a fresh DB and on an already-migrated DB.
        """
        try:
            with self._connect() as conn:
                existing = {
                    row["name"]
                    for row in conn.execute(
                        "PRAGMA table_info(journal_records)"
                    ).fetchall()
                }
                for col in _MIGRATION_LIST_COLUMNS:
                    if col not in existing:
                        conn.execute(
                            f"ALTER TABLE journal_records ADD COLUMN {col} TEXT DEFAULT '[]'"
                        )
                conn.commit()
        except Exception as exc:
            logger.exception("JournalDB migration failed: %s", exc)

    def insert(self, record: JournalRecord) -> None:
        """Upsert by id so re-processing the same entry hash updates in place."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO journal_records (
                        id, text, timestamp, emotion, emotion_confidence,
                        entities_people, entities_places, entities_orgs,
                        keywords, topics, habits, sentiment_compound, sentiment_valence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        text=excluded.text,
                        timestamp=excluded.timestamp,
                        emotion=excluded.emotion,
                        emotion_confidence=excluded.emotion_confidence,
                        entities_people=excluded.entities_people,
                        entities_places=excluded.entities_places,
                        entities_orgs=excluded.entities_orgs,
                        keywords=excluded.keywords,
                        topics=excluded.topics,
                        habits=excluded.habits,
                        sentiment_compound=excluded.sentiment_compound,
                        sentiment_valence=excluded.sentiment_valence
                    """,
                    (
                        record.id,
                        record.text,
                        record.timestamp,
                        record.emotion,
                        record.emotion_confidence,
                        json.dumps(record.entities_people),
                        json.dumps(record.entities_places),
                        json.dumps(record.entities_orgs),
                        json.dumps(record.keywords),
                        json.dumps(record.topics),
                        json.dumps(record.habits),
                        record.sentiment_compound,
                        record.sentiment_valence,
                    ),
                )
                conn.commit()
        except Exception as exc:
            logger.exception("JournalDB.insert failed: %s", exc)

    def _row_to_record(self, row: sqlite3.Row) -> JournalRecord:
        data: dict[str, Any] = dict(row)
        for field in _LIST_FIELDS:
            raw = data.get(field, "[]")
            try:
                data[field] = json.loads(raw) if isinstance(raw, str) else list(raw or [])
            except Exception:
                data[field] = []
        return JournalRecord(**data)

    def get_recent(self, limit: int = 50) -> list[JournalRecord]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM journal_records ORDER BY timestamp DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [self._row_to_record(r) for r in rows]
        except Exception as exc:
            logger.exception("JournalDB.get_recent failed: %s", exc)
            return []

    def get_all(self) -> list[JournalRecord]:
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT * FROM journal_records ORDER BY timestamp DESC"
                ).fetchall()
            return [self._row_to_record(r) for r in rows]
        except Exception as exc:
            logger.exception("JournalDB.get_all failed: %s", exc)
            return []
