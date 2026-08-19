from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import BaseModel

from backend.analytics._stats_utils import parse_ts, sort_key
from backend.retrieval.vector_store import FaissVectorStore
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)


class MemoryReplay(BaseModel):
    similar_entry_text: str
    similar_entry_timestamp: str
    similar_entry_emotion: str
    days_ago: int
    what_happened_next: str | None = None
    next_entry_emotion: str | None = None
    next_entry_sentiment: float | None = None
    recovery_hint: str
    confidence: float
    explanation: str


class ReplayEngine:
    """Surfaces a semantically similar past entry and what followed it."""

    def __init__(self, vector_store: FaissVectorStore, db: JournalDB) -> None:
        self.vector_store = vector_store
        self.db = db

    def find_replay(
        self, current_text: str, current_emotion: str
    ) -> MemoryReplay | None:
        try:
            index = getattr(self.vector_store, "index", None)
            if index is None or getattr(index, "ntotal", 0) < 5:
                return None

            results = self.vector_store.query(current_text, top_k=5)
            if not results:
                return None

            now = datetime.now(UTC)
            candidate = None
            for res in results:
                meta = res.get("metadata", {})
                ts = meta.get("timestamp")
                dt = parse_ts(ts) if ts else None
                if dt is None:
                    continue
                if (now - dt).total_seconds() < 48 * 3600:
                    continue  # skip last 48h
                candidate = (res, meta, dt)
                break

            if candidate is None:
                return None

            res, meta, sim_dt = candidate
            days_ago = (now - sim_dt).days

            nxt = self._next_entry(sim_dt)
            next_text = nxt.text if nxt else None
            next_emotion = nxt.emotion if nxt else None
            next_sentiment = nxt.sentiment_compound if nxt else None

            if next_sentiment is None:
                hint = f"This echoes a past entry from {days_ago} days ago."
            elif next_sentiment > 0:
                hint = "Last time you felt like this, things improved within a few days."
            else:
                hint = (
                    "Last time you felt like this, it was a tough stretch — "
                    "you got through it."
                )

            score = res.get("scores", {}).get("combined")
            if score is None:
                # query() returns distance; map to a rough 0..1 confidence.
                dist = res.get("distance", 1.0)
                score = max(0.0, min(1.0, 1.0 / (1.0 + float(dist))))
            confidence = round(min(1.0, float(score)), 4)

            return MemoryReplay(
                similar_entry_text=str(meta.get("text", ""))[:200],
                similar_entry_timestamp=meta.get("timestamp", ""),
                similar_entry_emotion=str(meta.get("emotion", "neutral")),
                days_ago=days_ago,
                what_happened_next=(next_text[:200] if next_text else None),
                next_entry_emotion=next_emotion,
                next_entry_sentiment=next_sentiment,
                recovery_hint=hint,
                confidence=confidence,
                explanation=(
                    f"Found a similar entry from {days_ago} days ago "
                    f"(emotion: {meta.get('emotion', 'neutral')})."
                ),
            )
        except Exception as exc:
            logger.exception("ReplayEngine.find_replay failed: %s", exc)
            return None

    def _next_entry(self, sim_dt):
        records = sorted(self.db.get_all(), key=lambda r: sort_key(r.timestamp))
        for r in records:
            dt = parse_ts(r.timestamp)
            if dt is None:
                continue
            delta_days = (dt - sim_dt).days
            if 1 <= delta_days <= 3:
                return r
        return None
