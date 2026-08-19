from __future__ import annotations

import logging

from pydantic import BaseModel

from backend.analytics._stats_utils import filter_window, sort_key
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)


class TimelineEvent(BaseModel):
    timestamp: str
    title: str
    description: str
    emotion: str
    sentiment: float
    significance_score: float
    event_type: str  # "positive_peak" | "negative_peak" | "normal"


class TimelineEngine:
    """Builds a significance-ranked emotional timeline. Deterministic, no LLM."""

    def __init__(self, db: JournalDB) -> None:
        self.db = db

    def build(self, lookback_days: int = 90, max_events: int = 20) -> list[TimelineEvent]:
        try:
            records = filter_window(self.db.get_all(), lookback_days)
            if not records:
                return []

            events: list[TimelineEvent] = []
            for r in records:
                s = r.sentiment_compound
                if s > 0.4:
                    etype = "positive_peak"
                elif s < -0.4:
                    etype = "negative_peak"
                else:
                    etype = "normal"
                topic = (r.topics or [None])[0]
                title = f"{topic if topic else r.emotion} — feeling {r.emotion}"
                events.append(
                    TimelineEvent(
                        timestamp=r.timestamp,
                        title=title,
                        description=(r.text or "")[:100],
                        emotion=r.emotion,
                        sentiment=round(s, 4),
                        significance_score=round(abs(s), 4),
                        event_type=etype,
                    )
                )

            # Keep all peaks + top-significance normals, dedupe by timestamp.
            peaks = [e for e in events if e.event_type != "normal"]
            normals = sorted(
                (e for e in events if e.event_type == "normal"),
                key=lambda e: e.significance_score,
                reverse=True,
            )
            selected = peaks + normals[: max(0, max_events - len(peaks))]

            seen = set()
            deduped = []
            for e in selected:
                if e.timestamp in seen:
                    continue
                seen.add(e.timestamp)
                deduped.append(e)

            deduped.sort(key=lambda e: sort_key(e.timestamp))
            return deduped
        except Exception as exc:
            logger.exception("TimelineEngine.build failed: %s", exc)
            return []
