from __future__ import annotations

import logging

from pydantic import BaseModel

from backend.analytics._stats_utils import filter_window, sort_key
from backend.analytics.presentation import mood_score, sentiment_label
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)


class TimelineEvent(BaseModel):
    timestamp: str
    title: str
    description: str
    emotion: str
    sentiment: float
    baseline_sentiment: float | None = None
    primary_person: str | None = None
    significance_score: float
    event_type: str  # "positive_peak" | "negative_peak" | "normal"
    mood_score: int = 50
    mood_label: str = "mixed"


class TimelineEngine:
    """Builds a significance-ranked emotional timeline. Deterministic, no LLM."""

    def __init__(self, db: JournalDB) -> None:
        self.db = db

    def build(self, lookback_days: int = 90, max_events: int = 20) -> list[TimelineEvent]:
        try:
            records = filter_window(self.db.get_all(), lookback_days)
            if not records:
                return []

            person_history: dict[str, list[float]] = {}
            events: list[TimelineEvent] = []
            for r in sorted(records, key=lambda row: sort_key(row.timestamp)):
                s = r.sentiment_compound
                primary_person = (r.entities_people or [None])[0]
                baseline = None
                significance = abs(s)
                if primary_person and person_history.get(primary_person):
                    prior = person_history[primary_person][-5:]
                    baseline = sum(prior) / len(prior)
                    significance = abs(s - baseline)

                if baseline is not None and s - baseline > 0.4:
                    etype = "positive_peak"
                elif baseline is not None and s - baseline < -0.4:
                    etype = "negative_peak"
                elif baseline is None and s > 0.4:
                    etype = "positive_peak"
                elif baseline is None and s < -0.4:
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
                        baseline_sentiment=(
                            round(baseline, 4) if baseline is not None else None
                        ),
                        primary_person=primary_person,
                        significance_score=round(significance, 4),
                        event_type=etype,
                        mood_score=mood_score(s),
                        mood_label=sentiment_label(s),
                    )
                )
                for person in r.entities_people or []:
                    person_history.setdefault(person, []).append(s)

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
