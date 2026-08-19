from __future__ import annotations

import logging
from collections import Counter, defaultdict

from pydantic import BaseModel

from backend.analytics._stats_utils import filter_window, half_split_trend, sort_key
from backend.analytics.models import compute_confidence
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)


class RelationshipProfile(BaseModel):
    person: str
    mention_count: int
    avg_sentiment: float
    dominant_emotion: str
    last_mentioned: str  # ISO timestamp of most recent mention
    trend: str  # "improving" | "declining" | "stable"
    confidence: float = 0.0
    explanation: str = ""


class RelationshipEngine:
    """Per-person emotional profile from entities_people. Deterministic, no LLM."""

    def __init__(self, db: JournalDB) -> None:
        self.db = db

    def analyze(self, lookback_days: int = 30) -> list[RelationshipProfile]:
        try:
            records = filter_window(self.db.get_all(), lookback_days)
            if not records:
                return []

            grouped: dict[str, list] = defaultdict(list)
            for r in records:
                for person in r.entities_people or []:
                    grouped[person].append(r)

            profiles: list[RelationshipProfile] = []
            for person, recs in grouped.items():
                if len(recs) < 2:
                    continue
                recs.sort(key=lambda r: sort_key(r.timestamp))

                sentiments = [r.sentiment_compound for r in recs]
                avg_sentiment = sum(sentiments) / len(sentiments)

                emotions = [r.emotion for r in recs if r.emotion]
                dominant = (
                    Counter(emotions).most_common(1)[0][0] if emotions else "neutral"
                )

                last_mentioned = recs[-1].timestamp
                trend = half_split_trend(
                    sentiments, up="improving", down="declining", stable="stable"
                )

                profiles.append(
                    RelationshipProfile(
                        person=person,
                        mention_count=len(recs),
                        avg_sentiment=round(avg_sentiment, 4),
                        dominant_emotion=dominant,
                        last_mentioned=last_mentioned,
                        trend=trend,
                        confidence=compute_confidence(len(recs)),
                        explanation=(
                            f"Based on {len(recs)} entries mentioning {person}; "
                            f"usually {dominant}, trend {trend}."
                        ),
                    )
                )

            profiles.sort(key=lambda p: p.mention_count, reverse=True)
            return profiles
        except Exception as exc:
            logger.exception("RelationshipEngine.analyze failed: %s", exc)
            return []
