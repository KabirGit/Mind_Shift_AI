from __future__ import annotations

import logging
from collections import Counter, defaultdict

from pydantic import BaseModel, Field

from backend.analytics._stats_utils import (
    filter_window,
    half_split_trend,
    recency_decay,
    sort_key,
)
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
    relationship_type: str = "unknown"
    relationship_type_confidence: float = 0.0
    relationship_type_ambiguity: str = ""
    closeness_score: float = 0.0
    sentiment_trend: str = "stable"
    co_mentioned_with: dict[str, int] = Field(default_factory=dict)
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
                rel_type, rel_conf, rel_ambiguity = self._relationship_type(person, recs)
                closeness = sum(
                    recency_decay(r.timestamp, half_life_hours=72.0) for r in recs
                )
                co_mentions: Counter[str] = Counter()
                for r in recs:
                    for other in r.entities_people or []:
                        if other != person:
                            co_mentions[other] += 1

                profiles.append(
                    RelationshipProfile(
                        person=person,
                        mention_count=len(recs),
                        avg_sentiment=round(avg_sentiment, 4),
                        dominant_emotion=dominant,
                        last_mentioned=last_mentioned,
                        trend=trend,
                        relationship_type=rel_type,
                        relationship_type_confidence=rel_conf,
                        relationship_type_ambiguity=rel_ambiguity,
                        closeness_score=round(closeness, 4),
                        sentiment_trend=trend,
                        co_mentioned_with=dict(co_mentions.most_common()),
                        confidence=compute_confidence(len(recs)),
                        explanation=(
                            f"Based on {len(recs)} entries mentioning {person}; "
                            f"usually {dominant}, trend {trend}."
                        ),
                    )
                )

            profiles.sort(key=lambda p: (p.mention_count, p.closeness_score), reverse=True)
            return profiles
        except Exception as exc:
            logger.exception("RelationshipEngine.analyze failed: %s", exc)
            return []

    @staticmethod
    def _relationship_type(person: str, recs: list) -> tuple[str, float, str]:
        counts: Counter[str] = Counter()
        for r in recs:
            rels = getattr(r, "person_relationship_types", None) or {}
            rel_type = rels.get(person) or rels.get(person.strip()) or "unknown"
            counts[rel_type] += 1
        if not counts:
            return "unknown", 0.0, ""

        rel_type, count = counts.most_common(1)[0]
        confidence = round(count / sum(counts.values()), 4)
        meaningful = {k: v for k, v in counts.items() if k != "unknown" and v > 0}
        ambiguity = ""
        if len(meaningful) > 1:
            parts = ", ".join(f"{k}:{v}" for k, v in sorted(meaningful.items()))
            ambiguity = f"Conflicting relationship cues across entries ({parts})."
        return rel_type, confidence, ambiguity
