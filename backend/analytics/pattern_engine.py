from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime, timedelta

from backend.analytics.models import PatternSummary, TriggerStat, compute_confidence
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord

logger = logging.getLogger(__name__)


class PatternEngine:
    """Deterministic, stats-only analytics over stored journal records.

    No LLM calls, no network. Pure counting and arithmetic.
    """

    def __init__(self, db: JournalDB) -> None:
        self.db = db

    def analyze(self, lookback_days: int = 30) -> PatternSummary:
        try:
            records = self.db.get_all()
            filtered = self._filter_window(records, lookback_days)

            if not filtered:
                return PatternSummary(period_entry_count=0)

            recurring_emotions = dict(Counter(r.emotion for r in filtered if r.emotion))

            topic_counter: Counter[str] = Counter()
            for r in filtered:
                topic_counter.update(r.topics or [])
            recurring_topics = dict(topic_counter)

            people_counter: Counter[str] = Counter()
            for r in filtered:
                people_counter.update(r.entities_people or [])
            recurring_people = dict(people_counter)

            triggers = self._build_triggers(filtered, topic_counter)

            return PatternSummary(
                recurring_emotions=recurring_emotions,
                recurring_topics=recurring_topics,
                recurring_people=recurring_people,
                triggers=triggers,
                period_entry_count=len(filtered),
            )
        except Exception as exc:
            logger.exception("PatternEngine.analyze failed: %s", exc)
            return PatternSummary(period_entry_count=0)

    def _filter_window(
        self, records: list[JournalRecord], lookback_days: int
    ) -> list[JournalRecord]:
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        out: list[JournalRecord] = []
        for r in records:
            dt = self._parse_ts(r.timestamp)
            if dt is not None and dt >= cutoff:
                out.append(r)
        return out

    def _build_triggers(
        self, records: list[JournalRecord], topic_counter: Counter[str]
    ) -> list[TriggerStat]:
        triggers: list[TriggerStat] = []
        for topic, freq in topic_counter.items():
            if freq < 2:
                continue
            topic_records = [r for r in records if topic in (r.topics or [])]
            topic_records.sort(key=lambda r: self._sort_key(r.timestamp))

            sentiments = [r.sentiment_compound for r in topic_records]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

            emotions = [r.emotion for r in topic_records if r.emotion]
            dominant_emotion = (
                Counter(emotions).most_common(1)[0][0] if emotions else "neutral"
            )

            trend = self._compute_trend(sentiments)

            triggers.append(
                TriggerStat(
                    topic=topic,
                    frequency=freq,
                    avg_sentiment=round(avg_sentiment, 4),
                    dominant_emotion=dominant_emotion,
                    trend=trend,
                    confidence=compute_confidence(freq),
                    explanation=(
                        f"Based on {freq} entries mentioning {topic}; "
                        f"avg sentiment {round(avg_sentiment, 2):+}, trend {trend}."
                    ),
                )
            )
        return triggers

    @staticmethod
    def _compute_trend(sentiments: list[float]) -> str:
        n = len(sentiments)
        if n < 2:
            return "stable"
        mid = n // 2
        first = sentiments[:mid]
        second = sentiments[mid:]
        first_avg = sum(first) / len(first) if first else 0.0
        second_avg = sum(second) / len(second) if second else 0.0
        diff = second_avg - first_avg
        if diff > 0.1:
            return "increasing"
        if diff < -0.1:
            return "decreasing"
        return "stable"

    @staticmethod
    def _parse_ts(timestamp: str):
        if not timestamp:
            return None
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except (ValueError, TypeError):
            return None

    @classmethod
    def _sort_key(cls, timestamp: str):
        dt = cls._parse_ts(timestamp)
        return dt if dt is not None else datetime.min.replace(tzinfo=UTC)
