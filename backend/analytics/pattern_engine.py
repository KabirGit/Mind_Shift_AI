from __future__ import annotations

import logging
from collections import Counter

from backend.analytics._stats_utils import filter_window, half_split_trend, sort_key
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
            filtered = filter_window(records, lookback_days)

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
            emotion_trends = self._trends_by_field(filtered, "emotion", recurring_emotions)
            topic_trends = self._trends_by_list_field(filtered, "topics", topic_counter)

            return PatternSummary(
                recurring_emotions=recurring_emotions,
                recurring_topics=recurring_topics,
                recurring_people=recurring_people,
                emotion_trends=emotion_trends,
                topic_trends=topic_trends,
                triggers=triggers,
                period_entry_count=len(filtered),
            )
        except Exception as exc:
            logger.exception("PatternEngine.analyze failed: %s", exc)
            return PatternSummary(period_entry_count=0)

    def _build_triggers(
        self, records: list[JournalRecord], topic_counter: Counter[str]
    ) -> list[TriggerStat]:
        triggers: list[TriggerStat] = []
        for topic, freq in topic_counter.items():
            if freq < 2:
                continue
            topic_records = [r for r in records if topic in (r.topics or [])]
            topic_records.sort(key=lambda r: sort_key(r.timestamp))

            sentiments = [r.sentiment_compound for r in topic_records]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

            emotions = [r.emotion for r in topic_records if r.emotion]
            dominant_emotion = (
                Counter(emotions).most_common(1)[0][0] if emotions else "neutral"
            )

            trend = half_split_trend(sentiments)

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
    def _trends_by_field(
        records: list[JournalRecord],
        field: str,
        counts: dict[str, int],
    ) -> dict[str, str]:
        trends: dict[str, str] = {}
        ordered = sorted(records, key=lambda r: sort_key(r.timestamp))
        for value, count in counts.items():
            if count < 2:
                continue
            sentiments = [
                r.sentiment_compound for r in ordered if getattr(r, field, None) == value
            ]
            trends[value] = half_split_trend(sentiments)
        return trends

    @staticmethod
    def _trends_by_list_field(
        records: list[JournalRecord],
        field: str,
        counts: Counter[str],
    ) -> dict[str, str]:
        trends: dict[str, str] = {}
        ordered = sorted(records, key=lambda r: sort_key(r.timestamp))
        for value, count in counts.items():
            if count < 2:
                continue
            sentiments = [
                r.sentiment_compound
                for r in ordered
                if value in (getattr(r, field, None) or [])
            ]
            trends[value] = half_split_trend(sentiments)
        return trends
