from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from backend.analytics._stats_utils import (
    filter_window,
    parse_ts,
    recovery_speed_days,
    sort_key,
)
from backend.analytics.goal_engine import GoalProgress
from backend.analytics.habit_engine import HabitCorrelation
from backend.analytics.models import TriggerStat
from backend.analytics.prediction_engine import BurnoutRisk, SentimentForecast
from backend.analytics.relationship_engine import RelationshipProfile
from backend.analytics.temporal_engine import TemporalPattern
from backend.analytics.timeline_engine import TimelineEvent
from backend.config.settings import get_settings

logger = logging.getLogger(__name__)

_settings = get_settings()

# 0.5 means at least half of the default confidence sample target is present;
# below this, a dashboard sentence would overstate a correlation.
MIN_INSIGHT_CONFIDENCE = _settings.dashboard_min_insight_confidence
# Three mentions is the first point where a person pattern is repeated, not anecdotal.
MIN_MENTION_COUNT = _settings.dashboard_min_mention_count
# Five entries gives the hero enough data to compare halves without sounding conclusive.
MIN_ENTRY_COUNT = _settings.dashboard_min_entry_count


class DashboardHeadline(BaseModel):
    baseline_sentiment: float = 0.0
    current_sentiment: float = 0.0
    sentiment_delta: float = 0.0
    dominant_emotion_start: str = "neutral"
    dominant_emotion_end: str = "neutral"
    recovery_speed_days_start: float = 0.0
    recovery_speed_days_end: float = 0.0
    entry_count: int = 0
    days_in_range: int = 0
    growth_score: float = 0.0
    growth_narrative: str = ""
    has_sufficient_data: bool = False
    minimum_entry_count: int = MIN_ENTRY_COUNT


class WeeklyBucket(BaseModel):
    label: str
    avg_sentiment: float
    dominant_emotion: str
    top_topic: str
    entry_count: int


class ForecastBlock(BaseModel):
    sentiment_forecast: SentimentForecast
    burnout_risk: BurnoutRisk


class StoryThresholds(BaseModel):
    min_insight_confidence: float = MIN_INSIGHT_CONFIDENCE
    min_mention_count: int = MIN_MENTION_COUNT
    min_entry_count: int = MIN_ENTRY_COUNT


class DashboardStoryResponse(BaseModel):
    range: str
    lookback_days: int
    headline: DashboardHeadline
    top_working: list[HabitCorrelation] = Field(default_factory=list)
    top_draining: list[TriggerStat] = Field(default_factory=list)
    people: list[RelationshipProfile] = Field(default_factory=list)
    rhythm: TemporalPattern | None = None
    weekly_buckets: list[WeeklyBucket] = Field(default_factory=list)
    forecast: ForecastBlock
    goals: list[GoalProgress] = Field(default_factory=list)
    highlight_memory: TimelineEvent | None = None
    thresholds: StoryThresholds = Field(default_factory=StoryThresholds)


class DashboardStoryComposer:
    """Composes existing analytics outputs into a narrative dashboard response."""

    def __init__(self, service: Any) -> None:
        self.service = service

    def compose(self, *, range_label: str, lookback_days: int) -> DashboardStoryResponse:
        records = filter_window(self.service.journal_db.get_all(), lookback_days)
        ordered = sorted(records, key=lambda r: sort_key(r.timestamp))
        profile = self.service.profile_manager.update()
        growth_narrative = self.service.growth_tracker.narrative()

        forecast = ForecastBlock(
            sentiment_forecast=self.service.prediction_engine.forecast_sentiment(),
            burnout_risk=self.service.prediction_engine.assess_burnout_risk(),
        )
        headline = self._headline(
            ordered,
            lookback_days=lookback_days,
            profile=profile,
            growth_narrative=growth_narrative,
        )

        summary = self.service.pattern_engine.analyze(lookback_days=lookback_days)
        habits = self.service.habit_engine.analyze(lookback_days=lookback_days)
        relationships = self.service.relationship_engine.analyze(
            lookback_days=lookback_days
        )
        temporal = self.service.temporal_engine.analyze(lookback_days=lookback_days)
        timeline = self.service.timeline_engine.build(lookback_days=lookback_days)

        return DashboardStoryResponse(
            range=range_label,
            lookback_days=lookback_days,
            headline=headline,
            top_working=self._top_working(habits),
            top_draining=self._top_draining(summary.triggers),
            people=self._people(relationships, profile.baseline_sentiment),
            rhythm=self._rhythm(temporal),
            weekly_buckets=self._weekly_buckets(ordered),
            forecast=forecast,
            goals=self.service.goal_engine.analyze(lookback_days=90),
            highlight_memory=self._highlight_memory(timeline),
        )

    def _headline(self, records, *, lookback_days: int, profile, growth_narrative: str):
        start, end = self._split(records)
        return DashboardHeadline(
            baseline_sentiment=profile.baseline_sentiment,
            current_sentiment=profile.current_sentiment,
            sentiment_delta=round(
                profile.current_sentiment - profile.baseline_sentiment, 4
            ),
            dominant_emotion_start=self._dominant_emotion(start),
            dominant_emotion_end=self._dominant_emotion(end),
            recovery_speed_days_start=round(recovery_speed_days(start), 2),
            recovery_speed_days_end=round(recovery_speed_days(end), 2),
            entry_count=len(records),
            days_in_range=lookback_days,
            growth_score=profile.growth_score,
            growth_narrative=growth_narrative,
            has_sufficient_data=len(records) >= MIN_ENTRY_COUNT,
            minimum_entry_count=MIN_ENTRY_COUNT,
        )

    @staticmethod
    def _split(records):
        if not records:
            return [], []
        mid = max(1, len(records) // 2)
        return records[:mid], records[mid:] or records[:mid]

    @staticmethod
    def _dominant_emotion(records) -> str:
        emotions = [r.emotion for r in records if r.emotion]
        return Counter(emotions).most_common(1)[0][0] if emotions else "neutral"

    @staticmethod
    def _top_working(habits: list[HabitCorrelation]) -> list[HabitCorrelation]:
        # Confidence below 0.5 is treated as exploratory, so it is not narrated.
        eligible = [
            h for h in habits
            if h.confidence >= MIN_INSIGHT_CONFIDENCE and h.delta > 0
        ]
        eligible.sort(key=lambda h: abs(h.delta), reverse=True)
        return eligible[:3]

    @staticmethod
    def _top_draining(triggers: list[TriggerStat]) -> list[TriggerStat]:
        # Confidence below 0.5 is treated as exploratory, so it is not narrated.
        eligible = [
            t for t in triggers
            if t.confidence >= MIN_INSIGHT_CONFIDENCE and t.avg_sentiment < 0
        ]
        eligible.sort(key=lambda t: abs(t.avg_sentiment), reverse=True)
        return eligible[:3]

    @staticmethod
    def _people(
        relationships: list[RelationshipProfile],
        baseline_sentiment: float,
    ) -> list[RelationshipProfile]:
        # Fewer than three mentions is not enough to call a person a recurring pattern.
        eligible = [
            p for p in relationships
            if p.mention_count >= MIN_MENTION_COUNT
        ]
        eligible.sort(
            key=lambda p: (abs(p.avg_sentiment - baseline_sentiment), p.mention_count),
            reverse=True,
        )
        return eligible[:3]

    @staticmethod
    def _rhythm(patterns: list[TemporalPattern]) -> TemporalPattern | None:
        if not patterns:
            return None
        return max(patterns, key=lambda p: (p.confidence, abs(p.delta)))

    @staticmethod
    def _highlight_memory(timeline: list[TimelineEvent]) -> TimelineEvent | None:
        if not timeline:
            return None
        return max(timeline, key=lambda e: e.significance_score)

    @staticmethod
    def _weekly_buckets(records) -> list[WeeklyBucket]:
        groups: dict[str, list] = defaultdict(list)
        for r in records:
            dt = parse_ts(r.timestamp)
            if dt is None:
                continue
            week_start = dt.date() - timedelta(days=dt.weekday())
            groups[week_start.isoformat()].append(r)

        buckets: list[WeeklyBucket] = []
        for week_start in sorted(groups):
            recs = groups[week_start]
            sentiments = [r.sentiment_compound for r in recs]
            emotions = [r.emotion for r in recs if r.emotion]
            topics: Counter[str] = Counter()
            for r in recs:
                topics.update(r.topics or [])

            start_dt = datetime.fromisoformat(week_start).replace(tzinfo=UTC)
            end_dt = start_dt + timedelta(days=6)
            label = f"{start_dt.strftime('%b %d')}-{end_dt.strftime('%b %d')}"
            buckets.append(
                WeeklyBucket(
                    label=label,
                    avg_sentiment=round(sum(sentiments) / len(sentiments), 4),
                    dominant_emotion=(
                        Counter(emotions).most_common(1)[0][0]
                        if emotions else "neutral"
                    ),
                    top_topic=topics.most_common(1)[0][0] if topics else "none",
                    entry_count=len(recs),
                )
            )
        return buckets
