from __future__ import annotations

import logging
from collections import defaultdict

from pydantic import BaseModel

from backend.analytics._stats_utils import filter_window, parse_ts
from backend.analytics.models import compute_confidence
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)

_DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class TemporalPattern(BaseModel):
    topic: str
    peak_day_of_week: str
    peak_time_of_day: str | None = None
    day_time_crossing: str | None = None
    day_time_sample_size: int = 0
    peak_day_avg_sentiment: float
    peak_time_avg_sentiment: float | None = None
    baseline_avg_sentiment: float
    delta: float  # peak - baseline
    confidence: float
    explanation: str


class TemporalEngine:
    """Finds day-of-week sentiment patterns per topic. Deterministic, no LLM."""

    def __init__(self, db: JournalDB) -> None:
        self.db = db

    def analyze(self, lookback_days: int = 60) -> list[TemporalPattern]:
        try:
            records = filter_window(self.db.get_all(), lookback_days)
            if not records:
                return []

            topic_records: dict[str, list] = defaultdict(list)
            for r in records:
                for topic in r.topics or []:
                    topic_records[topic].append(r)

            patterns: list[TemporalPattern] = []
            for topic, recs in topic_records.items():
                if len(recs) < 5:
                    continue
                baseline = sum(r.sentiment_compound for r in recs) / len(recs)

                by_day: dict[int, list[float]] = defaultdict(list)
                by_day_time: dict[tuple[int, str], list[float]] = defaultdict(list)
                for r in recs:
                    dt = parse_ts(r.timestamp)
                    if dt is not None:
                        by_day[dt.weekday()].append(r.sentiment_compound)
                        by_day_time[(dt.weekday(), self._time_bucket(dt.hour))].append(
                            r.sentiment_compound
                        )

                best_day, best_delta, best_avg = None, 0.0, baseline
                for day, sents in by_day.items():
                    day_avg = sum(sents) / len(sents)
                    delta = day_avg - baseline
                    if abs(delta) > abs(best_delta):
                        best_day, best_delta, best_avg = day, delta, day_avg

                if best_day is None or abs(best_delta) <= 0.15:
                    continue

                best_cross, best_cross_delta, best_cross_avg, best_cross_n = (
                    None,
                    0.0,
                    None,
                    0,
                )
                for key, sents in by_day_time.items():
                    if len(sents) < 2:
                        continue
                    avg = sum(sents) / len(sents)
                    delta = avg - baseline
                    if abs(delta) > abs(best_cross_delta):
                        best_cross, best_cross_delta, best_cross_avg, best_cross_n = (
                            key,
                            delta,
                            avg,
                            len(sents),
                        )

                day_name = _DAYS[best_day]
                peak_time = None
                crossing = None
                if best_cross is not None and abs(best_cross_delta) > 0.15:
                    cross_day, peak_time = best_cross
                    crossing = f"{_DAYS[cross_day]} {peak_time}"
                direction = "above" if best_delta > 0 else "below"
                patterns.append(
                    TemporalPattern(
                        topic=topic,
                        peak_day_of_week=day_name,
                        peak_time_of_day=peak_time,
                        day_time_crossing=crossing,
                        day_time_sample_size=best_cross_n,
                        peak_day_avg_sentiment=round(best_avg, 4),
                        peak_time_avg_sentiment=(
                            round(best_cross_avg, 4)
                            if best_cross_avg is not None
                            else None
                        ),
                        baseline_avg_sentiment=round(baseline, 4),
                        delta=round(best_delta, 4),
                        confidence=compute_confidence(len(recs)),
                        explanation=(
                            f"{topic.capitalize()} sentiment peaks on {day_name}s "
                            f"({round(best_delta, 2):+}, {direction} its baseline)."
                        ),
                    )
                )

            patterns.sort(key=lambda p: abs(p.delta), reverse=True)
            return patterns[:5]
        except Exception as exc:
            logger.exception("TemporalEngine.analyze failed: %s", exc)
            return []

    @staticmethod
    def _time_bucket(hour: int) -> str:
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 22:
            return "evening"
        return "night"
