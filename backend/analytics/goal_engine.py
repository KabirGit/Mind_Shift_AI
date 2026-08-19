from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel

from backend.analytics._stats_utils import filter_window, parse_ts, sort_key
from backend.analytics.models import compute_confidence
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)

GOAL_PATTERNS: dict[str, list[str]] = {
    "internship": ["internship", "intern", "application"],
    "job_search": ["job", "offer", "interview", "hiring"],
    "fitness": ["weight", "fitness", "diet", "gym goal"],
    "mental_health": ["therapy", "anxiety", "stress management"],
    "education": ["exam", "degree", "graduation", "course"],
    "promotion": ["promotion", "raise", "performance review"],
}


class GoalProgress(BaseModel):
    goal_keyword: str
    first_mentioned: str
    last_mentioned: str
    mention_count: int
    avg_sentiment: float
    sentiment_trend: str  # "improving" | "declining" | "stable"
    estimated_progress: float  # 0.0-1.0
    confidence: float
    explanation: str


class GoalEngine:
    """Tracks goal-related entries and a sentiment-based progress proxy."""

    def __init__(self, db: JournalDB) -> None:
        self.db = db

    def analyze(self, lookback_days: int = 90) -> list[GoalProgress]:
        try:
            records = filter_window(self.db.get_all(), lookback_days)
            if not records:
                return []

            results: list[GoalProgress] = []
            for goal, keywords in GOAL_PATTERNS.items():
                matched = [
                    r for r in records
                    if any(k in (r.text or "").lower() for k in keywords)
                ]
                if len(matched) < 2:
                    continue
                matched.sort(key=lambda r: sort_key(r.timestamp))

                sentiments = [r.sentiment_compound for r in matched]
                avg_sent = sum(sentiments) / len(sentiments)
                slope = self._slope(matched)

                if slope > 0.005:
                    trend = "improving"
                elif slope < -0.005:
                    trend = "declining"
                else:
                    trend = "stable"

                slope_norm = max(-0.5, min(0.5, slope * 20))
                progress = max(0.0, min(1.0, slope_norm + 0.5))

                results.append(
                    GoalProgress(
                        goal_keyword=goal,
                        first_mentioned=matched[0].timestamp,
                        last_mentioned=matched[-1].timestamp,
                        mention_count=len(matched),
                        avg_sentiment=round(avg_sent, 4),
                        sentiment_trend=trend,
                        estimated_progress=round(progress, 4),
                        confidence=compute_confidence(len(matched)),
                        explanation=(
                            f"{goal.replace('_', ' ').capitalize()} mentioned "
                            f"{len(matched)} times; sentiment trend {trend}."
                        ),
                    )
                )

            results.sort(key=lambda g: g.mention_count, reverse=True)
            return results
        except Exception as exc:
            logger.exception("GoalEngine.analyze failed: %s", exc)
            return []

    @staticmethod
    def _slope(records) -> float:
        t0 = parse_ts(records[0].timestamp)
        xs, ys = [], []
        for r in records:
            dt = parse_ts(r.timestamp)
            if dt is None or t0 is None:
                continue
            xs.append((dt - t0).total_seconds() / 86400.0)
            ys.append(r.sentiment_compound)
        if len(xs) < 2:
            return 0.0
        return float(np.polyfit(np.array(xs), np.array(ys), 1)[0])
