from __future__ import annotations

import logging

from pydantic import BaseModel

from backend.analytics._stats_utils import filter_window
from backend.analytics.models import compute_confidence
from backend.nlp.text_processor import HABIT_KEYWORDS
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)


class HabitCorrelation(BaseModel):
    habit: str
    mention_count: int
    avg_sentiment_when_mentioned: float
    avg_sentiment_other_days: float
    delta: float  # mentioned - other
    correlation_label: str  # "positive" | "negative" | "neutral"
    confidence: float = 0.0
    explanation: str = ""


class HabitEngine:
    """Correlates habit mentions with sentiment. Deterministic, no LLM."""

    def __init__(self, db: JournalDB) -> None:
        self.db = db

    def analyze(self, lookback_days: int = 30) -> list[HabitCorrelation]:
        try:
            records = filter_window(self.db.get_all(), lookback_days)
            if not records:
                return []

            results: list[HabitCorrelation] = []
            for habit in HABIT_KEYWORDS:
                mentioned = [r for r in records if habit in (r.habits or [])]
                if len(mentioned) < 2:
                    continue
                others = [r for r in records if habit not in (r.habits or [])]

                avg_when = self._avg([r.sentiment_compound for r in mentioned])
                avg_other = self._avg([r.sentiment_compound for r in others])
                delta = avg_when - avg_other

                if delta > 0.1:
                    label = "positive"
                elif delta < -0.1:
                    label = "negative"
                else:
                    label = "neutral"

                results.append(
                    HabitCorrelation(
                        habit=habit,
                        mention_count=len(mentioned),
                        avg_sentiment_when_mentioned=round(avg_when, 4),
                        avg_sentiment_other_days=round(avg_other, 4),
                        delta=round(delta, 4),
                        correlation_label=label,
                        confidence=compute_confidence(len(mentioned)),
                        explanation=(
                            f"{habit.capitalize()} mentioned {len(mentioned)} times; "
                            f"mood was {round(delta, 2):+} {'higher' if delta >= 0 else 'lower'} "
                            f"on those days."
                        ),
                    )
                )

            results.sort(key=lambda c: abs(c.delta), reverse=True)
            return results
        except Exception as exc:
            logger.exception("HabitEngine.analyze failed: %s", exc)
            return []

    @staticmethod
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0
