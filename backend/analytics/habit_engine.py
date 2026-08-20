from __future__ import annotations

import logging
from collections import defaultdict
from itertools import combinations

from pydantic import BaseModel, Field

from backend.analytics._stats_utils import filter_window, parse_ts
from backend.analytics.models import compute_confidence
from backend.nlp.text_processor import HABIT_KEYWORDS
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)


class HabitPairCorrelation(BaseModel):
    habit_a: str
    habit_b: str
    mention_count: int
    avg_sentiment_together: float
    avg_sentiment_either_alone: float
    delta_vs_either_alone: float
    confidence: float = 0.0
    explanation: str = ""


class HabitCorrelation(BaseModel):
    habit: str
    mention_count: int
    avg_sentiment_when_mentioned: float
    avg_sentiment_other_days: float
    delta: float  # mentioned - other
    correlation_label: str  # "positive" | "negative" | "neutral"
    streak_length: int = 0
    consistency_percentage: float = 0.0
    co_occurring_pairs: list[HabitPairCorrelation] = Field(default_factory=list)
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
            pairs = self.analyze_pairs_from_records(records)
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
                        streak_length=self._streak_length(mentioned),
                        consistency_percentage=self._consistency_percentage(
                            mentioned, records
                        ),
                        co_occurring_pairs=[
                            p for p in pairs if habit in {p.habit_a, p.habit_b}
                        ],
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

    def analyze_pairs(self, lookback_days: int = 30) -> list[HabitPairCorrelation]:
        try:
            records = filter_window(self.db.get_all(), lookback_days)
            return self.analyze_pairs_from_records(records)
        except Exception as exc:
            logger.exception("HabitEngine.analyze_pairs failed: %s", exc)
            return []

    def analyze_pairs_from_records(self, records) -> list[HabitPairCorrelation]:
        pair_records: dict[tuple[str, str], list] = defaultdict(list)
        for r in records:
            habits = sorted(set(r.habits or []))
            for pair in combinations(habits, 2):
                pair_records[pair].append(r)

        out: list[HabitPairCorrelation] = []
        for (a, b), together in pair_records.items():
            if len(together) < 2:
                continue
            either_alone = [
                r for r in records
                if (a in (r.habits or []) or b in (r.habits or []))
                and not (a in (r.habits or []) and b in (r.habits or []))
            ]
            avg_together = self._avg([r.sentiment_compound for r in together])
            avg_alone = self._avg([r.sentiment_compound for r in either_alone])
            delta = avg_together - avg_alone
            out.append(
                HabitPairCorrelation(
                    habit_a=a,
                    habit_b=b,
                    mention_count=len(together),
                    avg_sentiment_together=round(avg_together, 4),
                    avg_sentiment_either_alone=round(avg_alone, 4),
                    delta_vs_either_alone=round(delta, 4),
                    confidence=compute_confidence(len(together)),
                    explanation=(
                        f"{a} + {b} appeared together {len(together)} times; "
                        f"sentiment was {delta:+.2f} vs either habit alone."
                    ),
                )
            )
        out.sort(key=lambda p: abs(p.delta_vs_either_alone), reverse=True)
        return out

    @staticmethod
    def _streak_length(mentioned) -> int:
        days = sorted(
            {dt.date() for r in mentioned if (dt := parse_ts(r.timestamp)) is not None},
            reverse=True,
        )
        if not days:
            return 0
        streak = 1
        prev = days[0]
        for day in days[1:]:
            if (prev - day).days == 1:
                streak += 1
                prev = day
            else:
                break
        return streak

    @staticmethod
    def _consistency_percentage(mentioned, records) -> float:
        mention_days = {
            dt.date() for r in mentioned if (dt := parse_ts(r.timestamp)) is not None
        }
        all_days = {dt.date() for r in records if (dt := parse_ts(r.timestamp)) is not None}
        if not all_days:
            return 0.0
        return round(len(mention_days) / len(all_days) * 100.0, 2)

    @staticmethod
    def _avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0
