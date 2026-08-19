from __future__ import annotations

import logging

from pydantic import BaseModel

from backend.analytics._stats_utils import filter_window, parse_ts
from backend.analytics.models import compute_confidence
from backend.nlp.text_processor import HABIT_KEYWORDS
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)

_STRESSOR_TOPICS = ["career", "money", "health"]


class CausalLink(BaseModel):
    cause: str
    effect: str  # "positive_mood" | "negative_mood"
    probability: float  # P(effect | cause same or next day)
    base_rate: float    # P(effect) overall
    lift: float         # probability - base_rate
    sample_size: int
    confidence: float
    explanation: str


class CausalEngine:
    """Probabilistic (not ML) cause→mood links. Deterministic, no LLM."""

    def __init__(self, db: JournalDB) -> None:
        self.db = db

    def analyze(self, lookback_days: int = 60) -> list[CausalLink]:
        try:
            records = filter_window(self.db.get_all(), lookback_days)
            if not records:
                return []

            dated = []
            for r in records:
                dt = parse_ts(r.timestamp)
                if dt is not None:
                    dated.append((dt.date(), r))
            dated.sort(key=lambda x: x[0])
            if not dated:
                return []

            total = len(dated)
            pos_base = sum(1 for _, r in dated if r.sentiment_compound > 0.1) / total
            neg_base = sum(1 for _, r in dated if r.sentiment_compound < -0.1) / total

            links: list[CausalLink] = []

            # Habit -> positive mood (same or next calendar day).
            for habit in HABIT_KEYWORDS:
                cause_days = [d for d, r in dated if habit in (r.habits or [])]
                if not cause_days:
                    continue
                effect_count = 0
                for cd in cause_days:
                    if self._positive_on_or_after(dated, cd):
                        effect_count += 1
                sample = len(cause_days)
                prob = effect_count / sample if sample else 0.0
                lift = prob - pos_base
                if sample >= 3 and abs(lift) > 0.1:
                    links.append(
                        self._link(habit, "positive_mood", prob, pos_base, lift, sample)
                    )

            # Stressor topic -> negative mood.
            for topic in _STRESSOR_TOPICS:
                cause_days = [d for d, r in dated if topic in (r.topics or [])]
                if not cause_days:
                    continue
                effect_count = 0
                for cd in cause_days:
                    if self._negative_on_or_after(dated, cd):
                        effect_count += 1
                sample = len(cause_days)
                prob = effect_count / sample if sample else 0.0
                lift = prob - neg_base
                if sample >= 3 and abs(lift) > 0.1:
                    links.append(
                        self._link(topic, "negative_mood", prob, neg_base, lift, sample)
                    )

            links.sort(key=lambda c: abs(c.lift), reverse=True)
            return links[:6]
        except Exception as exc:
            logger.exception("CausalEngine.analyze failed: %s", exc)
            return []

    @staticmethod
    def _positive_on_or_after(dated, cause_date) -> bool:
        for d, r in dated:
            delta = (d - cause_date).days
            if 0 <= delta <= 1 and r.sentiment_compound > 0.1:
                return True
        return False

    @staticmethod
    def _negative_on_or_after(dated, cause_date) -> bool:
        for d, r in dated:
            delta = (d - cause_date).days
            if 0 <= delta <= 1 and r.sentiment_compound < -0.1:
                return True
        return False

    @staticmethod
    def _link(cause, effect, prob, base, lift, sample) -> CausalLink:
        mood = "improved" if effect == "positive_mood" else "dropped"
        return CausalLink(
            cause=cause,
            effect=effect,
            probability=round(prob, 4),
            base_rate=round(base, 4),
            lift=round(lift, 4),
            sample_size=sample,
            confidence=compute_confidence(sample),
            explanation=(
                f"Mood {mood} after {cause} in {round(prob * 100)}% of cases "
                f"(base {round(base * 100)}%)."
            ),
        )
