from __future__ import annotations

import logging
from dataclasses import dataclass

from backend.analytics.models import PatternSummary, compute_confidence
from backend.analytics.pattern_engine import PatternEngine

logger = logging.getLogger(__name__)

_NO_HISTORY = "Not enough journal history yet to generate insights."
MIN_INSIGHT_CONFIDENCE = 0.2
MAX_INSIGHTS = 5


@dataclass(frozen=True)
class _InsightCandidate:
    text: str
    confidence: float
    recency: float


class InsightEngine:
    """Turns deterministic analytics stats into plain-English strings.

    Uses string templates only. No LLM calls. Habit/relationship engines are
    optional so existing callers keep working unchanged.
    """

    def __init__(
        self,
        pattern_engine: PatternEngine,
        habit_engine=None,
        relationship_engine=None,
    ) -> None:
        self.pattern_engine = pattern_engine
        self.habit_engine = habit_engine
        self.relationship_engine = relationship_engine

    def generate(self, lookback_days: int = 30) -> list[str]:
        try:
            summary = self.pattern_engine.analyze(lookback_days=lookback_days)
            return self._from_summary(summary, lookback_days)
        except Exception as exc:
            logger.exception("InsightEngine.generate failed: %s", exc)
            return [_NO_HISTORY]

    def _from_summary(self, summary: PatternSummary, days: int) -> list[str]:
        if summary.period_entry_count == 0:
            return [_NO_HISTORY]

        candidates: list[_InsightCandidate] = []

        # Trigger-based insights (only for frequency >= 3 to avoid noise).
        triggers = sorted(summary.triggers, key=lambda t: t.frequency, reverse=True)
        for trig in triggers:
            if trig.frequency < 3:
                continue
            confidence = trig.confidence or compute_confidence(trig.frequency)
            candidates.append(
                _InsightCandidate(
                    text=(
                        f"You've mentioned {trig.topic} {trig.frequency} times in the last "
                        f"{days} days, often with {trig.dominant_emotion}."
                    ),
                    confidence=confidence,
                    recency=1.0,
                )
            )
            if trig.trend in {"increasing", "decreasing"}:
                candidates.append(
                    _InsightCandidate(
                        text=(
                            f"Your sentiment around {trig.topic} has been "
                            f"{trig.trend} recently."
                        ),
                        confidence=confidence,
                        recency=1.0,
                    )
                )

        # Recurring person insight.
        if summary.recurring_people:
            top_person, count = max(
                summary.recurring_people.items(), key=lambda kv: kv[1]
            )
            confidence = compute_confidence(count)
            if count >= 3:
                top_emotion = self._person_emotion(top_person, days)
                if top_emotion:
                    candidates.append(
                        _InsightCandidate(
                            text=(
                                f"{top_person} comes up often in your entries, usually "
                                f"with {top_emotion}."
                            ),
                            confidence=confidence,
                            recency=0.8,
                        )
                    )
                else:
                    candidates.append(
                        _InsightCandidate(
                            text=f"{top_person} appears frequently in your recent entries.",
                            confidence=confidence,
                            recency=0.8,
                        )
                    )
            elif count >= 2:
                candidates.append(
                    _InsightCandidate(
                        text=f"{top_person} appears frequently in your recent entries.",
                        confidence=confidence,
                        recency=0.8,
                    )
                )

        # Habit-correlation insights (up to 2; mention_count >= 3 and not neutral).
        candidates.extend(self._habit_insights(days))

        insights = self._rank_and_dedupe(candidates)
        if not insights:
            # Have entries but nothing crossed the noise thresholds.
            return [_NO_HISTORY]

        return insights

    def _habit_insights(self, days: int) -> list[_InsightCandidate]:
        if self.habit_engine is None:
            return []
        try:
            correlations = self.habit_engine.analyze(lookback_days=days)
        except Exception as exc:
            logger.exception("habit insight generation failed: %s", exc)
            return []

        out: list[_InsightCandidate] = []
        for corr in correlations:
            if corr.mention_count < 3 or corr.correlation_label == "neutral":
                continue
            direction = "more" if corr.correlation_label == "positive" else "less"
            out.append(
                _InsightCandidate(
                    text=(
                        f"Days you mention {corr.habit} tend to have {direction} "
                        f"positive sentiment ({corr.delta:+.2f})."
                    ),
                    confidence=corr.confidence or compute_confidence(corr.mention_count),
                    recency=0.7,
                )
            )
            if len(out) >= 2:
                break
        return out

    @staticmethod
    def _rank_and_dedupe(candidates: list[_InsightCandidate]) -> list[str]:
        filtered = [
            c for c in candidates
            if c.confidence >= MIN_INSIGHT_CONFIDENCE and c.text.strip()
        ]
        filtered.sort(key=lambda c: (c.confidence, c.recency), reverse=True)
        out: list[str] = []
        seen: set[str] = set()
        for c in filtered:
            signature = " ".join(
                word.lower().strip(".,;:!?")
                for word in c.text.split()
                if len(word) > 3
            )
            key = " ".join(signature.split()[:8])
            if key in seen:
                continue
            seen.add(key)
            out.append(c.text)
            if len(out) >= MAX_INSIGHTS:
                break
        return out

    def _person_emotion(self, person: str, days: int) -> str | None:
        if self.relationship_engine is None:
            return None
        try:
            profiles = self.relationship_engine.analyze(lookback_days=days)
        except Exception:
            return None
        for p in profiles:
            if p.person == person:
                return p.dominant_emotion
        return None
