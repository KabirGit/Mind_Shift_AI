from __future__ import annotations

import logging

from backend.analytics.models import PatternSummary
from backend.analytics.pattern_engine import PatternEngine

logger = logging.getLogger(__name__)

_NO_HISTORY = "Not enough journal history yet to generate insights."


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

        insights: list[str] = []

        # Trigger-based insights (only for frequency >= 3 to avoid noise).
        triggers = sorted(summary.triggers, key=lambda t: t.frequency, reverse=True)
        for trig in triggers:
            if trig.frequency < 3:
                continue
            insights.append(
                f"You've mentioned {trig.topic} {trig.frequency} times in the last "
                f"{days} days, often with {trig.dominant_emotion}."
            )
            if trig.trend in {"increasing", "decreasing"}:
                insights.append(
                    f"Your sentiment around {trig.topic} has been {trig.trend} recently."
                )
            if len(insights) >= 5:
                return insights[:5]

        # Recurring person insight.
        if summary.recurring_people:
            top_person, count = max(
                summary.recurring_people.items(), key=lambda kv: kv[1]
            )
            if count >= 3:
                top_emotion = self._person_emotion(top_person, days)
                if top_emotion:
                    insights.append(
                        f"{top_person} comes up often in your entries, usually with "
                        f"{top_emotion}."
                    )
                else:
                    insights.append(
                        f"{top_person} appears frequently in your recent entries."
                    )
            elif count >= 2:
                insights.append(
                    f"{top_person} appears frequently in your recent entries."
                )

        # Habit-correlation insights (up to 2; mention_count >= 3 and not neutral).
        insights.extend(self._habit_insights(days))

        if not insights:
            # Have entries but nothing crossed the noise thresholds.
            return [_NO_HISTORY]

        return insights[:5]

    def _habit_insights(self, days: int) -> list[str]:
        if self.habit_engine is None:
            return []
        try:
            correlations = self.habit_engine.analyze(lookback_days=days)
        except Exception as exc:
            logger.exception("habit insight generation failed: %s", exc)
            return []

        out: list[str] = []
        for corr in correlations:
            if corr.mention_count < 3 or corr.correlation_label == "neutral":
                continue
            direction = "more" if corr.correlation_label == "positive" else "less"
            out.append(
                f"Days you mention {corr.habit} tend to have {direction} positive "
                f"sentiment ({corr.delta:+.2f})."
            )
            if len(out) >= 2:
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
