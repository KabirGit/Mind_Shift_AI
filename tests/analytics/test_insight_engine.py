from __future__ import annotations

from backend.analytics.insight_engine import InsightEngine
from backend.analytics.models import PatternSummary, TriggerStat


class FakeEngine:
    def __init__(self, summary: PatternSummary):
        self._summary = summary

    def analyze(self, lookback_days: int = 30) -> PatternSummary:
        return self._summary


def test_known_summary_strings():
    summary = PatternSummary(
        recurring_emotions={"sadness": 3},
        recurring_topics={"career": 3},
        recurring_people={"Alice": 4},
        triggers=[
            TriggerStat(
                topic="career",
                frequency=3,
                avg_sentiment=-0.2,
                dominant_emotion="sadness",
                trend="decreasing",
            )
        ],
        period_entry_count=5,
    )
    out = InsightEngine(FakeEngine(summary)).generate(lookback_days=30)
    assert (
        "You've mentioned career 3 times in the last 30 days, often with sadness."
        in out
    )
    assert "Your sentiment around career has been decreasing recently." in out
    assert "Alice appears frequently in your recent entries." in out


def test_sparse_trigger_below_threshold_omitted():
    # frequency 2 trigger should NOT produce a trigger insight (needs >= 3).
    summary = PatternSummary(
        recurring_topics={"health": 2},
        recurring_people={},
        triggers=[
            TriggerStat(
                topic="health",
                frequency=2,
                avg_sentiment=0.1,
                dominant_emotion="joy",
                trend="stable",
            )
        ],
        period_entry_count=2,
    )
    out = InsightEngine(FakeEngine(summary)).generate()
    assert out == ["Not enough journal history yet to generate insights."]


def test_zero_entries_fallback():
    summary = PatternSummary(period_entry_count=0)
    out = InsightEngine(FakeEngine(summary)).generate()
    assert out == ["Not enough journal history yet to generate insights."]
