from __future__ import annotations

from pydantic import BaseModel, Field

# Default sample-count threshold for full confidence (confidence = count / THRESHOLD).
CONFIDENCE_THRESHOLD = 10


def compute_confidence(count: int, threshold: int = CONFIDENCE_THRESHOLD) -> float:
    return round(min(1.0, max(0, count) / float(threshold)), 4)


class TriggerStat(BaseModel):
    topic: str
    frequency: int
    avg_sentiment: float
    dominant_emotion: str
    trend: str  # "increasing" | "decreasing" | "stable"
    confidence: float = 0.0  # min(1.0, frequency / 10)
    explanation: str = ""


class PatternSummary(BaseModel):
    recurring_emotions: dict[str, int] = Field(default_factory=dict)
    recurring_topics: dict[str, int] = Field(default_factory=dict)
    recurring_people: dict[str, int] = Field(default_factory=dict)
    emotion_trends: dict[str, str] = Field(default_factory=dict)
    topic_trends: dict[str, str] = Field(default_factory=dict)
    triggers: list[TriggerStat] = Field(default_factory=list)
    period_entry_count: int = 0
