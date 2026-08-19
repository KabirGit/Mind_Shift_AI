from __future__ import annotations

from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    user_id: str = "default"
    baseline_sentiment: float = 0.0   # rolling avg of all-time sentiment_compound
    current_sentiment: float = 0.0    # avg of last-7-days entries
    dominant_emotion: str = "neutral"  # mode emotion across all entries
    recovery_speed_days: float = 0.0  # avg days from a negative entry to next positive
    top_triggers: list[str] = Field(default_factory=list)  # top 3 topics by frequency
    top_habits: list[str] = Field(default_factory=list)    # top 2 habits by positive delta
    top_people: list[str] = Field(default_factory=list)    # top 3 most-mentioned people
    entry_count: int = 0
    last_updated: str = ""            # ISO timestamp
    growth_score: float = 0.0         # clamped 0-1, higher = improvement over baseline
    communication_style: str = "reflective"  # brief | reflective | detailed
