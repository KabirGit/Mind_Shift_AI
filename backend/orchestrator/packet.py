from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.analytics.habit_engine import HabitCorrelation
from backend.analytics.models import TriggerStat
from backend.analytics.relationship_engine import RelationshipProfile
from backend.profile.models import UserProfile


class IntelligencePacket(BaseModel):
    """Shared state all engines contribute to before the single LLM call."""

    current_entry_emotion: str
    current_entry_sentiment: float
    insights: list[str] = Field(default_factory=list)
    reflection_prompts: list[str] = Field(default_factory=list)
    triggers: list[TriggerStat] = Field(default_factory=list)
    habits: list[HabitCorrelation] = Field(default_factory=list)
    relationships: list[RelationshipProfile] = Field(default_factory=list)
    user_profile: UserProfile | None = None
    proactive_alerts: list[str] = Field(default_factory=list)  # Phase 13
    temporal_patterns: list[Any] = Field(default_factory=list)  # Phase 13
    causal_links: list[Any] = Field(default_factory=list)       # Phase 13
    predictions: dict = Field(default_factory=dict)             # Phase 14
    goals: list[Any] = Field(default_factory=list)              # Phase 14
    memory_replay: dict | None = None                        # Phase 15
