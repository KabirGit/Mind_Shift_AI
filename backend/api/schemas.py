from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.analytics.goal_engine import GoalProgress
from backend.analytics.growth_tracker import GrowthSnapshot
from backend.analytics.habit_engine import HabitCorrelation
from backend.analytics.models import PatternSummary, TriggerStat
from backend.analytics.prediction_engine import BurnoutRisk, SentimentForecast
from backend.analytics.relationship_engine import RelationshipProfile
from backend.analytics.timeline_engine import TimelineEvent
from backend.orchestrator.packet import IntelligencePacket


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    text: str = Field(min_length=1)
    chat_history: list[ChatMessage] = Field(default_factory=list)
    top_k: int = Field(default=3, ge=1, le=20)
    tags: list[str] | None = None


class ChatResponse(BaseModel):
    emotion: dict[str, Any]
    response: str
    memory_replay: dict[str, Any] | None = None
    crisis: dict[str, Any]
    retrieved_memories: list[dict[str, Any]] = Field(default_factory=list)
    stored_entry: dict[str, Any] | None = None
    packet: IntelligencePacket | None = None
    prompt: str | None = None


class EmotionPoint(BaseModel):
    date: str
    emotion: str
    count: int


class DashboardSummaryResponse(BaseModel):
    range: str
    lookback_days: int
    emotion_over_time: list[EmotionPoint] = Field(default_factory=list)
    pattern_summary: PatternSummary
    recurring_topics: dict[str, int] = Field(default_factory=dict)
    triggers: list[TriggerStat] = Field(default_factory=list)
    habits: list[HabitCorrelation] = Field(default_factory=list)
    relationships: list[RelationshipProfile] = Field(default_factory=list)
    insights: list[str] = Field(default_factory=list)


class GoalsResponse(BaseModel):
    goals: list[GoalProgress] = Field(default_factory=list)


class PredictionsResponse(BaseModel):
    sentiment_forecast: SentimentForecast
    burnout_risk: BurnoutRisk


class TimelineResponse(BaseModel):
    events: list[TimelineEvent] = Field(default_factory=list)


class GrowthResponse(BaseModel):
    snapshots: list[GrowthSnapshot] = Field(default_factory=list)
    narrative: str


class GraphQueryResponse(BaseModel):
    node: str
    summary: str
    neighbors: list[str] = Field(default_factory=list)
    edge_data: list[dict[str, Any]] = Field(default_factory=list)


class GraphPeopleNode(BaseModel):
    id: str
    label: str
    type: str
    relationship_type: str = "unknown"
    mention_count: int = 0


class GraphPeopleEdge(BaseModel):
    source: str
    target: str
    sentiment: float = 0.0
    weight: int = 0
    closeness_score: float = 0.0


class GraphPeopleResponse(BaseModel):
    nodes: list[GraphPeopleNode] = Field(default_factory=list)
    edges: list[GraphPeopleEdge] = Field(default_factory=list)


class DiagnosticsResponse(BaseModel):
    retrieval_precision: dict[str, Any]
    emotion_confidence: dict[str, Any]
    latency: dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "ok"
