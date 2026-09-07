from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from backend.llm.models import LLMCallResult


def _clean_text(value: object, *, limit: int) -> str:
    return " ".join(str(value or "").strip().split())[:limit]


class SuggestedOption(BaseModel):
    """A possible option proposed during parsing, with explicit provenance."""

    model_config = ConfigDict(extra="ignore")

    title: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=400)
    benefits: list[str] = Field(default_factory=list, max_length=4)
    costs: list[str] = Field(default_factory=list, max_length=4)
    source: Literal["user", "generated", "fallback"] = "generated"

    @field_validator("title", mode="before")
    @classmethod
    def _clean_title(cls, value: object) -> str:
        return _clean_text(value, limit=160)

    @field_validator("description", mode="before")
    @classmethod
    def _clean_description(cls, value: object) -> str:
        return _clean_text(value, limit=400)

    @field_validator("benefits", "costs", mode="before")
    @classmethod
    def _clean_lists(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [_clean_text(item, limit=180) for item in value if _clean_text(item, limit=180)][:4]


class DecisionState(BaseModel):
    """Validated, bounded representation of the user's current decision."""

    model_config = ConfigDict(extra="ignore")

    problem: str = Field(min_length=1, max_length=1000)
    desired_outcome: str | None = Field(default=None, max_length=400)
    available_options: list[str] = Field(default_factory=list, max_length=4)
    fears: list[str] = Field(default_factory=list, max_length=5)
    constraints: list[str] = Field(default_factory=list, max_length=5)
    relevant_goals: list[str] = Field(default_factory=list, max_length=5)
    current_emotion: str = Field(default="neutral", max_length=80)
    emotion_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_information: list[str] = Field(default_factory=list, max_length=5)
    uncertainty: float = Field(default=0.5, ge=0.0, le=1.0)

    @field_validator("problem", mode="before")
    @classmethod
    def _clean_problem(cls, value: object) -> str:
        return _clean_text(value, limit=1000)

    @field_validator("desired_outcome", mode="before")
    @classmethod
    def _clean_outcome(cls, value: object) -> str | None:
        cleaned = _clean_text(value, limit=400)
        return cleaned or None

    @field_validator("current_emotion", mode="before")
    @classmethod
    def _clean_emotion(cls, value: object) -> str:
        return _clean_text(value, limit=80).lower() or "neutral"

    @field_validator(
        "available_options",
        "fears",
        "constraints",
        "relevant_goals",
        "missing_information",
        mode="before",
    )
    @classmethod
    def _clean_string_lists(
        cls, value: object, info: ValidationInfo
    ) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            text = _clean_text(item, limit=300)
            key = text.casefold()
            if text and key not in seen:
                cleaned.append(text)
                seen.add(key)
        limit = 4 if info.field_name == "available_options" else 5
        return cleaned[:limit]


class DecisionParseResult(BaseModel):
    """Outcome of routing and parsing one message."""

    is_decision: bool = False
    state: DecisionState | None = None
    suggested_options: list[SuggestedOption] = Field(default_factory=list, max_length=4)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    used_fallback: bool = False
    llm_called: bool = False
    llm_result: LLMCallResult | None = Field(default=None, exclude=True)


class GuidanceResponseDraft(BaseModel):
    """Bounded structured draft for the user-facing guidance response."""

    model_config = ConfigDict(extra="ignore")

    validation: str = Field(min_length=1, max_length=300)
    recommendation: str = Field(min_length=1, max_length=500)
    why: list[str] = Field(min_length=1, max_length=4)
    uncertainty: list[str] = Field(min_length=1, max_length=4)
    next_actions: list[str] = Field(min_length=2, max_length=4)

    @field_validator("validation", "recommendation", mode="before")
    @classmethod
    def _clean_draft_text(cls, value: object) -> str:
        return _clean_text(value, limit=500)

    @field_validator("why", "uncertainty", "next_actions", mode="before")
    @classmethod
    def _clean_draft_list(cls, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return [
            _clean_text(item, limit=300)
            for item in value
            if _clean_text(item, limit=300)
        ][:4]


class MemoryEvidence(BaseModel):
    memory_id: str | None = None
    text: str = Field(max_length=500)
    timestamp: str | None = None
    emotion: str = "neutral"
    relevance: float = Field(default=0.0, ge=0.0, le=1.0)
    topics: list[str] = Field(default_factory=list, max_length=5)


class EvidenceItem(BaseModel):
    source: Literal["pattern", "profile", "goal", "recent_decision"]
    summary: str = Field(max_length=500)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EmotionalContext(BaseModel):
    current_emotion: str = "neutral"
    current_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    recurring_emotions: dict[str, int] = Field(default_factory=dict)


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=500)


class ContextSelectionReport(BaseModel):
    included: dict[str, int] = Field(default_factory=dict)
    deduplicated: dict[str, int] = Field(default_factory=dict)
    truncated: dict[str, int] = Field(default_factory=dict)
    dropped: dict[str, int] = Field(default_factory=dict)
    estimated_tokens: int = Field(default=0, ge=0)
    budget_tokens: int = Field(default=1500, ge=1)


class PromptContext(BaseModel):
    """Globally budgeted prompt-only context with explicit source boundaries."""

    decision: dict[str, Any]
    retrieval_evidence: list[MemoryEvidence] = Field(default_factory=list, max_length=3)
    relevant_patterns: list[EvidenceItem] = Field(default_factory=list, max_length=3)
    goals: list[EvidenceItem] = Field(default_factory=list, max_length=2)
    profile_summary: dict[str, Any] | None = None
    related_prior_decisions: list[EvidenceItem] = Field(default_factory=list, max_length=2)
    conversation_history: list[ConversationTurn] = Field(default_factory=list, max_length=6)
    selection: ContextSelectionReport


class DecisionContext(BaseModel):
    """Small evidence packet assembled from existing project services."""

    current_decision: DecisionState
    similar_memories: list[MemoryEvidence] = Field(default_factory=list, max_length=3)
    emotional_context: EmotionalContext
    relevant_patterns: list[EvidenceItem] = Field(default_factory=list, max_length=5)
    profile: dict[str, Any] | None = None
    goals: list[EvidenceItem] = Field(default_factory=list, max_length=3)
    recent_decisions: list[EvidenceItem] = Field(default_factory=list, max_length=3)
    evidence_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_context: list[str] = Field(default_factory=list)
    prompt_context: PromptContext | None = None


class CandidateOption(BaseModel):
    id: str
    title: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=400)
    benefits: list[str] = Field(default_factory=list, max_length=4)
    costs: list[str] = Field(default_factory=list, max_length=4)
    source: Literal["user", "generated", "fallback"]


class OptionScores(BaseModel):
    goal_alignment: float = Field(ge=0.0, le=1.0)
    historical_evidence: float = Field(ge=0.0, le=1.0)
    feasibility: float = Field(ge=0.0, le=1.0)
    reversibility: float = Field(ge=0.0, le=1.0)
    downside_containment: float = Field(ge=0.0, le=1.0)
    total: float = Field(ge=0.0, le=1.0)


class ReversibilityAssessment(BaseModel):
    option_id: str
    reversible: bool
    score: float = Field(ge=0.0, le=1.0)
    cost_to_reverse: Literal["low", "medium", "high"]
    cooling_off_required: bool = False
    rationale: str = Field(max_length=300)


class ScoredOption(BaseModel):
    option: CandidateOption
    scores: OptionScores
    reversibility: ReversibilityAssessment


class ActionStep(BaseModel):
    order: int = Field(ge=1, le=4)
    action: str = Field(min_length=1, max_length=300)
    purpose: str = Field(default="", max_length=300)


class GuidanceResult(BaseModel):
    options: list[ScoredOption] = Field(default_factory=list, max_length=4)
    recommended_option_id: str | None = None
    recommendation: str
    strong_recommendation: bool = False
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: list[str] = Field(default_factory=list, max_length=4)
    uncertainty: list[str] = Field(default_factory=list, max_length=8)
    next_actions: list[ActionStep] = Field(default_factory=list, max_length=4)


class ToolObservation(BaseModel):
    tool: str
    success: bool
    result_count: int = Field(default=0, ge=0)
    duration_ms: float = Field(default=0.0, ge=0.0)
    failure_category: str | None = None
    error: str | None = None


class AgentResult(BaseModel):
    context: DecisionContext
    tools_called: list[str] = Field(default_factory=list, max_length=3)
    observations: list[ToolObservation] = Field(default_factory=list, max_length=3)
    agent_steps: int = Field(default=0, ge=0, le=3)
    accumulated_evidence: dict[str, int] = Field(default_factory=dict)
    termination_reason: Literal[
        "sufficient_context", "max_calls", "no_additional_tool_needed"
    ] = "no_additional_tool_needed"
    remaining_call_budget: int = Field(default=3, ge=0, le=3)
