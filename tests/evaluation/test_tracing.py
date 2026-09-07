from __future__ import annotations

import json

from backend.evaluation.tracing import RequestTrace, append_trace
from backend.guidance.models import (
    AgentResult,
    ContextSelectionReport,
    DecisionContext,
    DecisionState,
    EmotionalContext,
    ToolObservation,
)
from backend.llm.models import LLMCallResult, TokenUsage


def _context() -> DecisionContext:
    state = DecisionState(problem="private decision")
    return DecisionContext(
        current_decision=state,
        emotional_context=EmotionalContext(),
    )


def test_trace_aggregates_model_agent_context_and_tokens_without_content(tmp_path) -> None:
    trace = RequestTrace("trace-1")
    trace.record_llm(
        LLMCallResult(
            text="private model response",
            success=True,
            purpose="decision_parse",
            prompt_version="decision-v2",
            requested_model="requested",
            actual_model="actual",
            attempts=2,
            retry_count=1,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )
    )
    trace.record_agent(
        AgentResult(
            context=_context(),
            tools_called=["search_similar_memories"],
            observations=[
                ToolObservation(
                    tool="search_similar_memories",
                    success=True,
                    result_count=2,
                    duration_ms=4.5,
                )
            ],
            agent_steps=1,
            remaining_call_budget=2,
        )
    )
    trace.record_context(
        ContextSelectionReport(
            included={"memories": 2}, estimated_tokens=400, budget_tokens=1500
        )
    )
    trace.memories_retrieved = 2
    record = trace.build(mode="guidance", status="success", outcome="recommended")
    path = tmp_path / "trace.jsonl"

    assert append_trace(str(path), record) is True
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["logical_llm_calls"] == 1
    assert saved["provider_attempts"] == 2
    assert saved["retry_count"] == 1
    assert saved["token_usage"]["total_tokens"] == 15
    assert saved["context_selection"]["included"] == {"memories": 2}
    assert saved["tool_observations"][0]["result_count"] == 2
    assert "private decision" not in path.read_text(encoding="utf-8")
    assert "private model response" not in path.read_text(encoding="utf-8")


def test_failed_model_sets_sanitized_failure_category() -> None:
    trace = RequestTrace("trace-2")
    trace.record_llm(
        LLMCallResult(
            purpose="reflection",
            prompt_version="reflection-v1",
            attempts=1,
            failure_category="authentication",
            error_type="HTTPError",
        )
    )

    record = trace.build(mode="reflection", status="degraded", outcome="fallback")

    assert record.failure_category == "authentication"
    assert record.model_calls[0].failure_category == "authentication"


def test_trace_append_failure_is_non_fatal(tmp_path) -> None:
    record = RequestTrace("trace-3").build(
        mode="reflection", status="success", outcome="reflected"
    )

    assert append_trace(str(tmp_path), record) is False
