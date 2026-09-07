from __future__ import annotations

import logging
import time
from typing import Any, Literal

from backend.context.decision_context import DecisionContextBuilder
from backend.guidance.models import AgentResult, DecisionState, ToolObservation

logger = logging.getLogger(__name__)

_NEGATIVE_EMOTIONS = {
    "anger",
    "annoyance",
    "confusion",
    "disappointment",
    "fear",
    "grief",
    "nervousness",
    "sadness",
    "stress",
}


class GuidanceAgent:
    """Deterministic, bounded selector over four existing context capabilities."""

    ALLOWED_TOOLS = (
        "search_similar_memories",
        "get_emotional_patterns",
        "get_user_profile",
        "get_recent_decision_context",
    )
    MAX_TOOL_CALLS = 3
    EARLY_STOP_STRENGTH = 0.55

    def __init__(self, context_builder: DecisionContextBuilder) -> None:
        self.context_builder = context_builder

    def run(
        self,
        state: DecisionState,
        *,
        current_text: str,
        top_k: int = 3,
        recent_history: list[dict[str, str]] | None = None,
    ) -> AgentResult:
        results: dict[str, Any] = {}
        calls: list[str] = []
        observations: list[ToolObservation] = []

        self._observe(
            "search_similar_memories",
            state,
            current_text,
            top_k,
            results,
            calls,
            observations,
        )

        second = (
            "get_emotional_patterns"
            if state.fears or state.current_emotion in _NEGATIVE_EMOTIONS
            else "get_user_profile"
        )
        self._observe(
            second,
            state,
            current_text,
            top_k,
            results,
            calls,
            observations,
        )

        context = self._assemble(state, results, recent_history)
        termination_reason: Literal[
            "sufficient_context", "max_calls", "no_additional_tool_needed"
        ] = "sufficient_context"
        if context.evidence_strength < self.EARLY_STOP_STRENGTH:
            third = (
                "get_recent_decision_context"
                if context.evidence_strength < 0.35
                else "get_user_profile"
            )
            if third not in calls:
                self._observe(
                    third,
                    state,
                    current_text,
                    top_k,
                    results,
                    calls,
                    observations,
                )
                context = self._assemble(state, results, recent_history)
                termination_reason = (
                    "sufficient_context"
                    if context.evidence_strength >= self.EARLY_STOP_STRENGTH
                    else "max_calls"
                )
            else:
                termination_reason = "no_additional_tool_needed"

        accumulated_evidence = {
            observation.tool: observation.result_count
            for observation in observations
            if observation.success and observation.result_count
        }

        return AgentResult(
            context=context,
            tools_called=calls,
            observations=observations,
            agent_steps=len(calls),
            accumulated_evidence=accumulated_evidence,
            termination_reason=termination_reason,
            remaining_call_budget=self.MAX_TOOL_CALLS - len(calls),
        )

    def _assemble(
        self,
        state: DecisionState,
        results: dict[str, Any],
        recent_history: list[dict[str, str]] | None,
    ) -> Any:
        if recent_history is None:
            return self.context_builder.assemble(state, results)
        return self.context_builder.assemble(
            state, results, recent_history=recent_history
        )

    def _observe(
        self,
        name: str,
        state: DecisionState,
        current_text: str,
        top_k: int,
        results: dict[str, Any],
        calls: list[str],
        observations: list[ToolObservation],
    ) -> None:
        if name in calls or len(calls) >= self.MAX_TOOL_CALLS:
            return
        calls.append(name)
        started = time.perf_counter()
        try:
            value = self._execute(name, state, current_text=current_text, top_k=top_k)
            results[name] = value
            observations.append(
                ToolObservation(
                    tool=name,
                    success=True,
                    result_count=self._result_count(value),
                    duration_ms=(time.perf_counter() - started) * 1000,
                )
            )
        except Exception as exc:
            logger.info(
                "Guidance tool unavailable tool=%s error_type=%s",
                name,
                type(exc).__name__,
            )
            results[name] = self._empty_result(name)
            observations.append(
                ToolObservation(
                    tool=name,
                    success=False,
                    duration_ms=(time.perf_counter() - started) * 1000,
                    failure_category=self._failure_category(exc),
                    error=type(exc).__name__,
                )
            )

    def _execute(
        self,
        name: str,
        state: DecisionState,
        *,
        current_text: str,
        top_k: int,
    ) -> Any:
        if name not in self.ALLOWED_TOOLS:
            raise ValueError(f"Tool is not allowed: {name}")
        if name == "search_similar_memories":
            return self.context_builder.search_similar_memories(
                state, current_text=current_text, top_k=top_k
            )
        if name == "get_emotional_patterns":
            return self.context_builder.get_emotional_patterns(state)
        if name == "get_user_profile":
            return self.context_builder.get_user_profile(state)
        return self.context_builder.get_recent_decision_context(
            state, current_text=current_text
        )

    @staticmethod
    def _result_count(value: Any) -> int:
        if isinstance(value, list):
            return len(value)
        if isinstance(value, dict):
            return sum(bool(item) for item in value.values())
        return int(value is not None)

    @staticmethod
    def _empty_result(name: str) -> Any:
        if name == "get_emotional_patterns":
            return {"recurring_emotions": {}, "triggers": []}
        if name == "get_user_profile":
            return {"profile": None, "goals": []}
        return []

    @staticmethod
    def _failure_category(exc: Exception) -> str:
        if isinstance(exc, TimeoutError):
            return "timeout"
        if isinstance(exc, (ConnectionError, OSError)):
            return "unavailable"
        return "tool_error"
