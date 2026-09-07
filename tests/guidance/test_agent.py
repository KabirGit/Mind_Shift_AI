from __future__ import annotations

import pytest

from backend.guidance.agent import GuidanceAgent
from backend.guidance.models import DecisionContext, DecisionState, EmotionalContext


class FakeContextBuilder:
    def __init__(
        self,
        *,
        memories=True,
        patterns=True,
        profile=True,
        recent=True,
        fail_memory=False,
    ):
        self.memories = memories
        self.patterns = patterns
        self.profile = profile
        self.recent = recent
        self.fail_memory = fail_memory

    def search_similar_memories(self, state, *, current_text, top_k):
        if self.fail_memory:
            raise RuntimeError("retrieval unavailable")
        return [{"metadata": {"text": "past"}}] if self.memories else []

    def get_emotional_patterns(self, state):
        return {"triggers": [{"topic": "career"}]} if self.patterns else {}

    def get_user_profile(self, state):
        return {"profile": {"entry_count": 4}, "goals": []} if self.profile else {}

    def get_recent_decision_context(self, state, *, current_text):
        return [{"text": "past choice"}] if self.recent else []

    def assemble(self, state, results):
        has_memory = bool(results.get("search_similar_memories"))
        has_pattern = bool((results.get("get_emotional_patterns") or {}).get("triggers"))
        has_profile = bool((results.get("get_user_profile") or {}).get("profile"))
        has_recent = bool(results.get("get_recent_decision_context"))
        strength = 0.1
        if has_memory:
            strength += 0.3
        if has_pattern or has_profile:
            strength += 0.2
        if has_recent:
            strength += 0.2
        return DecisionContext(
            current_decision=state,
            emotional_context=EmotionalContext(
                current_emotion=state.current_emotion,
                current_confidence=state.emotion_confidence,
            ),
            evidence_strength=min(strength, 1.0),
        )


def _state(*, emotion="neutral", fears=None):
    return DecisionState(
        problem="Should I stay or leave?",
        available_options=["Stay", "Leave"],
        fears=fears or [],
        current_emotion=emotion,
        emotion_confidence=0.8,
    )


def test_fear_selects_emotional_patterns_and_stops_when_sufficient():
    agent = GuidanceAgent(FakeContextBuilder())
    result = agent.run(_state(emotion="fear", fears=["failure"]), current_text="now")
    assert result.tools_called == [
        "search_similar_memories",
        "get_emotional_patterns",
    ]
    assert result.agent_steps == 2
    assert result.termination_reason == "sufficient_context"
    assert result.remaining_call_budget == 1
    assert result.accumulated_evidence


def test_neutral_choice_uses_profile_second():
    agent = GuidanceAgent(FakeContextBuilder())
    result = agent.run(_state(), current_text="now")
    assert result.tools_called == ["search_similar_memories", "get_user_profile"]


def test_insufficient_evidence_uses_recent_decisions_third():
    agent = GuidanceAgent(
        FakeContextBuilder(memories=False, patterns=False, profile=False)
    )
    result = agent.run(_state(), current_text="now")
    assert result.tools_called == [
        "search_similar_memories",
        "get_user_profile",
        "get_recent_decision_context",
    ]
    assert result.agent_steps == 3
    assert result.termination_reason == "max_calls"
    assert result.remaining_call_budget == 0


def test_moderate_emotional_evidence_fetches_profile_third():
    agent = GuidanceAgent(FakeContextBuilder(patterns=False))
    result = agent.run(_state(emotion="fear"), current_text="now")
    assert result.tools_called[-1] == "get_user_profile"
    assert len(set(result.tools_called)) == result.agent_steps


def test_tool_failure_degrades_and_hard_limit_holds():
    agent = GuidanceAgent(
        FakeContextBuilder(
            memories=False,
            patterns=False,
            profile=False,
            recent=False,
            fail_memory=True,
        )
    )
    result = agent.run(_state(emotion="fear"), current_text="now")
    assert result.observations[0].success is False
    assert result.observations[0].error == "RuntimeError"
    assert result.observations[0].failure_category == "tool_error"
    assert result.observations[0].duration_ms >= 0
    assert result.agent_steps <= 3


def test_allow_list_rejects_unknown_tool():
    agent = GuidanceAgent(FakeContextBuilder())
    with pytest.raises(ValueError, match="not allowed"):
        agent._execute("delete_history", _state(), current_text="now", top_k=3)
