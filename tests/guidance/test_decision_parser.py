from __future__ import annotations

from backend.guidance.decision_parser import DecisionParser
from backend.guidance.models import DecisionState, SuggestedOption


class FakeLLM:
    def __init__(self, response: str = "", error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls = 0

    def generate(self, prompt: str) -> str:
        self.calls += 1
        if self.error:
            raise self.error
        return self.response


VALID_JSON = """{
  "is_decision": true,
  "problem": "Choose between two job offers",
  "desired_outcome": "Find sustainable career growth",
  "available_options": ["Join Alpha", "Join Beta"],
  "fears": ["Choosing the wrong role"],
  "constraints": ["Cannot relocate"],
  "relevant_goals": ["Career growth"],
  "missing_information": ["Manager expectations"],
  "uncertainty": 0.3,
  "suggested_options": [{
    "title": "Ask both managers follow-up questions",
    "description": "Clarify the unknowns first",
    "benefits": ["Better evidence"],
    "costs": ["Takes another day"]
  }],
  "confidence": 0.9
}"""


def test_normal_journaling_does_not_call_llm():
    llm = FakeLLM(VALID_JSON)
    result = DecisionParser(llm).parse("I had a tiring day at work.")

    assert result.is_decision is False
    assert result.llm_called is False
    assert llm.calls == 0


def test_valid_json_builds_bounded_state_and_preserves_option_provenance():
    llm = FakeLLM(f"```json\n{VALID_JSON}\n```")
    result = DecisionParser(llm).parse(
        "I have two job offers and cannot decide.",
        emotion={"emotion": "fear", "confidence": 0.82},
        extracted={"topics": ["career"]},
    )

    assert result.is_decision is True
    assert result.used_fallback is False
    assert result.state is not None
    assert result.state.current_emotion == "fear"
    assert result.state.available_options == ["Join Alpha", "Join Beta"]
    assert result.suggested_options[0].source == "generated"
    assert llm.calls == 1


def test_implicit_irreversible_action_is_a_decision_candidate():
    assert DecisionParser.is_decision_candidate(
        "I got criticised today and now I want to quit."
    )


def test_invalid_json_uses_deterministic_fallback():
    result = DecisionParser(FakeLLM("not json")).parse(
        "Should I accept the offer or keep searching?",
        emotion={"emotion": "nervousness", "confidence": 0.7},
    )

    assert result.is_decision is True
    assert result.used_fallback is True
    assert result.state is not None
    assert result.state.available_options == ["accept the offer", "keep searching"]
    assert result.confidence == 0.45


def test_llm_failure_uses_fallback_and_marks_missing_information():
    result = DecisionParser(FakeLLM(error=RuntimeError("offline"))).parse(
        "I want to resign after today's feedback.",
        emotion={"emotion": "fear", "confidence": 0.9},
    )

    assert result.is_decision is True
    assert result.used_fallback is True
    assert result.state is not None
    assert "at least two realistic alternatives" in result.state.missing_information
    assert result.suggested_options[0].source == "fallback"


def test_ambiguous_help_request_routes_but_retains_uncertainty():
    result = DecisionParser(FakeLLM("broken")).parse("What should I do?")

    assert result.is_decision is True
    assert result.state is not None
    assert result.state.uncertainty >= 0.7


def test_state_and_option_strings_are_clipped_to_contract_bounds():
    state = DecisionState(
        problem="Choose",
        available_options=[f"Option {index}" for index in range(6)],
    )
    option = SuggestedOption(title="x" * 200)

    assert len(state.available_options) == 4
    assert len(option.title) == 160
