from __future__ import annotations

from backend.guidance.decision_parser import DecisionParser
from backend.guidance.decision_scorer import DecisionScorer
from backend.guidance.models import (
    CandidateOption,
    DecisionContext,
    DecisionParseResult,
    DecisionState,
    EmotionalContext,
    MemoryEvidence,
    SuggestedOption,
)
from backend.guidance.option_generator import OptionGenerator
from backend.guidance.reversibility import ReversibilityChecker
from backend.guidance.service import GuidanceService
from backend.llm.prompt_builder import PromptBuilder


class UnusedLLM:
    def generate(self, prompt: str) -> str:
        raise AssertionError("not expected")


def _state() -> DecisionState:
    return DecisionState(
        problem="I was criticised and must decide whether to resign or ask for feedback.",
        desired_outcome="Protect stable income and make a sustainable career choice",
        available_options=["Resign now", "Ask for specific feedback"],
        constraints=["I need stable income"],
        relevant_goals=["Sustainable career growth"],
        current_emotion="fear",
        emotion_confidence=0.9,
    )


def _context(strength: float = 0.75) -> DecisionContext:
    state = _state()
    return DecisionContext(
        current_decision=state,
        emotional_context=EmotionalContext(
            current_emotion="fear", current_confidence=0.9
        ),
        similar_memories=[
            MemoryEvidence(
                text="Asking for specific feedback previously made the work situation clearer.",
                emotion="fear",
                relevance=0.9,
                topics=["career"],
            )
        ],
        profile={"entry_count": 5},
        evidence_strength=strength,
    )


def test_option_generation_deduplicates_and_adds_safe_generic_alternative():
    state = _state().model_copy(update={"available_options": ["Resign", "resign"]})
    options = OptionGenerator().generate(state)

    assert len(options) == 2
    assert options[0].title == "Resign"
    assert options[1].source == "fallback"
    assert [option.id for option in options] == ["option_1", "option_2"]


def test_irreversible_option_requires_cooling_off_under_high_negative_emotion():
    option = OptionGenerator().generate(_state())[0]
    assessment = ReversibilityChecker().assess(option, _state())

    assert assessment.reversible is False
    assert assessment.cooling_off_required is True
    assert assessment.cost_to_reverse == "high"


def test_scoring_favors_reversible_information_gathering_over_resignation():
    options = OptionGenerator().generate(_state())
    checker = ReversibilityChecker()
    assessments = {option.id: checker.assess(option, _state()) for option in options}
    scored = DecisionScorer().score(options, _context(), assessments)

    assert scored[0].option.title == "Ask for specific feedback"
    assert scored[-1].scores.total <= 0.54


def test_guidance_service_returns_strong_grounded_recommendation_and_actions():
    parsed = DecisionParseResult(
        is_decision=True,
        state=_state(),
        suggested_options=[
            SuggestedOption(
                title="Ask for specific feedback",
                benefits=["Creates clearer evidence"],
                costs=["Requires a difficult conversation"],
            )
        ],
        confidence=0.9,
        llm_called=True,
    )
    service = GuidanceService(DecisionParser(UnusedLLM()))
    result = service.guide(parsed, _context())

    assert result.strong_recommendation is True
    assert result.recommended_option_id is not None
    assert result.evidence
    assert 2 <= len(result.next_actions) <= 4


def test_low_evidence_returns_clarification_plan_not_forced_choice():
    parsed = DecisionParseResult(is_decision=True, state=_state(), confidence=0.5)
    service = GuidanceService(DecisionParser(UnusedLLM()))
    result = service.guide(parsed, _context(strength=0.2))

    assert result.strong_recommendation is False
    assert result.recommended_option_id is None
    assert "not enough evidence" in result.recommendation.lower()
    assert result.next_actions


def test_blocked_irreversible_choice_requires_cooling_off_action():
    class IrreversibleOnly:
        def generate(self, state, suggested):
            return [
                CandidateOption(
                    id="option_1",
                    title="Resign now",
                    source="user",
                )
            ]

    parsed = DecisionParseResult(is_decision=True, state=_state(), confidence=0.9)
    service = GuidanceService(
        DecisionParser(UnusedLLM()), option_generator=IrreversibleOnly()
    )
    result = service.guide(parsed, _context())

    assert result.strong_recommendation is False
    assert result.options[0].reversibility.cooling_off_required is True
    assert "Wait at least 24 hours" in result.next_actions[0].action


def test_guidance_prompt_and_fallback_have_required_sections():
    parsed = DecisionParseResult(is_decision=True, state=_state())
    service = GuidanceService(DecisionParser(UnusedLLM()))
    result = service.guide(parsed, _context(strength=0.2))
    prompt = PromptBuilder().build_guidance(
        state=_state(), context=_context(strength=0.2), guidance=result, recent_history=[]
    )
    fallback = service.render_fallback(result)

    assert "Never invent memories" in prompt
    for heading in ("Recommendation:", "Why:", "Uncertainty:", "Next steps:"):
        assert heading in fallback
