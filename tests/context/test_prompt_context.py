from __future__ import annotations

from backend.context.prompt_context import ContextPolicy, PromptContextPacker
from backend.guidance.models import DecisionState, EvidenceItem, MemoryEvidence


def _state() -> DecisionState:
    return DecisionState(
        problem="Should I remain in this role or request clearer feedback?",
        desired_outcome="Make a sustainable career decision",
        available_options=["Stay", "Request feedback"],
        constraints=["Stable income matters"],
        relevant_goals=["career growth"],
        current_emotion="fear",
    )


def test_context_is_globally_budgeted_capped_and_source_separated() -> None:
    memories = [
        MemoryEvidence(text=f"Distinct relevant career memory {index} " + "x" * 400)
        for index in range(6)
    ]
    patterns = [
        EvidenceItem(source="pattern", summary=f"Pattern {index}", confidence=0.7)
        for index in range(5)
    ]
    history = [
        {"role": "user", "content": f"Conversation turn {index} " + "y" * 300}
        for index in range(9)
    ]

    packed = PromptContextPacker().pack(
        state=_state(),
        memories=memories,
        patterns=patterns,
        goals=[],
        profile=None,
        recent_decisions=[],
        conversation_history=history,
    )

    assert packed.selection.estimated_tokens <= packed.selection.budget_tokens
    assert len(packed.retrieval_evidence) <= 3
    assert len(packed.relevant_patterns) <= 3
    assert len(packed.conversation_history) <= 6
    assert packed.selection.truncated["memories"] == 3
    assert packed.selection.truncated["conversation_history"] >= 3


def test_near_duplicates_are_removed_before_budget_allocation() -> None:
    memories = [
        MemoryEvidence(text="Feedback from my manager made me consider changing roles."),
        MemoryEvidence(text="Feedback from my manager made me consider changing roles!"),
        MemoryEvidence(text="I tested a different schedule before committing."),
    ]

    packed = PromptContextPacker().pack(
        state=_state(),
        memories=memories,
        patterns=[],
        goals=[],
        profile=None,
        recent_decisions=[],
    )

    assert len(packed.retrieval_evidence) == 2
    assert packed.selection.deduplicated["memories"] == 1


def test_profile_is_allowlisted_and_private_extra_fields_are_absent() -> None:
    packed = PromptContextPacker().pack(
        state=_state(),
        memories=[],
        patterns=[],
        goals=[],
        profile={
            "entry_count": 4,
            "dominant_emotion": "fear",
            "top_triggers": ["career"],
            "raw_entries": ["private journal text"],
            "arbitrary_secret": "never include",
        },
        recent_decisions=[],
    )

    assert packed.profile_summary == {
        "entry_count": 4,
        "dominant_emotion": "fear",
        "top_triggers": ["career"],
    }
    assert "private journal text" not in packed.model_dump_json()


def test_small_budget_drops_low_priority_context_not_the_decision() -> None:
    packed = PromptContextPacker(ContextPolicy(budget_tokens=200)).pack(
        state=_state(),
        memories=[MemoryEvidence(text="m" * 500)],
        patterns=[],
        goals=[],
        profile=None,
        recent_decisions=[],
        conversation_history=[{"role": "user", "content": "h" * 500}],
    )

    assert packed.decision["problem"] == _state().problem
    assert packed.selection.estimated_tokens <= 200
    assert packed.selection.dropped
