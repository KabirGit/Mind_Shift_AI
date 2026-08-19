from __future__ import annotations

from backend.llm.prompt_builder import PromptBuilder

_EMOTION = {"emotion": "joy", "confidence": 0.8}
_MEMORIES = [
    {"metadata": {"text": "past entry", "emotion": "joy", "timestamp": "2026-01-01T00:00:00Z"}}
]
_HISTORY = [{"role": "user", "content": "hi"}]


def test_prompt_without_insights_is_baseline():
    pb = PromptBuilder()
    baseline = pb.build(
        user_text="hello",
        current_emotion=_EMOTION,
        retrieved_memories=_MEMORIES,
        recent_history=_HISTORY,
    )
    none_variant = pb.build(
        user_text="hello",
        current_emotion=_EMOTION,
        retrieved_memories=_MEMORIES,
        recent_history=_HISTORY,
        insights=None,
    )
    empty_variant = pb.build(
        user_text="hello",
        current_emotion=_EMOTION,
        retrieved_memories=_MEMORIES,
        recent_history=_HISTORY,
        insights=[],
    )
    # Byte-identical when insights are absent/empty.
    assert baseline == none_variant == empty_variant
    assert "Long-term patterns you've noticed about yourself:" not in baseline


def test_prompt_with_reflection_adds_section():
    pb = PromptBuilder()
    prompt = pb.build(
        user_text="hello",
        current_emotion=_EMOTION,
        retrieved_memories=_MEMORIES,
        recent_history=_HISTORY,
        reflection_prompts=["Is there an exception, even a small one?"],
    )
    assert "Optional reflective questions" in prompt
    assert "- Is there an exception, even a small one?" in prompt
    # Reflection section sits before the past-entries section.
    assert prompt.index("Optional reflective questions") < prompt.index(
        "Relevant past entries:"
    )


def test_prompt_without_reflection_is_baseline():
    pb = PromptBuilder()
    baseline = pb.build(
        user_text="hello",
        current_emotion=_EMOTION,
        retrieved_memories=_MEMORIES,
        recent_history=_HISTORY,
        insights=["some insight"],
    )
    with_none = pb.build(
        user_text="hello",
        current_emotion=_EMOTION,
        retrieved_memories=_MEMORIES,
        recent_history=_HISTORY,
        insights=["some insight"],
        reflection_prompts=None,
    )
    assert baseline == with_none
    assert "Optional reflective questions" not in baseline


def test_prompt_with_insights_adds_section():
    pb = PromptBuilder()
    prompt = pb.build(
        user_text="hello",
        current_emotion=_EMOTION,
        retrieved_memories=_MEMORIES,
        recent_history=_HISTORY,
        insights=["You've mentioned career 3 times."],
    )
    assert "Long-term patterns you've noticed about yourself:" in prompt
    assert "- You've mentioned career 3 times." in prompt
    # New section sits before the past-entries section.
    assert prompt.index("Long-term patterns") < prompt.index("Relevant past entries:")
