from __future__ import annotations

from types import SimpleNamespace

from backend.context.decision_context import DecisionContextBuilder
from backend.guidance.models import DecisionState
from backend.profile.models import UserProfile


class FakeRetriever:
    def __init__(self, results=None, fail: bool = False):
        self.results = results or []
        self.fail = fail

    def retrieve(self, query, query_emotion="neutral", top_k=5):
        if self.fail:
            raise RuntimeError("retrieval unavailable")
        return self.results


class FakeDB:
    def __init__(self, records=None, fail: bool = False):
        self.records = records or []
        self.fail = fail

    def get_recent(self, limit=30):
        if self.fail:
            raise RuntimeError("db unavailable")
        return self.records[:limit]


class FakePatternEngine:
    def __init__(self, summary=None, fail: bool = False):
        self.summary = summary or SimpleNamespace(recurring_emotions={}, triggers=[])
        self.fail = fail

    def analyze(self, lookback_days=90):
        if self.fail:
            raise RuntimeError("patterns unavailable")
        return self.summary


class FakeProfileManager:
    def __init__(self, fail: bool = False):
        self.fail = fail

    def update(self):
        if self.fail:
            raise RuntimeError("profile unavailable")
        return UserProfile(entry_count=4, top_triggers=["career"])


class FakeGoalEngine:
    def __init__(self, goals=None, fail: bool = False):
        self.goals = goals or []
        self.fail = fail

    def analyze(self, lookback_days=90):
        if self.fail:
            raise RuntimeError("goals unavailable")
        return self.goals


def _state() -> DecisionState:
    return DecisionState(
        problem="Should I resign from my career job or ask for clearer feedback?",
        desired_outcome="Make a sustainable career decision",
        available_options=["Resign", "Ask for feedback"],
        constraints=["Need stable income"],
        current_emotion="fear",
        emotion_confidence=0.8,
    )


def _builder(**overrides):
    defaults = {
        "retriever": FakeRetriever(),
        "journal_db": FakeDB(),
        "pattern_engine": FakePatternEngine(),
        "profile_manager": FakeProfileManager(),
        "goal_engine": FakeGoalEngine(),
    }
    defaults.update(overrides)
    return DecisionContextBuilder(**defaults)


def test_current_message_is_excluded_from_similar_memories():
    current = "Should I resign from my career job or ask for clearer feedback?"
    prior = "Last month feedback at work made me want to resign."
    results = [
        {"metadata": {"text": current}, "scores": {"combined": 1.0}},
        {
            "metadata": {
                "text": prior,
                "emotion": "fear",
                "timestamp": "2026-07-01T00:00:00Z",
                "topics": ["career"],
            },
            "scores": {"combined": 0.82},
        },
    ]

    found = _builder(retriever=FakeRetriever(results)).search_similar_memories(
        _state(), current_text=current
    )

    assert [item["metadata"]["text"] for item in found] == [prior]


def test_build_normalizes_patterns_profile_goals_and_history():
    trigger = SimpleNamespace(
        topic="career",
        dominant_emotion="fear",
        trend="stable",
        confidence=0.7,
        explanation="Career feedback often coincides with fear.",
    )
    summary = SimpleNamespace(recurring_emotions={"fear": 3}, triggers=[trigger])
    goal = SimpleNamespace(
        model_dump=lambda: {
            "goal_keyword": "career",
            "confidence": 0.8,
            "explanation": "Career growth mentioned repeatedly.",
        }
    )
    record = SimpleNamespace(
        text="Should I change jobs or stay?",
        timestamp="2026-06-01T00:00:00Z",
        emotion="fear",
        topics=["career"],
    )
    context = _builder(
        pattern_engine=FakePatternEngine(summary),
        goal_engine=FakeGoalEngine([goal]),
        journal_db=FakeDB([record]),
    ).build(_state(), current_text=_state().problem)

    assert context.emotional_context.recurring_emotions == {"fear": 3}
    assert context.relevant_patterns[0].source == "pattern"
    assert context.goals[0].source == "goal"
    assert context.recent_decisions[0].source == "recent_decision"
    assert context.profile["entry_count"] == 4
    assert context.evidence_strength > 0.35


def test_empty_history_is_explicit_and_never_fabricated():
    context = _builder().build(_state(), current_text=_state().problem)

    assert context.similar_memories == []
    assert context.recent_decisions == []
    assert "relevant personal history" in context.missing_context
    assert not any("previous" in item.summary.lower() for item in context.relevant_patterns)


def test_partial_service_failures_degrade_to_missing_context():
    context = _builder(
        retriever=FakeRetriever(fail=True),
        journal_db=FakeDB(fail=True),
        pattern_engine=FakePatternEngine(fail=True),
        profile_manager=FakeProfileManager(fail=True),
        goal_engine=FakeGoalEngine(fail=True),
    ).build(_state(), current_text=_state().problem)

    assert context.similar_memories == []
    assert context.profile is None
    assert context.goals == []
    assert context.missing_context
