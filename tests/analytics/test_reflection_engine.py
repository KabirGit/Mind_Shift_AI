from __future__ import annotations

from backend.analytics.reflection_engine import ReflectionEngine

ABS_Q = (
    "You used some all-or-nothing language here — is there an exception to "
    "that, even a small one?"
)
CAT_Q = "What's the most likely outcome here, separate from the worst-case one?"
BLAME_Q = "What part of this was actually within your control, and what wasn't?"


def test_absolutist_detected():
    out = ReflectionEngine().detect("I always fail and no one ever helps me.")
    assert ABS_Q in out


def test_catastrophizing_detected():
    out = ReflectionEngine().detect("This is a complete disaster, everything is ruined.")
    assert CAT_Q in out


def test_self_blame_detected():
    out = ReflectionEngine().detect("It's all my fault, I'm such a failure.")
    assert BLAME_Q in out


def test_max_two_questions():
    # Matches all three patterns -> only first two (dict order) returned.
    text = "I always ruin everything and it's my fault, what a disaster."
    out = ReflectionEngine().detect(text)
    assert len(out) == 2
    assert out == [ABS_Q, CAT_Q]


def test_neutral_returns_empty():
    assert ReflectionEngine().detect("I had a calm and steady day today.") == []


def test_empty_returns_empty():
    assert ReflectionEngine().detect("") == []


def test_personalized_question_from_replay():
    eng = ReflectionEngine()
    replay = {"days_ago": 12, "next_entry_sentiment": 0.5}
    out = eng.detect("I feel a bit off today", replay=replay)
    assert any("12 days ago" in q for q in out)


def test_personalized_replaces_when_two_already():
    eng = ReflectionEngine()
    replay = {"days_ago": 9, "next_entry_sentiment": 0.4}
    # Text matches all 3 regex patterns -> would be 2 generic; personalized replaces one.
    text = "I always ruin everything and it's my fault, what a disaster."
    out = eng.detect(text, replay=replay)
    assert len(out) == 2
    assert any("9 days ago" in q for q in out)


def test_no_personalized_when_negative_outcome():
    eng = ReflectionEngine()
    replay = {"days_ago": 5, "next_entry_sentiment": -0.3}
    out = eng.detect("just a calm day", replay=replay)
    assert out == []
