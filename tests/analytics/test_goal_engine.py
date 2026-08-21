from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.goal_engine import GoalEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


def test_improving_goal_progress(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    vals = [(30, -0.3), (20, 0.1), (10, 0.4), (3, 0.6)]
    for i, (d, s) in enumerate(vals):
        db.insert(JournalRecord(
            id=f"g{i}",
            text="I am working on my job search and had a job interview today.",
            timestamp=_ts(d),
            emotion="joy",
            emotion_confidence=0.9,
            sentiment_compound=s,
        ))
    goals = GoalEngine(db).analyze(lookback_days=90)
    js = [g for g in goals if g.goal_keyword == "job_search"]
    assert js
    assert js[0].sentiment_trend == "improving"
    assert js[0].phase in {"starting", "ramping", "plateaued"}
    assert js[0].estimated_progress > 0.5


def test_single_mention_excluded(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    db.insert(JournalRecord(id="x", text="my exam is coming", timestamp=_ts(1),
                            emotion="fear", emotion_confidence=0.9,
                            sentiment_compound=-0.2))
    assert GoalEngine(db).analyze() == []


def test_incidental_third_party_mentions_are_not_user_goals(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    db.insert(JournalRecord(
        id="a",
        text="My cousin's course wraps up this week and she has an exam soon.",
        timestamp=_ts(5),
        emotion="neutral",
        emotion_confidence=0.8,
        sentiment_compound=0.1,
    ))
    db.insert(JournalRecord(
        id="b",
        text="She talked about graduation plans while I listened.",
        timestamp=_ts(2),
        emotion="neutral",
        emotion_confidence=0.8,
        sentiment_compound=0.2,
    ))

    assert all(goal.goal_keyword != "education" for goal in GoalEngine(db).analyze())


def test_first_person_fitness_goal_language_is_tracked(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    db.insert(JournalRecord(
        id="a",
        text="I am trying to keep my running streak going this month.",
        timestamp=_ts(8),
        emotion="determination",
        emotion_confidence=0.8,
        sentiment_compound=0.1,
    ))
    db.insert(JournalRecord(
        id="b",
        text="We are planning to sign up for a 10k, and my running feels steadier.",
        timestamp=_ts(1),
        emotion="joy",
        emotion_confidence=0.8,
        sentiment_compound=0.5,
    ))

    goals = GoalEngine(db).analyze()
    assert [goal.goal_keyword for goal in goals] == ["fitness"]
    assert goals[0].estimated_progress > 0.5


def test_zero_data(tmp_path):
    db = JournalDB(str(tmp_path / "empty.db"))
    assert GoalEngine(db).analyze() == []
