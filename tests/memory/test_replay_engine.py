from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.memory.replay_engine import ReplayEngine
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


class _FakeIndex:
    def __init__(self, n):
        self.ntotal = n


class _FakeStore:
    def __init__(self, n, results):
        self.index = _FakeIndex(n)
        self._results = results

    def query(self, text, top_k=5):
        return self._results


def test_replay_with_positive_next(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    # similar entry 10 days ago; next entry 2 days later positive.
    db.insert(JournalRecord(id="sim", text="I feel overwhelmed by work",
                            timestamp=_ts(10), emotion="fear",
                            emotion_confidence=0.9, sentiment_compound=-0.5))
    db.insert(JournalRecord(id="next", text="Work got better, I feel relieved",
                            timestamp=_ts(8), emotion="joy",
                            emotion_confidence=0.9, sentiment_compound=0.6))

    results = [{
        "metadata": {"text": "I feel overwhelmed by work", "emotion": "fear",
                     "timestamp": _ts(10)},
        "distance": 0.2,
        "scores": {"combined": 0.8},
    }]
    store = _FakeStore(n=10, results=results)
    replay = ReplayEngine(store, db).find_replay("overwhelmed at work", "fear")

    assert replay is not None
    assert replay.days_ago == 10
    assert replay.next_entry_sentiment == 0.6
    assert "improved" in replay.recovery_hint
    assert replay.confidence == 0.8


def test_returns_none_when_index_too_small(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    store = _FakeStore(n=4, results=[])
    assert ReplayEngine(store, db).find_replay("x", "joy") is None


def test_skips_last_48h(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    results = [{
        "metadata": {"text": "today entry", "emotion": "joy", "timestamp": _ts(0)},
        "distance": 0.1,
        "scores": {"combined": 0.9},
    }]
    store = _FakeStore(n=10, results=results)
    # Only candidate is within 48h -> filtered out -> None.
    assert ReplayEngine(store, db).find_replay("x", "joy") is None
