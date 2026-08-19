from __future__ import annotations

from datetime import UTC, datetime, timedelta

from backend.analytics.habit_engine import HabitEngine
from backend.analytics.insight_engine import InsightEngine
from backend.analytics.pattern_engine import PatternEngine
from backend.analytics.relationship_engine import RelationshipEngine
from backend.reports.report_generator import ReportGenerator
from backend.storage.db import JournalDB
from backend.storage.models import JournalRecord


def _ts(days_ago: int) -> str:
    return (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
        "+00:00", "Z"
    )


def _gen(db: JournalDB) -> ReportGenerator:
    pe = PatternEngine(db)
    he = HabitEngine(db)
    re_ = RelationshipEngine(db)
    ie = InsightEngine(pe, habit_engine=he, relationship_engine=re_)
    return ReportGenerator(pe, he, re_, ie)


def _seed(db: JournalDB) -> None:
    for i in range(4):
        db.insert(
            JournalRecord(
                id=f"r{i}", text="work and gym", timestamp=_ts(5 - i),
                emotion="joy" if i % 2 else "sadness", emotion_confidence=0.9,
                topics=["career"], habits=["exercise"], entities_people=["Alice"],
                sentiment_compound=0.1 * i,
            )
        )


def test_report_bytes_pdf_header(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    _seed(db)
    data = _gen(db).generate(lookback_days=30)
    assert isinstance(data, bytes)
    assert len(data) > 0
    assert data[:4] == b"%PDF"


def test_report_writes_to_disk(tmp_path):
    db = JournalDB(str(tmp_path / "j.db"))
    _seed(db)
    out = tmp_path / "report.pdf"
    data = _gen(db).generate(lookback_days=30, out_path=str(out))
    assert out.exists()
    assert out.read_bytes()[:4] == b"%PDF"
    assert out.read_bytes() == data


def test_report_empty_db_still_valid(tmp_path):
    db = JournalDB(str(tmp_path / "empty.db"))
    data = _gen(db).generate(lookback_days=7)
    assert data[:4] == b"%PDF"


def test_report_contains_story_and_predictions(tmp_path):
    from backend.analytics.growth_tracker import GrowthTracker
    from backend.analytics.prediction_engine import PredictionEngine
    from backend.reports.report_generator import ReportGenerator

    db = JournalDB(str(tmp_path / "j.db"))
    _seed(db)
    pe = PatternEngine(db)
    gen = ReportGenerator(
        pe, HabitEngine(db), RelationshipEngine(db),
        InsightEngine(pe),
        growth_tracker=GrowthTracker(db),
        prediction_engine=PredictionEngine(db),
    )
    data = gen.generate(lookback_days=30)
    text = data.decode("latin-1", "ignore")
    assert "Your Story This Week" in text
    assert "Predictions" in text
