from __future__ import annotations

from fastapi.testclient import TestClient

from backend.analytics.goal_engine import GoalProgress
from backend.analytics.growth_tracker import GrowthSnapshot
from backend.analytics.habit_engine import HabitCorrelation
from backend.analytics.models import PatternSummary, TriggerStat
from backend.analytics.prediction_engine import BurnoutRisk, SentimentForecast
from backend.analytics.relationship_engine import RelationshipProfile
from backend.analytics.timeline_engine import TimelineEvent
from backend.api.main import app, get_service
from backend.orchestrator.packet import IntelligencePacket
from backend.storage.models import JournalRecord

TS = "2026-08-19T10:00:00Z"


class FakeJournalDB:
    def __init__(self, records=None):
        self.records = records if records is not None else [_record()]

    def get_all(self):
        return self.records


class FakePatternEngine:
    def __init__(self, empty: bool = False):
        self.empty = empty

    def analyze(self, lookback_days: int = 30):
        if self.empty:
            return PatternSummary()
        return PatternSummary(
            recurring_emotions={"joy": 2},
            recurring_topics={"career": 2},
            recurring_people={"Alice": 2},
            period_entry_count=2,
            triggers=[
                TriggerStat(
                    topic="career",
                    frequency=2,
                    avg_sentiment=-0.2,
                    dominant_emotion="fear",
                    trend="stable",
                    confidence=0.2,
                    explanation="Career appears with heavier tone.",
                )
            ],
        )


class FakeHabitEngine:
    def __init__(self, empty: bool = False):
        self.empty = empty

    def analyze(self, lookback_days: int = 30):
        if self.empty:
            return []
        return [
            HabitCorrelation(
                habit="exercise",
                mention_count=2,
                avg_sentiment_when_mentioned=0.2,
                avg_sentiment_other_days=-0.1,
                delta=0.3,
                correlation_label="positive",
                confidence=0.2,
                explanation="Exercise seems helpful.",
            )
        ]


class FakeRelationshipEngine:
    def __init__(self, empty: bool = False):
        self.empty = empty

    def analyze(self, lookback_days: int = 30):
        if self.empty:
            return []
        return [
            RelationshipProfile(
                person="Alice",
                mention_count=2,
                avg_sentiment=0.1,
                dominant_emotion="joy",
                last_mentioned=TS,
                trend="stable",
                confidence=0.2,
                explanation="Alice appears supportive.",
            )
        ]


class FakeInsightEngine:
    def __init__(self, empty: bool = False):
        self.empty = empty

    def generate(self, lookback_days: int = 30):
        return [] if self.empty else ["Career pressure is the main theme."]


class FakeGoalEngine:
    def __init__(self, empty: bool = False):
        self.empty = empty

    def analyze(self, lookback_days: int = 90):
        if self.empty:
            return []
        return [
            GoalProgress(
                goal_keyword="job_search",
                first_mentioned=TS,
                last_mentioned=TS,
                mention_count=2,
                avg_sentiment=0.1,
                sentiment_trend="stable",
                estimated_progress=0.5,
                confidence=0.2,
                explanation="Job search mentioned twice.",
            )
        ]


class FakePredictionEngine:
    def forecast_sentiment(self):
        return SentimentForecast(
            horizon_days=7,
            predicted_sentiment=0.1,
            direction="stable",
            confidence=0.4,
            explanation="Based on sample entries.",
        )

    def assess_burnout_risk(self):
        return BurnoutRisk(
            risk_level="low",
            score=0.1,
            contributing_factors=[],
            confidence=0.4,
            explanation=(
                "Score 10% from 3 recent entries. "
                "This is a statistical pattern only, not a clinical assessment."
            ),
        )


class FakeTimelineEngine:
    def __init__(self, empty: bool = False):
        self.empty = empty

    def build(self, lookback_days: int = 90):
        if self.empty:
            return []
        return [
            TimelineEvent(
                timestamp=TS,
                title="career - feeling joy",
                description="A useful career day.",
                emotion="joy",
                sentiment=0.3,
                significance_score=0.3,
                event_type="normal",
            )
        ]


class FakeGrowthTracker:
    def __init__(self, empty: bool = False):
        self.empty = empty

    def compute_snapshots(self):
        if self.empty:
            return []
        return [
            GrowthSnapshot(
                period_label="2026-08",
                entry_count=2,
                avg_sentiment=0.2,
                dominant_emotion="joy",
                top_topic="career",
                snapshot_date=TS,
            )
        ]

    def narrative(self):
        return "Keep journaling - your growth story starts here."


class FakeKnowledgeGraph:
    def __init__(self, empty: bool = False):
        self.empty = empty

    def build(self, lookback_days: int = 90):
        return {}

    def query(self, graph, node: str):
        if self.empty:
            return {"neighbors": [], "edge_data": []}
        return {
            "neighbors": ["career"],
            "edge_data": [{"neighbor": "career", "weight": 2, "type": "topic"}],
        }

    def summarize_node(self, graph, node: str):
        if self.empty:
            return f"No connections found for '{node}'."
        return "'User' connects to: career."


class FakeReportGenerator:
    def generate(self, lookback_days: int = 30):
        return b"%PDF-1.4 fake"


class FakeEvalEngine:
    def retrieval_precision_at_k(self, k: int = 3):
        return {"precision_at_k": 1.0, "k": k, "n_samples_used": 1, "note": "ok"}

    def emotion_confidence_stats(self):
        return {
            "mean_confidence": 0.8,
            "min": 0.8,
            "max": 0.8,
            "low_confidence_ratio": 0.0,
        }

    def latency_summary(self):
        return {"avg_ms": 12.0, "p95_ms": 20.0, "sample_count": 2}


class FakeService:
    def __init__(self, empty: bool = False):
        self.empty = empty
        self.journal_db = FakeJournalDB([] if empty else [_record()])
        self.pattern_engine = FakePatternEngine(empty)
        self.habit_engine = FakeHabitEngine(empty)
        self.relationship_engine = FakeRelationshipEngine(empty)
        self.insight_engine = FakeInsightEngine(empty)
        self.goal_engine = FakeGoalEngine(empty)
        self.prediction_engine = FakePredictionEngine()
        self.timeline_engine = FakeTimelineEngine(empty)
        self.growth_tracker = FakeGrowthTracker(empty)
        self.knowledge_graph = FakeKnowledgeGraph(empty)
        self.report_generator = FakeReportGenerator()
        self.eval_engine = FakeEvalEngine()

    def run_pipeline(self, text, chat_history=None, top_k=3, tags=None):
        flagged = "suicide" in text.lower()
        packet = IntelligencePacket(
            current_entry_emotion="fear" if flagged else "joy",
            current_entry_sentiment=-0.7 if flagged else 0.2,
            insights=["Breathe and name the pressure."],
            reflection_prompts=["What is one controllable next step?"],
            proactive_alerts=["A gentle alert."],
            memory_replay={"days_ago": 3, "similar_entry_emotion": "fear"},
        )
        return {
            "emotion": {
                "emotion": packet.current_entry_emotion,
                "confidence": 0.9,
                "all_emotions": [{"emotion": "joy", "score": 0.8}],
            },
            "stored_entry": {"text": text, "emotion": packet.current_entry_emotion},
            "retrieved_memories": [{"metadata": {"text": "prior note"}}],
            "prompt": "constructed prompt",
            "response": "A grounded response.",
            "crisis": {
                "flagged": flagged,
                "matched_terms": ["suicide"] if flagged else [],
            },
            "packet": packet,
        }


def _record():
    return JournalRecord(
        id="r1",
        text="Career felt better after exercise with Alice.",
        timestamp=TS,
        emotion="joy",
        emotion_confidence=0.8,
        entities_people=["Alice"],
        topics=["career"],
        habits=["exercise"],
        sentiment_compound=0.2,
        sentiment_valence=0.2,
    )


def _client(service: FakeService):
    app.dependency_overrides[get_service] = lambda: service
    return TestClient(app)


def teardown_function():
    app.dependency_overrides.clear()


def test_health_endpoint():
    client = _client(FakeService())
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_happy_path():
    client = _client(FakeService())
    response = client.post("/api/chat", json={"text": "I had a good day"})

    assert response.status_code == 200
    body = response.json()
    assert body["emotion"]["emotion"] == "joy"
    assert body["response"] == "A grounded response."
    assert body["memory_replay"]["days_ago"] == 3
    assert body["crisis"]["flagged"] is False
    assert body["retrieved_memories"][0]["metadata"]["text"] == "prior note"
    assert body["packet"]["reflection_prompts"]


def test_chat_endpoint_allows_pages_cors_preflight():
    client = _client(FakeService())
    response = client.options(
        "/api/chat",
        headers={
            "Origin": "https://preview.mind-shift-ai.pages.dev",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == (
        "https://preview.mind-shift-ai.pages.dev"
    )
    assert "POST" in response.headers["access-control-allow-methods"]
    assert "content-type" in response.headers["access-control-allow-headers"]


def test_chat_endpoint_rejects_empty_text():
    client = _client(FakeService(empty=True))
    response = client.post("/api/chat", json={"text": ""})
    assert response.status_code == 422


def test_chat_endpoint_crisis_flag_pass_through():
    client = _client(FakeService())
    response = client.post("/api/chat", json={"text": "I am thinking about suicide"})

    assert response.status_code == 200
    crisis = response.json()["crisis"]
    assert crisis["flagged"] is True
    assert crisis["matched_terms"] == ["suicide"]
    assert "988" in crisis["resources"]


def test_dashboard_and_support_endpoints_happy_path():
    client = _client(FakeService())

    summary = client.get("/api/dashboard/summary?range=Last%2030%20days")
    assert summary.status_code == 200
    summary_body = summary.json()
    assert summary_body["lookback_days"] == 30
    assert summary_body["emotion_over_time"][0]["emotion"] == "joy"
    assert summary_body["triggers"][0]["confidence"] == 0.2
    assert summary_body["habits"][0]["confidence"] == 0.2
    assert summary_body["relationships"][0]["confidence"] == 0.2
    assert summary_body["insights"] == ["Career pressure is the main theme."]

    goals = client.get("/api/dashboard/goals")
    assert goals.status_code == 200
    assert goals.json()["goals"][0]["goal_keyword"] == "job_search"

    predictions = client.get("/api/dashboard/predictions")
    assert predictions.status_code == 200
    assert (
        "This is a statistical pattern only, not a clinical assessment."
        in predictions.json()["burnout_risk"]["explanation"]
    )

    timeline = client.get("/api/dashboard/timeline")
    assert timeline.status_code == 200
    assert timeline.json()["events"][0]["event_type"] == "normal"

    growth = client.get("/api/dashboard/growth")
    assert growth.status_code == 200
    assert growth.json()["snapshots"][0]["period_label"] == "2026-08"

    graph = client.get("/api/graph/query?node=User")
    assert graph.status_code == 200
    assert graph.json()["neighbors"] == ["career"]

    report = client.get("/api/report/weekly")
    assert report.status_code == 200
    assert report.headers["content-type"] == "application/pdf"
    assert report.content.startswith(b"%PDF")

    diagnostics = client.get("/api/diagnostics")
    assert diagnostics.status_code == 200
    assert diagnostics.json()["latency"]["sample_count"] == 2


def test_dashboard_and_support_endpoints_empty_data_path():
    client = _client(FakeService(empty=True))

    assert client.get("/api/dashboard/summary").json()["emotion_over_time"] == []
    assert client.get("/api/dashboard/summary").json()["triggers"] == []
    assert client.get("/api/dashboard/goals").json()["goals"] == []
    assert client.get("/api/dashboard/timeline").json()["events"] == []
    assert client.get("/api/dashboard/growth").json()["snapshots"] == []

    graph = client.get("/api/graph/query?node=missing")
    assert graph.status_code == 200
    assert graph.json()["neighbors"] == []

    report = client.get("/api/report/weekly")
    assert report.status_code == 200
    assert report.content.startswith(b"%PDF")

    diagnostics = client.get("/api/diagnostics")
    assert diagnostics.status_code == 200
    assert diagnostics.json()["retrieval_precision"]["precision_at_k"] == 1.0
