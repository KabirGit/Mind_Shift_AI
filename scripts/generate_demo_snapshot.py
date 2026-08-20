"""Generate recruiter-facing static demo JSON.

The generated files are served by /api/demo/* endpoints and never depend on
Render's ephemeral SQLite/FAISS state at request time.

Usage:
    python scripts/generate_demo_snapshot.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv  # noqa: E402

from backend.analytics.dashboard_story import DashboardStoryComposer  # noqa: E402
from backend.analytics.goal_engine import GoalEngine  # noqa: E402
from backend.analytics.growth_tracker import GrowthTracker  # noqa: E402
from backend.analytics.habit_engine import HabitEngine  # noqa: E402
from backend.analytics.insight_engine import InsightEngine  # noqa: E402
from backend.analytics.pattern_engine import PatternEngine  # noqa: E402
from backend.analytics.prediction_engine import PredictionEngine  # noqa: E402
from backend.analytics.relationship_engine import RelationshipEngine  # noqa: E402
from backend.analytics.temporal_engine import TemporalEngine  # noqa: E402
from backend.analytics.timeline_engine import TimelineEngine  # noqa: E402
from backend.api.schemas import (  # noqa: E402
    DashboardSummaryResponse,
    DiagnosticsResponse,
    EmotionPoint,
    GoalsResponse,
    GraphPeopleResponse,
    GraphQueryResponse,
    GrowthResponse,
    PredictionsResponse,
    TimelineResponse,
)
from backend.evaluation.eval_engine import EvalEngine  # noqa: E402
from backend.graph.knowledge_graph import KnowledgeGraph  # noqa: E402
from backend.llm.huggingface_client import HuggingFaceInferenceClient  # noqa: E402
from backend.profile.profile_manager import ProfileManager  # noqa: E402
from backend.storage.db import JournalDB  # noqa: E402
from backend.storage.models import JournalRecord  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent.parent / "backend" / "demo_data"


@dataclass(frozen=True)
class DemoEntry:
    days_ago: int
    text: str
    emotion: str
    emotion_confidence: float
    people: list[str]
    relationship_types: dict[str, str]
    topics: list[str]
    habits: list[str]
    sentiment: float


ENTRIES: list[DemoEntry] = [
    DemoEntry(29, "I started the month with a strong workout before work. The release plan feels big, but manageable.", "optimism", 0.86, [], {}, ["career", "health"], ["exercise"], 0.48),
    DemoEntry(28, "My manager Sarah liked the first project draft and said the promotion path is visible if I keep leading calmly.", "pride", 0.84, ["Sarah"], {"Sarah": "colleague"}, ["career"], [], 0.36),
    DemoEntry(27, "My friend Maya and I had dinner after a long day. It was easy to talk and I felt lighter afterward.", "joy", 0.88, ["Maya"], {"Maya": "friend"}, ["relationship"], [], 0.44),
    DemoEntry(26, "The deadline moved up. I skipped the gym and stayed at my desk late, which left me tense.", "stress", 0.82, [], {}, ["career", "health"], [], -0.25),
    DemoEntry(25, "I slept badly after thinking about the launch budget and rent. Money is starting to sit in the background.", "fear", 0.81, [], {}, ["money", "health"], ["sleep"], -0.36),
    DemoEntry(24, "Sarah asked for another revision. Her notes were fair, but I took them personally and felt deflated.", "sadness", 0.79, ["Sarah"], {"Sarah": "colleague"}, ["career"], [], -0.42),
    DemoEntry(23, "I went for a short run anyway. It did not fix work, but my body felt less braced by the end.", "relief", 0.83, [], {}, ["career", "health"], ["exercise"], 0.18),
    DemoEntry(22, "Maya checked in and I answered too quickly. I think I sounded distant, then felt guilty about it.", "remorse", 0.78, ["Maya"], {"Maya": "friend"}, ["relationship"], [], -0.28),
    DemoEntry(21, "The product meeting ran long. I drank too much coffee and could feel myself getting sharp.", "annoyance", 0.8, [], {}, ["career"], ["coffee"], -0.18),
    DemoEntry(20, "No workout again. I scrolled late while worrying about money, slept poorly, and woke up already behind.", "sadness", 0.82, [], {}, ["health", "money"], ["sleep", "social_media"], -0.46),
    DemoEntry(19, "Sarah and I disagreed about priorities. I left the call convinced I was failing the team.", "fear", 0.84, ["Sarah"], {"Sarah": "colleague"}, ["career"], [], -0.55),
    DemoEntry(18, "Maya invited me for a walk, but I cancelled because I felt too tired to explain myself.", "sadness", 0.79, ["Maya"], {"Maya": "friend"}, ["relationship", "health"], [], -0.38),
    DemoEntry(17, "I cooked dinner instead of ordering takeout. Small win, but it helped me feel less chaotic.", "relief", 0.8, [], {}, ["health"], ["cooking"], 0.16),
    DemoEntry(16, "The work backlog is real. I did not exercise and I can feel my patience thinning.", "anger", 0.77, [], {}, ["career", "health"], [], -0.33),
    DemoEntry(15, "I told Sarah I was overwhelmed and worried about the performance review. She was more understanding than I expected.", "realization", 0.81, ["Sarah"], {"Sarah": "colleague"}, ["career"], [], -0.12),
    DemoEntry(14, "Mid-month low point. I skipped the gym, worried about money, and avoided texting Maya back.", "sadness", 0.86, ["Maya"], {"Maya": "friend"}, ["money", "relationship", "health"], [], -0.62),
    DemoEntry(13, "I slept nine hours after putting my phone outside the bedroom. The morning felt quieter.", "relief", 0.84, [], {}, ["health"], ["sleep"], 0.24),
    DemoEntry(12, "Sarah helped me cut the project scope. I still felt behind, but at least there was a path.", "optimism", 0.82, ["Sarah"], {"Sarah": "colleague"}, ["career"], [], 0.08),
    DemoEntry(11, "Maya and I talked honestly. I apologized for disappearing and she said she understood.", "caring", 0.87, ["Maya"], {"Maya": "friend"}, ["relationship"], [], 0.34),
    DemoEntry(10, "First proper workout in days. My mood was noticeably steadier afterward.", "joy", 0.88, [], {}, ["health"], ["exercise"], 0.52),
    DemoEntry(9, "The budget still worries me, but I made a spreadsheet and stopped avoiding the numbers.", "realization", 0.79, [], {}, ["money"], [], -0.05),
    DemoEntry(8, "I did a morning run before the standup. Sarah noticed I was calmer during planning.", "pride", 0.86, ["Sarah"], {"Sarah": "colleague"}, ["career", "health"], ["exercise"], 0.46),
    DemoEntry(7, "Maya sent a voice note that made me laugh. The friendship feels repaired, not perfect, but warmer.", "joy", 0.89, ["Maya"], {"Maya": "friend"}, ["relationship"], [], 0.5),
    DemoEntry(6, "Work was still intense, yet I handled feedback without spiraling. That felt new.", "optimism", 0.84, ["Sarah"], {"Sarah": "colleague"}, ["career"], [], 0.28),
    DemoEntry(5, "Gym after work, simple dinner, early sleep. The combination made the day feel recoverable.", "relief", 0.87, [], {}, ["health"], ["exercise", "cooking", "sleep"], 0.58),
    DemoEntry(4, "I reviewed expenses and found a way to reduce two subscriptions. Money feels less foggy.", "optimism", 0.82, [], {}, ["money"], [], 0.22),
    DemoEntry(3, "Sarah approved the final project plan and said it supports my promotion case. I felt trusted again instead of just evaluated.", "pride", 0.9, ["Sarah"], {"Sarah": "colleague"}, ["career"], [], 0.62),
    DemoEntry(2, "Maya and I walked after dinner. We talked about stress without it taking over the whole evening.", "caring", 0.86, ["Maya"], {"Maya": "friend"}, ["relationship", "health"], ["exercise"], 0.49),
    DemoEntry(1, "This week feels better. Exercise is back, work is clearer, and I am not carrying everything alone.", "gratitude", 0.9, ["Sarah", "Maya"], {"Sarah": "colleague", "Maya": "friend"}, ["career", "relationship", "health"], ["exercise"], 0.68),
    DemoEntry(0, "I still have pressure around the launch, but I can see the pattern now: sleep, exercise, and honest conversations help.", "realization", 0.88, ["Sarah", "Maya"], {"Sarah": "colleague", "Maya": "friend"}, ["career", "relationship", "health"], ["exercise", "sleep"], 0.57),
]

CHAT_TURNS = [
    "I noticed I stopped exercising when work got intense. Why does it hit my mood so hard?",
    "I am nervous that Sarah's feedback means I am falling behind again.",
    "Maya and I are okay now, but I still feel bad that I disappeared when stressed.",
]


class _VectorProxy:
    def __init__(self, db: JournalDB) -> None:
        self.db = db

    def query(self, query_text: str, top_k: int = 3):
        query_terms = set(query_text.lower().split())
        ranked = []
        for idx, record in enumerate(self.db.get_all()):
            topic_overlap = len(query_terms & set(record.topics or []))
            text_overlap = len(query_terms & set((record.text or "").lower().split()))
            score = topic_overlap * 3 + text_overlap
            ranked.append(
                (
                    score,
                    {
                        "index": idx,
                        "distance": float(max(0, 10 - score)),
                        "metadata": {
                            "text": record.text,
                            "timestamp": record.timestamp,
                            "emotion": record.emotion,
                            "topics": record.topics,
                        },
                    },
                )
            )
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item for _, item in ranked[:top_k]]


class _DemoService:
    def __init__(self, db: JournalDB, vector_store: _VectorProxy) -> None:
        self.journal_db = db
        self.pattern_engine = PatternEngine(db)
        self.habit_engine = HabitEngine(db)
        self.relationship_engine = RelationshipEngine(db)
        self.insight_engine = InsightEngine(
            self.pattern_engine,
            habit_engine=self.habit_engine,
            relationship_engine=self.relationship_engine,
        )
        self.temporal_engine = TemporalEngine(db)
        self.prediction_engine = PredictionEngine(db)
        self.goal_engine = GoalEngine(db)
        self.timeline_engine = TimelineEngine(db)
        self.growth_tracker = GrowthTracker(db)
        self.profile_manager = ProfileManager(
            db,
            self.pattern_engine,
            self.habit_engine,
            self.relationship_engine,
        )
        self.knowledge_graph = KnowledgeGraph(db)
        self.eval_engine = EvalEngine(vector_store, db)


def _insert_entries(db: JournalDB) -> None:
    now = datetime.now(UTC)
    for i, entry in enumerate(ENTRIES):
        timestamp = (now - timedelta(days=entry.days_ago)).replace(
            hour=9 + (i % 10),
            minute=15,
            second=0,
            microsecond=0,
        ).isoformat().replace("+00:00", "Z")
        db.insert(
            JournalRecord(
                id=f"demo-{i:02d}",
                text=entry.text,
                timestamp=timestamp,
                emotion=entry.emotion,
                emotion_confidence=entry.emotion_confidence,
                entities_people=entry.people,
                topics=entry.topics,
                habits=entry.habits,
                keywords=_keywords(entry),
                person_relationship_types=entry.relationship_types,
                sentiment_compound=entry.sentiment,
                sentiment_valence=entry.sentiment,
            )
        )


def _keywords(entry: DemoEntry) -> list[str]:
    words = []
    for value in [*entry.topics, *entry.habits, *entry.people]:
        if value and value.lower() not in words:
            words.append(value.lower())
    return words[:10]


def _emotion_over_time(records) -> list[EmotionPoint]:
    counts: Counter[tuple[str, str]] = Counter()
    for record in records:
        date = (record.timestamp or "")[:10]
        if date:
            counts[(date, record.emotion or "neutral")] += 1
    return [
        EmotionPoint(date=date, emotion=emotion, count=count)
        for (date, emotion), count in sorted(counts.items())
    ]


def _dashboard_summary(service: _DemoService) -> DashboardSummaryResponse:
    lookback = 30
    records = service.journal_db.get_all()
    summary = service.pattern_engine.analyze(lookback_days=lookback)
    return DashboardSummaryResponse(
        range="Last 30 days",
        lookback_days=lookback,
        emotion_over_time=_emotion_over_time(records),
        pattern_summary=summary,
        recurring_topics=summary.recurring_topics,
        triggers=summary.triggers,
        habits=service.habit_engine.analyze(lookback_days=lookback),
        relationships=service.relationship_engine.analyze(lookback_days=lookback),
        insights=service.insight_engine.generate(lookback_days=lookback),
    )


def _chat_transcript(service: _DemoService, use_live_llm: bool) -> dict[str, Any]:
    messages = []
    history: list[dict[str, str]] = []
    client = _llm_client() if use_live_llm else None
    used_live_llm = False
    for turn in CHAT_TURNS:
        messages.append({"role": "user", "content": turn})
        response, live_response = _generate_chat_response(client, service, turn)
        used_live_llm = used_live_llm or live_response
        emotion = _emotion_for_text(turn)
        assistant = {
            "role": "assistant",
            "content": response,
            "emotion": emotion,
            "memory_replay": None,
            "crisis": {"flagged": False, "matched_terms": []},
            "retrieved_memories": service.eval_engine.vector_store.query(turn, top_k=3),
            "prompt": None,
        }
        messages.append(assistant)
        history.extend([
            {"role": "user", "content": turn},
            {"role": "assistant", "content": response},
        ])
    return {
        "mode": "demo",
        "persona": "Aarav, a product analyst navigating launch stress, exercise habits, and two recurring relationships.",
        "generated_with_live_llm": used_live_llm,
        "messages": messages,
    }


def _llm_client():
    load_dotenv()
    token = os.getenv("MISTRAL_API_KEY") or os.getenv("HF_API_TOKEN")
    if not token:
        return None
    model = os.getenv("MISTRAL_MODEL", os.getenv("HF_MODEL", "mistral-small"))
    return HuggingFaceInferenceClient(
        model_name=model,
        api_token=token,
        max_new_tokens=160,
        timeout_s=20,
        temperature=0.2,
    )


def _generate_chat_response(client, service: _DemoService, user_text: str) -> tuple[str, bool]:
    context = service.insight_engine.generate(lookback_days=30)[:3]
    prompt = (
        "Use the static Mind Shift AI demo persona and answer warmly in 2 short "
        "paragraphs. Ground the answer in these analytics insights: "
        f"{context}. User: {user_text}"
    )
    if client is not None:
        response = client.generate(prompt).strip()
        if response and response != "I'm here with you. Tell me more about what you're feeling.":
            return response, True
    lowered = user_text.lower()
    if "sarah" in lowered:
        fallback = (
            "The Sarah thread looks like it softened over time: earlier feedback landed "
            "as criticism, but later entries show clearer planning and trust returning. "
            "That suggests the pressure was real, but not the whole relationship.\n\n"
            "Before the next feedback moment, it may help to separate the task signal "
            "from the self-worth signal: what is she asking you to change, and what does "
            "that not say about your competence?"
        )
    elif "maya" in lowered:
        fallback = (
            "It makes sense that you still feel tender about that. The entries show you "
            "pulling back during the hardest stretch, then repairing with Maya through an "
            "honest conversation and a walk.\n\n"
            "The useful part is not that you handled it perfectly; it is that repair was "
            "possible. A small next step could be naming the pattern early, before stress "
            "turns into silence."
        )
    else:
        fallback = (
            "That pattern makes sense: when work pressure rose, the entries show exercise "
            "dropping off at the same time mood got heavier. It does not mean exercise is a "
            "magic fix, but it does look like one of your steadier supports.\n\n"
            "For today, I would treat it gently: choose the smallest version of the habit "
            "that still counts, then notice whether your body feels even a little less braced."
        )
    return fallback, False


def _emotion_for_text(text: str) -> dict[str, Any]:
    lowered = text.lower()
    if "nervous" in lowered or "falling behind" in lowered:
        return {
            "emotion": "fear",
            "confidence": 0.82,
            "all_emotions": [
                {"emotion": "fear", "score": 0.82},
                {"emotion": "realization", "score": 0.49},
            ],
        }
    if "bad" in lowered or "disappeared" in lowered:
        return {
            "emotion": "remorse",
            "confidence": 0.79,
            "all_emotions": [
                {"emotion": "remorse", "score": 0.79},
                {"emotion": "caring", "score": 0.55},
            ],
        }
    return {
        "emotion": "realization",
        "confidence": 0.84,
        "all_emotions": [
            {"emotion": "realization", "score": 0.84},
            {"emotion": "curiosity", "score": 0.54},
        ],
    }


def _dump(name: str, payload: Any) -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    path = DEMO_DIR / name
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json", exclude_none=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline-llm",
        action="store_true",
        help="Skip live Mistral calls for chat transcript generation.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="mind-shift-demo-",
        ignore_cleanup_errors=True,
    ) as tmp:
        db = JournalDB(str(Path(tmp) / "journal.db"))
        _insert_entries(db)
        service = _DemoService(db, _VectorProxy(db))
        graph = service.knowledge_graph.build(lookback_days=30)

        _dump("dashboard_summary.json", _dashboard_summary(service))
        _dump(
            "dashboard_story.json",
            DashboardStoryComposer(service).compose(
                range_label="Last 30 days",
                lookback_days=30,
            ),
        )
        _dump("goals.json", GoalsResponse(goals=service.goal_engine.analyze(lookback_days=90)))
        _dump(
            "predictions.json",
            PredictionsResponse(
                sentiment_forecast=service.prediction_engine.forecast_sentiment(),
                burnout_risk=service.prediction_engine.assess_burnout_risk(),
            ),
        )
        _dump("timeline.json", TimelineResponse(events=service.timeline_engine.build(lookback_days=30)))
        _dump(
            "growth.json",
            GrowthResponse(
                snapshots=service.growth_tracker.compute_snapshots(),
                narrative=service.growth_tracker.narrative(),
            ),
        )
        _dump(
            "diagnostics.json",
            DiagnosticsResponse(
                retrieval_precision=service.eval_engine.retrieval_precision_at_k(k=3),
                emotion_confidence=service.eval_engine.emotion_confidence_stats(),
                latency={"avg_ms": 0.0, "p95_ms": 0.0, "sample_count": 0},
            ),
        )
        _dump(
            "graph_people.json",
            GraphPeopleResponse(**service.knowledge_graph.people_graph(lookback_days=30)),
        )
        graph_queries = {}
        for node in ("User", "career", "exercise", "Maya", "Sarah"):
            result = service.knowledge_graph.query(graph, node)
            graph_queries[node.lower()] = GraphQueryResponse(
                node=node,
                summary=service.knowledge_graph.summarize_node(graph, node),
                neighbors=result.get("neighbors", []),
                edge_data=result.get("edge_data", []),
            ).model_dump(mode="json")
        _dump("graph_queries.json", graph_queries)
        _dump(
            "chat_transcript.json",
            _chat_transcript(service, use_live_llm=not args.offline_llm),
        )

    print(f"Wrote demo snapshot JSON to {DEMO_DIR}")


if __name__ == "__main__":
    main()
