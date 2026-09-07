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
    DemoJournalEntriesResponse,
    DemoJournalEntry,
    DiagnosticsResponse,
    EmotionPoint,
    GoalsResponse,
    GraphPeopleResponse,
    GraphQueryResponse,
    GrowthResponse,
    ObservabilityResponse,
    PredictionsResponse,
    TimelineResponse,
)
from backend.evaluation.eval_engine import EvalEngine  # noqa: E402
from backend.evaluation.observability import build_observability_snapshot  # noqa: E402
from backend.evaluation.tracing import TraceRecord  # noqa: E402
from backend.graph.knowledge_graph import KnowledgeGraph  # noqa: E402
from backend.llm.huggingface_client import HuggingFaceInferenceClient  # noqa: E402
from backend.profile.profile_manager import ProfileManager  # noqa: E402
from backend.storage.db import JournalDB  # noqa: E402
from backend.storage.models import JournalRecord  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent.parent / "backend" / "demo_data"
DEMO_ANCHOR = datetime(2026, 9, 6, 12, 0, tzinfo=UTC)


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

# Each base event receives a distinct reflective layer before it is inserted.
# This keeps the persona realistic enough to exercise retrieval, analytics,
# goals, relationships, and longitudinal change rather than using keyword-only
# one-liners.
DETAILS: dict[int, str] = {
    29: "I wrote down three priorities for the release and noticed that planning reduced the usual Sunday-night noise. My goal this month is to protect four movement sessions each week while still showing that I can lead the launch without becoming frantic.",
    28: "I felt proud, but I also caught myself immediately looking for what could go wrong. I asked Sarah which leadership behaviors would make the promotion case concrete, and she named clearer delegation, calmer status updates, and earlier risk escalation.",
    27: "We talked about how quickly I disappear into work when a deadline becomes personal. Maya did not try to fix it; she asked me to send a short message when I am overloaded instead of going silent, which felt both kind and practical.",
    26: "By late afternoon my shoulders were tight and I had eaten lunch at the laptop. I told myself that skipping one workout was efficient, but the evening felt restless and unfocused, so I want to watch whether this trade-off repeats.",
    25: "I kept calculating worst-case scenarios instead of checking the actual account balance. The fear was vague but physical, and I noticed it made the release feel more dangerous too. Tomorrow I will separate the work risk from the money facts.",
    24: "The first thought was that the promotion was slipping away, even though Sarah commented on the document rather than my ability. I saved the notes, took a ten-minute walk, and planned to sort them into factual changes and assumptions before replying.",
    23: "I only ran for eighteen minutes and almost dismissed it as too small to matter. Afterward I could read Sarah's comments without the same rush of defensiveness. That is useful evidence that movement changes how I approach pressure, even when it does not remove the problem.",
    22: "Maya's message was warm, but I treated it like another task competing for attention. I want to repair that quickly rather than wait until guilt turns into avoidance. A simple honest note would be more respectful than pretending I am fine.",
    21: "During the meeting I interrupted twice and later replayed both moments. The caffeine probably amplified an already tense morning, so I am going to cap coffee after lunch and prepare the decision points before the next planning session.",
    20: "The phone gave me an easy escape from the budget and the backlog, but it also kept my mind activated past midnight. I woke with less patience and more catastrophic thinking. Tonight I will charge it outside the bedroom and choose one financial task before scrolling.",
    19: "Sarah challenged the order of the launch milestones, and I heard it as evidence that I was not ready to lead. Looking back, she was asking for a clearer dependency map. I need to ask clarifying questions before turning disagreement into a verdict about myself.",
    18: "Cancelling gave short-term relief, then made the evening lonelier. I drafted a message explaining that I was depleted rather than uninterested. I am noticing that isolation feels protective in the moment but usually extends the difficult mood into the next day.",
    17: "Chopping vegetables and cleaning the kitchen gave the evening a beginning and an end. I did not solve the backlog, but I stopped carrying work into every room. Small routines may be useful because they restore a sense of sequence when everything feels urgent.",
    16: "I answered a teammate too abruptly and regretted it immediately. The pattern seems less about the person and more about accumulated sleep loss, missed movement, and unmade decisions. I blocked thirty minutes tomorrow to reduce the backlog before the first meeting.",
    15: "Saying the concern aloud was uncomfortable, but it replaced several imagined outcomes with specific information. Sarah said the review would focus on how I handled scope and communication, not whether the launch was flawless. I left with two concrete improvements instead of a vague threat.",
    14: "This was the clearest low point of the month: work, finances, health, and friendship all felt like one failure. I wrote down what was actually true and what I was predicting. The smallest next step is sleep, followed by one honest message to Maya.",
    13: "I expected to lie awake without the phone, but I fell asleep faster after reading for twenty minutes. The extra rest did not create instant optimism, yet I felt less reactive and more able to distinguish today's tasks from the whole future.",
    12: "We removed two low-value launch features and assigned a decision owner for each remaining risk. The workload became measurable rather than endless. I want to remember that asking for scope clarity is a leadership behavior, not an admission that I cannot cope.",
    11: "I told Maya that stress had made me withdraw and that the silence was not about her. She appreciated the directness and asked me not to wait for a perfect explanation next time. Repair felt lighter than the week of avoiding it.",
    10: "The session was moderate rather than ambitious: thirty minutes of strength work and a slow walk home. I concentrated better afterward and did not reread the same email repeatedly. Consistency appears more useful than waiting for enough energy to do an ideal workout.",
    9: "Listing rent, savings, and discretionary spending showed that the situation needs attention but is not an emergency. I scheduled a weekly fifteen-minute money review. Replacing background dread with a recurring check may stop financial fear from attaching itself to every work decision.",
    8: "The run created enough space to enter standup with a plan instead of an apology. Sarah's observation mattered because it connected a private habit with a visible work behavior. I want to test this on two more meeting days before treating it as a reliable pattern.",
    7: "I replied with a voice note instead of overthinking the perfect apology. We agreed to walk again this weekend. The relationship feels stronger when I share the messy middle, not only the polished version after I have already recovered.",
    6: "Sarah requested one more change and I asked which user risk it addressed before reacting. The conversation stayed specific, and I completed the revision without treating it as a referendum on the promotion. That felt like genuine progress under the same kind of pressure.",
    5: "None of the choices was dramatic, but together they lowered the sense of emergency. I prepared clothes for the morning and left the phone in the hallway. This combination may be a repeatable recovery routine after demanding workdays.",
    4: "The numbers showed a manageable gap rather than the disaster I had imagined. Cancelling unused subscriptions created a small monthly buffer, and I set a savings target for the next three pay cycles. Concrete facts made the career decision feel less emotionally loaded.",
    3: "Sarah highlighted the clearer ownership map and the calmer way I communicated trade-offs. I wrote down the specific evidence instead of converting it into a general need for approval. The promotion remains uncertain, but my next development steps are much clearer.",
    2: "I explained the launch pressure without asking Maya to absorb all of it. We also talked about her week and made plans unrelated to work. That balance felt important: support works better when the relationship remains larger than the current problem.",
    1: "The improvement is not a straight line, but the difficult stretch no longer feels random. Movement, sleep boundaries, smaller plans, and early communication each shorten the time I stay stuck. I want to keep tracking these supports rather than assuming the better week will maintain itself.",
    0: "I reviewed the month and can see a full sequence: pressure rose, routines disappeared, isolation increased, then small reversible actions helped. The launch and promotion are still meaningful uncertainties, but I now have evidence about what steadies me while I decide.",
}

CHAT_TURNS = [
    "I noticed I stopped exercising when work got intense. Why does it hit my mood so hard?",
    "I am nervous that Sarah's feedback means I am falling behind again.",
    "Maya and I are okay now, but I still feel bad that I disappeared when stressed.",
    (
        "I received another round of launch feedback. Should I resign now or ask "
        "Sarah for a two-week improvement plan while protecting my income?"
    ),
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
    def __init__(
        self,
        db: JournalDB,
        vector_store: _VectorProxy,
        *,
        latency_log_path: str,
    ) -> None:
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
        self.eval_engine = EvalEngine(vector_store, db, latency_log_path)


def _insert_entries(db: JournalDB) -> None:
    now = DEMO_ANCHOR
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
                text=f"{entry.text} {DETAILS[entry.days_ago]}",
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
        is_guidance = "should i" in turn.lower()
        assistant = {
            "role": "assistant",
            "content": response,
            "emotion": emotion,
            "memory_replay": None,
            "crisis": {"flagged": False, "matched_terms": []},
            "retrieved_memories": service.eval_engine.vector_store.query(turn, top_k=3),
            "prompt": None,
            "mode": "guidance" if is_guidance else "reflection",
            "trace_id": "demo-guidance-01" if is_guidance else None,
            "decision_state": (
                {
                    "problem": "How to respond to repeated launch feedback without making a fear-driven irreversible choice.",
                    "desired_outcome": "Protect income while testing whether clearer expectations improve the role.",
                    "user_stated_options": [
                        "Resign now",
                        "Ask Sarah for a two-week improvement plan",
                    ],
                    "fears": ["Falling behind", "Losing income"],
                    "constraints": ["Stable income", "Active product launch"],
                    "relevant_goals": ["Promotion", "Calmer leadership"],
                    "current_emotion": "fear",
                    "missing_information": [
                        "Specific success criteria",
                        "Available financial runway",
                    ],
                    "uncertainty": 0.42,
                }
                if is_guidance
                else None
            ),
            "guidance": (
                {
                    "recommended_option": "Ask Sarah for a two-week improvement plan",
                    "strong_recommendation": True,
                    "evidence_confidence": 0.68,
                    "uncertainty": [
                        "The next review criteria are not yet confirmed.",
                        "Resignation runway has not been calculated.",
                    ],
                    "next_actions": [
                        "Request two measurable success criteria.",
                        "Run the plan for two weeks and record outcomes.",
                        "Review finances before reconsidering resignation.",
                    ],
                }
                if is_guidance
                else None
            ),
            "tools_called": (
                [
                    "search_similar_memories",
                    "get_emotional_patterns",
                    "get_user_profile",
                ]
                if is_guidance
                else []
            ),
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
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HF_API_TOKEN")
        or os.getenv("HUGGINGFACE_API_KEY")
    )
    if not token:
        return None
    model = os.getenv("HF_MODEL", "arsoban/ocd-therapist-27b-v0.3")
    return HuggingFaceInferenceClient(
        model_name=model,
        api_token=token,
        provider=os.getenv("HF_INFERENCE_PROVIDER", "featherless-ai"),
        max_new_tokens=160,
        read_timeout_s=60,
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
    if "resign" in lowered:
        fallback = (
            "Validation:\nRepeated feedback can make leaving feel urgent, especially "
            "when the promotion and your sense of competence are involved.\n\n"
            "Recommendation:\nAsk Sarah for a two-week improvement plan before making "
            "an irreversible resignation decision.\n\n"
            "Why:\n- Recent entries show that specific scope and feedback conversations "
            "reduced uncertainty.\n- Stable income is a stated constraint, while a short "
            "trial is reversible.\n- Fear was high during earlier feedback, but later evidence "
            "showed clearer planning and trust.\n\n"
            "Uncertainty:\n- The next review criteria are not yet confirmed.\n- Your "
            "financial runway has not been calculated.\n\n"
            "Next steps:\n1. Ask for two measurable success criteria.\n2. Run the plan "
            "for two weeks and record outcomes.\n3. Review your finances before "
            "reconsidering resignation."
        )
    elif "sarah" in lowered:
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


def _journal_entries_response(db: JournalDB) -> DemoJournalEntriesResponse:
    records = sorted(db.get_all(), key=lambda record: record.timestamp or "")
    word_counts = [len((record.text or "").split()) for record in records]
    entries = [
        DemoJournalEntry(
            id=record.id,
            date=(record.timestamp or "")[:10],
            text=record.text,
            emotion=record.emotion,
            emotion_confidence=record.emotion_confidence,
            sentiment=record.sentiment_compound,
            topics=record.topics or [],
            habits=record.habits or [],
            people=record.entities_people or [],
        )
        for record in records
    ]
    return DemoJournalEntriesResponse(
        persona=(
            "Aarav, a product analyst navigating a release, promotion goal, "
            "health routines, finances, and relationship repair."
        ),
        entry_count=len(entries),
        days_covered=len({entry.date for entry in entries}),
        average_words_per_entry=round(
            sum(word_counts) / len(word_counts) if word_counts else 0.0,
            1,
        ),
        entries=entries,
    )


def _write_demo_traces(path: Path) -> None:
    model = "arsoban/ocd-therapist-27b-v0.3:featherless-ai"
    rows = [
        _demo_trace("reflection-01", "reflection", "success", "reflected", 1, [], 742.4),
        _demo_trace(
            "guidance-01",
            "guidance",
            "success",
            "recommended",
            2,
            ["search_similar_memories", "get_emotional_patterns", "get_user_profile"],
            1684.2,
        ),
        _demo_trace("safety-01", "safety", "blocked", "blocked_high_risk", 0, [], 41.8),
        _demo_trace(
            "guidance-02",
            "guidance",
            "degraded",
            "guidance_response_fallback",
            2,
            ["search_similar_memories", "get_user_profile"],
            1326.6,
            failure_category="invalid_structured_output",
        ),
        _demo_trace("reflection-02", "reflection", "success", "reflected", 1, [], 696.1),
        _demo_trace(
            "guidance-03",
            "guidance",
            "success",
            "insufficient_evidence",
            2,
            [
                "search_similar_memories",
                "get_emotional_patterns",
                "get_recent_decision_context",
            ],
            1518.9,
        ),
    ]
    payloads = []
    for row in rows:
        data = row.model_dump(mode="json")
        if data["logical_llm_calls"]:
            data["requested_models"] = [model] * data["logical_llm_calls"]
            data["actual_models"] = [
                "arsoban/ocd-therapist-27b-v0.3"
            ] * data["logical_llm_calls"]
        payloads.append(json.dumps(data))
    path.write_text("\n".join(payloads) + "\n", encoding="utf-8")


def _demo_trace(
    trace_id: str,
    mode: str,
    status: str,
    outcome: str,
    llm_calls: int,
    tools: list[str],
    latency_ms: float,
    *,
    failure_category: str | None = None,
) -> TraceRecord:
    prompts = (
        ["reflection-v1"]
        if mode == "reflection"
        else ["decision-parse-v2", "guidance-response-v2"]
        if mode == "guidance"
        else []
    )
    tool_observations = [
        {
            "tool": tool,
            "success": True,
            "result_count": 3 if tool == "search_similar_memories" else 2,
            "duration_ms": round(7.5 + index * 2.2, 2),
            "failure_category": None,
        }
        for index, tool in enumerate(tools)
    ]
    return TraceRecord(
        timestamp=DEMO_ANCHOR.isoformat().replace("+00:00", "Z"),
        trace_id=f"demo-{trace_id}",
        mode=mode,  # type: ignore[arg-type]
        prompt_versions=prompts,
        logical_llm_calls=llm_calls,
        llm_calls=llm_calls,
        provider_attempts=llm_calls,
        retry_count=0,
        token_usage={
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        },
        tools_called=tools,
        tool_observations=tool_observations,
        memories_retrieved=3 if mode == "guidance" else 0,
        agent_steps=len(tools),
        agent_termination_reason=(
            "max_calls"
            if len(tools) == 3 and "get_recent_decision_context" in tools
            else "sufficient_context"
            if mode == "guidance"
            else None
        ),
        context_selection=(
            {
                "estimated_tokens": 1084,
                "budget_tokens": 1500,
                "included": 9,
                "deduplicated": 2,
                "truncated": 0,
                "dropped": 3,
            }
            if mode == "guidance"
            else {}
        ),
        stage_latencies_ms={
            "safety": 3.4,
            "preprocessing": 28.2,
            "persistence": 16.1,
            "decision_parsing": 402.0 if mode == "guidance" else 0.0,
            "context_agent": 31.7 if mode == "guidance" else 0.0,
            "guidance_scoring": 4.6 if mode == "guidance" else 0.0,
            "final_generation": max(0.0, latency_ms - 84.0),
            "total": latency_ms,
        },
        failure_category=failure_category,
        status=status,  # type: ignore[arg-type]
        outcome=outcome,
        latency_ms=latency_ms,
        elapsed_ms=latency_ms,
    )


def _dump(name: str, payload: Any) -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    path = DEMO_DIR / name
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json", exclude_none=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _growth_response(service: _DemoService) -> GrowthResponse:
    snapshots = service.growth_tracker.compute_snapshots()
    fixed_timestamp = DEMO_ANCHOR.isoformat().replace("+00:00", "Z")
    for snapshot in snapshots:
        snapshot.snapshot_date = fixed_timestamp
    return GrowthResponse(
        snapshots=snapshots,
        narrative=service.growth_tracker.narrative(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--offline-llm",
        action="store_true",
        help="Skip live hosted-model calls for chat transcript generation.",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="mind-shift-demo-",
        ignore_cleanup_errors=True,
    ) as tmp:
        temp_root = Path(tmp)
        db = JournalDB(str(temp_root / "journal.db"))
        _insert_entries(db)
        trace_path = temp_root / "traces.jsonl"
        _write_demo_traces(trace_path)
        service = _DemoService(
            db,
            _VectorProxy(db),
            latency_log_path=str(trace_path),
        )
        graph = service.knowledge_graph.build(lookback_days=30)

        _dump("journal_entries.json", _journal_entries_response(db))
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
            _growth_response(service),
        )
        _dump(
            "diagnostics.json",
            DiagnosticsResponse(
                retrieval_precision=service.eval_engine.retrieval_precision_at_k(k=3),
                emotion_confidence=service.eval_engine.emotion_confidence_stats(),
                latency=service.eval_engine.latency_summary(),
                trace_health=service.eval_engine.trace_health_summary(),
            ),
        )
        _dump(
            "observability.json",
            ObservabilityResponse.model_validate(
                build_observability_snapshot(
                    service,
                    source="deterministic 30-day recruiter fixture",
                    model="arsoban/ocd-therapist-27b-v0.3",
                    provider="Hugging Face router / Featherless AI",
                    generated_at=DEMO_ANCHOR.isoformat().replace("+00:00", "Z"),
                    evaluation={
                        "status": "passed",
                        "case_count": 15,
                        "passed": 15,
                        "coverage": [
                            "career decision routing",
                            "anxiety-driven decision",
                            "ambiguous and insufficient information",
                            "empty history",
                            "high-risk and crisis safety",
                            "normal reflection",
                            "retrieval relevance and duplicate suppression",
                            "structured-schema fallback",
                            "transient retry and permanent failure",
                            "unnecessary-tool prevention",
                            "three-call termination",
                        ],
                    },
                )
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
