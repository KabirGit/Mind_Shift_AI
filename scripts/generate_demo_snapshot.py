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

from backend.analytics._stats_utils import recency_decay  # noqa: E402
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
    DemoEntry(29, "The month began with a sunrise run before I drafted the launch plan with Sarah.", "optimism", 0.86, ["Sarah"], {"Sarah": "manager"}, ["career", "health", "learning"], ["exercise", "planning"], 0.48),
    DemoEntry(28, "Arjun and I tried a phone-free dinner, although I kept thinking about unfinished work.", "caring", 0.83, ["Arjun"], {"Arjun": "partner"}, ["relationship", "home", "career"], ["cooking", "social_media"], 0.22),
    DemoEntry(27, "Maya convinced me to take photographs by the lake instead of working through Saturday.", "joy", 0.88, ["Maya"], {"Maya": "friend"}, ["relationship", "creativity", "health"], ["exercise"], 0.58),
    DemoEntry(26, "Neha called about Dad's follow-up appointment with Dr Mehta, and I immediately volunteered to manage everything.", "caring", 0.81, ["Neha", "Raj", "Dr Mehta"], {"Neha": "sibling", "Raj": "parent", "Dr Mehta": "doctor"}, ["family", "health", "home"], ["planning"], 0.05),
    DemoEntry(25, "Vikram missed an integration deadline and I reacted as though his delay proved I was failing as lead.", "anger", 0.84, ["Vikram"], {"Vikram": "colleague"}, ["career", "relationship"], ["coffee"], -0.44),
    DemoEntry(24, "I stayed online past midnight, scrolled between messages, and slept badly before the review.", "stress", 0.86, ["Arjun"], {"Arjun": "partner"}, ["career", "health", "relationship"], ["sleep", "social_media", "coffee"], -0.52),
    DemoEntry(23, "Sarah returned the proposal with another round of edits and mentioned that promotion evidence still looked uneven.", "fear", 0.88, ["Sarah"], {"Sarah": "manager"}, ["career", "learning"], [], -0.58),
    DemoEntry(22, "Arjun said he was carrying most of the meals and housework while I lived inside the launch.", "remorse", 0.84, ["Arjun"], {"Arjun": "partner"}, ["relationship", "home", "career"], ["cooking"], -0.41),
    DemoEntry(21, "Raj asked whether I could cover a larger family expense, and I felt guilty for wanting limits.", "fear", 0.82, ["Raj", "Neha"], {"Raj": "parent", "Neha": "sibling"}, ["family", "money", "relationship"], ["budgeting"], -0.37),
    DemoEntry(20, "A short walk with Maya interrupted three days of sitting and gave me room to admit how brittle I felt.", "relief", 0.86, ["Maya"], {"Maya": "friend"}, ["health", "relationship", "career"], ["exercise"], 0.24),
    DemoEntry(19, "Vikram and I reviewed the failed handoff, and I apologized for turning a process gap into a personal accusation.", "remorse", 0.83, ["Vikram"], {"Vikram": "colleague"}, ["career", "relationship", "learning"], ["planning"], -0.08),
    DemoEntry(18, "I accompanied Neha and Raj to Dr Mehta, then noticed I had silently expected Neha to read my exhaustion.", "sadness", 0.82, ["Neha", "Raj", "Dr Mehta"], {"Neha": "sibling", "Raj": "parent", "Dr Mehta": "doctor"}, ["family", "health", "relationship"], ["cooking"], 0.0),
    DemoEntry(17, "At a team dinner, Sarah praised Vikram's technical recovery while I struggled not to compare it with my own review.", "envy", 0.79, ["Sarah", "Vikram"], {"Sarah": "manager", "Vikram": "colleague"}, ["career", "relationship"], [], -0.21),
    DemoEntry(16, "This was the low point: I cancelled Maya, snapped at Arjun, ignored the budget, and worked without making progress.", "sadness", 0.91, ["Maya", "Arjun"], {"Maya": "friend", "Arjun": "partner"}, ["career", "relationship", "money", "health"], ["social_media", "coffee"], -0.72),
    DemoEntry(15, "I put the phone outside the bedroom, slept eight hours, and tried ten minutes of breathing before email.", "relief", 0.85, ["Arjun"], {"Arjun": "partner"}, ["health", "home", "relationship"], ["sleep", "meditation", "social_media"], 0.31),
    DemoEntry(14, "Arjun and I divided household tasks explicitly instead of arguing about who should have noticed them.", "caring", 0.87, ["Arjun"], {"Arjun": "partner"}, ["relationship", "home"], ["cooking", "planning"], 0.42),
    DemoEntry(13, "Sarah reduced the release scope, and Vikram agreed to own the risky integration with daily checkpoints.", "optimism", 0.86, ["Sarah", "Vikram"], {"Sarah": "manager", "Vikram": "colleague"}, ["career", "learning"], ["planning"], 0.29),
    DemoEntry(12, "Leena challenged my belief that good leadership means absorbing every uncertainty before anyone else sees it.", "realization", 0.84, ["Leena"], {"Leena": "mentor"}, ["career", "learning", "relationship"], ["reading", "journaling"], 0.26),
    DemoEntry(11, "Neha and I made a shared calendar for Raj's care instead of treating the more anxious person as the default organizer.", "relief", 0.86, ["Neha", "Raj"], {"Neha": "sibling", "Raj": "parent"}, ["family", "health", "home"], ["planning"], 0.34),
    DemoEntry(10, "Maya and I walked after work, and I listened to her career news before talking about my launch.", "caring", 0.88, ["Maya"], {"Maya": "friend"}, ["relationship", "health", "career"], ["exercise"], 0.51),
    DemoEntry(9, "I reviewed the budget, set a clear amount for helping Raj, and showed the plan to Neha.", "realization", 0.83, ["Raj", "Neha"], {"Raj": "parent", "Neha": "sibling"}, ["money", "family", "home"], ["budgeting"], 0.17),
    DemoEntry(8, "A run before planning helped me ask Sarah and Vikram clarifying questions rather than arriving defensive.", "pride", 0.89, ["Sarah", "Vikram"], {"Sarah": "manager", "Vikram": "colleague"}, ["career", "health", "learning"], ["exercise", "sleep"], 0.56),
    DemoEntry(7, "Vikram completed the integration and credited the smaller checkpoints rather than individual heroics.", "gratitude", 0.88, ["Vikram"], {"Vikram": "colleague"}, ["career", "learning", "relationship"], ["planning"], 0.54),
    DemoEntry(6, "Arjun and I went out without our phones and talked about travel, not just chores or my job.", "joy", 0.9, ["Arjun"], {"Arjun": "partner"}, ["relationship", "creativity", "home"], ["social_media", "meditation"], 0.64),
    DemoEntry(5, "Neha, Raj, and I cooked together, and I let an imperfect family plan remain shared rather than taking it back.", "gratitude", 0.89, ["Neha", "Raj"], {"Neha": "sibling", "Raj": "parent"}, ["family", "health", "relationship"], ["cooking"], 0.59),
    DemoEntry(4, "Leena reviewed my promotion examples and Sarah helped turn them into measurable leadership evidence.", "pride", 0.87, ["Leena", "Sarah"], {"Leena": "mentor", "Sarah": "manager"}, ["career", "learning"], ["reading", "planning"], 0.47),
    DemoEntry(3, "A late QA regression tested the new habits, but Vikram and I paused, assigned owners, and avoided blame.", "optimism", 0.88, ["Vikram"], {"Vikram": "colleague"}, ["career", "health", "learning"], ["meditation", "coffee"], 0.38),
    DemoEntry(2, "Maya and Arjun joined me for dinner, and I stayed present even when a launch notification appeared.", "joy", 0.91, ["Maya", "Arjun"], {"Maya": "friend", "Arjun": "partner"}, ["relationship", "home", "creativity"], ["cooking", "social_media"], 0.67),
    DemoEntry(1, "Sarah scheduled the promotion review and offered two weeks to demonstrate the final leadership criteria.", "anticipation", 0.86, ["Sarah"], {"Sarah": "manager"}, ["career", "money", "learning"], ["planning", "budgeting"], 0.19),
    DemoEntry(0, "Looking back with Arjun, Maya, Neha, and Raj in mind, I can see that progress came from sharing responsibility rather than controlling every outcome.", "realization", 0.92, ["Arjun", "Maya", "Neha", "Raj"], {"Arjun": "partner", "Maya": "friend", "Neha": "sibling", "Raj": "parent"}, ["career", "relationship", "family", "health", "money"], ["exercise", "sleep", "meditation", "budgeting"], 0.61),
]

# Each base event receives a distinct reflective layer before it is inserted.
# This keeps the persona realistic enough to exercise retrieval, analytics,
# goals, relationships, and longitudinal change rather than using keyword-only
# one-liners.
DETAILS: dict[int, str] = {
    29: "I was excited by the promotion possibility, but I noticed the old urge to prove myself by owning every dependency. My goal is to earn the promotion through visible delegation, predictable updates, and early escalation rather than heroics. I wrote those behaviors beside the release milestones and protected three exercise blocks, treating energy as part of leadership rather than a reward after work.",
    28: "Arjun was patient, yet he could tell I was only half listening. I initially defended checking notifications as responsibility, then realized I was asking him to share an evening with my anxiety. I want to protect our partnership with one twenty-minute work check after dinner and a shared cooking night, which feels more realistic than promising never to think about work at home.",
    27: "Photography used to make me curious, while lately I have judged it as unproductive. Maya noticed that I became playful once we stopped discussing the launch. I also listened to her uncertainty about changing jobs, which reminded me that friendship cannot be only a recovery service for my stress. I came home physically tired and mentally wider.",
    26: "Neha sounded worried and Raj minimized the issue, so I slipped into organizer mode before asking what either of them wanted. Dr Mehta had described the appointment as routine, but my mind turned it into an emergency. I created a checklist, then wondered whether taking control was care or a way to avoid feeling uncertain. Neha and I agreed to divide calls after we had more facts.",
    25: "The integration dependency was genuinely late, but my tone in the meeting made collaboration harder. I had consumed three coffees and entered without asking what had blocked Vikram. Later I could see a familiar attitude: when I fear being judged by Sarah, I become controlling with the team. I scheduled a calmer technical review and wrote down process questions instead of accusations.",
    24: "Arjun asked me to close the laptop twice and I said I was almost done, though I was mostly refreshing messages. The combination of caffeine, scrolling, and shallow sleep left me impatient the next morning. I skipped movement and interpreted neutral comments as criticism. This is the first clear cluster I want to track rather than treating each bad reaction as a separate personality failure.",
    23: "I heard the word uneven and translated it into not leadership material. After the meeting, Sarah clarified that my technical delivery was strong but that delegation evidence was thin. The distinction mattered, although fear remained high. I resisted drafting a resignation message and instead listed what could be demonstrated within two weeks, what required longer, and what was outside my control.",
    22: "My first response was to list everything I had done recently, which made Arjun feel even less seen. When I slowed down, I could admit that work had become an excuse to opt out of ordinary shared responsibility. We chose three fixed tasks I would own and one evening with no launch discussion. His frustration was not rejection; it was information about an imbalance we could change.",
    21: "I want to support Raj, and I also have rent, savings goals, and uncertainty about work. Neha assumed I had more financial room because I usually say yes quickly. I opened the actual numbers instead of arguing from guilt. We postponed the decision for forty-eight hours, agreed to ask Raj what portion was essential, and decided that care should not require either sibling to hide resentment.",
    20: "Maya did not minimize the work or family pressure, but she questioned why rest always had to be earned. During the walk my breathing settled, and I noticed I could describe Vikram's delay without calling it a disaster. I sent him a brief message confirming the review agenda. The problem was unchanged, yet my stance moved from prosecution toward investigation after twenty-five minutes outside.",
    19: "Vikram explained that the handoff lacked an owner and that he had been reluctant to challenge my optimistic estimate. That was uncomfortable evidence about the environment I had created. We mapped the dependency, added a daily checkpoint, and agreed he would raise risk directly. Apologizing did not weaken my authority; it made the next action clearer and reduced the private shame I had been carrying.",
    18: "The appointment went well, but the day exposed a family pattern. Neha waits until she is certain before asking for help, while I volunteer early and later feel unappreciated. Raj tries to protect us by downplaying his needs, which creates more guessing. Over dinner I named my fatigue without blaming Neha, and we assigned the next two tasks explicitly instead of relying on silent expectations.",
    17: "I was genuinely pleased for Vikram and still felt threatened by Sarah's praise. Both reactions were true. Instead of withdrawing, I asked what had helped the recovery and learned that the checkpoints I suggested were useful. Comparison softened when I focused on the shared system. I left early enough to keep my plan with Arjun, which was a small test of whether work success had to consume the whole evening.",
    16: "I had slept poorly, used coffee instead of breakfast, and moved between work, family messages, and the bank app without finishing anything. Maya's invitation felt like another demand, and Arjun's question about dinner triggered a sharp response. The old attitude was that everyone needed something from me; the harder truth was that I had agreed to too much without communicating limits. I wrote apologies but waited until calmer to send them.",
    15: "Rest did not solve the launch, yet it changed the scale of every problem. During the breathing practice I noticed how quickly fear became a story about permanent failure. Arjun and I ate breakfast without discussing logistics, and I entered work with one priority list. I still felt embarrassed about yesterday, but the embarrassment led to repair rather than more hiding and frantic activity.",
    14: "We discussed emotional labor as well as visible chores. Arjun said uncertainty was easier than being repeatedly promised help that did not appear. I chose cooking and the weekly shopping list, while he chose laundry and bills. The conversation moved from who cares more to what reliability looks like. I felt less defensive once the expectations were observable instead of moral judgments.",
    13: "Cutting two features disappointed me because I had attached ambition to scope. Sarah framed the smaller release as judgment, not retreat, and asked me to make ownership visible. Vikram chose the integration plan and I kept review responsibility without rewriting his work. The workload became bounded, and I noticed curiosity replacing the urge to monitor every detail.",
    12: "Leena asked whether I wanted to be indispensable or trusted, and the difference stayed with me. We reviewed moments where I had hidden risk, absorbed family tasks, or avoided disappointing friends. I plan to use the leadership course to practice delegation, while the evening journal tracks facts, assumptions, and the next reversible action. The new approach feels more honest than performing certainty.",
    11: "The calendar included appointments, transport, medication pickup, and who would update relatives. Raj could see the plan and correct what we had assumed about him. My goal is to keep family care shared: Neha took two calls I would normally grab, while I chose transport and one expense. Control has often been my way of showing love, but the plan left space for all three of us to have preferences.",
    10: "Maya told me about an interview she had not mentioned because our recent conversations centered on my crises. I felt ashamed for a moment, then chose interest over self-punishment. I want to keep our friendship mutual, so we walked for forty minutes, discussed her news, and made a photography plan. Repair includes making room for the other person's life, not only explaining my absence.",
    9: "The spreadsheet showed I could contribute a fixed amount without using emergency savings, but not the larger amount I had almost promised. I plan to keep a weekly budget review so guilt does not decide first. Neha was relieved to see the constraint, and Raj chose the smaller immediate expense. Budgeting turned a loyalty test into a shared decision with boundaries and dates.",
    8: "I slept seven and a half hours, ran slowly, and limited coffee before the planning meeting. When Sarah questioned the sequence, I asked which customer risk concerned her. Vikram answered part of it and I did not interrupt. The meeting produced a better dependency order without the familiar aftermath of replaying every sentence. I want to test whether preparation and movement consistently reduce defensiveness.",
    7: "My first impulse was to claim the recovery as proof of my leadership, but Vikram's comment corrected that story. The system worked because ownership was distributed and risk became discussable. I thanked him publicly and asked what process to keep after launch. Pride felt steadier when it included the team rather than protecting a heroic self-image.",
    6: "We chose a small neighborhood restaurant and left both phones in Arjun's bag. He talked about a travel course he wants to take, and I noticed how little attention I had given his ambitions this month. I shared the promotion uncertainty without asking him to decide for me. The evening felt intimate because neither of us had to become manager, patient, or problem solver.",
    5: "Raj wanted to cook an old family recipe and initially rejected help. Neha became impatient, while I nearly took over. We slowed down, divided the steps, and let the meal take longer. The same lesson appeared outside work: coordination does not require control. I left without checking the care calendar twice, trusting that Neha would handle her part and Raj would call if his needs changed.",
    4: "Leena helped me select examples of delegation, recovery after conflict, and clearer boundaries rather than presenting only output volume. Sarah converted them into two measurable criteria for the promotion review. I plan to finish the leadership course this quarter, and I scheduled reading time instead of adding another late work session. Clear standards now feel more useful than guessing at approval.",
    3: "The regression arrived at 9 p.m., exactly when my old pattern would have produced blame and an all-night rescue. I paused for five breaths, asked Vikram to assess impact, and informed Sarah with a bounded plan. We fixed the highest-risk path and deferred the rest. I drank one coffee rather than three and still slept before midnight. The calmer response was imperfect but repeatable.",
    2: "Maya brought recent photographs and Arjun described his travel idea. A launch alert appeared on my watch, and I felt the familiar pull to leave the table mentally. I checked that it was nonurgent, silenced it, and returned to the conversation. Both of them noticed. Being present felt less like willpower and more like a series of small environmental choices made before stress peaked.",
    1: "The review is still uncertain, which makes resignation fantasies tempting because they offer an immediate end to evaluation. I plan to use Sarah's two-week window to build promotion evidence, while remembering that leaving would affect rent, the family support budget, and plans with Arjun. I listed all options and their constraints. Anticipation and fear are present, but neither has to make the irreversible choice alone.",
    0: "The month was not a simple improvement story. Work pressure exposed defensiveness, family care exposed control, money exposed guilt, and close relationships exposed avoidance. Exercise, sleep, meditation, budgeting, and explicit conversations helped, but only when I used them to engage rather than escape. My attitude is shifting from proving I can carry everything to asking what can be shared, measured, tested, or repaired.",
}

CHAT_TURNS = [
    "When deadlines tighten I stop moving, sleep less, and get sharper with Arjun. Is that pattern actually in my history?",
    "I am nervous that Sarah's feedback and Vikram's delay mean I am failing as a lead.",
    "I want to help Neha and Raj without letting family responsibility consume every evening. What has changed?",
    "Maya and Arjun both say I am more present now, but I worry I will disappear again under pressure.",
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


class _DemoRelationshipEngine(RelationshipEngine):
    """Use the fixture anchor for stable relationship-closeness evidence."""

    def analyze(self, lookback_days: int = 30):
        profiles = super().analyze(lookback_days=lookback_days)
        records = self.db.get_all()
        for profile in profiles:
            mentions = [
                record
                for record in records
                if profile.person in (record.entities_people or [])
            ]
            profile.closeness_score = round(
                sum(
                    recency_decay(
                        record.timestamp,
                        half_life_hours=72.0,
                        now=DEMO_ANCHOR,
                    )
                    for record in mentions
                ),
                4,
            )
        return profiles


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
        self.relationship_engine = _DemoRelationshipEngine(db)
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
                    "constraints": [
                        "Stable income",
                        "Active product launch",
                        "Family support commitments",
                    ],
                    "relevant_goals": [
                        "Promotion",
                        "Calmer leadership",
                        "Reliable relationships",
                    ],
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
        "persona": (
            "Aarav, a product analyst balancing leadership pressure, a partnership, "
            "friendship, shared family care, finances, health, and creativity across "
            "eight recurring relationships."
        ),
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
            "reduced uncertainty.\n- Stable income and family support are stated "
            "constraints, while a short trial is reversible.\n- Fear was high during "
            "earlier feedback, but later evidence showed clearer planning, delegation, "
            "and trust.\n\n"
            "Uncertainty:\n- The next review criteria are not yet confirmed.\n- Your "
            "financial runway has not been calculated.\n\n"
            "Next steps:\n1. Ask for two measurable success criteria.\n2. Run the plan "
            "for two weeks and record outcomes.\n3. Review your finances before "
            "reconsidering resignation."
        )
    elif "sarah" in lowered or "vikram" in lowered:
        fallback = (
            "The history shows a more complex pattern than personal failure. Sarah's "
            "feedback initially triggered defensiveness, while the missed handoff with "
            "Vikram exposed unclear ownership. Once you apologized, reduced scope, and "
            "used daily checkpoints, both the work and your attitude improved.\n\n"
            "The useful question is not whether you felt anxious, but what you did next. "
            "Ask for the specific leadership criterion, keep ownership distributed, and "
            "judge the next two weeks using observable evidence rather than one meeting."
        )
    elif "neha" in lowered or "raj" in lowered:
        fallback = (
            "Earlier entries show you volunteering before Neha or Raj had clarified what "
            "they needed, then feeling trapped by the responsibility. The later calendar "
            "and budget conversations changed that: tasks, costs, and preferences became "
            "shared instead of silently assigned.\n\n"
            "The behavioral shift is from proving care through control to making care "
            "explicit and sustainable. Keep the shared calendar, state your available "
            "time and money before agreeing, and let Neha and Raj own their parts."
        )
    elif "maya" in lowered or "arjun" in lowered:
        fallback = (
            "The concern is understandable because the withdrawal pattern was real: Maya "
            "was cancelled on, and Arjun was left carrying more of home life. The later "
            "entries also contain concrete counter-evidence—repair, shared chores, mutual "
            "conversation, and phone-free time that both relationships noticed.\n\n"
            "Treat presence as a maintained behavior, not a personality verdict. Keep one "
            "small agreement with each person and name overload early, before silence or "
            "overpromising becomes the signal they receive."
        )
    else:
        fallback = (
            "That pattern is present, but it is broader than exercise alone. During the "
            "hardest stretch, sleep shortened, coffee and scrolling increased, movement "
            "dropped, and your tone with Arjun and Vikram became sharper. Later entries "
            "show better reactions when sleep, movement, and short pauses returned.\n\n"
            "This is correlation, not proof of a single cause. Test the smallest version: "
            "protect sleep tonight, take a short walk before the next difficult conversation, "
            "and record whether your tone and recovery time change."
        )
    return fallback, False


def _emotion_for_text(text: str) -> dict[str, Any]:
    lowered = text.lower()
    if "nervous" in lowered or "failing" in lowered or "resign" in lowered:
        return {
            "emotion": "fear",
            "confidence": 0.82,
            "all_emotions": [
                {"emotion": "fear", "score": 0.82},
                {"emotion": "realization", "score": 0.49},
            ],
        }
    if "bad" in lowered or "disappear" in lowered:
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


def _people_graph_response(service: _DemoService) -> GraphPeopleResponse:
    profiles = service.relationship_engine.analyze(lookback_days=30)
    nodes = [
        {
            "id": "User",
            "label": "You",
            "type": "user",
            "relationship_type": "self",
            "mention_count": 0,
        },
        *[
            {
                "id": profile.person,
                "label": profile.person,
                "type": "person",
                "relationship_type": profile.relationship_type,
                "mention_count": profile.mention_count,
            }
            for profile in profiles
        ],
    ]
    edges = [
        {
            "source": "User",
            "target": profile.person,
            "sentiment": profile.avg_sentiment,
            "weight": profile.mention_count,
            "closeness_score": profile.closeness_score,
        }
        for profile in profiles
    ]
    return GraphPeopleResponse.model_validate({"nodes": nodes, "edges": edges})


def _stabilize_graph_relationships(service: _DemoService, graph: Any) -> None:
    for profile in service.relationship_engine.analyze(lookback_days=30):
        if graph.has_edge("User", profile.person):
            graph["User"][profile.person]["closeness_score"] = (
                profile.closeness_score
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
        _stabilize_graph_relationships(service, graph)

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
            _people_graph_response(service),
        )
        graph_queries = {}
        for node in (
            "User",
            "career",
            "health",
            "relationship",
            "family",
            "money",
            "exercise",
            "sleep",
            "planning",
            "Arjun",
            "Maya",
            "Neha",
            "Raj",
            "Sarah",
            "Vikram",
            "Leena",
            "Dr Mehta",
        ):
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
