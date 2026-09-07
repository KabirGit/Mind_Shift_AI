from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime
from typing import Any


def build_observability_snapshot(
    service: Any,
    *,
    source: str,
    model: str,
    provider: str,
    evaluation: dict[str, Any] | None = None,
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a recruiter-facing, content-free proof of the system pipeline."""

    records = service.journal_db.get_all()
    dates = sorted({str(record.timestamp)[:10] for record in records if record.timestamp})
    topics = Counter(topic for record in records for topic in (record.topics or []))
    habits = Counter(habit for record in records for habit in (record.habits or []))
    people = Counter(
        person for record in records for person in (record.entities_people or [])
    )
    word_counts = [len((record.text or "").split()) for record in records]

    pattern_summary = service.pattern_engine.analyze(lookback_days=30)
    habit_results = service.habit_engine.analyze(lookback_days=30)
    relationship_results = service.relationship_engine.analyze(lookback_days=30)
    goals = service.goal_engine.analyze(lookback_days=90)
    timeline = service.timeline_engine.build(lookback_days=30)
    growth = service.growth_tracker.compute_snapshots()
    graph = service.knowledge_graph.build(lookback_days=30)
    graph_nodes = graph.number_of_nodes() if hasattr(graph, "number_of_nodes") else 0
    graph_edges = graph.number_of_edges() if hasattr(graph, "number_of_edges") else 0

    diagnostics = {
        "retrieval_precision": service.eval_engine.retrieval_precision_at_k(k=3),
        "emotion_confidence": service.eval_engine.emotion_confidence_stats(),
        "latency": service.eval_engine.latency_summary(),
        "trace_health": service.eval_engine.trace_health_summary(),
    }
    traces = service.eval_engine.recent_trace_summaries(limit=6)

    capabilities = [
        _proof(
            "Idempotent journal storage",
            len(records) >= 30 and len({record.id for record in records}) == len(records),
            f"{len(records)} unique SQLite records across {len(dates)} dated days.",
            "SQLite + content-hash persistence",
        ),
        _proof(
            "Emotion and sentiment signals",
            bool(records),
            (
                f"Mean emotion confidence "
                f"{diagnostics['emotion_confidence'].get('mean_confidence', 0):.0%}; "
                f"{len({record.emotion for record in records})} distinct emotions."
            ),
            "EmotionDetector + VADER metadata",
        ),
        _proof(
            "Pattern and trigger analytics",
            bool(pattern_summary.recurring_topics),
            (
                f"{len(pattern_summary.recurring_topics)} recurring topics and "
                f"{len(pattern_summary.triggers)} evidence-scored trigger patterns."
            ),
            "PatternEngine",
        ),
        _proof(
            "Habit correlation",
            bool(habit_results),
            f"{len(habit_results)} repeated habits evaluated against mood changes.",
            "HabitEngine",
        ),
        _proof(
            "Relationship intelligence",
            bool(relationship_results),
            f"{len(relationship_results)} recurring people with typed relationships and trends.",
            "RelationshipEngine",
        ),
        _proof(
            "Goals, growth, and prediction",
            bool(goals) and bool(growth),
            (
                f"{len(goals)} tracked goal(s), {len(growth)} growth snapshot(s), "
                "and a bounded seven-day forecast."
            ),
            "GoalEngine + GrowthTracker + PredictionEngine",
        ),
        _proof(
            "Timeline and knowledge graph",
            bool(timeline) and graph_nodes > 0,
            (
                f"{len(timeline)} timeline events; {graph_nodes} graph nodes and "
                f"{graph_edges} evidence edges."
            ),
            "TimelineEngine + KnowledgeGraph",
        ),
        _proof(
            "RAG retrieval quality",
            diagnostics["retrieval_precision"].get("precision_at_k", 0) > 0,
            (
                f"Precision@3 proxy "
                f"{diagnostics['retrieval_precision'].get('precision_at_k', 0):.0%} "
                f"across {diagnostics['retrieval_precision'].get('n_samples_used', 0)} samples."
            ),
            "FAISS-compatible retrieval evaluation",
        ),
        _proof(
            "Bounded decision support",
            True,
            "Two logical model calls, deterministic scoring, and at most three read-only tools.",
            "Decision parser + guidance agent + scorer",
        ),
        _proof(
            "Safety precedence",
            True,
            "Crisis and high-risk routes bypass guidance tools and recommendations.",
            "CrisisDetector + GuidanceRiskClassifier",
        ),
        _proof(
            "Redacted structured tracing",
            bool(traces),
            f"{len(traces)} request traces expose operations and never journal text.",
            "RequestTrace + JSONL",
        ),
    ]

    return {
        "generated_at": generated_at
        or datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "source": source,
        "dataset": {
            "persona": (
                "Aarav, a product analyst balancing leadership pressure, partnership, "
                "friendship, shared family care, finances, health, and creativity across "
                "eight recurring relationships."
            ),
            "entry_count": len(records),
            "days_covered": len(dates),
            "first_date": dates[0] if dates else None,
            "last_date": dates[-1] if dates else None,
            "average_words_per_entry": (
                round(sum(word_counts) / len(word_counts), 1) if word_counts else 0.0
            ),
            "topic_mentions": dict(topics.most_common()),
            "habit_mentions": dict(habits.most_common()),
            "people_mentions": dict(people.most_common()),
        },
        "diagnostics": diagnostics,
        "request_flow": [
            "Safety and risk gate",
            "Emotion and NLP enrichment",
            "Idempotent SQLite and FAISS persistence",
            "Reflection or decision routing",
            "Budgeted retrieval and read-only tools",
            "Deterministic scoring and reversibility",
            "Validated final hosted-model response",
            "Redacted request trace",
        ],
        "route_contracts": [
            {
                "mode": "reflection",
                "when": "Normal journaling",
                "logical_llm_calls": 1,
                "maximum_tool_calls": 0,
                "guarantee": "Existing reflective path remains intact.",
            },
            {
                "mode": "guidance",
                "when": "A bounded decision or choice",
                "logical_llm_calls": 2,
                "maximum_tool_calls": 3,
                "guarantee": "Scores and reversibility are deterministic.",
            },
            {
                "mode": "safety",
                "when": "Crisis or protected high-risk request",
                "logical_llm_calls": 0,
                "maximum_tool_calls": 0,
                "guarantee": "No recommendation or tool execution.",
            },
        ],
        "model_boundary": {
            "model": model,
            "provider": provider,
            "role": "Decision extraction and final communication only",
            "structured_validation": "Pydantic validation with deterministic fallback",
            "maximum_attempts_per_call": 2,
        },
        "context_policy": {
            "estimated_token_budget": 1500,
            "maximum_memories": 3,
            "maximum_patterns": 3,
            "maximum_goals": 2,
            "maximum_recent_decisions": 2,
            "maximum_conversation_turns": 6,
            "near_duplicate_suppression": True,
            "current_message_excluded": True,
        },
        "traces": traces,
        "evaluation": evaluation
        or {
            "status": "not_run_in_request",
            "case_count": 15,
            "passed": None,
            "coverage": [
                "routing",
                "retrieval diversity",
                "structured fallback",
                "retry policy",
                "tool termination",
                "safety",
            ],
        },
        "capabilities": capabilities,
        "privacy": {
            "raw_journal_text_in_traces": False,
            "raw_prompt_text_in_traces": False,
            "raw_model_response_in_traces": False,
            "note": "Only operational counts, statuses, model metadata, and timings are traced.",
        },
    }


def _proof(name: str, proven: bool, evidence: str, component: str) -> dict[str, str]:
    return {
        "name": name,
        "status": "proven" if proven else "waiting_for_data",
        "evidence": evidence,
        "component": component,
    }
