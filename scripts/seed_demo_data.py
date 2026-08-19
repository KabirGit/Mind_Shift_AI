"""Seed the journal with realistic synthetic entries for a demo-ready dashboard.

Runs each entry through the real RAGService.run_pipeline() so FAISS, SQLite,
and every analytics engine get populated exactly as in real usage.

The LLM is stubbed locally here so seeding never depends on the external API
(seeding only needs storage + analytics populated, not generated replies).

Usage:
    python scripts/seed_demo_data.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.api.rag_service import RAGService  # noqa: E402
from backend.safety.crisis_detector import CrisisDetector  # noqa: E402


class _StubLLM:
    def generate(self, prompt: str) -> str:
        return "Thanks for sharing this. That sounds meaningful — be kind to yourself."


# (days_ago, text) — varied topics, emotions, repeated people + habits, varied sentiment.
ENTRIES: list[tuple[int, str]] = [
    (29, "Started the week with a great gym session, I feel energized about my career goals."),
    (28, "Long day at the office, my manager Sarah praised my project and I felt proud."),
    (26, "Couldn't sleep again, insomnia is creeping back and work stress isn't helping."),
    (24, "Coffee with my friend Alex, we talked about money worries and rent going up."),
    (22, "Went for a run this morning, the exercise really lifted my mood."),
    (20, "Tough meeting with Sarah at work, I worry the deadline will be a disaster."),
    (18, "Read a good book before bed and actually slept well for once."),
    (16, "Spent the evening cooking with Alex, felt calm and connected."),
    (14, "Money is tight this month, the bills are piling up and I feel anxious."),
    (12, "Hit the gym again, exercise is becoming a steady habit and I feel good."),
    (10, "Sarah gave me more responsibility at work, nervous but hopeful about the promotion."),
    (8, "Scrolling social media too late, slept badly and felt foggy all day."),
    (6, "Coffee and a long walk with Alex, we laughed a lot, a really joyful day."),
    (5, "Studied for my course exam, stressed but the reading is paying off."),
    (4, "Good workout and an early night, sleep and exercise together really help my mood."),
    (3, "Quiet day, cooked a new recipe and felt content and relaxed."),
    (2, "Work went smoothly, Sarah and I planned the next project, feeling optimistic."),
    (1, "Reflecting on the month: career is improving, exercise helps, money still a worry."),
]


def main() -> None:
    service = RAGService(llm_client=_StubLLM())
    detector = CrisisDetector()

    for days_ago, text in ENTRIES:
        # Crisis-detector false-positive sanity check on the seed corpus itself.
        assert not detector.check(text)["flagged"], f"Seed entry tripped crisis detector: {text!r}"

        # Backdate the record so trends/lookback windows show meaningful data.
        ts = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat().replace(
            "+00:00", "Z"
        )
        out = service.run_pipeline(text=text, chat_history=[], top_k=3)
        rid = service.vector_store._hash_entry(text, None)
        record = service.journal_db.get_all()
        # Re-stamp this entry's timestamp to the backdated value.
        for r in record:
            if r.id == rid:
                r.timestamp = ts
                service.journal_db.insert(r)
                break
        # Also backdate the FAISS metadata timestamp so memory replay (which
        # ignores entries from the last 48h) has meaningful history to surface.
        for meta in service.vector_store.metadata:
            if meta.get("entry_hash") == rid or (
                meta.get("text", "").strip() == text.strip()
            ):
                meta["timestamp"] = ts
                break
        assert out["response"], "Pipeline returned empty response during seeding."

    # Persist the backdated FAISS metadata.
    service.vector_store.save()

    total = len(service.journal_db.get_all())
    print(f"Seeded {len(ENTRIES)} entries. Journal DB now has {total} records.")
    summary = service.pattern_engine.analyze(lookback_days=30)
    print(f"Triggers detected: {len(summary.triggers)}")
    print("Insights:")
    for line in service.insight_engine.generate(lookback_days=30):
        print(f"  - {line}")

    # Phase 17: show a quality metric + growth narrative right after seeding.
    precision = service.eval_engine.retrieval_precision_at_k(k=3)
    print(
        f"Retrieval Precision@3: {precision['precision_at_k']:.1%} "
        f"({precision['n_samples_used']} samples; {precision['note']})"
    )
    snapshots = service.growth_tracker.compute_snapshots()
    print(f"Growth snapshots: {len(snapshots)} month(s)")
    print(f"Growth narrative: {service.growth_tracker.narrative()}")


if __name__ == "__main__":
    main()
