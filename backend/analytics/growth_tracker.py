from __future__ import annotations

import logging
from collections import Counter, defaultdict
from datetime import UTC, datetime

from pydantic import BaseModel

from backend.analytics._stats_utils import parse_ts
from backend.analytics.presentation import delta_summary, mood_score
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)


class GrowthSnapshot(BaseModel):
    period_label: str  # "YYYY-MM"
    entry_count: int
    avg_sentiment: float
    dominant_emotion: str
    top_topic: str
    snapshot_date: str


class GrowthTracker:
    """Month-over-month growth snapshots + narrative. Deterministic, no LLM."""

    def __init__(self, db: JournalDB) -> None:
        self.db = db

    def compute_snapshots(self) -> list[GrowthSnapshot]:
        try:
            records = self.db.get_all()
            groups: dict[str, list] = defaultdict(list)
            for r in records:
                dt = parse_ts(r.timestamp)
                if dt is None:
                    continue
                groups[dt.strftime("%Y-%m")].append(r)

            return self._snapshots_from_groups(groups)
        except Exception as exc:
            logger.exception("GrowthTracker.compute_snapshots failed: %s", exc)
            return []

    def compute_growth_deltas(self) -> list[dict]:
        snaps = self.compute_snapshots()
        deltas = []
        for a, b in zip(snaps, snaps[1:], strict=False):
            deltas.append(
                {
                    "from": a.period_label,
                    "to": b.period_label,
                    "sentiment_delta": round(b.avg_sentiment - a.avg_sentiment, 4),
                    "entry_count_delta": b.entry_count - a.entry_count,
                    "emotion_changed": a.dominant_emotion != b.dominant_emotion,
                }
            )
        return deltas

    def narrative(self) -> str:
        snaps = self.compute_snapshots()
        weekly = self._weekly_snapshots()
        total = sum(s.entry_count for s in snaps)
        suffix = f" You've written {total} entries total."
        if len(weekly) >= 2:
            first = weekly[0]
            current = weekly[-1]
            sent_delta = current.avg_sentiment - first.avg_sentiment
            entry_delta = current.entry_count - first.entry_count
            return (
                f"From {first.period_label} to {current.period_label}, mood score "
                f"moved from {mood_score(first.avg_sentiment)}% to "
                f"{mood_score(current.avg_sentiment)}%, a {delta_summary(sent_delta)} "
                f"shift. Entry volume changed by {entry_delta:+d}, and the leading topic moved from "
                f"{first.top_topic} to {current.top_topic}."
            ) + suffix
        if len(snaps) < 2:
            return "Keep journaling - your growth story starts here." + suffix
        deltas = self.compute_growth_deltas()
        latest = deltas[-1]["sentiment_delta"]
        if latest > 0.1:
            base = f"Your mood has been improving compared to last month (+{latest:.2f})."
        elif latest < -0.1:
            base = "This has been a harder month emotionally than the last."
        else:
            base = "Your emotional patterns have been consistent across months."
        return base + suffix

    def _weekly_snapshots(self) -> list[GrowthSnapshot]:
        records = self.db.get_all()
        groups: dict[str, list] = defaultdict(list)
        for r in records:
            dt = parse_ts(r.timestamp)
            if dt is None:
                continue
            iso = dt.isocalendar()
            groups[f"{iso.year}-W{iso.week:02d}"].append(r)
        return self._snapshots_from_groups(groups)

    def _snapshots_from_groups(self, groups: dict[str, list]) -> list[GrowthSnapshot]:
        snapshots: list[GrowthSnapshot] = []
        for label in sorted(groups.keys()):
            recs = groups[label]
            sentiments = [r.sentiment_compound for r in recs]
            emotions = [r.emotion for r in recs if r.emotion]
            topics: Counter[str] = Counter()
            for r in recs:
                topics.update(r.topics or [])
            snapshots.append(
                GrowthSnapshot(
                    period_label=label,
                    entry_count=len(recs),
                    avg_sentiment=round(sum(sentiments) / len(sentiments), 4),
                    dominant_emotion=(
                        Counter(emotions).most_common(1)[0][0]
                        if emotions else "neutral"
                    ),
                    top_topic=topics.most_common(1)[0][0] if topics else "none",
                    snapshot_date=self._now(),
                )
            )
        return snapshots

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")
