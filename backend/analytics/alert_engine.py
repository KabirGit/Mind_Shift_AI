from __future__ import annotations

import logging
from collections import Counter
from datetime import UTC, datetime

from pydantic import BaseModel

from backend.analytics._stats_utils import parse_ts
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)

_SEVERITY_RANK = {"flag": 0, "watch": 1, "info": 2}


class ProactiveAlert(BaseModel):
    severity: str  # "info" | "watch" | "flag"
    message: str
    evidence: str


class AlertEngine:
    """Rule-based proactive alerts over recent records. Deterministic, no LLM."""

    def __init__(self, db: JournalDB, pattern_engine) -> None:
        self.db = db
        self.pattern_engine = pattern_engine

    def check(self) -> list[ProactiveAlert]:
        recent = self.db.get_recent(limit=50)  # newest first
        alerts: list[ProactiveAlert] = []
        for rule in (
            self._consecutive_stress,
            self._trigger_spike,
            self._positive_streak,
            self._habit_absence,
        ):
            try:
                a = rule(recent)
                if a:
                    alerts.append(a)
            except Exception as exc:
                logger.exception("alert rule %s failed: %s", rule.__name__, exc)
        alerts.sort(key=lambda a: _SEVERITY_RANK.get(a.severity, 9))
        return alerts

    def _consecutive_stress(self, recent):
        n = 0
        for r in recent:  # newest first
            stressed = (
                r.emotion in {"sadness", "fear"} or r.sentiment_compound < -0.2
            )
            if stressed:
                n += 1
            else:
                break
        if n >= 3:
            return ProactiveAlert(
                severity="watch",
                message=f"You've had {n} consecutive difficult entries. How are you holding up?",
                evidence=f"{n} most-recent entries were sad/fearful or sentiment < -0.2.",
            )
        return None

    def _positive_streak(self, recent):
        n = 0
        for r in recent:
            if r.sentiment_compound > 0.2:
                n += 1
            else:
                break
        if n >= 4:
            return ProactiveAlert(
                severity="info",
                message=f"You've had {n} positive entries in a row — something's going well.",
                evidence=f"{n} most-recent entries had sentiment > 0.2.",
            )
        return None

    def _trigger_spike(self, recent):
        now = datetime.now(UTC)
        this_week: Counter[str] = Counter()
        last_week: Counter[str] = Counter()
        for r in recent:
            dt = parse_ts(r.timestamp)
            if dt is None:
                continue
            age = (now - dt).days
            if age < 7:
                this_week.update(r.topics or [])
            elif age < 14:
                last_week.update(r.topics or [])
        for topic, cur in this_week.items():
            prior = last_week.get(topic, 0)
            if cur > 2 * max(prior, 1) and cur >= 2 and prior >= 1:
                return ProactiveAlert(
                    severity="watch",
                    message=f"{topic} stress has doubled this week compared to last week.",
                    evidence=f"{topic}: {cur} mentions this week vs {prior} last week.",
                )
        return None

    def _habit_absence(self, recent):
        if len(recent) < 5:
            return None
        now = datetime.now(UTC)
        last14 = [r for r in recent if (dt := parse_ts(r.timestamp)) and (now - dt).days < 14]
        habit_days: Counter[str] = Counter()
        for r in last14:
            habit_days.update(set(r.habits or []))
        last5_habits = set()
        for r in recent[:5]:
            last5_habits.update(r.habits or [])
        for habit, days in habit_days.items():
            if days >= 3 and habit not in last5_habits:
                return ProactiveAlert(
                    severity="info",
                    message=(
                        f"You haven't mentioned {habit} recently — "
                        "you'd noted it helps your mood."
                    ),
                    evidence=f"{habit} appeared in {days} of the last 14 days, none of the last 5.",
                )
        return None
