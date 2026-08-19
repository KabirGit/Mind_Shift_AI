from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from datetime import UTC, datetime

from backend.analytics._stats_utils import filter_window, parse_ts, sort_key
from backend.profile.models import UserProfile
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)


class ProfileManager:
    """Maintains a single-row persistent UserProfile in the JournalDB file."""

    def __init__(
        self,
        db: JournalDB,
        pattern_engine,
        habit_engine,
        relationship_engine,
    ) -> None:
        self.db = db
        self.pattern_engine = pattern_engine
        self.habit_engine = habit_engine
        self.relationship_engine = relationship_engine
        self._last_known = UserProfile()
        try:
            self._create_table()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("ProfileManager init failed: %s", exc)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_table(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS user_profiles "
                "(id TEXT PRIMARY KEY, profile_data TEXT)"
            )
            conn.commit()

    def load(self) -> UserProfile:
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT profile_data FROM user_profiles WHERE id = ?", ("default",)
                ).fetchone()
            if not row:
                return UserProfile()
            profile = UserProfile(**json.loads(row["profile_data"]))
            self._last_known = profile
            return profile
        except Exception as exc:
            logger.exception("ProfileManager.load failed: %s", exc)
            return self._last_known

    def _save(self, profile: UserProfile) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO user_profiles (id, profile_data) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET profile_data = excluded.profile_data",
                (profile.user_id, profile.model_dump_json()),
            )
            conn.commit()

    def update(self) -> UserProfile:
        try:
            records = self.db.get_all()
            if not records:
                profile = UserProfile(last_updated=self._now())
                self._save(profile)
                self._last_known = profile
                return profile

            sentiments = [r.sentiment_compound for r in records]
            baseline = sum(sentiments) / len(sentiments)

            last7 = filter_window(records, 7)
            current = (
                sum(r.sentiment_compound for r in last7) / len(last7) if last7 else baseline
            )

            emotions = [r.emotion for r in records if r.emotion]
            dominant = Counter(emotions).most_common(1)[0][0] if emotions else "neutral"

            avg_len = sum(len(r.text or "") for r in records) / len(records)
            if avg_len < 100:
                style = "brief"
            elif avg_len <= 300:
                style = "reflective"
            else:
                style = "detailed"

            recovery = self._recovery_speed(records)

            summary = self.pattern_engine.analyze(lookback_days=90)
            top_triggers = [
                t for t, _ in sorted(
                    summary.recurring_topics.items(), key=lambda kv: kv[1], reverse=True
                )
            ][:3]
            top_people = [
                p for p, _ in sorted(
                    summary.recurring_people.items(), key=lambda kv: kv[1], reverse=True
                )
            ][:3]

            habit_corrs = self.habit_engine.analyze(lookback_days=90)
            top_habits = [
                c.habit for c in sorted(
                    (c for c in habit_corrs if c.delta > 0.1),
                    key=lambda c: c.delta,
                    reverse=True,
                )
            ][:2]

            growth = max(0.0, min(1.0, (current - baseline + 1) / 2))

            profile = UserProfile(
                baseline_sentiment=round(baseline, 4),
                current_sentiment=round(current, 4),
                dominant_emotion=dominant,
                recovery_speed_days=round(recovery, 2),
                top_triggers=top_triggers,
                top_habits=top_habits,
                top_people=top_people,
                entry_count=len(records),
                last_updated=self._now(),
                growth_score=round(growth, 4),
                communication_style=style,
            )
            self._save(profile)
            self._last_known = profile
            return profile
        except Exception as exc:
            logger.exception("ProfileManager.update failed: %s", exc)
            return self._last_known

    def _recovery_speed(self, records) -> float:
        ordered = sorted(records, key=lambda r: sort_key(r.timestamp))
        deltas: list[float] = []
        for i in range(len(ordered) - 1):
            cur = ordered[i]
            nxt = ordered[i + 1]
            if cur.sentiment_compound < 0 and nxt.sentiment_compound > 0.1:
                t0 = parse_ts(cur.timestamp)
                t1 = parse_ts(nxt.timestamp)
                if t0 and t1:
                    deltas.append((t1 - t0).total_seconds() / 86400.0)
        return sum(deltas) / len(deltas) if deltas else 0.0

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")
