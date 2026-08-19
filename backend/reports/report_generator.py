from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime

from fpdf import FPDF

logger = logging.getLogger(__name__)

_DISCLAIMER = "This is a reflection tool, not a medical or psychological diagnosis."
_NO_DATA = "Not enough data yet."


def _ascii(text: str) -> str:
    """fpdf2 core fonts are latin-1; strip anything outside it defensively."""
    return str(text).encode("latin-1", "replace").decode("latin-1")


class ReportGenerator:
    """Builds a local PDF weekly report from the analytics engines. No API."""

    def __init__(
        self,
        pattern_engine,
        habit_engine,
        relationship_engine,
        insight_engine,
        growth_tracker=None,
        prediction_engine=None,
    ) -> None:
        self.pattern_engine = pattern_engine
        self.habit_engine = habit_engine
        self.relationship_engine = relationship_engine
        self.insight_engine = insight_engine
        self.growth_tracker = growth_tracker
        self.prediction_engine = prediction_engine

    def generate(self, lookback_days: int = 7, out_path: str | None = None) -> bytes:
        try:
            summary = self.pattern_engine.analyze(lookback_days=lookback_days)
            habits = self.habit_engine.analyze(lookback_days=lookback_days)
            people = self.relationship_engine.analyze(lookback_days=lookback_days)
            insights = self.insight_engine.generate(lookback_days=lookback_days)
        except Exception as exc:
            logger.exception("Report data gathering failed: %s", exc)
            summary, habits, people, insights = None, [], [], []

        pdf = FPDF()
        pdf.set_compression(False)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("Helvetica", "B", 18)
        pdf.cell(0, 10, _ascii("Weekly Reflection Report"), ln=True)
        pdf.set_font("Helvetica", "", 10)
        now = datetime.now(UTC).strftime("%Y-%m-%d")
        pdf.cell(
            0, 7,
            _ascii(f"Date range: last {lookback_days} days  (generated {now})"),
            ln=True,
        )
        pdf.ln(3)

        self._story(pdf, summary, insights, lookback_days)
        self._emotional_summary(pdf, summary)
        self._triggers(pdf, summary)
        self._habits(pdf, habits)
        self._people(pdf, people)
        self._insights(pdf, insights)
        self._predictions(pdf)

        pdf.ln(6)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 5, _ascii(_DISCLAIMER))

        raw = pdf.output()
        data = bytes(raw)

        if out_path:
            try:
                with open(out_path, "wb") as f:
                    f.write(data)
            except Exception as exc:
                logger.exception("Failed to write report to %s: %s", out_path, exc)

        return data

    # --- sections ---

    def _section_title(self, pdf: FPDF, title: str) -> None:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 8, _ascii(title), ln=True)
        pdf.set_font("Helvetica", "", 10)

    def _story(self, pdf: FPDF, summary, insights, lookback_days: int) -> None:
        self._section_title(pdf, "Your Story This Week")
        parts: list[str] = []
        entry_count = summary.period_entry_count if summary else 0
        parts.append(f"You wrote {entry_count} entries this week.")
        if summary and summary.recurring_emotions:
            dom = max(summary.recurring_emotions.items(), key=lambda kv: kv[1])[0]
            parts.append(f"Your dominant mood was {dom}.")
        if summary and summary.triggers:
            top = max(summary.triggers, key=lambda t: t.frequency)
            parts.append(
                f"{top.topic.capitalize()} came up most, typically with "
                f"{top.avg_sentiment:+.2f} sentiment."
            )
        if self.growth_tracker is not None:
            with contextlib.suppress(Exception):
                parts.append(self.growth_tracker.narrative())
        if self.prediction_engine is not None:
            try:
                risk = self.prediction_engine.assess_burnout_risk()
                if risk.risk_level != "low":
                    parts.append("Your stress indicators are worth watching this week.")
            except Exception:
                pass
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, _ascii(" ".join(parts)))

    def _predictions(self, pdf: FPDF) -> None:
        if self.prediction_engine is None:
            return
        self._section_title(pdf, "Predictions")
        try:
            fc = self.prediction_engine.forecast_sentiment()
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                0, 6,
                _ascii(
                    f"Based on recent trends, your mood is {fc.direction} over the "
                    f"next {fc.horizon_days} days."
                ),
            )
            pdf.set_font("Helvetica", "I", 8)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                0, 5,
                _ascii("This is a statistical pattern only, not a clinical assessment."),
            )
            pdf.set_font("Helvetica", "", 10)
        except Exception as exc:
            logger.exception("report predictions section failed: %s", exc)

    def _emotional_summary(self, pdf: FPDF, summary) -> None:
        self._section_title(pdf, "Emotional Summary")
        if not summary or summary.period_entry_count == 0 or not summary.recurring_emotions:
            pdf.cell(0, 6, _ascii(_NO_DATA), ln=True)
            return
        pdf.cell(0, 6, _ascii(f"Entries in period: {summary.period_entry_count}"), ln=True)
        for emotion, count in sorted(
            summary.recurring_emotions.items(), key=lambda kv: kv[1], reverse=True
        ):
            pdf.cell(0, 6, _ascii(f"  - {emotion}: {count}"), ln=True)

    def _triggers(self, pdf: FPDF, summary) -> None:
        self._section_title(pdf, "Top Triggers")
        triggers = summary.triggers if summary else []
        if not triggers:
            pdf.cell(0, 6, _ascii(_NO_DATA), ln=True)
            return
        for t in triggers:
            pdf.cell(
                0, 6,
                _ascii(
                    f"  - {t.topic}: {t.frequency}x, avg sentiment "
                    f"{t.avg_sentiment:+.2f}, {t.dominant_emotion}, trend {t.trend}"
                ),
                ln=True,
            )

    def _habits(self, pdf: FPDF, habits) -> None:
        self._section_title(pdf, "Habits & Mood")
        if not habits:
            pdf.cell(0, 6, _ascii(_NO_DATA), ln=True)
            return
        for h in habits:
            pdf.cell(
                0, 6,
                _ascii(
                    f"  - {h.habit}: {h.mention_count}x, delta {h.delta:+.2f} "
                    f"({h.correlation_label})"
                ),
                ln=True,
            )

    def _people(self, pdf: FPDF, people) -> None:
        self._section_title(pdf, "People")
        if not people:
            pdf.cell(0, 6, _ascii(_NO_DATA), ln=True)
            return
        for p in people:
            pdf.cell(
                0, 6,
                _ascii(
                    f"  - {p.person}: {p.mention_count}x, avg sentiment "
                    f"{p.avg_sentiment:+.2f}, {p.dominant_emotion}, trend {p.trend}"
                ),
                ln=True,
            )

    def _insights(self, pdf: FPDF, insights) -> None:
        self._section_title(pdf, "Insights")
        if not insights:
            pdf.cell(0, 6, _ascii(_NO_DATA), ln=True)
            return
        for line in insights:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 6, _ascii(f"  - {line}"))
