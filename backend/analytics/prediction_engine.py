from __future__ import annotations

import logging

import numpy as np
from pydantic import BaseModel, Field

from backend.analytics._stats_utils import filter_window, parse_ts, sort_key
from backend.analytics.models import compute_confidence
from backend.storage.db import JournalDB

logger = logging.getLogger(__name__)

_DISCLAIMER = "This is a statistical pattern only, not a clinical assessment."


class SentimentForecast(BaseModel):
    horizon_days: int = 7
    predicted_sentiment: float = 0.0
    direction: str = "stable"  # "improving" | "declining" | "stable"
    confidence: float = 0.0
    forecast_accuracy_note: str = (
        "Forecast accuracy is not tracked yet because historical forecasts are not "
        "persisted for later comparison."
    )
    explanation: str = ""


class BurnoutRisk(BaseModel):
    risk_level: str = "low"  # "low" | "medium" | "high"
    score: float = 0.0
    contributing_factors: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    explanation: str = ""


class PredictionEngine:
    """Forward-looking forecasts via linear fit + rule-based burnout score.

    Deterministic, no LLM. numpy used for the linear regression only.
    """

    def __init__(self, db: JournalDB) -> None:
        self.db = db

    def _xy(self, records):
        ordered = sorted(records, key=lambda r: sort_key(r.timestamp))
        t0 = parse_ts(ordered[0].timestamp)
        xs, ys = [], []
        for r in ordered:
            dt = parse_ts(r.timestamp)
            if dt is None or t0 is None:
                continue
            xs.append((dt - t0).total_seconds() / 86400.0)
            ys.append(r.sentiment_compound)
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    def forecast_sentiment(self, days_back: int = 14, horizon: int = 7) -> SentimentForecast:
        try:
            records = filter_window(self.db.get_all(), days_back)
            if len(records) < 3:
                return SentimentForecast(
                    horizon_days=horizon,
                    confidence=0.0,
                    direction="stable",
                    explanation="Not enough data yet.",
                )
            x, y = self._xy(records)
            if len(x) < 3:
                return SentimentForecast(
                    horizon_days=horizon, explanation="Not enough data yet."
                )
            m, b = np.polyfit(x, y, 1)
            pred = m * (x.max() + horizon) + b
            ss_res = float(np.sum((y - (m * x + b)) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = 0.0 if ss_tot == 0 else 1 - ss_res / ss_tot

            if m > 0.005:
                direction = "improving"
            elif m < -0.005:
                direction = "declining"
            else:
                direction = "stable"

            return SentimentForecast(
                horizon_days=horizon,
                predicted_sentiment=round(float(max(-1.0, min(1.0, pred))), 4),
                direction=direction,
                confidence=round(max(0.0, min(1.0, r2)), 4),
                explanation=(
                    f"Based on {len(records)} entries. Trend slope: {m:+.3f}/day. "
                    f"R\u00b2: {r2:.2f}."
                ),
            )
        except Exception as exc:
            logger.exception("forecast_sentiment failed: %s", exc)
            return SentimentForecast(horizon_days=horizon, explanation="Not enough data yet.")

    def assess_burnout_risk(self) -> BurnoutRisk:
        try:
            records = filter_window(self.db.get_all(), 14)
            total = len(records)
            if total < 3:
                return BurnoutRisk(
                    risk_level="low",
                    score=0.0,
                    confidence=0.0,
                    explanation="Not enough data to assess. " + _DISCLAIMER,
                )

            neg_ratio = sum(1 for r in records if r.sentiment_compound < -0.1) / total
            stress_ratio = sum(
                1 for r in records if r.emotion in {"sadness", "fear", "anger"}
            ) / total

            x, y = self._xy(records)
            m = float(np.polyfit(x, y, 1)[0]) if len(x) >= 2 else 0.0
            slope_component = max(0.0, -m * 10)

            career_ratio = sum(1 for r in records if "career" in (r.topics or [])) / total
            high_freq_career = 1.0 if career_ratio > 0.4 else 0.0

            c_neg = 0.35 * neg_ratio
            c_stress = 0.30 * stress_ratio
            c_slope = 0.20 * slope_component
            c_career = 0.15 * high_freq_career
            score = max(0.0, min(1.0, c_neg + c_stress + c_slope + c_career))

            if score > 0.65:
                level = "high"
            elif score > 0.35:
                level = "medium"
            else:
                level = "low"

            factors = []
            if c_neg > 0.05:
                factors.append("frequent negative entries")
            if c_stress > 0.05:
                factors.append("stress-linked emotions")
            if c_slope > 0.05:
                factors.append("declining sentiment trend")
            if c_career > 0.05:
                factors.append("high career focus")

            return BurnoutRisk(
                risk_level=level,
                score=round(score, 4),
                contributing_factors=factors,
                confidence=compute_confidence(total),
                explanation=(
                    f"Score {score:.0%} from {total} recent entries. " + _DISCLAIMER
                ),
            )
        except Exception as exc:
            logger.exception("assess_burnout_risk failed: %s", exc)
            return BurnoutRisk(explanation="Not enough data to assess. " + _DISCLAIMER)
