# Dashboard Story Manual QA

## Purpose

Confirm the redesigned dashboard renders only backend-backed story claims from
`GET /api/dashboard/story?range=Last%2030%20days`.

## Mock Payload Checks

Use a mock response shaped like:

```json
{
  "range": "Last 30 days",
  "lookback_days": 30,
  "headline": {
    "baseline_sentiment": 0.1,
    "current_sentiment": 0.35,
    "sentiment_delta": 0.25,
    "dominant_emotion_start": "fear",
    "dominant_emotion_end": "joy",
    "recovery_speed_days_start": 2.0,
    "recovery_speed_days_end": 1.0,
    "entry_count": 6,
    "days_in_range": 30,
    "growth_score": 0.65,
    "growth_narrative": "From 2026-W32 to 2026-W33, average sentiment changed by +0.25.",
    "has_sufficient_data": true,
    "minimum_entry_count": 5
  },
  "top_working": [
    { "habit": "exercise", "mention_count": 5, "delta": 0.5, "confidence": 0.6, "explanation": "Exercise seems helpful." }
  ],
  "top_draining": [
    { "topic": "career", "frequency": 6, "avg_sentiment": -0.35, "dominant_emotion": "fear", "trend": "decreasing", "confidence": 0.6, "explanation": "Career has been draining." }
  ],
  "people": [
    { "person": "Alice", "mention_count": 4, "avg_sentiment": 0.4, "dominant_emotion": "joy", "last_mentioned": "2026-08-19T10:00:00Z", "trend": "improving", "relationship_type": "friend", "confidence": 0.6, "explanation": "Alice appears supportive." }
  ],
  "weekly_buckets": [
    { "label": "Aug 10-Aug 16", "avg_sentiment": 0.2, "dominant_emotion": "joy", "top_topic": "career", "entry_count": 4 }
  ],
  "forecast": {
    "sentiment_forecast": { "horizon_days": 7, "predicted_sentiment": 0.2, "direction": "stable", "confidence": 0.4, "explanation": "Based on sample entries." },
    "burnout_risk": { "risk_level": "low", "score": 0.1, "contributing_factors": [], "confidence": 0.4, "explanation": "Score 10% from 3 recent entries. This is a statistical pattern only, not a clinical assessment." }
  },
  "goals": [],
  "thresholds": { "min_insight_confidence": 0.5, "min_mention_count": 3, "min_entry_count": 5 }
}
```

Expected rendering:

- Hero sentence uses `dominant_emotion_start`, `dominant_emotion_end`,
  `days_in_range`, `sentiment_delta`, `baseline_sentiment`, and
  `current_sentiment` exactly from the payload.
- "What's Working For You" shows exercise because confidence `0.6 >= 0.5`.
- Any item with confidence `< thresholds.min_insight_confidence` does not render.
- Any person with mentions `< thresholds.min_mention_count` does not render.
- Burnout disclaimer text remains visible exactly as returned.

## Empty State Checks

Use `top_working: []`, `top_draining: []`, `people: []`, and
`weekly_buckets: []`.

Expected rendering:

- Working, draining, people, and week sections render explicit empty states.
- If `headline.has_sufficient_data` is false, the hero renders the low-data state
  and does not render a mood-shift conclusion.
