from __future__ import annotations


def clamp_sentiment(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(-1.0, min(1.0, float(value)))


def mood_score(value: float | None) -> int:
    """Translate VADER compound sentiment (-1..1) into a 0..100 mood score."""
    return int(round((clamp_sentiment(value) + 1.0) * 50.0))


def sentiment_label(value: float | None) -> str:
    raw = clamp_sentiment(value)
    if raw >= 0.55:
        return "strongly positive"
    if raw >= 0.2:
        return "positive"
    if raw > -0.2:
        return "mixed"
    if raw > -0.55:
        return "heavy"
    return "very heavy"


def delta_direction(delta: float | None) -> str:
    raw = float(delta or 0.0)
    if raw > 0.05:
        return "lighter"
    if raw < -0.05:
        return "heavier"
    return "steady"


def delta_band(delta: float | None) -> str:
    raw = abs(float(delta or 0.0))
    if raw < 0.05:
        return "steady"
    if raw < 0.15:
        return "slight"
    if raw < 0.35:
        return "noticeable"
    return "strong"


def delta_summary(delta: float | None) -> str:
    direction = delta_direction(delta)
    band = delta_band(delta)
    if direction == "steady":
        return "about steady"
    adverbs = {
        "slight": "slightly",
        "noticeable": "noticeably",
        "strong": "strongly",
    }
    return f"{adverbs.get(band, band)} {direction}"


def sentiment_summary(value: float | None) -> str:
    return f"{sentiment_label(value)} mood score {mood_score(value)}%"


def presence_label(mention_count: int) -> str:
    """Qualitative mention-frequency bands for relationship cards.

    Thresholds are intentionally simple and count-based:
    10+ mentions = often, 5-9 = recurring, 3-4 = occasional but repeated.
    """
    if mention_count >= 10:
        return "someone you mention often"
    if mention_count >= 5:
        return "a recurring presence"
    if mention_count >= 3:
        return "an occasional but repeated presence"
    return "an occasional presence"


def trend_phrase(sentiment_trend: str) -> str:
    article = "an" if sentiment_trend[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    return f"{article} {sentiment_trend} trend"


def relationship_impact_summary(
    *,
    person: str,
    relationship_type: str,
    mention_count: int,
    avg_sentiment: float,
    dominant_emotion: str,
    sentiment_trend: str,
    closeness_score: float,
) -> str:
    relation_labels = {
        "family": "family",
        "friend": "a friend",
        "colleague": "a colleague",
        "partner": "a partner",
        "other": "a support contact",
    }
    relation = relation_labels.get(relationship_type, "a recurring connection")
    return (
        f"{person} appears as {relation} across {mention_count} entries; "
        f"the pattern is {sentiment_label(avg_sentiment)}, most often {dominant_emotion}, "
        f"with {trend_phrase(sentiment_trend)}. "
        f"This reads as {presence_label(mention_count)}."
    )
