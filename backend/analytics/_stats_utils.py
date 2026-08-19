from __future__ import annotations

from datetime import UTC, datetime, timedelta


def parse_ts(timestamp: str):
    if not timestamp:
        return None
    try:
        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError):
        return None


def sort_key(timestamp: str):
    dt = parse_ts(timestamp)
    return dt if dt is not None else datetime.min.replace(tzinfo=UTC)


def filter_window(records, lookback_days: int):
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    out = []
    for r in records:
        dt = parse_ts(r.timestamp)
        if dt is not None and dt >= cutoff:
            out.append(r)
    return out


def half_split_trend(
    sentiments: list[float],
    threshold: float = 0.1,
    up: str = "increasing",
    down: str = "decreasing",
    stable: str = "stable",
) -> str:
    """First-half vs second-half sentiment comparison (Phase 3 trend logic)."""
    n = len(sentiments)
    if n < 2:
        return stable
    mid = n // 2
    first = sentiments[:mid]
    second = sentiments[mid:]
    first_avg = sum(first) / len(first) if first else 0.0
    second_avg = sum(second) / len(second) if second else 0.0
    diff = second_avg - first_avg
    if diff > threshold:
        return up
    if diff < -threshold:
        return down
    return stable
