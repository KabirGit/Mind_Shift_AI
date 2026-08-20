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


def recency_decay(
    timestamp: str,
    half_life_hours: float = 72.0,
    *,
    now: datetime | None = None,
    default: float = 0.3,
) -> float:
    """Recency score using the retriever's half-life formula."""
    dt = parse_ts(timestamp)
    if dt is None:
        return default
    anchor = now or datetime.now(UTC)
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=UTC)
    age_hours = max(0.0, (anchor - dt).total_seconds() / 3600.0)
    return 0.5 ** (age_hours / max(1.0, half_life_hours))


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


def recovery_speed_days(records) -> float:
    """Average days from a negative entry to the next positive entry."""
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
