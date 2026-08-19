from __future__ import annotations

import re

# Conservative, fixed set of high-risk phrases. Case-insensitive substring/
# regex matching. No ML model, no network — runs in microseconds.
# NOTE: For a real deployment, this list and the resource message below should
# be localized to the user's country/region and reviewed by professionals.
_CRISIS_PATTERNS: list[str] = [
    r"\bkill myself\b",
    r"\bkilling myself\b",
    r"\bend my life\b",
    r"\bending my life\b",
    r"\btake my own life\b",
    r"\bsuicidal\b",
    r"\bsuicide\b",
    r"\bwant to die\b",
    r"\bwanna die\b",
    r"\bdon'?t want to live\b",
    r"\bno reason to live\b",
    r"\bnothing to live for\b",
    r"\bharm myself\b",
    r"\bhurt myself\b",
    r"\bself[- ]harm\b",
    r"\bcut myself\b",
    r"\bbetter off dead\b",
    r"\bcan'?t go on\b",
    r"\bgive up on life\b",
]

# Calm, non-diagnostic message. Hotline is a placeholder to be localized.
CRISIS_MESSAGE = (
    "It sounds like you may be going through something extremely painful right now, "
    "and I want you to know you are not alone. I'm not able to provide crisis care, "
    "but please consider reaching out to a trained person who can: contact your local "
    "emergency number, or a crisis line such as 988 (US Suicide & Crisis Lifeline) or "
    "your country's equivalent. If you are in immediate danger, please call emergency "
    "services now."
)


class CrisisDetector:
    """Rule-based crisis language detector. Pure local matching, no LLM."""

    def __init__(self) -> None:
        self._regexes = [re.compile(p, re.IGNORECASE) for p in _CRISIS_PATTERNS]

    def check(self, text: str) -> dict:
        if not text or not text.strip():
            return {"flagged": False, "matched_terms": []}
        matched: list[str] = []
        for rx in self._regexes:
            m = rx.search(text)
            if m:
                matched.append(m.group(0).lower())
        return {"flagged": bool(matched), "matched_terms": matched}
