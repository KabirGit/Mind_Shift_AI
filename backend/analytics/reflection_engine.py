from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# Conservative, non-clinical language-pattern flags. Order matters: when more
# than two patterns match, the first two in this dict are used.
PATTERNS: dict[str, list[str]] = {
    "absolutist_language": [r"\balways\b", r"\bnever\b", r"\beveryone\b", r"\bno one\b"],
    "catastrophizing": [r"\bruined\b", r"\bdisaster\b", r"\bcan'?t handle\b", r"\bworst\b"],
    "self_blame": [r"\bmy fault\b", r"\bi always mess\b", r"\bi'?m such a\b"],
}

# One soft, pre-written reflective question per pattern (templates, not LLM).
_QUESTIONS: dict[str, str] = {
    "absolutist_language": (
        "You used some all-or-nothing language here — is there an exception to "
        "that, even a small one?"
    ),
    "catastrophizing": (
        "What's the most likely outcome here, separate from the worst-case one?"
    ),
    "self_blame": (
        "What part of this was actually within your control, and what wasn't?"
    ),
}


class ReflectionEngine:
    """Per-entry soft reflective-question generator. Rule-based, no LLM.

    Runs on the current message only (not aggregate history). Optionally adds
    one personalized question derived from a MemoryReplay (Phase 15b).
    """

    def __init__(self, replay_engine=None) -> None:
        self.replay_engine = replay_engine
        self._compiled = {
            name: [re.compile(p, re.IGNORECASE) for p in patterns]
            for name, patterns in PATTERNS.items()
        }

    def detect(self, text: str, replay: dict | None = None) -> list[str]:
        if not text or not text.strip():
            return []
        try:
            questions: list[str] = []
            for name, regexes in self._compiled.items():
                if any(rx.search(text) for rx in regexes):
                    questions.append(_QUESTIONS[name])
                if len(questions) >= 2:
                    break

            personalized = self._personalized_question(replay)
            if personalized:
                if len(questions) >= 2:
                    # Prefer the personalized one over a generic regex match.
                    questions[-1] = personalized
                else:
                    questions.append(personalized)
            return questions[:2]
        except Exception as exc:
            logger.exception("ReflectionEngine.detect failed: %s", exc)
            return []

    @staticmethod
    def _personalized_question(replay: dict | None) -> str | None:
        if not replay:
            return None
        days_ago = replay.get("days_ago")
        nxt = replay.get("next_entry_sentiment")
        # Only when the past situation resolved more positively.
        if days_ago is not None and nxt is not None and nxt > 0:
            return (
                f"You navigated something like this {days_ago} days ago. "
                "What was different about how you handled it then?"
            )
        return None
