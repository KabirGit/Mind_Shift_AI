from __future__ import annotations

import re
from typing import Literal

from backend.guidance.models import CandidateOption, DecisionState, ReversibilityAssessment

_IRREVERSIBLE = re.compile(
    r"\b(?:quit|resign|leave|end|break\s*up|divorce|sell|buy|purchase|sign|"
    r"relocate|move\s+away|publish)\b",
    re.IGNORECASE,
)
_LOW_COST = re.compile(
    r"\b(?:ask|clarify|gather|pause|review|wait|trial|test|research|compare|talk)\b",
    re.IGNORECASE,
)
_NEGATIVE_EMOTIONS = {
    "anger",
    "annoyance",
    "confusion",
    "disappointment",
    "fear",
    "grief",
    "nervousness",
    "sadness",
    "stress",
}


class ReversibilityChecker:
    def assess(
        self, option: CandidateOption, state: DecisionState
    ) -> ReversibilityAssessment:
        text = f"{option.title} {option.description}"
        if _IRREVERSIBLE.search(text):
            reversible = False
            score = 0.2
            cost: Literal["low", "medium", "high"] = "high"
            rationale = "This appears difficult or costly to undo once committed."
        elif _LOW_COST.search(text):
            reversible = True
            score = 0.95
            cost = "low"
            rationale = "This is an information-gathering or trial action that is easy to revise."
        else:
            reversible = True
            score = 0.65
            cost = "medium"
            rationale = "This appears adjustable, but the cost of changing course is not fully known."

        cooling_off = (
            not reversible
            and state.current_emotion in _NEGATIVE_EMOTIONS
            and state.emotion_confidence >= 0.70
        )
        return ReversibilityAssessment(
            option_id=option.id,
            reversible=reversible,
            score=score,
            cost_to_reverse=cost,
            cooling_off_required=cooling_off,
            rationale=rationale,
        )
