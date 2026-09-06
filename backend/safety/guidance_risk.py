from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel

RiskCategory = Literal["medical", "immediate_danger", "severe_mental_health"]


class GuidanceRiskResult(BaseModel):
    flagged: bool = False
    category: RiskCategory | None = None
    response: str | None = None


_DECISION_ACTION = re.compile(
    r"\b(?:should\s+i|can\s+i|do\s+i|stop|start|increase|decrease|change|"
    r"skip|double|take|quit|refuse|avoid)\b",
    re.IGNORECASE,
)
_MEDICAL = re.compile(
    r"\b(?:medication|medicine|prescription|dose|dosage|antidepressant|"
    r"antipsychotic|insulin|diagnos(?:is|ed)|treatment|therapy|therapist|"
    r"doctor|psychiatrist|pregnan(?:t|cy))\b",
    re.IGNORECASE,
)
_IMMEDIATE_DANGER = re.compile(
    r"\b(?:immediate\s+danger|not\s+safe|unsafe\s+(?:at\s+home|right\s+now)|"
    r"threaten(?:ed|ing)?\s+(?:me|us)|abusive|domestic\s+(?:abuse|violence)|"
    r"hit(?:ting)?\s+me|hurt(?:ing)?\s+me|stalk(?:ing|er|ed)?)\b",
    re.IGNORECASE,
)
_SEVERE_MENTAL_HEALTH = re.compile(
    r"\b(?:hearing\s+voices|voices\s+(?:tell|telling|command)|hallucinat(?:e|ing|ions?)|"
    r"psychosis|psychotic|manic\s+episode|delusion(?:s|al)?|"
    r"haven'?t\s+slept\s+(?:for\s+)?(?:three|four|five|six|seven|\d+)\s+days)\b",
    re.IGNORECASE,
)


_MESSAGES: dict[RiskCategory, str] = {
    "medical": (
        "This decision needs qualified medical guidance because changing medication, "
        "treatment, or a dose can carry serious risks. Please contact the prescribing "
        "clinician or a pharmacist before making the change. If severe symptoms are "
        "happening now, use local urgent or emergency care."
    ),
    "immediate_danger": (
        "Your immediate safety matters more than weighing decision options here. Move to "
        "a safer place if you can, contact someone you trust, and use local emergency or "
        "domestic-violence support if danger may be immediate. Avoid confronting the "
        "person alone."
    ),
    "severe_mental_health": (
        "These symptoms need prompt support from a qualified mental-health professional, "
        "not automated decision guidance. Contact a clinician or trusted person now; if "
        "you may be unable to stay safe or symptoms are escalating, use local emergency "
        "services."
    ),
}


class GuidanceRiskClassifier:
    """Conservative local gate for decisions outside this feature's safe scope."""

    def classify(self, text: str) -> GuidanceRiskResult:
        clean = " ".join((text or "").split())
        if not clean:
            return GuidanceRiskResult()
        if _IMMEDIATE_DANGER.search(clean):
            return self._result("immediate_danger")
        if _SEVERE_MENTAL_HEALTH.search(clean):
            return self._result("severe_mental_health")
        if _MEDICAL.search(clean) and _DECISION_ACTION.search(clean):
            return self._result("medical")
        return GuidanceRiskResult()

    @staticmethod
    def _result(category: RiskCategory) -> GuidanceRiskResult:
        return GuidanceRiskResult(
            flagged=True,
            category=category,
            response=_MESSAGES[category],
        )
