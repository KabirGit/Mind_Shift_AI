from __future__ import annotations

import re

from backend.guidance.models import (
    CandidateOption,
    DecisionContext,
    OptionScores,
    ReversibilityAssessment,
    ScoredOption,
)

_STOP = {"and", "before", "from", "into", "that", "the", "this", "with"}


def _tokens(value: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(value or "").casefold())
        if len(token) > 2 and token not in _STOP
    }


class DecisionScorer:
    WEIGHTS = {
        "goal_alignment": 0.30,
        "historical_evidence": 0.25,
        "feasibility": 0.20,
        "reversibility": 0.15,
        "downside_containment": 0.10,
    }

    def score(
        self,
        options: list[CandidateOption],
        context: DecisionContext,
        reversibility: dict[str, ReversibilityAssessment],
    ) -> list[ScoredOption]:
        scored = [
            self._score_one(option, context, reversibility[option.id])
            for option in options
        ]
        return sorted(scored, key=lambda item: item.scores.total, reverse=True)

    def _score_one(
        self,
        option: CandidateOption,
        context: DecisionContext,
        assessment: ReversibilityAssessment,
    ) -> ScoredOption:
        option_tokens = _tokens(f"{option.title} {option.description}")
        goal_alignment = self._goal_alignment(option_tokens, context)
        historical = self._historical_evidence(option_tokens, context)
        feasibility = self._feasibility(option_tokens, context)
        downside = 0.9 if assessment.cost_to_reverse == "low" else (
            0.6 if assessment.cost_to_reverse == "medium" else 0.3
        )
        values = {
            "goal_alignment": goal_alignment,
            "historical_evidence": historical,
            "feasibility": feasibility,
            "reversibility": assessment.score,
            "downside_containment": downside,
        }
        total = sum(values[key] * weight for key, weight in self.WEIGHTS.items())
        if assessment.cooling_off_required:
            total = min(total, 0.54)
        scores = OptionScores(
            **{key: round(value, 4) for key, value in values.items()},
            total=round(total, 4),
        )
        return ScoredOption(option=option, scores=scores, reversibility=assessment)

    @staticmethod
    def _goal_alignment(option_tokens: set[str], context: DecisionContext) -> float:
        state = context.current_decision
        goal_text = " ".join(
            [state.desired_outcome or "", *state.relevant_goals]
            + [goal.summary for goal in context.goals]
        )
        goal_tokens = _tokens(goal_text)
        if not goal_tokens:
            return 0.5
        overlap = len(option_tokens & goal_tokens) / max(1, len(goal_tokens))
        return min(1.0, 0.3 + overlap * 2.0)

    @staticmethod
    def _historical_evidence(
        option_tokens: set[str], context: DecisionContext
    ) -> float:
        matches = []
        for memory in context.similar_memories:
            if option_tokens & _tokens(f"{memory.text} {' '.join(memory.topics)}"):
                matches.append(memory.relevance)
        for item in [*context.relevant_patterns, *context.recent_decisions]:
            if option_tokens & _tokens(f"{item.summary} {item.metadata}"):
                matches.append(item.confidence)
        if matches:
            return min(1.0, max(matches))
        if context.similar_memories or context.relevant_patterns:
            return min(0.4, context.evidence_strength)
        return 0.0

    @staticmethod
    def _feasibility(option_tokens: set[str], context: DecisionContext) -> float:
        state = context.current_decision
        constraint_text = " ".join(state.constraints).casefold()
        irreversible_job_action = bool(
            option_tokens & {"quit", "resign", "leave"}
            and _tokens(constraint_text) & {"income", "money", "salary", "stable"}
        )
        if irreversible_job_action:
            return 0.25
        if option_tokens & {"gather", "ask", "clarify", "trial", "research", "compare"}:
            return 0.9
        return 0.65 if state.constraints else 0.55
