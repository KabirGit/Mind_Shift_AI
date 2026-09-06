from __future__ import annotations

from typing import Any

from backend.guidance.action_planner import ActionPlanner
from backend.guidance.decision_parser import DecisionParser
from backend.guidance.decision_scorer import DecisionScorer
from backend.guidance.models import (
    DecisionContext,
    DecisionParseResult,
    GuidanceResult,
    ScoredOption,
)
from backend.guidance.option_generator import OptionGenerator
from backend.guidance.reversibility import ReversibilityChecker


class GuidanceService:
    """Coordinates the decision-support path as capabilities are added."""

    def __init__(
        self,
        decision_parser: DecisionParser,
        *,
        option_generator: OptionGenerator | None = None,
        scorer: DecisionScorer | None = None,
        reversibility_checker: ReversibilityChecker | None = None,
        action_planner: ActionPlanner | None = None,
    ) -> None:
        self.decision_parser = decision_parser
        self.option_generator = option_generator or OptionGenerator()
        self.scorer = scorer or DecisionScorer()
        self.reversibility_checker = reversibility_checker or ReversibilityChecker()
        self.action_planner = action_planner or ActionPlanner()

    def parse_decision(
        self,
        text: str,
        *,
        emotion: dict[str, Any] | None = None,
        extracted: dict[str, Any] | None = None,
    ) -> DecisionParseResult:
        return self.decision_parser.parse(
            text, emotion=emotion, extracted=extracted
        )

    def guide(
        self,
        parsed: DecisionParseResult,
        context: DecisionContext,
    ) -> GuidanceResult:
        if not parsed.is_decision or parsed.state is None:
            raise ValueError("A valid decision state is required for guidance")
        options = self.option_generator.generate(parsed.state, parsed.suggested_options)
        assessments = {
            option.id: self.reversibility_checker.assess(option, parsed.state)
            for option in options
        }
        scored = self.scorer.score(options, context, assessments)
        recommended = self._select_recommendation(scored, context.evidence_strength)
        strong = recommended is not None
        if recommended:
            recommendation = (
                f"The best-supported next step is {recommended.option.title}."
            )
            confidence = round(
                recommended.scores.total * (0.5 + context.evidence_strength * 0.5),
                4,
            )
        else:
            recommendation = "There is not enough evidence for a strong recommendation yet."
            confidence = round(
                min(scored[0].scores.total if scored else 0.0, context.evidence_strength),
                4,
            )
        evidence = self._evidence(context)
        uncertainty = list(
            dict.fromkeys([*parsed.state.missing_information, *context.missing_context])
        )[:8]
        blocked_option = (
            scored[0]
            if scored and scored[0].reversibility.cooling_off_required
            else None
        )
        return GuidanceResult(
            options=scored,
            recommended_option_id=recommended.option.id if recommended else None,
            recommendation=recommendation,
            strong_recommendation=strong,
            confidence=confidence,
            evidence=evidence,
            uncertainty=uncertainty,
            next_actions=self.action_planner.plan(
                context, recommended, blocked_option=blocked_option
            ),
        )

    @staticmethod
    def _select_recommendation(
        scored: list[ScoredOption], evidence_strength: float
    ) -> ScoredOption | None:
        if not scored or evidence_strength < 0.35 or scored[0].scores.total < 0.55:
            return None
        runner_up = scored[1].scores.total if len(scored) > 1 else 0.0
        if scored[0].scores.total - runner_up < 0.08:
            return None
        if scored[0].reversibility.cooling_off_required:
            return None
        return scored[0]

    @staticmethod
    def _evidence(context: DecisionContext) -> list[str]:
        evidence = [
            f"A similar entry was retrieved with relevance {memory.relevance:.2f}: "
            f"{memory.text[:140]}"
            for memory in context.similar_memories[:2]
        ]
        evidence.extend(item.summary for item in context.relevant_patterns[:1])
        evidence.extend(item.summary for item in context.goals[:1])
        return evidence[:4]

    @staticmethod
    def render_fallback(result: GuidanceResult) -> str:
        evidence = (
            "; ".join(result.evidence)
            if result.evidence
            else "No relevant personal history was available, so this is tentative."
        )
        uncertainty = (
            "; ".join(result.uncertainty[:3])
            if result.uncertainty
            else "No major missing information was identified."
        )
        actions = " ".join(
            f"{step.order}. {step.action}" for step in result.next_actions
        )
        return (
            "Validation: It makes sense to want a clearer, lower-risk way to approach "
            "this choice.\n\n"
            f"Recommendation: {result.recommendation}\n\n"
            f"Why: {evidence}\n\n"
            f"Uncertainty: {uncertainty}\n\n"
            f"Next steps: {actions}"
        )
