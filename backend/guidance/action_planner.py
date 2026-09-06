from __future__ import annotations

from backend.guidance.models import ActionStep, DecisionContext, ScoredOption


class ActionPlanner:
    def plan(
        self,
        context: DecisionContext,
        recommended: ScoredOption | None,
        blocked_option: ScoredOption | None = None,
    ) -> list[ActionStep]:
        if recommended is None:
            unknowns = context.missing_context[:2] or ["the most important trade-off"]
            clarification_steps: list[ActionStep] = []
            if blocked_option and blocked_option.reversibility.cooling_off_required:
                clarification_steps.append(
                    ActionStep(
                        order=1,
                        action=(
                            "Wait at least 24 hours before making the irreversible move, "
                            "then review it when the emotional intensity is lower."
                        ),
                        purpose="Avoid committing during a high-confidence negative emotion.",
                    )
                )
            clarification_steps.extend(
                ActionStep(
                    order=len(clarification_steps) + index,
                    action=f"Clarify {unknown}.",
                    purpose="Replace uncertainty with a concrete fact.",
                )
                for index, unknown in enumerate(unknowns, start=1)
            )
            clarification_steps.append(
                ActionStep(
                    order=len(clarification_steps) + 1,
                    action="Compare the options again using the new information.",
                    purpose="Revisit the decision before making a commitment.",
                )
            )
            return clarification_steps[:4]

        steps: list[ActionStep] = []
        if recommended.reversibility.cooling_off_required:
            steps.append(
                ActionStep(
                    order=1,
                    action="Wait at least 24 hours before making the irreversible move.",
                    purpose="Let the current emotional spike settle before committing.",
                )
            )
        steps.append(
            ActionStep(
                order=len(steps) + 1,
                action=f"Take one small reversible step toward: {recommended.option.title}.",
                purpose="Create evidence without overcommitting.",
            )
        )
        if context.missing_context:
            steps.append(
                ActionStep(
                    order=len(steps) + 1,
                    action=f"Verify {context.missing_context[0]}.",
                    purpose="Check the largest remaining uncertainty.",
                )
            )
        steps.append(
            ActionStep(
                order=len(steps) + 1,
                action="Set a review point and reconsider after observing the result.",
                purpose="Update the decision using what actually happens.",
            )
        )
        return steps[:4]
