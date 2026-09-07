from __future__ import annotations

from backend.guidance.models import CandidateOption, DecisionState, SuggestedOption


class OptionGenerator:
    """Normalizes stated/model-proposed choices and adds generic safe alternatives."""

    MAX_OPTIONS = 4

    def generate(
        self,
        state: DecisionState,
        suggested: list[SuggestedOption] | None = None,
    ) -> list[CandidateOption]:
        candidates: list[CandidateOption] = []
        for title in state.available_options:
            candidates.append(
                CandidateOption(
                    id="pending",
                    title=title,
                    benefits=["Keeps a choice explicitly raised by the user in view"],
                    costs=["Its trade-offs still need to be validated"],
                    source="user",
                )
            )
        for option in suggested or []:
            candidates.append(
                CandidateOption(
                    id="pending",
                    title=option.title,
                    description=option.description,
                    benefits=option.benefits,
                    costs=option.costs,
                    source=option.source,
                )
            )

        candidates = self._dedupe(candidates)
        generic = (
            CandidateOption(
                id="pending",
                title="Gather key information before deciding",
                description="Resolve the most important unknowns before committing.",
                benefits=["Reduces uncertainty", "Avoids acting on an untested assumption"],
                costs=["Delays the final decision"],
                source="fallback",
            ),
            CandidateOption(
                id="pending",
                title="Run a small reversible trial",
                description="Test part of the preferred direction without a full commitment.",
                benefits=["Creates direct evidence", "Limits downside"],
                costs=["May not answer every question"],
                source="fallback",
            ),
        )
        for generic_option in generic:
            if len(candidates) >= 2:
                break
            candidates.append(generic_option)

        return [
            option.model_copy(update={"id": f"option_{index}"})
            for index, option in enumerate(candidates[: self.MAX_OPTIONS], start=1)
        ]

    @staticmethod
    def _dedupe(options: list[CandidateOption]) -> list[CandidateOption]:
        seen: set[str] = set()
        result = []
        for option in options:
            key = " ".join(option.title.casefold().split())
            if not key or key in seen:
                continue
            seen.add(key)
            result.append(option)
        return result
