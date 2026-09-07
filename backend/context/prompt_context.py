from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Literal

from backend.guidance.models import (
    ContextSelectionReport,
    ConversationTurn,
    DecisionState,
    EvidenceItem,
    MemoryEvidence,
    PromptContext,
)

_PROFILE_ALLOWLIST = (
    "entry_count",
    "growth_score",
    "dominant_emotion",
    "recovery_speed_days",
    "top_triggers",
    "communication_style",
)


@dataclass(frozen=True)
class ContextPolicy:
    budget_tokens: int = 1_500
    characters_per_token: int = 4
    max_memories: int = 3
    max_patterns: int = 3
    max_goals: int = 2
    max_recent_decisions: int = 2
    max_conversation_turns: int = 6
    near_duplicate_threshold: float = 0.82

    @property
    def budget_characters(self) -> int:
        return self.budget_tokens * self.characters_per_token


class PromptContextPacker:
    """Apply source caps, deduplication, and one global guidance budget."""

    def __init__(self, policy: ContextPolicy | None = None) -> None:
        self.policy = policy or ContextPolicy()

    def pack(
        self,
        *,
        state: DecisionState,
        memories: list[MemoryEvidence],
        patterns: list[EvidenceItem],
        goals: list[EvidenceItem],
        profile: dict[str, Any] | None,
        recent_decisions: list[EvidenceItem],
        conversation_history: list[dict[str, str]] | None = None,
    ) -> PromptContext:
        included: dict[str, int] = {}
        deduplicated: dict[str, int] = {}
        truncated: dict[str, int] = {}
        dropped: dict[str, int] = {}
        seen_text: list[str] = []

        decision = self._decision_payload(state, truncated)
        payload: dict[str, Any] = {
            "decision": decision,
            "retrieval_evidence": [],
            "relevant_patterns": [],
            "goals": [],
            "profile_summary": None,
            "related_prior_decisions": [],
            "conversation_history": [],
        }

        def add_items(
            source: str,
            destination: str,
            items: list[Any],
            cap: int,
            text_getter: Any,
        ) -> None:
            if len(items) > cap:
                truncated[source] = truncated.get(source, 0) + len(items) - cap
            for item in items[:cap]:
                text = str(text_getter(item))
                if self._is_duplicate(text, seen_text):
                    deduplicated[source] = deduplicated.get(source, 0) + 1
                    continue
                dumped = item.model_dump(mode="json")
                candidate = {**payload, destination: [*payload[destination], dumped]}
                if self._size(candidate) > self.policy.budget_characters:
                    dropped[source] = dropped.get(source, 0) + 1
                    continue
                payload[destination].append(dumped)
                seen_text.append(text)
                included[source] = included.get(source, 0) + 1

        add_items(
            "memories",
            "retrieval_evidence",
            memories,
            self.policy.max_memories,
            lambda item: item.text,
        )
        add_items(
            "patterns",
            "relevant_patterns",
            patterns,
            self.policy.max_patterns,
            lambda item: item.summary,
        )
        add_items(
            "goals",
            "goals",
            goals,
            self.policy.max_goals,
            lambda item: item.summary,
        )

        profile_summary = self.allowlisted_profile(profile)
        if profile_summary:
            candidate = {**payload, "profile_summary": profile_summary}
            if self._size(candidate) <= self.policy.budget_characters:
                payload["profile_summary"] = profile_summary
                included["profile"] = 1
            else:
                dropped["profile"] = 1

        add_items(
            "recent_decisions",
            "related_prior_decisions",
            recent_decisions,
            self.policy.max_recent_decisions,
            lambda item: item.summary,
        )

        history = list(conversation_history or [])[-self.policy.max_conversation_turns :]
        if len(conversation_history or []) > len(history):
            truncated["conversation_history"] = len(conversation_history or []) - len(history)
        for raw in history:
            raw_role = raw.get("role", "user")
            role: Literal["user", "assistant"] = (
                "assistant" if raw_role == "assistant" else "user"
            )
            content = " ".join(str(raw.get("content", "")).split())
            clipped = content[:500]
            if len(clipped) < len(content):
                truncated["conversation_history"] = (
                    truncated.get("conversation_history", 0) + 1
                )
            if not clipped:
                dropped["conversation_history"] = (
                    dropped.get("conversation_history", 0) + 1
                )
                continue
            if self._is_duplicate(clipped, seen_text):
                deduplicated["conversation_history"] = (
                    deduplicated.get("conversation_history", 0) + 1
                )
                continue
            turn = ConversationTurn(role=role, content=clipped)
            dumped = turn.model_dump(mode="json")
            candidate = {
                **payload,
                "conversation_history": [*payload["conversation_history"], dumped],
            }
            if self._size(candidate) > self.policy.budget_characters:
                dropped["conversation_history"] = (
                    dropped.get("conversation_history", 0) + 1
                )
                continue
            payload["conversation_history"].append(dumped)
            seen_text.append(clipped)
            included["conversation_history"] = included.get("conversation_history", 0) + 1

        estimated_tokens = math.ceil(
            self._size(payload) / self.policy.characters_per_token
        )
        return PromptContext.model_validate(
            {
                **payload,
                "selection": ContextSelectionReport(
                    included=included,
                    deduplicated=deduplicated,
                    truncated=truncated,
                    dropped=dropped,
                    estimated_tokens=estimated_tokens,
                    budget_tokens=self.policy.budget_tokens,
                ),
            }
        )

    @staticmethod
    def allowlisted_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
        if not profile:
            return None
        safe = {key: profile[key] for key in _PROFILE_ALLOWLIST if key in profile}
        if isinstance(safe.get("top_triggers"), list):
            safe["top_triggers"] = [
                " ".join(str(item).split())[:120]
                for item in safe["top_triggers"][:5]
            ]
        return safe or None

    @staticmethod
    def _decision_payload(
        state: DecisionState, truncated: dict[str, int]
    ) -> dict[str, Any]:
        def clip(value: str, limit: int) -> str:
            cleaned = " ".join(value.split())
            if len(cleaned) > limit:
                truncated["decision"] = truncated.get("decision", 0) + 1
            return cleaned[:limit]

        return {
            "problem": clip(state.problem, 1_200),
            "desired_outcome": (
                clip(state.desired_outcome, 500) if state.desired_outcome else None
            ),
            "available_options": [clip(item, 300) for item in state.available_options[:4]],
            "constraints": [clip(item, 250) for item in state.constraints[:5]],
            "relevant_goals": [clip(item, 250) for item in state.relevant_goals[:5]],
            "fears": [clip(item, 250) for item in state.fears[:5]],
            "missing_information": [
                clip(item, 250) for item in state.missing_information[:5]
            ],
            "current_emotion": state.current_emotion,
            "emotion_confidence": state.emotion_confidence,
            "uncertainty": state.uncertainty,
        }

    def _is_duplicate(self, text: str, seen: list[str]) -> bool:
        normalized = self._normalized(text)
        if not normalized:
            return True
        tokens = set(re.findall(r"[a-z0-9]+", normalized))
        for prior in seen:
            prior_normalized = self._normalized(prior)
            if normalized == prior_normalized:
                return True
            prior_tokens = set(re.findall(r"[a-z0-9]+", prior_normalized))
            union = tokens | prior_tokens
            if union and len(tokens & prior_tokens) / len(union) >= self.policy.near_duplicate_threshold:
                return True
        return False

    @staticmethod
    def _normalized(text: str) -> str:
        return " ".join(text.casefold().split())

    @staticmethod
    def _size(payload: dict[str, Any]) -> int:
        return len(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
