from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable
from typing import Any

from backend.context.prompt_context import ContextPolicy, PromptContextPacker
from backend.guidance.decision_parser import DecisionParser
from backend.guidance.models import (
    DecisionContext,
    DecisionState,
    EmotionalContext,
    EvidenceItem,
    MemoryEvidence,
)

logger = logging.getLogger(__name__)


def _normalized(text: object) -> str:
    return " ".join(str(text or "").casefold().split())


def _tokens(text: object) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", _normalized(text))
        if len(token) > 2
    }


class DecisionContextBuilder:
    """Read-only adapters over existing retrieval, journal, and analytics services."""

    def __init__(
        self,
        *,
        retriever: Any,
        journal_db: Any,
        pattern_engine: Any,
        profile_manager: Any,
        goal_engine: Any,
        context_policy: ContextPolicy | None = None,
    ) -> None:
        self.retriever = retriever
        self.journal_db = journal_db
        self.pattern_engine = pattern_engine
        self.profile_manager = profile_manager
        self.goal_engine = goal_engine
        self.context_packer = PromptContextPacker(context_policy)

    def search_similar_memories(
        self,
        state: DecisionState,
        *,
        current_text: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        query = self._retrieval_query(state)
        current_hash = self._content_hash(current_text)
        try:
            results = self.retriever.retrieve(
                query=query,
                query_emotion=state.current_emotion,
                top_k=max(top_k + 4, 7),
                exclude_entry_hashes={current_hash},
                diversify=True,
            )
        except TypeError:
            # Backward compatibility for externally injected legacy retrievers.
            results = self.retriever.retrieve(
                query=query,
                query_emotion=state.current_emotion,
                top_k=max(top_k + 4, 7),
            )
        current = _normalized(current_text)
        return [
            item
            for item in results
            if _normalized(item.get("metadata", {}).get("text")) != current
        ][:top_k]

    def get_emotional_patterns(self, state: DecisionState) -> dict[str, Any]:
        summary = self.pattern_engine.analyze(lookback_days=90)
        decision_tokens = self._decision_tokens(state)
        triggers: list[dict[str, Any]] = []
        for trigger in getattr(summary, "triggers", []) or []:
            topic = str(getattr(trigger, "topic", ""))
            if decision_tokens and not (_tokens(topic) & decision_tokens):
                continue
            triggers.append(
                {
                    "topic": topic,
                    "dominant_emotion": str(
                        getattr(trigger, "dominant_emotion", "neutral")
                    ),
                    "trend": str(getattr(trigger, "trend", "stable")),
                    "confidence": float(getattr(trigger, "confidence", 0.0)),
                    "explanation": str(getattr(trigger, "explanation", "")),
                }
            )
        triggers.sort(key=lambda item: float(item["confidence"]), reverse=True)
        return {
            "recurring_emotions": dict(
                getattr(summary, "recurring_emotions", {}) or {}
            ),
            "triggers": triggers[:3],
        }

    def get_user_profile(self, state: DecisionState) -> dict[str, Any]:
        """Read persisted profile and goals; this tool never updates storage."""

        profile = self.profile_manager.load()
        profile_data = (
            profile.model_dump() if hasattr(profile, "model_dump") else dict(profile)
        )
        raw_goals = self.goal_engine.analyze(lookback_days=90)
        decision_tokens = self._decision_tokens(state)
        goals = []
        for goal in raw_goals:
            data = goal.model_dump() if hasattr(goal, "model_dump") else dict(goal)
            label = str(data.get("goal_keyword", ""))
            if decision_tokens and _tokens(label) & decision_tokens:
                goals.append(data)
        if not goals:
            goals = [
                goal.model_dump() if hasattr(goal, "model_dump") else dict(goal)
                for goal in raw_goals[:2]
            ]
        return {"profile": profile_data, "goals": goals[:2]}

    def get_recent_decision_context(
        self,
        state: DecisionState,
        *,
        current_text: str,
    ) -> list[dict[str, Any]]:
        current = _normalized(current_text)
        decision_tokens = self._decision_tokens(state)
        matches = []
        for record in self.journal_db.get_recent(limit=30):
            if _normalized(record.text) == current:
                continue
            if not DecisionParser.is_decision_candidate(record.text):
                continue
            record_topics = set(getattr(record, "topics", []) or [])
            overlap = bool(decision_tokens & _tokens(" ".join(record_topics)))
            if decision_tokens and not overlap and not (
                decision_tokens & _tokens(record.text)
            ):
                continue
            matches.append(
                {
                    "text": record.text,
                    "timestamp": record.timestamp,
                    "emotion": record.emotion,
                    "topics": list(record_topics),
                }
            )
        return matches[:2]

    def build(
        self,
        state: DecisionState,
        *,
        current_text: str,
        top_k: int = 3,
        recent_history: list[dict[str, str]] | None = None,
    ) -> DecisionContext:
        """Non-agent convenience path that degrades each read independently."""

        tool_results = {
            "search_similar_memories": self._safe(
                lambda: self.search_similar_memories(
                    state, current_text=current_text, top_k=top_k
                ),
                [],
            ),
            "get_emotional_patterns": self._safe(
                lambda: self.get_emotional_patterns(state),
                {"recurring_emotions": {}, "triggers": []},
            ),
            "get_user_profile": self._safe(
                lambda: self.get_user_profile(state), {"profile": None, "goals": []}
            ),
            "get_recent_decision_context": self._safe(
                lambda: self.get_recent_decision_context(
                    state, current_text=current_text
                ),
                [],
            ),
        }
        return self.assemble(state, tool_results, recent_history=recent_history)

    def assemble(
        self,
        state: DecisionState,
        tool_results: dict[str, Any],
        *,
        recent_history: list[dict[str, str]] | None = None,
    ) -> DecisionContext:
        memories = [
            self._memory_evidence(item)
            for item in tool_results.get("search_similar_memories", [])[:3]
        ]
        emotional = tool_results.get("get_emotional_patterns", {}) or {}
        patterns = [
            EvidenceItem(
                source="pattern",
                summary=str(item.get("explanation") or item.get("topic") or ""),
                confidence=float(item.get("confidence", 0.0)),
                metadata={
                    key: item.get(key)
                    for key in ("topic", "dominant_emotion", "trend")
                },
            )
            for item in emotional.get("triggers", [])[:3]
            if item.get("explanation") or item.get("topic")
        ]
        profile_result = tool_results.get("get_user_profile", {}) or {}
        profile = profile_result.get("profile")
        goals = [
            self._goal_evidence(item) for item in profile_result.get("goals", [])[:2]
        ]
        recent = [
            EvidenceItem(
                source="recent_decision",
                summary=str(item.get("text", ""))[:500],
                confidence=0.5,
                metadata={
                    "timestamp": item.get("timestamp"),
                    "emotion": item.get("emotion", "neutral"),
                    "topics": item.get("topics", []),
                },
            )
            for item in tool_results.get("get_recent_decision_context", [])[:2]
            if item.get("text")
        ]

        strength = self._evidence_strength(state, memories, patterns, profile, goals, recent)
        missing = list(state.missing_information)
        if not memories and not recent:
            missing.append("relevant personal history")
        if not patterns:
            missing.append("relevant emotional or behavioral pattern")
        if not profile and not goals:
            missing.append("profile or goal evidence")

        context = DecisionContext(
            current_decision=state,
            similar_memories=memories,
            emotional_context=EmotionalContext(
                current_emotion=state.current_emotion,
                current_confidence=state.emotion_confidence,
                recurring_emotions=emotional.get("recurring_emotions", {}),
            ),
            relevant_patterns=patterns,
            profile=profile,
            goals=goals,
            recent_decisions=recent,
            evidence_strength=strength,
            missing_context=list(dict.fromkeys(missing))[:8],
        )
        return context.model_copy(
            update={
                "prompt_context": self.context_packer.pack(
                    state=state,
                    memories=memories,
                    patterns=patterns,
                    goals=goals,
                    profile=profile,
                    recent_decisions=recent,
                    conversation_history=recent_history,
                )
            }
        )

    @staticmethod
    def _safe(fn: Callable[[], Any], default: Any) -> Any:
        try:
            return fn()
        except Exception as exc:
            logger.info(
                "Decision context source unavailable error_type=%s", type(exc).__name__
            )
            return default

    @staticmethod
    def _memory_evidence(item: dict[str, Any]) -> MemoryEvidence:
        meta = item.get("metadata", {})
        scores = item.get("scores", {})
        combined = scores.get("combined")
        if combined is None:
            combined = scores.get("semantic", 0.0)
        return MemoryEvidence(
            memory_id=(
                str(meta.get("id") or meta.get("entry_hash"))
                if meta.get("id") or meta.get("entry_hash")
                else None
            ),
            text=str(meta.get("text", ""))[:500],
            timestamp=meta.get("timestamp"),
            emotion=str(meta.get("emotion", "neutral")),
            relevance=max(0.0, min(1.0, float(combined or 0.0))),
            topics=[str(topic) for topic in meta.get("topics", [])[:5]],
        )

    @staticmethod
    def _goal_evidence(item: dict[str, Any]) -> EvidenceItem:
        label = str(item.get("goal_keyword", "goal")).replace("_", " ")
        explanation = str(item.get("explanation") or f"Goal: {label}")
        return EvidenceItem(
            source="goal",
            summary=explanation,
            confidence=max(0.0, min(1.0, float(item.get("confidence", 0.0)))),
            metadata={"goal_keyword": item.get("goal_keyword")},
        )

    @staticmethod
    def _decision_tokens(state: DecisionState) -> set[str]:
        return _tokens(
            " ".join(
                [
                    state.problem,
                    state.desired_outcome or "",
                    *state.available_options,
                    *state.relevant_goals,
                ]
            )
        )

    @staticmethod
    def _retrieval_query(state: DecisionState) -> str:
        parts = [state.problem, *state.available_options, *state.relevant_goals]
        return " ".join(" ".join(parts).split())[:2_000]

    @staticmethod
    def _content_hash(text: str) -> str:
        normalized = " ".join(str(text).strip().lower().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    @staticmethod
    def _evidence_strength(
        state: DecisionState,
        memories: list[MemoryEvidence],
        patterns: list[EvidenceItem],
        profile: dict[str, Any] | None,
        goals: list[EvidenceItem],
        recent: list[EvidenceItem],
    ) -> float:
        history = min(1.0, (len(memories) + len(recent)) / 3.0) * 0.35
        pattern_score = max((p.confidence for p in patterns), default=0.0) * 0.20
        profile_score = 0.10 if profile else 0.0
        goal_score = max((g.confidence for g in goals), default=0.0) * 0.10
        completeness_parts = (
            int(len(state.available_options) >= 2),
            int(bool(state.desired_outcome or state.relevant_goals)),
            int(bool(state.constraints)),
        )
        completeness = sum(completeness_parts) / len(completeness_parts) * 0.25
        return round(
            min(
                1.0,
                history + pattern_score + profile_score + goal_score + completeness,
            ),
            4,
        )
