from __future__ import annotations

import logging
from typing import Any

from backend.orchestrator.packet import IntelligencePacket

logger = logging.getLogger(__name__)


class Orchestrator:
    """Collects already-computed engine outputs into one IntelligencePacket.

    Does NOT re-run engines; it assembles results passed in by the caller. Any
    individual piece failing degrades the packet gracefully (partial is fine).
    """

    def assemble(
        self,
        *,
        text: str,
        emotion_result: dict,
        sentiment: float,
        insights: list[str] | None = None,
        reflection_prompts: list[str] | None = None,
        triggers: list[Any] | None = None,
        habits: list[Any] | None = None,
        relationships: list[Any] | None = None,
        user_profile: Any | None = None,
        proactive_alerts: list[str] | None = None,
        temporal_patterns: list[Any] | None = None,
        causal_links: list[Any] | None = None,
        predictions: dict | None = None,
        goals: list[Any] | None = None,
        memory_replay: dict | None = None,
    ) -> IntelligencePacket:
        def _safe(value, default):
            return value if value is not None else default

        try:
            return IntelligencePacket(
                current_entry_emotion=emotion_result.get("emotion", "neutral"),
                current_entry_sentiment=sentiment,
                insights=_safe(insights, []),
                reflection_prompts=_safe(reflection_prompts, []),
                triggers=_safe(triggers, []),
                habits=_safe(habits, []),
                relationships=_safe(relationships, []),
                user_profile=user_profile,
                proactive_alerts=_safe(proactive_alerts, []),
                temporal_patterns=_safe(temporal_patterns, []),
                causal_links=_safe(causal_links, []),
                predictions=_safe(predictions, {}),
                goals=_safe(goals, []),
                memory_replay=memory_replay,
            )
        except Exception as exc:
            logger.exception("Orchestrator.assemble failed: %s", exc)
            # Minimal valid packet so the pipeline can still run.
            return IntelligencePacket(
                current_entry_emotion=emotion_result.get("emotion", "neutral"),
                current_entry_sentiment=sentiment,
            )
