from __future__ import annotations

from collections import Counter
from typing import Any


class PromptBuilder:
    def build(
        self,
        user_text: str,
        current_emotion: dict[str, Any],
        retrieved_memories: list[dict[str, Any]],
        recent_history: list[dict[str, str]],
        insights: list[str] | None = None,
        reflection_prompts: list[str] | None = None,
        packet: Any | None = None,
    ) -> str:
        emotion_name = current_emotion.get("emotion", "neutral")
        emotion_conf = current_emotion.get("confidence", 0.0)
        memory_lines = self._format_memory_lines(retrieved_memories)
        pattern_lines = self._detect_patterns(retrieved_memories)
        history_lines = self._format_history(recent_history)

        insights_block = ""
        if insights:
            insight_lines = "\n".join(f"- {s}" for s in insights)
            insights_block = (
                "Long-term patterns you've noticed about yourself:\n"
                f"{insight_lines}\n\n"
            )

        intelligence_block = self._intelligence_block(packet)
        replay_block = self._replay_block(packet)

        reflection_block = ""
        if reflection_prompts:
            reflection_lines = "\n".join(f"- {s}" for s in reflection_prompts)
            reflection_block = (
                "Optional reflective questions (use at most one, gently, only if it "
                "fits naturally):\n"
                f"{reflection_lines}\n\n"
            )

        return (
            "You are a journaling companion. Write empathetic, calm, and human-like responses.\n"
            "Safety and style rules:\n"
            "- Validate feelings without judging.\n"
            "- Offer gentle reflection and optional small next step.\n"
            "- Avoid toxic positivity and absolute claims.\n"
            "- Keep tone consistent, warm, and concise.\n\n"
            f"Current user emotion: {emotion_name} (confidence={emotion_conf})\n\n"
            "Detected patterns:\n"
            f"{pattern_lines}\n\n"
            f"{insights_block}"
            f"{intelligence_block}"
            f"{replay_block}"
            f"{reflection_block}"
            "Relevant past entries:\n"
            f"{memory_lines}\n\n"
            "Recent chat history:\n"
            f"{history_lines}\n\n"
            f"Current user message:\n{user_text}\n\n"
            "Respond in 4-7 sentences. Include emotional validation and one advice or suggestion and avoid user from taking extreme actions."
        )

    def _intelligence_block(self, packet: Any | None) -> str:
        if packet is None:
            return ""
        profile = getattr(packet, "user_profile", None)
        if profile is None:
            return ""
        triggers = ", ".join(profile.top_triggers) if profile.top_triggers else "none yet"
        return (
            "Intelligence Context\n"
            f"User Profile Snapshot: {profile.entry_count} entries. "
            f"Growth score: {profile.growth_score:.2f}. "
            f"Dominant emotion: {profile.dominant_emotion}.\n"
            f"Recovery speed: {profile.recovery_speed_days:.1f} days. "
            f"Top triggers: {triggers}. "
            f"Communication style: {profile.communication_style}.\n\n"
        )

    def _replay_block(self, packet: Any | None) -> str:
        if packet is None:
            return ""
        replay = getattr(packet, "memory_replay", None)
        if not replay:
            return ""
        text = str(replay.get("similar_entry_text", ""))[:120]
        emotion = replay.get("similar_entry_emotion", "neutral")
        days_ago = replay.get("days_ago", "?")
        hint = replay.get("recovery_hint", "")
        return (
            f"Memory Replay — Similar past experience ({days_ago} days ago, "
            f"emotion: {emotion}):\n"
            f'"{text}..."\n'
            f"What happened next: {hint} "
            "(Reference this gently if it feels relevant to the user's current situation.)\n\n"
        )

    def _format_memory_lines(self, retrieved_memories: list[dict[str, Any]]) -> str:
        if not retrieved_memories:
            return "- No prior memory retrieved."
        lines: list[str] = []
        for item in retrieved_memories[:5]:
            meta = item.get("metadata", {})
            text = str(meta.get("text", "")).strip().replace("\n", " ")
            emotion = meta.get("emotion", "neutral")
            timestamp = meta.get("timestamp", "unknown_time")
            snippet = text[:160] + ("..." if len(text) > 160 else "")
            lines.append(f"- You previously mentioned ({timestamp}, {emotion}): {snippet}")
        return "\n".join(lines)

    def _detect_patterns(self, retrieved_memories: list[dict[str, Any]]) -> str:
        if not retrieved_memories:
            return "- Not enough memory to infer patterns yet."

        emotions = []
        triggers = []
        for item in retrieved_memories:
            meta = item.get("metadata", {})
            emotions.append(str(meta.get("emotion", "neutral")))
            text = str(meta.get("text", ""))
            if "when " in text.lower():
                fragment = text.lower().split("when ", 1)[1][:80].strip()
                if fragment:
                    triggers.append(fragment)

        top_emotion = Counter(emotions).most_common(1)[0][0]
        lines = [f"- You often feel {top_emotion} in recent memories."]
        if triggers:
            lines.append(f"- Recurring trigger hint: when {Counter(triggers).most_common(1)[0][0]}")
        else:
            lines.append("- Trigger pattern: no strong recurring 'when ...' trigger detected.")
        return "\n".join(lines)

    def _format_history(self, recent_history: list[dict[str, str]]) -> str:
        if not recent_history:
            return "- No prior chat turns in this session."
        lines = []
        for turn in recent_history[-6:]:
            role = turn.get("role", "user").capitalize()
            text = turn.get("content", "").replace("\n", " ")
            lines.append(f"- {role}: {text[:120]}")
        return "\n".join(lines)
