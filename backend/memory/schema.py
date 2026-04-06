from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, UTC


@dataclass
class MemoryEntry:
    text: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat().replace("+00:00", "Z")
    )
    emotion: str = "neutral"
    emotion_intensity: float = 0.0
    tags: list[str] = field(default_factory=list)

    def to_metadata(self) -> dict:
        return {
            "text": self.text,
            "timestamp": self.timestamp,
            "emotion": self.emotion,
            "emotion_intensity": float(self.emotion_intensity),
            # Compatibility alias for downstream consumers expecting "score".
            "emotion_score": float(self.emotion_intensity),
            "tags": self.tags,
        }
