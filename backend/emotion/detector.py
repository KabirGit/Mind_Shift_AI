from __future__ import annotations

import logging
import threading
from typing import Any

from transformers import pipeline

logger = logging.getLogger(__name__)


class EmotionDetector:
    """
    Standalone emotion detector.
    Output contract:
    {
      "emotion": str,   # sadness|joy|anger|fear|neutral
      "confidence": float
    }
    """

    _SUPPORTED = {"sadness", "joy", "anger", "fear", "neutral"}

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._classifier = None
        self._lock = threading.Lock()

    def _get_classifier(self):
        if self._classifier is None:
            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
                return_all_scores=False,
            )
        return self._classifier

    def _normalize_label(self, label: str) -> str:
        normalized = label.strip().lower()
        if normalized in self._SUPPORTED:
            return normalized
        if normalized in {"happiness", "love"}:
            return "joy"
        if normalized in {"disgust"}:
            return "anger"
        if normalized in {"surprise"}:
            return "neutral"
        return "neutral"

    def detect(self, text: str) -> dict[str, Any]:
        # Fallback output keeps downstream schema stable.
        fallback = {"emotion": "neutral", "confidence": 0.0}
        if not text or not text.strip():
            return fallback

        try:
            classifier = self._get_classifier()
            with self._lock:
                result = classifier(text)[0]
            emotion = self._normalize_label(str(result.get("label", "neutral")))
            confidence = float(result.get("score", 0.0))
            return {"emotion": emotion, "confidence": round(confidence, 4)}
        except Exception as exc:
            logger.exception("Emotion detection failed, using fallback: %s", exc)
            return fallback
