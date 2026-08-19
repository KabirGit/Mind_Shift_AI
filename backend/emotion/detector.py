from __future__ import annotations

import logging
import threading
from typing import Any

logger = logging.getLogger(__name__)


class EmotionDetector:
    """
    Standalone emotion detector.

    Defaults to the GoEmotions taxonomy (28 fine-grained emotions) via
    `SamLowe/roberta-base-go_emotions`, but works with any HF
    text-classification model (the label set adapts to whatever the model emits).

    Output contract (stable):
    {
      "emotion": str,        # top predicted label, lowercased
      "confidence": float,   # score of the top label, 0..1
      "all_emotions": [      # additive: full ranked spread (top N)
          {"emotion": str, "score": float}, ...
      ]
    }
    """

    # GoEmotions (SamLowe/roberta-base-go_emotions) label set, for reference.
    GO_EMOTIONS = {
        "admiration", "amusement", "anger", "annoyance", "approval", "caring",
        "confusion", "curiosity", "desire", "disappointment", "disapproval",
        "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
        "joy", "love", "nervousness", "optimism", "pride", "realization",
        "relief", "remorse", "sadness", "surprise", "neutral",
    }

    # Number of ranked emotions to surface in `all_emotions`.
    _TOP_N = 5

    def __init__(self, model_name: str, top_n: int = _TOP_N) -> None:
        self.model_name = model_name
        self.top_n = top_n
        self._classifier = None
        self._lock = threading.Lock()

    def _get_classifier(self):
        if self._classifier is None:
            try:
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "transformers is not installed; use EMOTION_MODEL=rule-based "
                    "or install requirements-ml.txt"
                ) from exc

            # top_k=None returns scores for every label so we can rank them.
            self._classifier = pipeline(
                "text-classification",
                model=self.model_name,
                top_k=None,
            )
        return self._classifier

    @property
    def _uses_rule_based(self) -> bool:
        return self.model_name.lower() in {"rule-based", "rules", "lightweight"}

    def _detect_rule_based(self, text: str) -> dict[str, Any]:
        lowered = text.lower()
        rules = [
            ("joy", {"happy", "great", "good", "joy", "excited", "proud", "grateful"}),
            ("optimism", {"hope", "hopeful", "better", "improving", "progress"}),
            ("fear", {"worried", "anxious", "scared", "afraid", "nervous"}),
            ("sadness", {"sad", "tired", "lonely", "upset", "heavy"}),
            ("anger", {"angry", "frustrated", "annoyed", "furious"}),
            ("stress", {"stress", "deadline", "pressure", "overwhelmed"}),
        ]
        matches = []
        for label, words in rules:
            count = sum(1 for word in words if word in lowered)
            if count:
                matches.append((label, min(0.95, 0.55 + count * 0.1)))
        if not matches:
            matches = [("neutral", 0.65)]
        ranked = [
            {"emotion": label, "score": round(score, 4)}
            for label, score in sorted(matches, key=lambda item: item[1], reverse=True)
        ]
        top = ranked[0]
        return {
            "emotion": top["emotion"],
            "confidence": top["score"],
            "all_emotions": ranked[: self.top_n],
        }

    @staticmethod
    def _normalize_label(label: str) -> str:
        return str(label).strip().lower()

    @staticmethod
    def _coerce_scores(raw: Any) -> list[dict[str, Any]]:
        """Normalize the various shapes a HF pipeline may return into a flat
        list of {"label": str, "score": float} dicts.
        """
        # With top_k=None and a single string, pipelines typically return
        # [[{label, score}, ...]]; older/other configs may return [{...}].
        if isinstance(raw, list) and raw and isinstance(raw[0], list):
            return raw[0]
        if isinstance(raw, list):
            return raw
        if isinstance(raw, dict):
            return [raw]
        return []

    def detect(self, text: str) -> dict[str, Any]:
        fallback = {"emotion": "neutral", "confidence": 0.0, "all_emotions": []}
        if not text or not text.strip():
            return fallback

        try:
            if self._uses_rule_based:
                return self._detect_rule_based(text)

            classifier = self._get_classifier()
            with self._lock:
                # truncation guards against inputs longer than the model's limit.
                raw = classifier(text, truncation=True)

            scores = self._coerce_scores(raw)
            if not scores:
                return fallback

            ranked: list[dict[str, Any]] = sorted(
                (
                    {
                        "emotion": self._normalize_label(item.get("label", "neutral")),
                        "score": round(float(item.get("score", 0.0)), 4),
                    }
                    for item in scores
                ),
                key=lambda d: float(d["score"]),
                reverse=True,
            )

            top = ranked[0]
            return {
                "emotion": top["emotion"],
                "confidence": top["score"],
                "all_emotions": ranked[: self.top_n],
            }
        except Exception as exc:
            logger.exception("Emotion detection failed, using rule-based fallback: %s", exc)
            return self._detect_rule_based(text)
