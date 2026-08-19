from __future__ import annotations

from backend.emotion.detector import EmotionDetector


class _FakePipe:
    """Mimics a HF text-classification pipeline with top_k=None output."""

    def __init__(self, rows):
        self._rows = rows

    def __call__(self, text, **kwargs):
        return [self._rows]


def _detector(rows):
    d = EmotionDetector("fake-model")
    d._classifier = _FakePipe(rows)
    return d


def test_returns_top_emotion_and_ranked_spread():
    rows = [
        {"label": "gratitude", "score": 0.1},
        {"label": "pride", "score": 0.7},
        {"label": "joy", "score": 0.2},
    ]
    out = _detector(rows).detect("I'm so proud")
    assert out["emotion"] == "pride"
    assert out["confidence"] == 0.7
    # Ranked descending, top label first.
    labels = [e["emotion"] for e in out["all_emotions"]]
    assert labels[0] == "pride"
    assert set(labels) == {"pride", "joy", "gratitude"}


def test_top_n_limit():
    rows = [{"label": f"e{i}", "score": i / 100} for i in range(20)]
    out = EmotionDetector("fake-model", top_n=5)
    out._classifier = _FakePipe(rows)
    res = out.detect("x")
    assert len(res["all_emotions"]) == 5


def test_empty_text_fallback():
    out = _detector([{"label": "joy", "score": 1.0}]).detect("")
    assert out == {"emotion": "neutral", "confidence": 0.0, "all_emotions": []}


def test_label_normalized_lowercase():
    out = _detector([{"label": "Nervousness", "score": 0.9}]).detect("worried")
    assert out["emotion"] == "nervousness"


def test_fine_grained_labels_supported():
    # Labels outside the old 5-set are preserved, not collapsed.
    out = _detector([{"label": "disgust", "score": 0.8}]).detect("gross")
    assert out["emotion"] == "disgust"  # previously collapsed to "anger"


def test_rule_based_detector_avoids_hf_pipeline():
    out = EmotionDetector("rule-based").detect("I feel hopeful but stressed.")
    assert out["emotion"] == "optimism"
    assert out["all_emotions"]
