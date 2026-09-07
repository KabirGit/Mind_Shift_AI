from __future__ import annotations

from backend.config.debug import sanitize_for_logging


def test_recursive_log_sanitizer_removes_private_text_and_keeps_metrics() -> None:
    private = "I want to resign after private feedback"
    payload = {
        "trace_id": "trace-1",
        "latency_ms": 10,
        "nested": {
            "prompt_preview": private,
            "memories": [{"text": private, "score": 0.8}],
            "response": private,
        },
    }

    safe = sanitize_for_logging(payload)

    assert private not in str(safe)
    assert safe["trace_id"] == "trace-1"
    assert safe["latency_ms"] == 10
    assert safe["nested"]["prompt_preview"] == "<redacted>"
