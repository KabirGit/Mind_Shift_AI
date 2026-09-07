from __future__ import annotations

import json
import logging
from typing import Any

_SENSITIVE_KEY_PARTS = (
    "content",
    "context",
    "input",
    "journal",
    "matched",
    "memory",
    "message",
    "prompt",
    "response",
    "result",
    "text",
)
_REDACTED = "<redacted>"


def sanitize_for_logging(value: Any, *, parent_key: str = "") -> Any:
    """Recursively retain operational metadata while removing private text."""

    normalized_key = parent_key.casefold()
    if any(part in normalized_key for part in _SENSITIVE_KEY_PARTS):
        if value is None:
            return None
        if isinstance(value, (list, tuple, set, dict)):
            return {"redacted": True, "count": len(value)}
        return _REDACTED
    if isinstance(value, dict):
        return {
            str(key): sanitize_for_logging(item, parent_key=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitize_for_logging(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return type(value).__name__


def log_stage(stage: str, payload: dict[str, Any], logger_name: str = "pipeline") -> None:
    logger = logging.getLogger(logger_name)
    safe_payload = sanitize_for_logging(payload)
    logger.info(
        "%s | %s", stage, json.dumps(safe_payload, ensure_ascii=True, default=str)
    )
