from __future__ import annotations

import json
import logging
from typing import Any


def log_stage(stage: str, payload: dict[str, Any], logger_name: str = "pipeline") -> None:
    logger = logging.getLogger(logger_name)
    logger.info("%s | %s", stage, json.dumps(payload, ensure_ascii=True, default=str))
