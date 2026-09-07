from __future__ import annotations

import time
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

LLMPurpose = Literal[
    "decision_parse",
    "guidance_response",
    "reflection",
    "legacy",
]
LLMResponseFormat = Literal["text", "json_object"]


class LLMRequest(BaseModel):
    """One logical model request, independent from transport attempts."""

    purpose: LLMPurpose
    system_prompt: str = Field(min_length=1, max_length=4_000)
    user_prompt: str = Field(min_length=1, max_length=30_000)
    prompt_version: str = Field(min_length=1, max_length=80)
    response_format: LLMResponseFormat = "text"
    temperature: float = Field(default=0.2, ge=0.0, le=1.5)
    max_tokens: int = Field(default=220, ge=1, le=4_096)


class TokenUsage(BaseModel):
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)


class LLMCallResult(BaseModel):
    """Sanitized result metadata returned by every model boundary."""

    text: str = ""
    success: bool = False
    purpose: LLMPurpose
    prompt_version: str
    requested_model: str = "unknown"
    actual_model: str | None = None
    status_code: int | None = None
    attempts: int = Field(default=0, ge=0, le=2)
    retry_count: int = Field(default=0, ge=0, le=1)
    latency_ms: float = Field(default=0.0, ge=0.0)
    finish_reason: str | None = None
    usage: TokenUsage = Field(default_factory=TokenUsage)
    failure_category: str | None = None
    error_type: str | None = None


class CompletionClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMCallResult: ...


def invoke_completion(client: Any, request: LLMRequest) -> LLMCallResult:
    """Invoke a typed client while preserving legacy injected ``generate`` fakes."""

    complete = getattr(client, "complete", None)
    if callable(complete):
        try:
            result = complete(request)
        except Exception as exc:
            return _invalid_result(request, "client_error", type(exc).__name__)
        if isinstance(result, LLMCallResult):
            return result
        return _invalid_result(request, "invalid_client_result", type(result).__name__)

    generate = getattr(client, "generate", None)
    if not callable(generate):
        return _invalid_result(request, "invalid_client", "MissingGenerateMethod")

    started = time.perf_counter()
    try:
        text = generate(request.user_prompt)
    except Exception as exc:
        return LLMCallResult(
            purpose=request.purpose,
            prompt_version=request.prompt_version,
            attempts=1,
            latency_ms=(time.perf_counter() - started) * 1000,
            failure_category="client_error",
            error_type=type(exc).__name__,
        )
    latency_ms = (time.perf_counter() - started) * 1000
    if not isinstance(text, str) or not text.strip():
        return LLMCallResult(
            purpose=request.purpose,
            prompt_version=request.prompt_version,
            attempts=1,
            latency_ms=latency_ms,
            failure_category="invalid_response",
            error_type="EmptyText",
        )
    return LLMCallResult(
        text=text.strip(),
        success=True,
        purpose=request.purpose,
        prompt_version=request.prompt_version,
        attempts=1,
        latency_ms=latency_ms,
    )


def failed_result(
    result: LLMCallResult,
    failure_category: str,
    error_type: str,
) -> LLMCallResult:
    """Turn a transport success into a visible application-validation failure."""

    return result.model_copy(
        update={
            "success": False,
            "failure_category": failure_category,
            "error_type": error_type,
        }
    )


def _invalid_result(
    request: LLMRequest,
    failure_category: str,
    error_type: str,
) -> LLMCallResult:
    return LLMCallResult(
        purpose=request.purpose,
        prompt_version=request.prompt_version,
        failure_category=failure_category,
        error_type=error_type,
    )
