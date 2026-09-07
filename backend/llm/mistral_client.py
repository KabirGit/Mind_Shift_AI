from __future__ import annotations

import logging
import time
from collections.abc import Callable
from contextlib import suppress
from typing import Any

import requests

from backend.llm.models import LLMCallResult, LLMRequest, TokenUsage

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class MistralInferenceClient:
    """Typed Mistral chat-completions boundary with one bounded retry."""

    endpoint = "https://api.mistral.ai/v1/chat/completions"
    provider_name = "Mistral"

    def __init__(
        self,
        model_name: str,
        api_token: str | None,
        max_new_tokens: int = 220,
        timeout_s: float | None = None,
        temperature: float = 0.2,
        *,
        connect_timeout_s: float = 5.0,
        read_timeout_s: float = 30.0,
        max_attempts: int = 2,
        retry_backoff_s: float = 0.25,
        session: Any | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model_name = model_name
        self.api_token = api_token
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.connect_timeout_s = max(0.1, float(connect_timeout_s))
        self.read_timeout_s = max(
            0.1, float(timeout_s if timeout_s is not None else read_timeout_s)
        )
        self.max_attempts = min(2, max(1, int(max_attempts)))
        self.retry_backoff_s = max(0.0, float(retry_backoff_s))
        self.session = session or requests.Session()
        self.sleep_fn = sleep_fn

    def complete(self, request: LLMRequest) -> LLMCallResult:
        started = time.perf_counter()
        if not self.api_token:
            return self._failure(
                request,
                started,
                attempts=0,
                category="configuration",
                error_type="MissingAPIKey",
            )

        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.response_format == "json_object":
            payload["response_format"] = {"type": "json_object"}

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        attempts = 0
        while attempts < self.max_attempts:
            attempts += 1
            try:
                response = self.session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=(self.connect_timeout_s, self.read_timeout_s),
                )
                status_code = int(response.status_code)
                if status_code in _RETRYABLE_STATUS_CODES and attempts < self.max_attempts:
                    self._sleep_before_retry(response, attempts)
                    continue
                response.raise_for_status()
                return self._success_result(request, response, started, attempts)
            except requests.Timeout as exc:
                if attempts < self.max_attempts:
                    self._sleep_before_retry(None, attempts)
                    continue
                return self._failure(
                    request,
                    started,
                    attempts,
                    "timeout",
                    type(exc).__name__,
                )
            except requests.ConnectionError as exc:
                if attempts < self.max_attempts:
                    self._sleep_before_retry(None, attempts)
                    continue
                return self._failure(
                    request,
                    started,
                    attempts,
                    "network",
                    type(exc).__name__,
                )
            except requests.HTTPError as exc:
                status_code = getattr(exc.response, "status_code", None)
                category = self._http_failure_category(status_code)
                return self._failure(
                    request,
                    started,
                    attempts,
                    category,
                    type(exc).__name__,
                    status_code=status_code,
                )
            except (KeyError, TypeError, ValueError) as exc:
                return self._failure(
                    request,
                    started,
                    attempts,
                    "invalid_response",
                    type(exc).__name__,
                )
            except requests.RequestException as exc:
                return self._failure(
                    request,
                    started,
                    attempts,
                    "client_error",
                    type(exc).__name__,
                )

        return self._failure(
            request,
            started,
            attempts,
            "client_error",
            "UnexpectedAttemptState",
        )

    def generate(self, prompt: str) -> str:
        """Backward-compatible string API for legacy callers."""

        request = LLMRequest(
            purpose="legacy",
            system_prompt="You are an empathetic journaling assistant.",
            user_prompt=prompt,
            prompt_version="legacy-v1",
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        result = self.complete(request)
        if result.success:
            return result.text
        return "I'm here with you. Tell me more about what you're feeling."

    def _success_result(
        self,
        request: LLMRequest,
        response: Any,
        started: float,
        attempts: int,
    ) -> LLMCallResult:
        data = response.json()
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("missing choices")
        choice = choices[0]
        if not isinstance(choice, dict) or not isinstance(choice.get("message"), dict):
            raise ValueError("invalid choice shape")
        content = self._extract_content(choice["message"].get("content"))
        if not content:
            raise ValueError("empty completion")

        usage_data = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        usage = TokenUsage(
            prompt_tokens=self._optional_int(usage_data.get("prompt_tokens")),
            completion_tokens=self._optional_int(usage_data.get("completion_tokens")),
            total_tokens=self._optional_int(usage_data.get("total_tokens")),
        )
        return LLMCallResult(
            text=content,
            success=True,
            purpose=request.purpose,
            prompt_version=request.prompt_version,
            requested_model=self.model_name,
            actual_model=str(data.get("model")) if data.get("model") else None,
            status_code=int(response.status_code),
            attempts=attempts,
            retry_count=attempts - 1,
            latency_ms=(time.perf_counter() - started) * 1000,
            finish_reason=(
                str(choice.get("finish_reason")) if choice.get("finish_reason") else None
            ),
            usage=usage,
        )

    @staticmethod
    def _extract_content(content: Any) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts).strip()
        return ""

    def _failure(
        self,
        request: LLMRequest,
        started: float,
        attempts: int,
        category: str,
        error_type: str,
        *,
        status_code: int | None = None,
    ) -> LLMCallResult:
        logger.warning(
            "%s request failed category=%s error_type=%s status=%s attempts=%s",
            self.provider_name,
            category,
            error_type,
            status_code,
            attempts,
        )
        return LLMCallResult(
            purpose=request.purpose,
            prompt_version=request.prompt_version,
            requested_model=self.model_name,
            status_code=status_code,
            attempts=attempts,
            retry_count=max(0, attempts - 1),
            latency_ms=(time.perf_counter() - started) * 1000,
            failure_category=category,
            error_type=error_type,
        )

    def _sleep_before_retry(self, response: Any | None, attempt: int) -> None:
        delay = self.retry_backoff_s * attempt
        if response is not None:
            raw = response.headers.get("Retry-After")
            with suppress(TypeError, ValueError):
                delay = min(2.0, max(0.0, float(raw))) if raw is not None else delay
        if delay:
            self.sleep_fn(delay)

    @staticmethod
    def _http_failure_category(status_code: int | None) -> str:
        if status_code in {401, 403}:
            return "authentication"
        if status_code == 429:
            return "rate_limit"
        if status_code in {500, 502, 503, 504}:
            return "provider_unavailable"
        if status_code is not None and 400 <= status_code < 500:
            return "invalid_request"
        return "provider_error"

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed >= 0 else None
